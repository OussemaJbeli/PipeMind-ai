from app.config import settings
from app.core.errors import LLMRateLimited, LLMTimeout, LLMUnavailable
from app.core.logging import log
from app.providers.base import LLMProvider, LLMResponse
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.providers.stub import StubProvider


class FallbackProvider:
    """Primary provider, with a secondary used only when a retry could help.

    Never falls back silently: `analyses.model_provider` records which provider
    actually answered, and the UI shows it. An analysis produced by a 3B local
    model when the user expected Gemini is not wrong — presenting it as though
    nothing changed is.
    """

    name = "fallback"

    def __init__(self, primary: LLMProvider, secondary: LLMProvider) -> None:
        self._primary = primary
        self._secondary = secondary

    async def complete(self, **kwargs) -> LLMResponse:
        try:
            return await self._primary.complete(**kwargs)
        except (LLMUnavailable, LLMTimeout, LLMRateLimited) as exc:
            # Deliberately NOT InvalidLLMResponse (a second model will likely
            # fail the same way on the same input) or BudgetExceeded (the point
            # of a budget is that it stops you).
            log.warning(
                "llm.fallback",
                primary=self._primary.name,
                secondary=self._secondary.name,
                reason=type(exc).__name__,
            )

            response = await self._secondary.complete(**kwargs)
            response.raw["fell_back_from"] = self._primary.name

            return response

    async def health(self) -> bool:
        return await self._primary.health() or await self._secondary.health()

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return self._primary.cost(prompt_tokens, completion_tokens)


def build_provider(override: dict | None = None) -> LLMProvider:
    """Resolve a provider from the team's ai_providers row, else env defaults.

    `override` is sent per request by Laravel so this service holds no team's
    credentials at rest.
    """
    config = settings()
    options = override or {}
    kind = options.get("provider") or config.llm_provider

    if kind == "stub":
        if config.app_env not in {"local", "testing"}:
            raise LLMUnavailable("The stub provider is not permitted outside local/testing.")

        return StubProvider()

    if kind == "gemini":
        return GeminiProvider(
            api_key=options.get("api_key") or config.gemini_api_key or "",
            model=options.get("model") or config.gemini_model,
            input_cost_per_1k=float(options.get("input_cost_per_1k", 0.0)),
            output_cost_per_1k=float(options.get("output_cost_per_1k", 0.0)),
        )

    if kind == "ollama":
        return OllamaProvider(
            base_url=options.get("base_url") or config.ollama_base_url,
            model=options.get("model") or config.ollama_model,
        )

    if kind in {"openai", "openai_compatible", "azure_openai"}:
        return OpenAICompatibleProvider(
            api_key=options.get("api_key") or config.openai_api_key or "",
            model=options.get("model") or config.openai_model,
            base_url=options.get("base_url") or config.openai_base_url,
            input_cost_per_1k=float(options.get("input_cost_per_1k", 0.0)),
            output_cost_per_1k=float(options.get("output_cost_per_1k", 0.0)),
        )

    raise LLMUnavailable(f"Unknown provider: {kind}")


def build_provider_with_fallback(override: dict | None = None) -> LLMProvider:
    primary = build_provider(override)
    fallback_config = (override or {}).get("fallback")

    if not fallback_config:
        return primary

    return FallbackProvider(primary, build_provider(fallback_config))


def available_providers() -> dict[str, bool]:
    """Which providers this deployment could actually use right now.

    Surfaced on /v1/info so a misconfigured key shows up as a red dot in the UI
    rather than as a failed analysis an hour later.
    """
    config = settings()

    return {
        "stub": config.app_env in {"local", "testing"},
        "gemini": bool(config.gemini_api_key),
        "openai": bool(config.openai_api_key),
        "ollama": bool(config.ollama_base_url),
    }
