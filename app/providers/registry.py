from app.config import settings
from app.core.errors import LLMRateLimited, LLMTimeout, LLMUnavailable
from app.core.logging import log
from app.providers.base import LLMProvider, LLMResponse, ProviderCheck
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

    async def verify(self) -> ProviderCheck:
        """Reports on the PRIMARY, and says so when only the secondary answers.

        Silently reporting the secondary as healthy would tell the user their
        Gemini key works when it does not — the fallback exists to keep analyses
        running, not to hide a broken credential.
        """
        primary = await self._primary.verify()

        if primary.ok:
            return primary

        secondary = await self._secondary.verify()

        if not secondary.ok:
            return primary

        return ProviderCheck(
            ok=True,
            provider=f"{self._primary.name} → {self._secondary.name}",
            model=secondary.model,
            models=secondary.models,
            latency_ms=secondary.latency_ms,
            message=(
                f"{self._primary.name} is not usable ({primary.message}) — "
                f"analyses will run on {self._secondary.name} instead."
            ),
        )

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return self._primary.cost(prompt_tokens, completion_tokens)


def build_provider(override: dict | None = None, *, allow_env_fallback: bool = True) -> LLMProvider:
    """Resolve a provider from the team's ai_providers row, else env defaults.

    `override` is sent per request by Laravel so this service holds no team's
    credentials at rest.

    `allow_env_fallback=False` is for credential testing: falling back to the
    server's own key there would report "your key works" about a key the user
    never entered, which is worse than any error message.
    """
    config = settings()
    options = override or {}
    kind = options.get("provider") or config.llm_provider

    def credential(key: str, fallback: str | None) -> str:
        return options.get(key) or (fallback if allow_env_fallback else "") or ""

    if kind == "stub":
        if config.app_env not in {"local", "testing"}:
            raise LLMUnavailable("The stub provider is not permitted outside local/testing.")

        return StubProvider()

    if kind == "gemini":
        return GeminiProvider(
            api_key=credential("api_key", config.gemini_api_key),
            model=options.get("model") or config.gemini_model,
            input_cost_per_1k=float(options.get("input_cost_per_1k", 0.0)),
            output_cost_per_1k=float(options.get("output_cost_per_1k", 0.0)),
            thinking_level=options.get("thinking_level") or config.gemini_thinking_level,
        )

    if kind == "ollama":
        return OllamaProvider(
            base_url=options.get("base_url") or config.ollama_base_url,
            model=options.get("model") or config.ollama_model,
        )

    if kind in {"openai", "openai_compatible", "azure_openai"}:
        return OpenAICompatibleProvider(
            api_key=credential("api_key", config.openai_api_key),
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
