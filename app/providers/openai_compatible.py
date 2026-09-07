"""One implementation for every provider that speaks the OpenAI chat schema.

Covers OpenAI, Azure OpenAI, vLLM, LM Studio, OpenRouter — and Ollama's /v1
endpoint, though OllamaProvider adds the leniency local models need.
"""

import json
import time
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.errors import (
    InvalidLLMResponse,
    LLMAuthenticationFailed,
    LLMRateLimited,
    LLMTimeout,
    LLMUnavailable,
)
from app.providers.base import LLMResponse, ProviderCheck


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0,
    ) -> None:
        self._api_key = api_key or "not-needed"
        self._model = model
        self._base_url = base_url
        self._input_cost = input_cost_per_1k
        self._output_cost = output_cost_per_1k
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise LLMUnavailable("openai is not installed. Install the 'llm' extra.") from exc

            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

        return self._client

    @retry(
        retry=retry_if_exception_type((LLMRateLimited, LLMTimeout)),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: dict | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 90,
    ) -> LLMResponse:
        started = time.perf_counter()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "analysis", "schema": json_schema, "strict": True},
            }

        try:
            response = await self._get_client().chat.completions.create(**kwargs)
        except Exception as exc:
            raise self._translate(exc) from exc

        choice = response.choices[0]
        text = choice.message.content or ""
        parsed = None

        if json_schema:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise InvalidLLMResponse(f"non-JSON response: {text[:200]}") from exc

        usage = response.usage

        return LLMResponse(
            text=text,
            parsed=parsed,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=self._model,
            provider=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=choice.finish_reason,
        )

    @staticmethod
    def _translate(exc: Exception) -> Exception:
        message = str(exc).lower()

        # Authentication first, and never on a bare substring: see the note in
        # gemini.py about "rate" matching inside "GenerateContent".
        #
        # Non-retryable: a wrong key is a configuration mistake, and retrying it
        # through tenacity, the fallback provider and the queue buries the one
        # message that tells the user what to fix.
        if any(token in message for token in (
            "api key", "api_key", "401", "403", "authentication", "unauthorized",
            "invalid_api_key",
        )):
            return LLMAuthenticationFailed(f"Provider rejected the credentials: {exc}")

        if any(token in message for token in ("rate limit", "rate_limit", "429")):
            return LLMRateLimited(str(exc))

        if "timeout" in message:
            return LLMTimeout(str(exc))

        return LLMUnavailable(str(exc))

    async def health(self) -> bool:
        return (await self.verify()).ok

    async def verify(self) -> ProviderCheck:
        import time

        started = time.perf_counter()

        try:
            client = self._get_client()
            names = [m.id for m in (await client.models.list()).data]
        except Exception as exc:
            return ProviderCheck(
                ok=False, provider=self.name, model=self._model,
                message=str(self._translate(exc)),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        elapsed = int((time.perf_counter() - started) * 1000)

        # Self-hosted gateways often expose no model list at all; an empty list
        # is not evidence the model is missing, so only judge when we have one.
        if names and self._model not in names:
            return ProviderCheck(
                ok=False, provider=self.name, model=self._model, models=sorted(names),
                message=f"The key works, but '{self._model}' is not available to it.",
                latency_ms=elapsed,
            )

        return ProviderCheck(
            ok=True, provider=self.name, model=self._model, models=sorted(names),
            message=f"Connected. {len(names)} models available.", latency_ms=elapsed,
        )

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(
            prompt_tokens / 1000 * self._input_cost
            + completion_tokens / 1000 * self._output_cost,
            6,
        )
