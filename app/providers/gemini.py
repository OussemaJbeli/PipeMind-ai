"""Google Gemini.

Uses schema-constrained decoding: with `response_schema` set, invalid JSON is
impossible by construction, which removes an entire class of parse-and-retry
failure. Local models cannot offer that — see ollama.py.
"""

import json
import time
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.errors import InvalidLLMResponse, LLMRateLimited, LLMTimeout, LLMUnavailable
from app.providers.base import LLMResponse


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0,
    ) -> None:
        if not api_key:
            raise LLMUnavailable("No Gemini API key configured.")

        self._api_key = api_key
        self._model = model
        self._input_cost = input_cost_per_1k
        self._output_cost = output_cost_per_1k
        self._client: Any = None

    def _get_client(self) -> Any:
        # Imported lazily so the llm extra stays optional.
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise LLMUnavailable(
                    "google-genai is not installed. Install the 'llm' extra."
                ) from exc

            self._client = genai.Client(api_key=self._api_key)

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
        from google.genai import types

        started = time.perf_counter()
        client = self._get_client()

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_schema else "text/plain",
            response_schema=json_schema,
        )

        try:
            response = await client.aio.models.generate_content(
                model=self._model, contents=prompt, config=config
            )
        except Exception as exc:
            raise self._translate(exc) from exc

        text = response.text or ""
        parsed = None

        if json_schema:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise InvalidLLMResponse(
                    f"non-JSON response despite a schema: {text[:200]}"
                ) from exc

        usage = getattr(response, "usage_metadata", None)

        return LLMResponse(
            text=text,
            parsed=parsed,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            model=self._model,
            provider=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _translate(exc: Exception) -> Exception:
        message = str(exc).lower()

        if any(token in message for token in ("rate", "429", "quota", "resource_exhausted")):
            return LLMRateLimited(str(exc))

        if any(token in message for token in ("timeout", "deadline")):
            return LLMTimeout(str(exc))

        if any(token in message for token in ("api key", "401", "permission", "unauthenticated")):
            return LLMUnavailable(f"Gemini rejected the credentials: {exc}")

        return LLMUnavailable(str(exc))

    async def health(self) -> bool:
        try:
            await self.complete(system="Reply with ok.", prompt="ok", max_tokens=8, timeout=10)
            return True
        except Exception:
            return False

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(
            prompt_tokens / 1000 * self._input_cost
            + completion_tokens / 1000 * self._output_cost,
            6,
        )
