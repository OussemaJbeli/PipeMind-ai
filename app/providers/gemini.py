"""Google Gemini.

Uses schema-constrained decoding: with `response_schema` set, invalid JSON is
impossible by construction, which removes an entire class of parse-and-retry
failure. Local models cannot offer that — see ollama.py.
"""

import json
import time
from typing import Any, cast

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.errors import (
    InvalidLLMResponse,
    LLMAuthenticationFailed,
    LLMRateLimited,
    LLMTimeout,
    LLMUnavailable,
)
from app.providers.base import LLMResponse, ProviderCheck


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0,
        thinking_level: str | None = None,
    ) -> None:
        if not api_key:
            raise LLMUnavailable("No Gemini API key configured.")

        self._api_key = api_key
        self._model = model
        self._input_cost = input_cost_per_1k
        self._output_cost = output_cost_per_1k
        self._thinking_level = thinking_level
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

        options: dict[str, Any] = dict(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_schema else "text/plain",
            response_schema=json_schema,
        )

        # Thinking tokens dominate latency on Gemini 3.x. They are billed as
        # output but never appear in the response, so an unbounded budget is a
        # cost and a delay with nothing to show for either.
        if self._thinking_level and self._thinking_level != "default":
            options["thinking_config"] = types.ThinkingConfig(
                # The SDK types this as a Literal; the value is validated by
                # config, so a cast is honest rather than a suppression.
                thinking_level=cast(Any, self._thinking_level),
            )

        config = types.GenerateContentConfig(**options)

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

        # Thinking tokens are billed as output but are absent from the response.
        # Counting only visible tokens would understate every cost figure in the
        # reports — by roughly 4x on this model at the default thinking level.
        completion = (getattr(usage, "candidates_token_count", 0) or 0) + (
            getattr(usage, "thoughts_token_count", 0) or 0
        )

        return LLMResponse(
            text=text,
            parsed=parsed,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=completion,
            model=self._model,
            provider=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _translate(exc: Exception) -> Exception:
        message = str(exc).lower()

        # Authentication is checked FIRST and with precise tokens.
        #
        # A bare `"rate" in message` test used to run ahead of this and matched
        # the substring inside "GenerateContent" — which appears in the method
        # name of every Gemini error. Every failure was therefore reported as a
        # rate limit, and rate limits are retryable, so a wrong API key was
        # retried three times with backoff and surfaced as the wrong problem.
        if any(token in message for token in (
            "access_token_type_unsupported", "api key", "api_key", "unauthenticated",
            "unauthorized", "permission_denied", "401", "403",
        )):
            if "access_token_type_unsupported" in message:
                # Google returns this for a MALFORMED key as well as a wrong-type
                # credential — a single stray character on an otherwise valid
                # "AQ.Ab8…" key produces exactly this reason. Do not tell the user
                # their key is the wrong kind: that sends them off to regenerate a
                # key that was fine, when the real fault was a truncated paste.
                return LLMAuthenticationFailed(
                    "Google rejected this credential. Check the key was copied whole — "
                    "a single extra or missing character produces this exact error. "
                    "Keys come from https://aistudio.google.com/apikey and look like "
                    "'AQ.Ab8…' or 'AIza…'."
                )

            return LLMAuthenticationFailed(f"Gemini rejected the credentials: {exc}")

        if any(token in message for token in (
            "rate limit", "rate_limit", "ratelimit", "429", "quota", "resource_exhausted",
        )):
            return LLMRateLimited(str(exc))

        if any(token in message for token in ("timeout", "deadline")):
            return LLMTimeout(str(exc))

        return LLMUnavailable(str(exc))

    async def health(self) -> bool:
        return (await self.verify()).ok

    async def verify(self) -> ProviderCheck:
        """Proves the credentials work by actually generating.

        ListModels alone is NOT sufficient and must never be the only check:
        Google lists models that `generateContent` then rejects with 404 "no
        longer available to new users". Verifying against the list reported
        "Connected" for a model that could not be called, which is exactly the
        false confidence this endpoint exists to prevent.

        So the list is fetched for the model picker, and a real one-token
        generation decides `ok`. Seven input tokens is a rounding error against
        shipping a provider that fails on its first real analysis.
        """
        from google.genai import types

        started = time.perf_counter()

        try:
            client = self._get_client()
            names = [
                m.name.removeprefix("models/")
                for m in await client.aio.models.list()
                if "generateContent" in (getattr(m, "supported_actions", None) or [])
            ]
        except Exception as exc:
            return ProviderCheck(
                ok=False, provider=self.name, model=self._model,
                message=str(self._translate(exc)),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        try:
            await client.aio.models.generate_content(
                model=self._model,
                contents="ok",
                config=types.GenerateContentConfig(max_output_tokens=300),
            )
        except Exception as exc:
            message = str(exc)
            usable = sorted(names)

            # Google's 404 names the replacement model. Passing that through is
            # far more useful than a generic "not available".
            if "no longer available" in message.lower():
                message = (
                    f"The key works, but '{self._model}' is retired for this key. "
                    f"{self._suggested(message)}"
                )
            else:
                message = str(self._translate(exc))

            return ProviderCheck(
                ok=False, provider=self.name, model=self._model, models=usable, message=message,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        return ProviderCheck(
            ok=True, provider=self.name, model=self._model, models=sorted(names),
            message=f"Connected. '{self._model}' answered.",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _suggested(message: str) -> str:
        """Pulls the replacement model out of Google's deprecation message."""
        import re

        match = re.search(r"use\s+models/([\w.\-]+)", message)

        return f"Google suggests '{match.group(1)}'." if match else "Pick another model."

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(
            prompt_tokens / 1000 * self._input_cost
            + completion_tokens / 1000 * self._output_cost,
            6,
        )
