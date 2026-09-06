"""Local models via Ollama.

Ollama's JSON mode is a HINT, not a constraint — unlike Gemini's schema-
constrained decoding, where invalid JSON is impossible by construction. Local
models genuinely do wrap their output in prose or code fences, so the parser
here has to be forgiving where the cloud one can be strict.

That asymmetry is a real cost of the local option and belongs in the evaluation,
not hidden behind a shrug.
"""

import json
import re
import time
from typing import Any

from app.core.errors import InvalidLLMResponse, LLMTimeout, LLMUnavailable
from app.providers.base import LLMResponse

FENCED_JSON = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")
BRACED_JSON = re.compile(r"\{[\s\S]*\}")


def salvage_json(text: str) -> dict | None:
    """Recover JSON from a chattier model's output."""
    candidates: list[str] = []

    if match := FENCED_JSON.search(text):
        candidates.append(match.group(1))

    if match := BRACED_JSON.search(text):
        candidates.append(match.group(0))

    candidates.append(text.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed

    return None


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5-coder:7b") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

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
        import httpx

        started = time.perf_counter()

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        if json_schema:
            payload["format"] = "json"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self._base_url}/api/generate", json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"Ollama did not respond within {timeout}s.") from exc
        except Exception as exc:
            raise LLMUnavailable(f"Ollama is unreachable: {exc}") from exc

        text = body.get("response", "")
        parsed = None

        if json_schema:
            parsed = salvage_json(text)

            if parsed is None:
                raise InvalidLLMResponse(
                    f"{self._model} did not return usable JSON: {text[:200]}"
                )

        return LLMResponse(
            text=text,
            parsed=parsed,
            prompt_tokens=int(body.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(body.get("eval_count", 0) or 0),
            model=self._model,
            provider=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=body.get("done_reason"),
        )

    async def health(self) -> bool:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                return (await client.get(f"{self._base_url}/api/tags")).is_success
        except Exception:
            return False

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # Local inference has no marginal cost. That is its whole argument.
        return 0.0
