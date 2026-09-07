"""Provider behaviour under HTTP conditions a live key cannot reproduce.

Rate limits, timeouts, malformed JSON and 5xx are exactly the paths that decide
whether a bad afternoon degrades or cascades — and exactly the paths a working
credential will never exercise. respx mocks the transport so they are testable
without a server and without waiting for a real outage.
"""

import httpx
import pytest
import respx

from app.core.errors import InvalidLLMResponse, LLMTimeout, LLMUnavailable
from app.providers.ollama import OllamaProvider
from app.services.prompts import ANALYZE_SCHEMA

BASE = "http://localhost:11434"
GENERATE = f"{BASE}/api/generate"
TAGS = f"{BASE}/api/tags"

VALID = {
    "category": "DATABASE",
    "severity": "high",
    "confidence": 0.9,
    "summary": "Database unreachable.",
    "root_cause": "The database container was not running.",
}


def provider() -> OllamaProvider:
    return OllamaProvider(base_url=BASE, model="qwen2.5-coder:7b")


async def complete(**kwargs):
    return await provider().complete(
        system="s", prompt="p", json_schema=ANALYZE_SCHEMA, timeout=5, **kwargs
    )


class TestSuccess:
    @respx.mock
    @pytest.mark.asyncio
    async def test_a_good_response_is_parsed_with_token_counts(self):
        import json

        respx.post(GENERATE).mock(return_value=httpx.Response(200, json={
            "response": json.dumps(VALID),
            "prompt_eval_count": 412,
            "eval_count": 180,
            "done_reason": "stop",
        }))

        result = await complete()

        assert result.parsed["category"] == "DATABASE"
        assert result.prompt_tokens == 412
        assert result.completion_tokens == 180
        assert result.total_tokens == 592
        assert result.provider == "ollama"

    @respx.mock
    @pytest.mark.asyncio
    async def test_json_wrapped_in_prose_is_salvaged(self):
        """Local models have no schema-constrained decoding, so they narrate.

        Gemini cannot emit invalid JSON with a schema set; Ollama can and does.
        salvage_json is why one provider's habits do not become a failed analysis.
        """
        import json

        respx.post(GENERATE).mock(return_value=httpx.Response(200, json={
            "response": f"Sure! Here is the analysis:\n```json\n{json.dumps(VALID)}\n```\nHope that helps.",
        }))

        assert (await complete()).parsed["root_cause"].startswith("The database container")


class TestFailureModes:
    @respx.mock
    @pytest.mark.asyncio
    async def test_a_timeout_is_typed_as_a_timeout(self):
        respx.post(GENERATE).mock(side_effect=httpx.ReadTimeout("too slow"))

        with pytest.raises(LLMTimeout) as caught:
            await complete()

        # Retryable: a slow local model is a transient condition.
        assert caught.value.retryable is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_rate_limit_is_not_swallowed(self):
        respx.post(GENERATE).mock(return_value=httpx.Response(429, json={"error": "busy"}))

        with pytest.raises(LLMUnavailable) as caught:
            await complete()

        assert caught.value.retryable is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_5xx_degrades_rather_than_crashing(self):
        respx.post(GENERATE).mock(return_value=httpx.Response(503, text="overloaded"))

        with pytest.raises(LLMUnavailable):
            await complete()

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_refused_connection_names_the_endpoint(self):
        respx.post(GENERATE).mock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(LLMUnavailable, match="unreachable"):
            await complete()

    @respx.mock
    @pytest.mark.asyncio
    async def test_unsalvageable_output_is_not_retried_as_an_outage(self):
        """A second call to the same model on the same input fails the same way.

        InvalidLLMResponse is distinct from LLMUnavailable precisely so the
        fallback provider does not spend a second call proving that.
        """
        respx.post(GENERATE).mock(return_value=httpx.Response(200, json={
            "response": "I'm afraid I can't help with that.",
        }))

        with pytest.raises(InvalidLLMResponse):
            await complete()

    @respx.mock
    @pytest.mark.asyncio
    async def test_an_empty_body_is_not_mistaken_for_a_valid_answer(self):
        respx.post(GENERATE).mock(return_value=httpx.Response(200, json={"response": ""}))

        with pytest.raises(InvalidLLMResponse):
            await complete()


class TestVerify:
    @respx.mock
    @pytest.mark.asyncio
    async def test_a_pulled_model_verifies(self):
        respx.get(TAGS).mock(return_value=httpx.Response(200, json={
            "models": [{"name": "qwen2.5-coder:7b"}, {"name": "llama3.1:8b"}],
        }))

        check = await provider().verify()

        assert check.ok
        assert "qwen2.5-coder:7b" in check.models

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_running_server_missing_the_model_says_how_to_fix_it(self):
        """The usual local failure: configured but never pulled, and invisible
        until the first analysis."""
        respx.get(TAGS).mock(return_value=httpx.Response(200, json={
            "models": [{"name": "llama3.1:8b"}],
        }))

        check = await provider().verify()

        assert not check.ok
        assert "ollama pull qwen2.5-coder:7b" in check.message

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_bare_model_name_matches_a_tagged_one(self):
        respx.get(TAGS).mock(return_value=httpx.Response(200, json={
            "models": [{"name": "llama3.1:8b"}],
        }))

        check = await OllamaProvider(base_url=BASE, model="llama3.1").verify()

        assert check.ok

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_dead_server_reports_the_address_it_tried(self):
        respx.get(TAGS).mock(side_effect=httpx.ConnectError("refused"))

        check = await provider().verify()

        assert not check.ok
        assert BASE in check.message
