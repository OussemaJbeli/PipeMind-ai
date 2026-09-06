import pytest

from app.core.errors import LLMUnavailable
from app.providers.registry import FallbackProvider, build_provider, build_provider_with_fallback
from app.providers.stub import StubProvider
from app.services.prompts import ANALYZE_SCHEMA, SYSTEM_PROMPT


def test_default_provider_is_stub():
    assert isinstance(build_provider(), StubProvider)


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMUnavailable):
        build_provider({"provider": "definitely-not-a-provider"})


def test_override_selects_provider_without_storing_credentials():
    provider = build_provider({"provider": "gemini", "api_key": "k-123", "model": "gemini-2.5-flash"})

    assert provider.name == "gemini"
    assert provider._model == "gemini-2.5-flash"


def test_fallback_only_wraps_when_configured():
    assert not isinstance(build_provider_with_fallback(), FallbackProvider)
    assert isinstance(
        build_provider_with_fallback({"provider": "stub", "fallback": {"provider": "stub"}}),
        FallbackProvider,
    )


@pytest.mark.asyncio
async def test_stub_returns_schema_shaped_json():
    response = await StubProvider().complete(
        system=SYSTEM_PROMPT,
        prompt="SQLSTATE[HY000] [2002] Connection refused",
        json_schema=ANALYZE_SCHEMA,
    )

    assert response.parsed
    for field in ("category", "summary", "root_cause", "confidence", "recommendations"):
        assert field in response.parsed


@pytest.mark.asyncio
async def test_fallback_uses_secondary_when_primary_is_unavailable():
    from app.core.errors import LLMUnavailable as Unavailable

    class Broken:
        name, model = "broken", "none"

        async def complete(self, **kwargs):
            raise Unavailable("down")

        def cost(self, *_):
            return 0.0

    response = await FallbackProvider(Broken(), StubProvider()).complete(
        system="s", prompt="p", json_schema=ANALYZE_SCHEMA
    )

    # The caller must be able to tell that a different model answered.
    assert response.raw["fell_back_from"] == "broken"


@pytest.mark.asyncio
async def test_fallback_does_not_mask_a_bad_response():
    """A second model will fail the same way on the same input — retrying is waste."""
    from app.core.errors import InvalidLLMResponse

    class Garbage:
        name, model = "garbage", "none"

        async def complete(self, **kwargs):
            raise InvalidLLMResponse("not json")

        def cost(self, *_):
            return 0.0

    with pytest.raises(InvalidLLMResponse):
        await FallbackProvider(Garbage(), StubProvider()).complete(
            system="s", prompt="p", json_schema=ANALYZE_SCHEMA
        )
