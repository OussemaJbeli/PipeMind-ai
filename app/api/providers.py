from fastapi import APIRouter

from app.core.errors import LLMUnavailable
from app.core.logging import log
from app.models.requests import TestProviderRequest
from app.models.responses import TestProviderResponse
from app.providers.registry import build_provider

router = APIRouter(tags=["providers"])


@router.post("/providers/test", response_model=TestProviderResponse)
async def test_provider(request: TestProviderRequest) -> TestProviderResponse:
    """Tests credentials before anything is saved.

    The same contract as the integration wizard: prove the connection works,
    then persist. A key that only fails at analysis time fails an hour later,
    inside a queue, with nobody watching — and what the user sees is not "your
    key is wrong" but "analysis failed".

    Always 200. A rejected credential is a valid answer to "does this work?",
    not a server error, and the wizard needs the message either way.
    """
    try:
        provider = build_provider(
            request.model_dump(exclude_none=True),
            # Never fall back to the server's own credentials here: the wizard
            # would report success for a key the user never entered.
            allow_env_fallback=False,
        )
        check = await provider.verify()
    except LLMUnavailable as exc:
        return TestProviderResponse(ok=False, provider=request.provider,
                                    model=request.model or "", message=str(exc))

    log.info("provider.tested", provider=check.provider, ok=check.ok)

    return TestProviderResponse(
        ok=check.ok,
        provider=check.provider,
        model=check.model,
        message=check.message,
        models=check.models,
        latency_ms=check.latency_ms,
    )
