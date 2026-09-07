from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.router import router
from app.config import settings
from app.core.errors import PipeMindAIError
from app.core.logging import configure_logging, log
from app.core.security import require_service_token


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    # Compile every regex at boot rather than on the first real request.
    from app.services.classifier.rules import RuleClassifier

    RuleClassifier.warm()

    embedder_ms = _warm_embedder()

    log.info(
        "ai.started",
        version=settings().service_version,
        llm_provider=settings().llm_provider,
        embedder_warm_ms=embedder_ms,
    )
    yield
    log.info("ai.stopped")


def _warm_embedder() -> int | None:
    """Loads the sentence-transformers model at boot, not on first use.

    It costs ~10 s to load. Lazily, that 10 s lands on the first real analysis
    after every restart — measured at 13.9 s end to end against 4 s warm, which
    looks like a slow model rather than a cold start and is the single worst
    number a reviewer would see.

    Failure here is not fatal: retrieval degrades to nothing (`retrieve_safely`),
    and classification plus a cloud LLM still work. A service that refuses to
    boot because an optional extra is missing is worse than a slow one.
    """
    import time

    from app.services.embeddings import get_embedder

    started = time.perf_counter()

    try:
        get_embedder()
    except Exception as exc:
        log.warning("embedder.warm_failed", error=str(exc), exc_type=type(exc).__name__)

        return None

    return int((time.perf_counter() - started) * 1000)


app = FastAPI(
    title="PipeMind AI",
    version=settings().service_version,
    lifespan=lifespan,
    docs_url="/docs" if settings().app_env == "local" else None,
    redoc_url=None,
)

app.include_router(router, prefix="/v1", dependencies=[Depends(require_service_token)])


@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok", "version": settings().service_version}


@app.exception_handler(PipeMindAIError)
async def handle_domain_error(_: Request, exc: PipeMindAIError) -> JSONResponse:
    log.warning("ai.error", code=exc.code, message=str(exc))

    # Same shape Laravel returns, so it can be forwarded to the browser as-is.
    return JSONResponse(
        status_code=exc.status,
        content={"message": str(exc), "error_code": exc.code, "retryable": exc.retryable},
    )
