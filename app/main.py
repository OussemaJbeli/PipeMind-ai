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

    log.info(
        "ai.started",
        version=settings().service_version,
        llm_provider=settings().llm_provider,
    )
    yield
    log.info("ai.stopped")


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
