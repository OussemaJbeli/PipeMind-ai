from fastapi import APIRouter

from app.config import settings
from app.models.responses import ServiceInfo
from app.providers.registry import available_providers
from app.services.classifier.ml import MLClassifier

router = APIRouter(tags=["system"])


@router.get("/info", response_model=ServiceInfo)
async def info() -> ServiceInfo:
    config = settings()

    return ServiceInfo(
        version=config.service_version,
        contract_version=config.contract_version,
        llm_provider=config.llm_provider,
        # `stub` is always ready; a cloud provider needs a key.
        llm_ready=config.llm_provider == "stub"
        or bool(config.gemini_api_key or config.openai_api_key or config.llm_provider == "ollama"),
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
        embeddings_ready=_embeddings_ready(),
        database=bool(config.database_url),
        providers=available_providers(),
    )


@router.get("/classifier", tags=["system"])
async def classifier_status() -> dict:
    from app.services.classifier.rules import RULES

    return {
        "rules": len(RULES),
        "categories": sorted({rule.category for rule in RULES}),
        # False until a dataset exists; the hybrid falls back to rules.
        "ml_model_loaded": MLClassifier().available(),
    }


def _embeddings_ready() -> bool:
    """Reports whether the ml extra is installed, without importing torch.

    find_spec only looks for the module; importing it would cost ~8 s on a
    health check that is polled every few seconds.
    """
    from importlib.util import find_spec

    return find_spec("sentence_transformers") is not None
