from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging import log
from app.db.queries import find_knowledge_chunks, get_signature_history
from app.models.responses import SimilarFailure
from app.services.embeddings import embed_text
from app.services.similarity import find_similar


@dataclass
class RetrievedContext:
    similar_failures: list[SimilarFailure] = field(default_factory=list)
    knowledge_chunks: list[dict] = field(default_factory=list)
    signature_history: dict | None = None
    used: bool = False

    def is_empty(self) -> bool:
        return not (self.similar_failures or self.knowledge_chunks or self.signature_history)


async def retrieve(
    session: AsyncSession, *, team_id: int, project_id: int | None,
    error_message: str, category: str | None, ecosystem: str | None,
    job_name: str | None, signature_hash: str | None,
    exclude_failure_id: int | None = None,
) -> RetrievedContext:
    cfg = settings()

    # 1. Exact signature match short-circuits everything. If this precise error
    #    already has a confirmed resolution, retrieval is done — and the LLM's job
    #    becomes "confirm and explain", not "guess".
    history = (await get_signature_history(session, team_id, signature_hash)
               if signature_hash else None)

    if history and history.get("is_known"):
        return RetrievedContext(signature_history=history, used=True)

    # 2. Nearest historical failures
    similar = await find_similar(
        session, team_id=team_id, error_message=error_message, category=category,
        ecosystem=ecosystem, job_name=job_name, exclude_failure_id=exclude_failure_id,
    )

    # 3. Project documentation / runbooks
    query_vec = embed_text(f"{category or ''} {error_message}"[:1000])
    chunks = await find_knowledge_chunks(
        session, team_id=team_id, project_id=project_id,
        embedding=query_vec, limit=cfg.max_knowledge_chunks, threshold=0.60,
    )

    log.info("rag.retrieved", similar=len(similar), chunks=len(chunks),
             signature_seen=bool(history))

    return RetrievedContext(
        similar_failures=similar,
        knowledge_chunks=chunks,
        signature_history=history,
        used=bool(similar or chunks or history),
    )


async def retrieve_safely(session: AsyncSession, **kwargs) -> RetrievedContext:
    """Retrieval is an enhancement, never a precondition.

    A dead pgvector index or a cold model must degrade the analysis, not fail it
    — an answer without history beats no answer.
    """
    try:
        return await retrieve(session, **kwargs)
    except Exception as exc:
        log.warning("rag.failed", error=str(exc), exc_type=type(exc).__name__)
        return RetrievedContext(used=False)
