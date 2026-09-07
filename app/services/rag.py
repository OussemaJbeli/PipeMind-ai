from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging import log
from app.db.queries import (
    find_chunk_neighbours,
    find_knowledge_chunks,
    get_signature_history,
)
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

    # 3. Project documentation / runbooks.
    #    Deliberately NOT embedding_input(): measured against a 6-probe corpus,
    #    category + raw error beat the structured composition on recall at every
    #    threshold. Docs are prose written by humans, not normalized error rows,
    #    so the "ecosystem: php | job: test" suffix is noise on this side.
    query_vec = embed_text(f"{category or ''} {error_message}"[:1000])
    seeds = await find_knowledge_chunks(
        session, team_id=team_id, project_id=project_id, embedding=query_vec,
        limit=cfg.max_knowledge_seeds, threshold=cfg.knowledge_threshold,
    )
    chunks = await _with_neighbours(session, team_id=team_id, seeds=seeds, cfg=cfg)

    log.info("rag.retrieved", similar=len(similar), chunks=len(chunks),
             seeds=len(seeds), signature_seen=bool(history))

    return RetrievedContext(
        similar_failures=similar,
        knowledge_chunks=chunks,
        signature_history=history,
        used=bool(similar or chunks or history),
    )


async def _with_neighbours(
    session: AsyncSession, *, team_id: int, seeds: list[dict], cfg: Any,
) -> list[dict]:
    """Grows each matched chunk into the passage around it.

    Retrieval finds the section that names the error; the section that fixes it
    sits next to that one and scores near zero against the same query, because
    it describes a remedy rather than a symptom. Returning the hit alone gives
    the model a restatement of the problem it was already shown.

    Chunks are returned in document order rather than by score: a runbook read
    out of sequence reads as contradictory advice, and the model has no way to
    tell which fragment came first.
    """
    if not seeds:
        return []

    radius = cfg.knowledge_neighbour_radius
    ranked_documents: list[int] = []
    wanted: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for seed in seeds:
        document, index = seed["document_id"], seed["chunk_index"]

        if document not in ranked_documents:
            ranked_documents.append(document)

        for offset in range(-radius, radius + 1):
            # Negative indexes simply do not exist; the query returns whatever
            # is real, so there is no need to know each document's length here.
            key = (document, index + offset)

            if key not in seen and key[1] >= 0:
                seen.add(key)
                wanted.append(key)

    rows = await find_chunk_neighbours(session, team_id, wanted)

    # Best-matching document first, then that document read top to bottom.
    similarity = {s["document_id"]: s["similarity"] for s in seeds}
    rows.sort(key=lambda r: (ranked_documents.index(r["document_id"]), r["chunk_index"]))

    for row in rows:
        # The seed's score, carried so provenance can show why the passage was
        # pulled in. A neighbour has no similarity of its own worth reporting.
        row["similarity"] = similarity.get(row["document_id"])

    return rows[:cfg.max_knowledge_chunks]


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
