from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.queries import find_similar_failures
from app.models.responses import SimilarFailure
from app.services.embeddings import embed_text, embedding_input


async def find_similar(
    session: AsyncSession, *, team_id: int, error_message: str,
    category: str | None = None, ecosystem: str | None = None,
    job_name: str | None = None, exclude_failure_id: int | None = None,
    limit: int | None = None, threshold: float | None = None,
) -> list[SimilarFailure]:
    cfg = settings()

    vector = embed_text(embedding_input(
        error_message=error_message, category=category,
        ecosystem=ecosystem, job_name=job_name,
    ))

    rows = await find_similar_failures(
        session, team_id=team_id, embedding=vector,
        limit=limit or cfg.max_similar_failures,
        threshold=threshold if threshold is not None else cfg.similarity_threshold,
        exclude_failure_id=exclude_failure_id,
        category=category if category and category != "UNKNOWN" else None,
    )

    return [
        SimilarFailure(
            uuid=r["uuid"],
            similarity=round(float(r["similarity"]), 4),
            project_name=r["project_name"],
            occurred_at=r["failed_at"].isoformat() if r["failed_at"] else None,
            root_cause=r["known_root_cause"],
            resolution=r["known_resolution"] or r["resolution_note"],
            resolved=r["resolved_at"] is not None,
        )
        for r in rows
    ]
