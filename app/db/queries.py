"""Raw SQL against Laravel's schema.

Raw, not ORM: the migrations own these tables. Mapping them a second time in
Python would mean two definitions of one truth, and the copy would rot.

Every query filters on team_id. Laravel enforces tenancy with a global scope;
here there is no ORM to hang one on, so it is stated explicitly in each WHERE.
"""


from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def find_similar_failures(
    session: AsyncSession,
    team_id: int,
    embedding: list[float],
    limit: int = 5,
    threshold: float = 0.55,
    exclude_failure_id: int | None = None,
    category: str | None = None,
) -> list[dict]:
    """Cosine similarity over pgvector, resolved failures preferred.

    `<=>` is cosine distance, so similarity = 1 - distance.
    """
    sql = text("""
        SELECT
            f.id, f.uuid::text AS uuid, f.category, f.subcategory,
            f.error_message, f.failed_at, f.resolved_at, f.resolution_type,
            f.resolution_note, f.time_to_resolution_seconds,
            p.name AS project_name,
            s.known_root_cause, s.known_resolution, s.is_known,
            1 - (fe.embedding <=> CAST(:emb AS vector)) AS similarity
        FROM failure_embeddings fe
        JOIN failures f  ON f.id = fe.failure_id
        JOIN projects  p ON p.id = f.project_id
        LEFT JOIN failure_signatures s ON s.id = f.signature_id
        WHERE fe.team_id = :team_id
          AND (CAST(:exclude AS bigint) IS NULL OR f.id <> CAST(:exclude AS bigint))
          AND 1 - (fe.embedding <=> CAST(:emb AS vector)) >= :threshold
        ORDER BY
            -- a resolved neighbour is worth more than a marginally closer unresolved one
            (f.resolved_at IS NOT NULL) DESC,
            -- and a same-category neighbour beats a closer cross-category one:
            -- embeddings put two node failures near each other because they are
            -- both node, not because they share a cause
            (CAST(:category AS text) IS NULL OR f.category = CAST(:category AS text)) DESC,
            fe.embedding <=> CAST(:emb AS vector)
        LIMIT :limit
    """)

    rows = await session.execute(sql, {
        "team_id": team_id,
        "emb": _vector(embedding),
        "limit": limit,
        "threshold": threshold,
        "exclude": exclude_failure_id,
        "category": category,
    })

    return [dict(r) for r in rows.mappings()]


async def get_signature_history(
    session: AsyncSession, team_id: int, signature_hash: str
) -> dict | None:
    """The exact-match path. A confirmed resolution here skips the LLM entirely."""
    sql = text("""
        SELECT hash, category, subcategory, occurrence_count, projects_affected,
               first_seen_at, last_seen_at, is_known, known_root_cause,
               known_resolution, avg_resolution_seconds
        FROM failure_signatures
        WHERE team_id = :team_id AND hash = :hash
        LIMIT 1
    """)

    row = (await session.execute(sql, {"team_id": team_id, "hash": signature_hash})).mappings().first()

    if not row:
        return None

    history = dict(row)

    # is_known is the flag a human sets when confirming a fix. Without an actual
    # resolution text it means nothing, so require both before short-circuiting.
    history["is_known"] = bool(history["is_known"] and history["known_resolution"])

    return history


async def find_knowledge_chunks(
    session: AsyncSession,
    team_id: int,
    embedding: list[float],
    project_id: int | None = None,
    limit: int = 4,
    threshold: float = 0.60,
) -> list[dict]:
    """Project docs and runbooks.

    Chunks scoped to a project rank above team-wide ones at equal distance:
    this project's own runbook beats a generic team note.
    """
    sql = text("""
        SELECT kc.content, kc.metadata, kc.chunk_index,
               kd.title, kd.type AS source_type, kd.source_url, kd.project_id,
               1 - (kc.embedding <=> CAST(:emb AS vector)) AS similarity
        FROM knowledge_chunks kc
        JOIN knowledge_documents kd ON kd.id = kc.document_id
        WHERE kc.team_id = :team_id
          AND kd.is_active = TRUE
          AND (kd.project_id IS NULL OR kd.project_id = CAST(:project_id AS bigint))
          AND 1 - (kc.embedding <=> CAST(:emb AS vector)) >= :threshold
        ORDER BY
            (kd.project_id = CAST(:project_id AS bigint)) DESC,
            kc.embedding <=> CAST(:emb AS vector)
        LIMIT :limit
    """)

    rows = await session.execute(sql, {
        "team_id": team_id,
        "emb": _vector(embedding),
        "project_id": project_id,
        "limit": limit,
        "threshold": threshold,
    })

    return [dict(r) for r in rows.mappings()]


async def insert_failure_embedding(
    session: AsyncSession,
    *,
    failure_id: int,
    team_id: int,
    embedding: list[float],
    source_text: str,
    signature_id: int | None = None,
    model: str = "all-MiniLM-L6-v2",
) -> None:
    """The one write this service makes.

    ON CONFLICT DO UPDATE, not DO NOTHING: re-embedding is how a corrected
    category or a better source_text propagates into the vector.
    """
    sql = text("""
        INSERT INTO failure_embeddings
            (failure_id, signature_id, team_id, embedding, model, dimensions, source_text, created_at)
        VALUES
            (:failure_id, :signature_id, :team_id, CAST(:emb AS vector), :model, :dims, :source_text, now())
        ON CONFLICT (failure_id, model) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                source_text = EXCLUDED.source_text
    """)

    await session.execute(sql, {
        "failure_id": failure_id,
        "signature_id": signature_id,
        "team_id": team_id,
        "emb": _vector(embedding),
        "model": model,
        "dims": len(embedding),
        "source_text": source_text[:4000],
    })
    await session.commit()


async def get_project_context(session: AsyncSession, team_id: int, project_uuid: str) -> dict | None:
    sql = text("""
        SELECT id, name, tech_stack, default_branch, settings
        FROM projects
        WHERE team_id = :team_id AND uuid = CAST(:uuid AS uuid) AND deleted_at IS NULL
        LIMIT 1
    """)

    row = (await session.execute(sql, {"team_id": team_id, "uuid": project_uuid})).mappings().first()

    return dict(row) if row else None


async def month_to_date_cost(session: AsyncSession, team_id: int) -> float:
    """Spend since the first of the month, for the budget guard.

    Failed requests are excluded — a provider error we never got tokens back
    from should not eat the budget.
    """
    sql = text("""
        SELECT COALESCE(SUM(cost_usd), 0) AS total
        FROM ai_requests
        WHERE team_id = :team_id
          AND status = 'success'
          AND created_at >= date_trunc('month', now())
    """)

    return float((await session.execute(sql, {"team_id": team_id})).scalar_one())


def _vector(embedding: list[float]) -> str:
    """pgvector's text input format: [0.1,0.2,...] with no spaces."""
    return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
