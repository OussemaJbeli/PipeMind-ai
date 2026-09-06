from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_session
from app.models.requests import ChunkEmbedRequest, EmbedRequest, SimilarRequest
from app.models.responses import ChunkEmbedResponse, EmbeddedChunk, EmbedResponse, SimilarFailure
from app.services.chunker import chunk_document
from app.services.embeddings import embed_batch, embed_text, embedding_input
from app.services.similarity import find_similar

router = APIRouter(tags=["vectors"])


@router.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    text = embedding_input(
        error_message=request.text,
        category=request.category,
        ecosystem=request.ecosystem,
        job_name=request.job_name,
    )

    vector = embed_text(text)

    return EmbedResponse(
        embedding=vector,
        dimensions=len(vector),
        model=settings().embedding_model,
        source_text=text,
    )


@router.post("/similar", response_model=list[SimilarFailure])
async def similar(
    request: SimilarRequest, session: AsyncSession = Depends(get_session)
) -> list[SimilarFailure]:
    return await find_similar(
        session,
        team_id=request.team_id,
        error_message=request.error_message,
        category=request.category,
        ecosystem=request.ecosystem,
        job_name=request.job_name,
        exclude_failure_id=request.exclude_failure_id,
        limit=request.limit,
        threshold=request.threshold,
    )


@router.post("/knowledge/chunk-embed", response_model=ChunkEmbedResponse)
async def chunk_embed(request: ChunkEmbedRequest) -> ChunkEmbedResponse:
    """Splits a document and embeds every chunk in one batch.

    One call rather than one per chunk: batching is where nearly all of the
    throughput is, and a 40-chunk runbook would otherwise mean 40 round trips.
    """
    config = settings()

    chunks = chunk_document(
        request.text,
        target_tokens=request.target_tokens,
        overlap_tokens=request.overlap_tokens,
    )

    if not chunks:
        return ChunkEmbedResponse(chunks=[], model=config.embedding_model,
                                  dimensions=config.embedding_dim)

    # Prefixing the title gives an otherwise context-free chunk something to
    # anchor on: "restart the worker" means little until you know which runbook
    # it came from.
    prefix = f"{request.title}: " if request.title else ""
    vectors = embed_batch([f"{prefix}{c.content}" for c in chunks])

    return ChunkEmbedResponse(
        chunks=[
            EmbeddedChunk(index=c.index, content=c.content,
                          token_count=c.token_count, embedding=v)
            for c, v in zip(chunks, vectors, strict=True)
        ],
        model=config.embedding_model,
        dimensions=len(vectors[0]),
    )
