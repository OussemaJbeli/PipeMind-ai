"""Vector embeddings.

What you embed matters more than which model you use.
"""

from functools import lru_cache
from typing import Any

from app.config import settings

MAX_CHARS = 1000


@lru_cache(maxsize=1)
def get_embedder() -> Any:
    """Loaded once, on first use. ~90 MB, ~8 s cold.

    Imported inside the function on purpose: sentence-transformers pulls in
    torch, and a service running on rules and a cloud LLM should not pay an
    8-second import it never uses.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings().embedding_model, device="cpu")


def embed_text(text: str) -> list[float]:
    import numpy as np

    vec = get_embedder().encode(
        _prepare(text),
        normalize_embeddings=True,   # cosine distance assumes unit vectors
        show_progress_bar=False,
    )

    return np.asarray(vec, dtype=np.float32).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    import numpy as np

    if not texts:
        return []

    vecs = get_embedder().encode(
        [_prepare(t) for t in texts],
        batch_size=settings().embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return [np.asarray(v, dtype=np.float32).tolist() for v in vecs]


def _prepare(text: str) -> str:
    """all-MiniLM truncates at 256 word-pieces. Choose those 256 deliberately."""
    return " ".join(text.split())[:MAX_CHARS]


def embedding_input(*, error_message: str, category: str | None = None,
                    ecosystem: str | None = None, job_name: str | None = None) -> str:
    """Compose what actually gets embedded.

    Raw log excerpts embed badly — timestamps, paths and IDs dominate the vector
    and everything ends up 0.8-similar to everything else. Embed the normalized
    error plus a little structured context instead.
    """
    parts = [error_message.strip()[:600]]

    if category and category != "UNKNOWN":
        parts.append(f"category: {category}")

    if ecosystem:
        parts.append(f"ecosystem: {ecosystem}")

    if job_name:
        parts.append(f"job: {job_name}")

    return " | ".join(parts)
