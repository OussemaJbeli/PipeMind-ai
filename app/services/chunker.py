"""Splits documents for embedding.

Chunk size is a retrieval decision, not a formatting one. Too large and the
vector averages several ideas into something that matches nothing precisely;
too small and a chunk loses the context that made it meaningful. ~400 tokens
with 50 tokens of overlap is the usual compromise, and the overlap exists so a
passage that straddles a boundary still appears whole in one chunk.
"""

import re
from dataclasses import dataclass

# all-MiniLM truncates at 256 word-pieces, so a 400-token chunk is already past
# what the model reads. Sized for the LLM's context, not the embedder's: the
# chunk shown to the model and the chunk that was embedded must be the same text,
# or a citation points at something the model never saw.
TARGET_TOKENS = 400
OVERLAP_TOKENS = 50

# Rough but stable: English averages ~4 characters per token.
CHARS_PER_TOKEN = 4

_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    index: int
    content: str
    token_count: int


def chunk_document(text: str, *, target_tokens: int = TARGET_TOKENS,
                   overlap_tokens: int = OVERLAP_TOKENS) -> list[Chunk]:
    text = text.strip()

    if not text:
        return []

    target_chars = target_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    # Split on paragraphs first: a runbook step cut in half mid-sentence
    # retrieves badly no matter how good the model is.
    paragraphs = [p.strip() for p in _PARAGRAPH.split(text) if p.strip()]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > target_chars:
            # A single oversized paragraph (a long code block, usually) still has
            # to be split. Nothing better to split on, so split on size.
            if current:
                chunks.append(current)
                current = ""

            chunks.extend(_split_oversized(paragraph, target_chars, overlap_chars))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(candidate) <= target_chars:
            current = candidate
        else:
            chunks.append(current)
            current = _tail(current, overlap_chars) + "\n\n" + paragraph if overlap_chars else paragraph

    if current:
        chunks.append(current)

    return [
        Chunk(index=i, content=c.strip(), token_count=estimate_tokens(c))
        for i, c in enumerate(chunks)
        if c.strip()
    ]


def _split_oversized(paragraph: str, target_chars: int, overlap_chars: int) -> list[str]:
    pieces, start = [], 0
    step = max(1, target_chars - overlap_chars)

    while start < len(paragraph):
        pieces.append(paragraph[start:start + target_chars])
        start += step

    return pieces


def _tail(text: str, chars: int) -> str:
    """The trailing overlap, cut on a word boundary rather than mid-word."""
    if chars <= 0 or len(text) <= chars:
        return text

    tail = text[-chars:]
    space = tail.find(" ")

    return tail[space + 1:] if space != -1 else tail


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)
