"""Splits documents for embedding.

Chunk size is a retrieval decision, not a formatting one, and it is bounded by
the embedder rather than by taste. Measured on a two-section runbook whose first
heading is the exact error it documents (see
PipeMind-data/experiments/knowledge-retrieval.md):

    target   chunks  max word-pieces  truncated   score for that exact error
      400         2              400          1   0.334  <- below threshold: MISS
      220         3              219          0   0.349
      150         5              146          0   0.573  <- retrieved
       90         7               97          0   0.573, but distractors reach 0.362

Two failure modes bracket the choice. Above ~230 tokens the chunk exceeds
all-MiniLM's 256 word-piece window and the tail is silently dropped — at 400 a
third of each chunk never reaches the model, so the fix section of a runbook is
unretrievable no matter how well it is written. Below ~120 the chunk loses the
context that made it specific and starts matching unrelated errors.
"""

import re
from dataclasses import dataclass

# Must stay under the embedder's window with room for the ~4-char-per-token
# estimate below to be wrong on code-heavy text, where yaml and identifiers
# tokenize closer to 3.6 chars per word-piece. 150 measured at 146 word-pieces
# worst-case, a 43% margin against the 256 limit.
#
# Raising this is the tempting mistake: a chunk the embedder truncates cannot be
# retrieved, so its extra text never reaches the LLM either — it is not "more
# context for the model", it is dead weight in the vector. test_chunker.py
# asserts the invariant against the live tokenizer.
EMBEDDER_WORD_PIECE_LIMIT = 256
TARGET_TOKENS = 150
OVERLAP_TOKENS = 25

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
