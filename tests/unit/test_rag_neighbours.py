"""Neighbour expansion: the part that decides whether an answer is actionable.

Similarity retrieves the section of a runbook that names the error, because that
is the only section written in the same words as the error. The section holding
the fix describes a remedy instead and scores near zero against the same query —
0.059 against 0.573 on a real runbook. Returning the hit alone hands the model a
restatement of the problem it was already shown.
"""

import pytest

from app.config import settings
from app.services import rag


def seed(document: int, index: int, similarity: float = 0.55) -> dict:
    return {"document_id": document, "chunk_index": index, "similarity": similarity}


def chunk(document: int, index: int) -> dict:
    return {"document_id": document, "chunk_index": index,
            "content": f"doc{document} chunk{index}", "title": "Runbook"}


@pytest.fixture
def captured(monkeypatch):
    """Records what the expansion asked the database for."""
    asked: list[tuple[int, int]] = []

    async def fake(_session, _team_id, wanted):
        asked.extend(wanted)

        return [chunk(*pair) for pair in sorted(wanted) if pair[1] >= 0]

    monkeypatch.setattr(rag, "find_chunk_neighbours", fake)

    return asked


async def expand(seeds):
    return await rag._with_neighbours(None, team_id=1, seeds=seeds, cfg=settings())


async def test_a_seed_pulls_in_the_chunks_around_it(captured):
    chunks = await expand([seed(1, 4)])
    indexes = [c["chunk_index"] for c in chunks]

    radius = settings().knowledge_neighbour_radius
    assert indexes == list(range(4 - radius, 4 + radius + 1)), (
        "the fix sits next to the symptom, so the passage around a hit is the answer"
    )


async def test_negative_indexes_are_never_requested(captured):
    await expand([seed(1, 0)])

    assert all(index >= 0 for _, index in captured), (
        f"asked the database for chunks that cannot exist: {captured}"
    )


async def test_overlapping_seeds_are_not_requested_twice(captured):
    # Two hits one apart in the same document: their windows overlap heavily,
    # and a duplicated chunk would be quoted to the model twice.
    await expand([seed(1, 3), seed(1, 4)])

    assert len(captured) == len(set(captured)), f"duplicate requests: {captured}"


async def test_chunks_are_ordered_by_document_then_position(captured):
    # 0.9 second on purpose: the better-matching document must still come first,
    # but each document has to read top to bottom. A runbook quoted out of
    # sequence reads as contradictory advice and the model cannot tell which
    # fragment came first.
    chunks = await expand([seed(7, 5, 0.5), seed(3, 2, 0.9)])
    order = [(c["document_id"], c["chunk_index"]) for c in chunks]

    documents = list(dict.fromkeys(document for document, _ in order))
    assert documents == [7, 3], "seed order decides which document leads"

    for document in documents:
        positions = [index for doc, index in order if doc == document]
        assert positions == sorted(positions), f"document {document} is out of order"


async def test_expansion_never_exceeds_the_prompt_budget(captured):
    # Three seeds in three documents, each widened by the radius, is far more
    # than the prompt should carry.
    chunks = await expand([seed(1, 10), seed(2, 10), seed(3, 10)])

    assert len(chunks) <= settings().max_knowledge_chunks


async def test_neighbours_carry_the_seed_score_not_a_score_of_their_own(captured):
    chunks = await expand([seed(1, 4, 0.61)])

    # Provenance has to explain why a passage was included. A neighbour was not
    # matched, so reporting a similarity of its own would be a fabricated number.
    assert {c["similarity"] for c in chunks} == {0.61}


async def test_no_seeds_means_no_query_at_all(captured):
    assert await expand([]) == []
    assert captured == [], "expanded nothing into a database round trip"
