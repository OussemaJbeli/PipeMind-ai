from app.services.chunker import CHARS_PER_TOKEN, chunk_document, estimate_tokens


def _doc(paragraphs: int, words: int = 60) -> str:
    return "\n\n".join(f"Paragraph {i}. " + "word " * words for i in range(paragraphs))


def test_a_short_document_is_one_chunk():
    chunks = chunk_document("A single short runbook step.")

    assert len(chunks) == 1
    assert chunks[0].index == 0


def test_an_empty_document_produces_nothing():
    assert chunk_document("") == []
    assert chunk_document("   \n\n  ") == []


def test_chunks_stay_near_the_target_size():
    chunks = chunk_document(_doc(12), target_tokens=400)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 400 * 1.1


def test_chunks_overlap_so_a_straddling_passage_survives_whole():
    """A step split across a boundary would otherwise be retrievable as neither half."""
    chunks = chunk_document(_doc(12), target_tokens=400, overlap_tokens=50)

    tail = chunks[0].content[-80:].strip()[:30]

    assert tail in chunks[1].content


def test_paragraph_boundaries_are_preferred_over_size():
    doc = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_document(doc, target_tokens=400)

    # Everything fits, so nothing should be cut.
    assert len(chunks) == 1
    assert "First paragraph." in chunks[0].content
    assert "Third paragraph." in chunks[0].content


def test_an_oversized_paragraph_is_still_split():
    """A long code block has no paragraph breaks to split on."""
    chunks = chunk_document("x" * (400 * CHARS_PER_TOKEN * 3), target_tokens=400)

    assert len(chunks) >= 3


def test_indexes_are_sequential_and_gapless():
    chunks = chunk_document(_doc(20))

    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_token_estimate_is_monotonic():
    assert estimate_tokens("word") >= 1
    assert estimate_tokens("word " * 100) > estimate_tokens("word " * 10)
