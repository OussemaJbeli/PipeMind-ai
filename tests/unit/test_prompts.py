import pytest
from jinja2 import UndefinedError

from app.models.responses import SimilarFailure
from app.services.classifier.hybrid import Classification
from app.services.prompts import ANALYZE_SCHEMA, SYSTEM_PROMPT, render_analyze_prompt
from app.services.rag import RetrievedContext
from tests.unit.test_analyzer import _request


def _classification():
    return Classification(category="DATABASE", subcategory="ConnectionRefused",
                          confidence=0.98, source="rules", matched_rules=["DATABASE/ConnectionRefused"])


def test_renders_without_any_retrieval():
    prompt = render_analyze_prompt(_request(), _classification(), None, "Connection refused")

    assert "Connection refused" in prompt
    assert "DATABASE" in prompt


def test_known_signature_is_stated_up_front():
    rag = RetrievedContext(
        signature_history={"is_known": True, "occurrence_count": 12,
                           "known_root_cause": "Postgres service not linked",
                           "known_resolution": "Add the postgres service to the job"},
        used=True,
    )

    prompt = render_analyze_prompt(_request(), _classification(), rag, "Connection refused")

    assert "KNOWN ISSUE" in prompt
    assert "Add the postgres service to the job" in prompt


def test_unresolved_neighbours_are_labelled_as_such():
    """The model must not present an unresolved neighbour as a proven fix."""
    rag = RetrievedContext(
        similar_failures=[
            SimilarFailure(uuid="a", similarity=0.91, project_name="api",
                           occurred_at="2026-03-01T10:00:00+00:00", resolved=False),
        ],
        used=True,
    )

    prompt = render_analyze_prompt(_request(), _classification(), rag, "Connection refused")

    assert "91% similar" in prompt
    assert "(never resolved)" in prompt


def test_a_neighbour_with_no_date_does_not_break_rendering():
    rag = RetrievedContext(
        similar_failures=[SimilarFailure(uuid="a", similarity=0.8, project_name="api",
                                         occurred_at=None)],
        used=True,
    )

    assert "unknown date" in render_analyze_prompt(_request(), _classification(), rag, "x")


def test_a_missing_variable_fails_loudly():
    """StrictUndefined: a silently blank prompt section is worse than a crash."""
    from app.services.prompts import _env

    with pytest.raises(UndefinedError):
        _env.from_string("{{ nope.at_all }}").render()


def test_system_prompt_and_schema_agree_on_categories():
    for category in ("DATABASE", "DEPENDENCY", "UNKNOWN"):
        assert category in ANALYZE_SCHEMA["properties"]["category"]["enum"]

    assert "Ground every claim in the evidence" in SYSTEM_PROMPT
