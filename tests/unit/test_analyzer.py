import pytest

from app.models.requests import AnalyzeRequest
from app.services.analyzer import _calibrate, _valid_evidence, analyze
from app.services.classifier.hybrid import Classification
from app.services.rag import RetrievedContext


def _classification(category="DATABASE", confidence=0.95, source="rules"):
    return Classification(category=category, subcategory="ConnectionRefused",
                          confidence=confidence, source=source)


class TestCalibration:
    def test_a_known_signature_raises_confidence(self):
        history = RetrievedContext(signature_history={"is_known": True}, used=True)

        assert _calibrate(0.80, _classification(), history) > 0.80

    def test_retrieval_that_found_nothing_caps_confidence(self):
        """No history at all — the model cannot be more sure than the evidence."""
        assert _calibrate(0.99, _classification(), RetrievedContext(used=False)) == 0.85

    def test_retrieval_never_running_is_not_evidence_of_absence(self):
        assert _calibrate(0.99, _classification(), None) == 0.99

    def test_unknown_category_caps_confidence(self):
        assert _calibrate(0.99, _classification(category="UNKNOWN"), None) == 0.70

    def test_garbage_confidence_falls_back_to_neutral(self):
        assert _calibrate("very sure", _classification(), None) == 0.5
        assert _calibrate(None, _classification(), None) == 0.5

    def test_confidence_is_clamped_to_range(self):
        assert _calibrate(4.2, _classification(), None) == 1.0
        assert _calibrate(-1, _classification(), None) == 0.0


class TestEvidence:
    def test_malformed_entries_are_dropped_not_fatal(self):
        """Losing one citation beats losing the whole analysis."""
        items = [
            {"type": "log_line", "content": "real"},
            {"type": "log_line"},          # no content
            {"content": "no type"},
            "not a dict",
            None,
        ]

        kept = _valid_evidence(items)

        assert len(kept) == 1
        assert kept[0]["content"] == "real"

    def test_weight_is_clamped(self):
        kept = _valid_evidence([{"type": "log_line", "content": "x", "weight": 99}])

        assert kept[0]["weight"] == 1.0


def _request(**overrides) -> AnalyzeRequest:
    payload = {
        "team_id": 1,
        "failure": {"uuid": "f-1", "signature_hash": "abc123", "ecosystem": "php"},
        "project": {"uuid": "p-1", "name": "api", "default_branch": "main"},
        "pipeline": {"iid": 1, "ref": "main"},
        "job": {"name": "test"},
        "log_excerpt": "SQLSTATE[HY000] [2002] Connection refused",
        "error_block": "SQLSTATE[HY000] [2002] Connection refused",
        "use_rag": False,
    }
    payload.update(overrides)

    return AnalyzeRequest(**payload)


class TestAnalyze:
    @pytest.mark.asyncio
    async def test_classification_only_makes_no_model_call(self):
        result = await analyze(_request(use_llm=False))

        assert result.category == "DATABASE"
        assert result.usage.provider == "none"
        assert result.usage.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_secrets_are_redacted_even_if_laravel_forgot(self):
        """Defence in depth: the analyzer never trusts that upstream cleaned up."""
        leaked = "connect failed with password=hunter2SuperSecret"
        result = await analyze(_request(log_excerpt=leaked, error_block=None, use_llm=True))

        assert "hunter2SuperSecret" not in result.root_cause
        assert "hunter2SuperSecret" not in result.summary

    @pytest.mark.asyncio
    async def test_every_recommendation_carries_a_code_assigned_risk(self):
        from app.services.recommender import ACTION_RISK

        result = await analyze(_request())

        assert result.recommendations
        for rec in result.recommendations:
            assert rec.action_type in ACTION_RISK
            assert rec.risk in {"low", "medium", "high", "critical"}

    @pytest.mark.asyncio
    async def test_usage_reports_the_provider_that_actually_answered(self):
        result = await analyze(_request())

        assert result.usage.provider == "stub"
        assert result.usage.prompt_tokens > 0
