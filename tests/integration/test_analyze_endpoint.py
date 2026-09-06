import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "analyze-request-db.json"


@pytest.fixture
def payload() -> dict:
    return json.loads(FIXTURE.read_text())


def test_analyze_requires_the_service_token(client, payload):
    assert client.post("/v1/analyze", json=payload).status_code == 401


def test_analyze_returns_a_complete_analysis(client, auth, payload):
    response = client.post("/v1/analyze", json=payload, headers=auth)

    assert response.status_code == 200
    body = response.json()

    assert body["category"] == "DATABASE"
    assert body["subcategory"] == "ConnectionRefused"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["summary"] and body["root_cause"]
    assert body["classification_source"] in {"rules", "ml", "hybrid", "llm"}
    assert body["usage"]["provider"] == "stub"


def test_every_evidence_item_cites_something(client, auth, payload):
    """A citation that points nowhere is worse than no citation: it looks checkable."""
    body = client.post("/v1/analyze", json=payload, headers=auth).json()

    for item in body["evidence"]:
        assert item["type"]
        assert item["content"]
        assert item["source_ref"], "evidence must say where it came from"


def test_recommendations_carry_a_risk_the_model_did_not_choose(client, auth, payload):
    from app.services.recommender import ACTION_RISK

    body = client.post("/v1/analyze", json=payload, headers=auth).json()

    for rec in body["recommendations"]:
        assert rec["action_type"] in ACTION_RISK
        assert rec["risk"] == ACTION_RISK[rec["action_type"]] or rec["risk"] in {
            "medium", "high", "critical"
        }


def test_contract_version_is_echoed(client, auth, payload):
    """Laravel checks this: a mismatch means the two services disagree on shape."""
    body = client.post("/v1/analyze", json=payload, headers=auth).json()

    assert body["contract_version"] == "v1"


def test_a_malformed_request_is_rejected_before_any_work(client, auth):
    response = client.post("/v1/analyze", json={"team_id": 1}, headers=auth)

    assert response.status_code == 422


def test_info_reports_provider_and_embedding_readiness(client, auth):
    body = client.get("/v1/info", headers=auth).json()

    assert "providers" in body
    assert body["providers"]["stub"] is True
    assert isinstance(body["embeddings_ready"], bool)
    assert body["embedding_dim"] == 384
