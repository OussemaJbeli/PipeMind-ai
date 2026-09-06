import pytest
from fastapi.testclient import TestClient

from app.main import app

TOKEN = {"X-PipeMind-Token": "change-me-shared-secret"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_needs_no_token(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-PipeMind-Token": "wrong"}], ids=["none", "wrong"])
def test_v1_requires_the_service_token(client: TestClient, headers: dict) -> None:
    assert client.get("/v1/info", headers=headers).status_code == 401


def test_info_reports_what_is_configured(client: TestClient) -> None:
    body = client.get("/v1/info", headers=TOKEN).json()

    assert body["contract_version"] == "v1"
    assert body["embedding_dim"] == 384


def test_process_log_redacts_before_anything_else(client: TestClient) -> None:
    raw = (
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
        "DB_PASSWORD=hunter2\n"
        "$ php artisan test\n"
        "SQLSTATE[HY000] [2002] Connection refused\n"
        "ERROR: Job failed: exit code 1\n"
    )

    body = client.post("/v1/logs/process", headers=TOKEN, json={"raw_log": raw}).json()
    serialised = str(body)

    # Nothing secret may appear anywhere in the response, including the excerpt.
    assert "AKIAIOSFODNN7EXAMPLE" not in serialised
    assert "hunter2" not in serialised
    assert body["is_redacted"] is True
    assert body["redaction_count"] >= 2

    # ...and the actual error must survive.
    assert "SQLSTATE[HY000] [2002] Connection refused" in body["excerpt"]
    assert body["ecosystem"] == "db"
    assert body["exit_code"] == 1
    assert len(body["signature_hash"]) == 64


def test_process_log_is_deterministic(client: TestClient) -> None:
    """The same failure must produce the same signature, or dedup, caching and
    similarity all break at once."""
    raw = "SQLSTATE[HY000] [2002] Connection refused at /app/Db.php:42\nexit code 1\n"

    first = client.post("/v1/logs/process", headers=TOKEN, json={"raw_log": raw}).json()
    second = client.post("/v1/logs/process", headers=TOKEN, json={"raw_log": raw}).json()

    assert first["signature_hash"] == second["signature_hash"]


def test_signature_ignores_run_specific_noise(client: TestClient) -> None:
    a = "SQLSTATE[HY000] [2002] Connection refused at /app/src/Db.php:42 at 2026-08-30T14:00:00Z"
    b = "SQLSTATE[HY000] [2002] Connection refused at /srv/lib/Db.php:87 at 2026-09-01T09:14:22Z"

    first = client.post("/v1/logs/process", headers=TOKEN, json={"raw_log": a}).json()
    second = client.post("/v1/logs/process", headers=TOKEN, json={"raw_log": b}).json()

    assert first["signature_hash"] == second["signature_hash"]


def test_classify_endpoint(client: TestClient) -> None:
    body = client.post(
        "/v1/classify",
        headers=TOKEN,
        json={"text": "npm ERR! ERESOLVE unable to resolve dependency tree"},
    ).json()

    assert body["category"] == "DEPENDENCY"
    assert body["source"] == "rules"
    assert body["confidence"] >= 0.85


def test_large_log_is_reduced_but_keeps_the_error(client: TestClient) -> None:
    noise = "\n".join(f"[{i}] downloading package-{i}" for i in range(5000))
    raw = f"{noise}\nSQLSTATE[HY000] [2002] Connection refused\nexit code 1\n"

    body = client.post("/v1/logs/process", headers=TOKEN, json={"raw_log": raw}).json()

    # Reduction is the easy metric; retention is the one that matters.
    assert body["reduction_ratio"] > 0.9
    assert "SQLSTATE[HY000] [2002] Connection refused" in body["excerpt"]


def test_process_log_returns_a_classification(client: TestClient) -> None:
    """One round trip, not two.

    Classification is regex over data the endpoint already has; making the
    caller ask again would double the latency of every ingest for nothing.
    """
    raw = "$ php artisan test\nSQLSTATE[HY000] [2002] Connection refused\nexit code 1\n"

    body = client.post("/v1/logs/process", headers=TOKEN, json={"raw_log": raw}).json()

    assert body["category"] == "DATABASE"
    assert body["subcategory"] == "ConnectionRefused"
    assert body["classification_source"] == "rules"
    assert body["classification_confidence"] >= 0.85


def test_unclassifiable_log_says_so(client: TestClient) -> None:
    body = client.post(
        "/v1/logs/process", headers=TOKEN, json={"raw_log": "Build completed in 42s\n"}
    ).json()

    # UNKNOWN is a real answer. Forcing a guess pollutes every other class.
    assert body["category"] == "UNKNOWN"
    assert body["classification_confidence"] == 0.0
