from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"

# (fixture, ecosystem, a phrase from the root cause that MUST survive)
CASES = [
    ("failed-db-test.log", "db", "SQLSTATE[HY000] [2002] Connection refused"),
    ("failed-npm-dependency.log", "node", "ERESOLVE unable to resolve dependency tree"),
]


def test_requires_the_service_token(client) -> None:
    assert client.post("/v1/logs/process", json={"raw_log": "x"}).status_code == 401
    assert (
        client.post(
            "/v1/logs/process", headers={"X-PipeMind-Token": "wrong"}, json={"raw_log": "x"}
        ).status_code
        == 401
    )


@pytest.mark.parametrize("name,ecosystem,root_cause", CASES, ids=[c[0] for c in CASES])
def test_keeps_the_root_cause(client, auth, name: str, ecosystem: str, root_cause: str) -> None:
    raw = (FIXTURES / name).read_text()

    response = client.post("/v1/logs/process", headers=auth, json={"raw_log": raw})
    assert response.status_code == 200

    body = response.json()

    # Reduction is the easy metric and the least important one. An excerpt 99.9%
    # smaller that drops the actual error has made the system worse, not cheaper.
    assert root_cause in body["excerpt"], "root cause was lost"
    assert body["ecosystem"] == ecosystem
    assert len(body["signature_hash"]) == 64
    assert body["matched_lines"]


def test_the_same_failure_twice_produces_one_signature(client, auth) -> None:
    raw = (FIXTURES / "failed-db-test.log").read_text()

    first = client.post("/v1/logs/process", headers=auth, json={"raw_log": raw}).json()
    # A second run with different paths and timestamps is still the same bug.
    varied = raw.replace("/app/", "/srv/").replace(":42", ":915")
    second = client.post("/v1/logs/process", headers=auth, json={"raw_log": varied}).json()

    assert first["signature_hash"] == second["signature_hash"]


def test_secrets_never_reach_the_excerpt(client, auth) -> None:
    raw = (
        "$ deploy\n"
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
        "DATABASE_PASSWORD=hunter2\n"
        "export GH_TOKEN=ghp_" + "a" * 36 + "\n"
        "ERROR: deployment failed\n"
        "ERROR: Job failed: exit code 1\n"
    )

    body = client.post("/v1/logs/process", headers=auth, json={"raw_log": raw}).json()
    serialised = str(body)

    for secret in ("AKIAIOSFODNN7EXAMPLE", "hunter2", "ghp_aaaa"):
        assert secret not in serialised, f"{secret} leaked into the response"

    assert body["is_redacted"] is True
    assert body["redaction_count"] >= 3


def test_reduces_a_large_log(client, auth) -> None:
    # 20k lines of noise around one real error.
    raw = "\n".join(
        ["installing dependency package-" + str(i) for i in range(10_000)]
        + ["SQLSTATE[HY000] [2002] Connection refused"]
        + ["cleanup step " + str(i) for i in range(10_000)]
        + ["ERROR: Job failed: exit code 1"]
    )

    body = client.post("/v1/logs/process", headers=auth, json={"raw_log": raw}).json()

    assert body["reduction_ratio"] > 0.95
    assert "SQLSTATE[HY000] [2002] Connection refused" in body["excerpt"]
