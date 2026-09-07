"""The exported contract must match the models.

Laravel cannot see Python type hints, so PipeMind-data/contracts/v1 is the only
place the agreement between the two services is written down. A model change that
does not reach the export means the contract silently describes something that no
longer exists — and the mismatch surfaces as a 422 in production rather than a
diff in review.
"""

import json
from pathlib import Path

import pytest

from app.models import requests, responses

CONTRACTS = Path(__file__).resolve().parents[3] / "PipeMind-data" / "contracts" / "v1"

EXPORTS = [
    (requests, "ProcessLogRequest", "logs-process.request.json"),
    (responses, "ProcessLogResponse", "logs-process.response.json"),
    (requests, "ClassifyRequest", "classify.request.json"),
    (responses, "ClassifyResponse", "classify.response.json"),
    (requests, "AnalyzeRequest", "analyze.request.json"),
    (responses, "AnalyzeResponse", "analyze.response.json"),
    (requests, "EmbedRequest", "embed.request.json"),
    (responses, "EmbedResponse", "embed.response.json"),
    (requests, "SimilarRequest", "similar.request.json"),
    (requests, "ChunkEmbedRequest", "knowledge-chunk-embed.request.json"),
    (responses, "ChunkEmbedResponse", "knowledge-chunk-embed.response.json"),
    (requests, "TestProviderRequest", "providers-test.request.json"),
    (responses, "TestProviderResponse", "providers-test.response.json"),
    (responses, "ServiceInfo", "info.response.json"),
]


@pytest.mark.parametrize(("module", "name", "filename"), EXPORTS, ids=[e[1] for e in EXPORTS])
def test_the_exported_schema_matches_the_model(module, name, filename):
    path = CONTRACTS / filename

    assert path.exists(), f"{filename} missing — run scripts/export_contracts.py"

    expected = getattr(module, name).model_json_schema()
    actual = json.loads(path.read_text())

    assert actual == expected, (
        f"{filename} is stale. Run:\n"
        f"  ./.venv/bin/python scripts/export_contracts.py\n"
        f"then add a CHANGELOG.md entry."
    )


def test_the_changelog_exists():
    """A shape change without an entry is a change nobody can review."""
    assert (CONTRACTS / "CHANGELOG.md").exists()


def test_contract_version_travels_in_both_directions():
    """A mismatch must be a loud error, not a silent wrong answer."""
    assert "contract_version" in requests.AnalyzeRequest.model_fields
    assert "contract_version" in responses.AnalyzeResponse.model_fields


def test_config_loads_from_any_working_directory():
    """`env_file=".env"` resolves relative to CWD.

    Launching uvicorn from the Laravel directory silently loaded *its* .env
    instead, and the service booted with an unparseable DATABASE_URL — surfacing
    as a 500 on /v1/analyze rather than as a configuration problem.
    """
    import os
    import tempfile

    from app.config import Settings

    original = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            config = Settings()

            assert config.database_url.startswith("postgresql")
            assert config.embedding_dim == 384
    finally:
        os.chdir(original)
