"""Exports every request/response model as JSON Schema.

The contract between Laravel and this service is defined by Pydantic models
here, and Laravel has no way to see them. Exporting the schemas puts the shape
somewhere both sides can read, and makes a breaking change visible as a diff in
review rather than as a 422 in production.

Run after any model change:

    ./.venv/bin/python scripts/export_contracts.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import requests, responses

OUT = Path(__file__).resolve().parent.parent.parent / "PipeMind-data" / "contracts" / "v1"

# (module, model name, filename). Named explicitly rather than discovered, so a
# new internal model does not silently become part of the public contract.
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written, changed = 0, []

    for module, name, filename in EXPORTS:
        model = getattr(module, name, None)

        if model is None:
            print(f"  MISSING MODEL: {module.__name__}.{name}")
            return 1

        schema = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        path = OUT / filename

        if path.exists() and path.read_text() != schema:
            changed.append(filename)

        path.write_text(schema)
        written += 1
        print(f"  {filename}")

    print(f"\n{written} schemas written to {OUT}")

    if changed:
        print("\nCHANGED — add a CHANGELOG.md entry for each:")
        for filename in changed:
            print(f"  · {filename}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
