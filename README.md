# PipeMind AI

The intelligence layer: log processing, secret redaction, failure classification,
and (from roadmap 09) embeddings, similarity search and LLM reasoning.

Owns no application state. It reads Postgres for similarity and knowledge, writes
only embeddings, and never receives infrastructure credentials — Laravel remains
the execution boundary.

## Setup

```bash
cp .env.example .env
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/uvicorn app.main:app --port 8001
```

`torch` and `sentence-transformers` are in the optional `ml` extra (~2 GB) and are
not needed for log processing or classification:

```bash
./.venv/bin/pip install -e ".[dev,ml,llm]"   # once embeddings are in play
```

## Endpoints

| | | |
|---|---|---|
| `GET` | `/health` | liveness, no auth |
| `GET` | `/v1/info` | version, contract, configured provider |
| `GET` | `/v1/classifier` | rule count, categories, whether an ML model is loaded |
| `POST` | `/v1/logs/process` | redact → clean → extract → signature → classify |
| `POST` | `/v1/classify` | category only |

Everything under `/v1` requires `X-PipeMind-Token`, checked with
`hmac.compare_digest`. The service is never exposed publicly; its only client is
Laravel.

## The order is not negotiable

```
redact  →  clean  →  extract  →  signature  →  classify
```

Redaction runs **first** because every later stage copies text around: a secret
that survives stage one is in four places by stage four. `RedactionFailed` is
deliberately non-retryable — a retry loop around a redaction bug is a loop that
keeps trying to ship secrets to a third party.

## Design notes

**Window the first error, never the tail.** A dependency conflict at line 400
produces a build failure at line 8,000 and a "job failed" at 8,100. Leading with
the tail yields a confident, useless analysis of the symptom.

**Symptoms don't dilute causes.** A log containing "connection refused" *and*
"1 test failed" is a confident `DATABASE` failure, not a coin toss. `TEST` and
`BUILD` are excluded from the confidence denominator when a cause category wins —
otherwise an obvious case escalates to the LLM for nothing.

**Dominance is measured per category, not per rule.** Two `TEST` rules firing is
corroboration, not ambiguity.

**`UNKNOWN` is a real answer.** Forcing a guess pollutes every other class.

**The ML classifier is optional.** It does not exist until `scripts/train.py` has
run against a labelled dataset — and that dataset comes from
`analysis_feedback.correct_category`, i.e. from people using the product. Until
then the hybrid falls back to rules, which is why M3 does not depend on it.

## Which LLM

`LLM_PROVIDER=stub` is the default: deterministic, no key, no cost. It builds the
whole pipeline end to end and labels its own output so a stubbed analysis can
never be mistaken for a real one. Switch to `gemini` by setting
`GEMINI_API_KEY` — nothing else changes.

Embeddings are a **separate** choice and are local by default (`all-MiniLM-L6-v2`,
CPU, 384 dimensions, no key). Similarity search and RAG cost nothing and work
offline regardless of which LLM is configured.

## Tests

```bash
./.venv/bin/python -m pytest tests -q
./.venv/bin/python -m ruff check .
```

The redaction matrix carries a positive case per rule **and** the negatives that
matter: a commit SHA, an image digest, a version string and a stack-trace path
must all survive. Over-redaction removes the most useful tokens in the log and
makes the analysis worse, not safer.

Build plan: `../PipeMind-data/roadmaps/06-ai-service-setup.md` and `07-ai-log-processing.md`.
