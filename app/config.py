"""Runtime configuration, read once from the environment."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to the package, not the working directory.
#
# A bare ".env" resolves relative to CWD, so launching uvicorn from the Laravel
# directory silently loaded *its* .env instead — and the service booted with a
# DATABASE_URL it could not parse, reporting a 500 on /v1/analyze rather than a
# configuration problem. The service must find its own config wherever it is
# started from.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    app_env: Literal["local", "testing", "staging", "production"] = "local"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    service_version: str = "0.1.0"
    contract_version: str = "v1"

    # Must equal AI_SERVICE_TOKEN in the Laravel .env.
    service_token: str = "change-me-shared-secret"

    database_url: str = ""
    redis_url: str = "redis://localhost:6380/3"

    # --- LLM ---------------------------------------------------------------
    # `stub` is deterministic and needs no key: files 07-16 are fully buildable
    # without one. Switching to gemini is a single env var.
    llm_provider: Literal["stub", "gemini", "ollama", "openai_compatible"] = "stub"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"
    # Gemini 3.x reasons before answering, and that reasoning is most of the wall
    # clock: measured on one real failure, unset = 11.1 s / 1462 thinking tokens,
    # "high" = 16.4 s / 2330, "low" = 3.2 s / 0. CI failure analysis is a
    # short-context, heavily-grounded task where the rule classifier has already
    # narrowed the category, so the extra reasoning buys little. Raise it for a
    # project whose failures are genuinely ambiguous.
    gemini_thinking_level: Literal["low", "high", "default"] = "low"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    llm_timeout_seconds: int = 90
    llm_max_retries: int = 2
    llm_temperature: float = 0.2
    max_context_tokens: int = 8000

    # --- Embeddings (independent of the LLM: local, CPU, no key) ------------
    embedding_provider: Literal["local", "gemini"] = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    # --- Log processing ----------------------------------------------------
    max_log_chars: int = 60_000
    excerpt_context_lines: int = 15
    redaction_strict: bool = True

    # --- Retrieval ---------------------------------------------------------
    # 0.75 was the initial guess and returns nothing: MiniLM scores genuinely
    # related-but-not-identical CI failures in the 0.50-0.65 band, so the guess
    # made retrieval silently return zero rows. See
    # PipeMind-data/experiments/similarity-threshold.md. Provisional until
    # calibrated on labelled pairs from real data.
    similarity_threshold: float = 0.55
    max_similar_failures: int = 5

    # Knowledge chunks are scored against a short error string, so they sit
    # lower than failure-to-failure pairs: a runbook that names the exact error
    # lands at 0.42-0.80, unrelated docs at -0.07-0.45. 0.60 was the initial
    # guess and hid a third of genuine matches. 0.40 recovered 6/6 true matches
    # for 1 weak extra chunk in 48. See
    # PipeMind-data/experiments/knowledge-retrieval.md.
    knowledge_threshold: float = 0.40

    # Seeds are chunks that matched on similarity; the neighbours pulled in
    # around them are what usually hold the fix. Kept separate because raising
    # the seed count widens the topics considered, while raising the radius
    # deepens the context on topics already found — different decisions.
    max_knowledge_seeds: int = 3
    # 2, because the radius is a function of chunk size and must move with it.
    # At 150 tokens a runbook's symptom -> cause -> fix arc spans about three
    # chunks, and the seed almost always lands on the symptom: measured on a
    # real runbook, radius 1 reached the restatement of the problem and stopped
    # one chunk short of the remedy.
    knowledge_neighbour_radius: int = 2

    # The cap on what actually reaches the prompt, after expansion. Chunks are
    # 150 tokens now (see chunker.py), so six is ~900 prompt tokens — close to
    # the old four-at-400 budget, but each one earned its place.
    max_knowledge_chunks: int = 6

    # --- Cost --------------------------------------------------------------
    monthly_cost_ceiling_usd: float = 25.0


@lru_cache
def settings() -> Settings:
    return Settings()
