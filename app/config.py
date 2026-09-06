"""Runtime configuration, read once from the environment."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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
    gemini_model: str = "gemini-2.0-flash"
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
    max_knowledge_chunks: int = 4

    # --- Cost --------------------------------------------------------------
    monthly_cost_ceiling_usd: float = 25.0


@lru_cache
def settings() -> Settings:
    return Settings()
