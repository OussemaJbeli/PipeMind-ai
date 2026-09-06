"""Request contract. Versioned — see PipeMind-data/contracts/v1/."""

from pydantic import BaseModel, Field


class ChangedFile(BaseModel):
    path: str
    change_type: str = "modified"
    additions: int = 0
    deletions: int = 0
    is_config: bool = False
    is_dependency: bool = False


class ProjectContext(BaseModel):
    uuid: str
    name: str
    tech_stack: list[str] = Field(default_factory=list)
    provider: str = "unknown"
    default_branch: str = "main"


class PipelineContext(BaseModel):
    iid: int | None = None
    ref: str = "unknown"
    source: str = "push"
    commit_sha: str | None = None
    commit_message: str | None = None
    duration_seconds: int | None = None
    # The single highest-signal fact available: regression vs pre-existing is
    # most of the diagnosis.
    previous_status: str | None = None
    changed_files: list[ChangedFile] = Field(default_factory=list)


class JobContext(BaseModel):
    name: str | None = None
    stage_name: str | None = None
    status: str = "failed"
    exit_code: int | None = None
    duration_seconds: int | None = None
    failure_reason: str | None = None
    baseline_duration_seconds: float | None = None


class FailureContext(BaseModel):
    uuid: str
    occurrence_index: int = 1
    is_flaky: bool = False
    signature_hash: str | None = None
    # Detected once at ingestion and persisted. Retrieval must embed the same
    # context the signature was hashed with, or neighbours drift.
    ecosystem: str | None = None


class ProcessLogRequest(BaseModel):
    raw_log: str
    job_name: str | None = None
    stage_name: str | None = None
    exit_code: int | None = None
    strict_redaction: bool = True


class ClassifyRequest(BaseModel):
    text: str
    ecosystem: str | None = None


class EmbedRequest(BaseModel):
    text: str
    # When given, the vector is composed the same way retrieval composes it,
    # so a text embedded here matches one embedded during analysis.
    category: str | None = None
    ecosystem: str | None = None
    job_name: str | None = None


class ChunkEmbedRequest(BaseModel):
    text: str
    title: str | None = None
    target_tokens: int = 400
    overlap_tokens: int = 50


class SimilarRequest(BaseModel):
    team_id: int
    error_message: str
    category: str | None = None
    ecosystem: str | None = None
    job_name: str | None = None
    exclude_failure_id: int | None = None
    limit: int | None = None
    threshold: float | None = None


class AnalyzeRequest(BaseModel):
    contract_version: str = "v1"
    team_id: int

    failure: FailureContext
    project: ProjectContext
    pipeline: PipelineContext
    job: JobContext

    # Already redacted by /v1/logs/process — the analyzer re-checks anyway.
    log_excerpt: str = ""
    error_block: str | None = None
    stack_trace: str | None = None

    use_rag: bool = True
    use_llm: bool = True
    # Bypasses the known-signature short-circuit. Set by the "Re-analyze" action
    # when a user disagrees with the stored resolution.
    force_llm: bool = False
    # Provider + key travel per request so this service stores no team's
    # credentials at rest.
    llm_override: dict | None = None
