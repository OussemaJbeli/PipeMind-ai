"""Request contract. Versioned — see PipeMind-data/contracts/v1/."""

from pydantic import BaseModel, Field

from app.services.chunker import OVERLAP_TOKENS, TARGET_TOKENS


class ChangedFile(BaseModel):
    path: str
    change_type: str = "modified"
    additions: int = 0
    deletions: int = 0
    is_config: bool = False
    is_dependency: bool = False
    # The unified diff hunk. Both GitHub and GitLab return it in the same call
    # that produces the counts above; without it the model is given a filename
    # and asked to explain what went wrong inside it.
    patch: str | None = None
    patch_truncated: bool = False


class SourceWindow(BaseModel):
    """Lines around a location a stack trace pointed at.

    The `line` is marked with `>` in `content`, because a model asked to count
    lines to find the referenced one counts badly.
    """

    path: str
    line: int
    start_line: int
    end_line: int
    content: str


class ProjectContext(BaseModel):
    # Numeric id alongside the uuid: knowledge_documents.project_id is a bigint
    # and the AI service reads that table directly, so retrieval needs the id to
    # scope project documents. Optional for backwards compatibility — a payload
    # without it falls back to team-wide knowledge only.
    id: int | None = None
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


class TestProviderRequest(BaseModel):
    """The same shape as `llm_override`, so the wizard tests exactly what will
    later be sent — testing a different shape proves nothing."""

    provider: str
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None


class ChunkEmbedRequest(BaseModel):
    text: str
    title: str | None = None
    # Bound to the chunker's constants, never restated. These were literal
    # 400/50 copies, which silently shadowed the chunker: retuning chunk size
    # there changed nothing for the only caller that matters, because every
    # request arrived carrying the stale defaults explicitly.
    target_tokens: int = TARGET_TOKENS
    overlap_tokens: int = OVERLAP_TOKENS


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

    # Resolved from the stack trace at request time. Empty is normal: the file
    # may be gitignored, generated, or the provider may host no repository.
    source_context: list[SourceWindow] = Field(default_factory=list)

    use_rag: bool = True
    use_llm: bool = True
    # Bypasses the known-signature short-circuit. Set by the "Re-analyze" action
    # when a user disagrees with the stored resolution.
    force_llm: bool = False
    # Provider + key travel per request so this service stores no team's
    # credentials at rest.
    llm_override: dict | None = None
