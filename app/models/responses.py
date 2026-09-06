"""Response contract. Versioned — see PipeMind-data/contracts/v1/."""

from pydantic import BaseModel, Field


class ProcessLogResponse(BaseModel):
    excerpt: str
    excerpt_start_line: int
    excerpt_end_line: int
    error_block: str | None = None
    stack_trace: str | None = None
    error_message: str | None = None
    ecosystem: str | None = None
    exit_code: int | None = None
    # Lines the processor identified as the error, so the UI can highlight them
    # and scroll there instead of showing line 1 of 48,000.
    matched_lines: list[int] = Field(default_factory=list)

    signature_hash: str
    normalized_error: str

    # Classified here rather than in a second call: it is ~2 ms of regex over
    # data this endpoint already has, and one round trip beats two.
    category: str = "UNKNOWN"
    subcategory: str | None = None
    classification_confidence: float = 0.0
    classification_source: str = "rules"

    is_redacted: bool = False
    redaction_count: int = 0
    redaction_types: list[str] = Field(default_factory=list)

    original_chars: int
    excerpt_chars: int
    reduction_ratio: float


class ClassifyResponse(BaseModel):
    category: str
    subcategory: str | None = None
    confidence: float = Field(ge=0, le=1)
    source: str
    matched_rules: list[str] = Field(default_factory=list)
    probabilities: dict[str, float] = Field(default_factory=dict)


class EmbedResponse(BaseModel):
    embedding: list[float]
    dimensions: int
    model: str
    # Echoed back so Laravel stores exactly what was embedded, not what it sent.
    source_text: str = ""


class EmbeddedChunk(BaseModel):
    index: int
    content: str
    token_count: int
    embedding: list[float]


class ChunkEmbedResponse(BaseModel):
    chunks: list[EmbeddedChunk]
    model: str
    dimensions: int


class Evidence(BaseModel):
    type: str
    content: str
    source_ref: str | None = None
    line_number: int | None = None
    related_failure_uuid: str | None = None
    weight: float = Field(0.5, ge=0, le=1)


class Recommendation(BaseModel):
    title: str
    description: str | None = None
    rationale: str | None = None
    action_type: str = "investigate"
    # Assigned from action_type by code, never taken from the model.
    risk: str = "medium"
    confidence: float = Field(0.5, ge=0, le=1)
    affected_files: list[str] = Field(default_factory=list)
    patch: str | None = None


class SimilarFailure(BaseModel):
    uuid: str
    similarity: float
    project_name: str
    occurred_at: str | None = None
    root_cause: str | None = None
    resolution: str | None = None
    resolved: bool = False


class Usage(BaseModel):
    provider: str = "none"
    model: str = "none"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cache_hit: bool = False


class AnalyzeResponse(BaseModel):
    contract_version: str = "v1"
    service_version: str

    category: str
    subcategory: str | None = None
    severity: str = "medium"
    confidence: float = Field(ge=0, le=1)

    summary: str
    root_cause: str
    explanation: str | None = None

    is_transient: bool = False
    retry_recommended: bool = False

    evidence: list[Evidence] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    similar_failures: list[SimilarFailure] = Field(default_factory=list)

    # Provenance: answers "how often did we actually need the LLM?" with data.
    classification_source: str = "rules"
    classification_confidence: float = 0.0
    used_rag: bool = False

    usage: Usage = Field(default_factory=Usage)


class ServiceInfo(BaseModel):
    service: str = "pipemind-ai"
    version: str
    contract_version: str
    llm_provider: str
    llm_ready: bool
    embedding_model: str
    embedding_dim: int
    embeddings_ready: bool = False
    database: bool
    providers: dict[str, bool] = {}
