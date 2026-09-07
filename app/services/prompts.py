"""Prompt construction.

The quality of the analysis is decided here, not in the model choice. Every
element below is present for a reason recorded next to it.
"""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATES = Path(__file__).resolve().parent.parent / "prompts"

_env = Environment(
    loader=FileSystemLoader(_TEMPLATES),
    undefined=StrictUndefined,   # a missing variable must fail loudly, not silently
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,            # this is a prompt, not HTML
)

SYSTEM_PROMPT = """You are PipeMind, a CI/CD failure analyst. You explain why pipelines fail.

Rules you must follow:
- Ground every claim in the evidence provided. Never invent file names, line
  numbers, package versions, or error text that does not appear in the input.
- Distinguish what you OBSERVE from what you INFER. State inferences as inferences.
- If the evidence does not support a confident conclusion, say so and lower your
  confidence. A calibrated 0.55 is more useful than a fabricated 0.95.
- Prefer the simplest explanation consistent with all the evidence.
- When a historical failure with a confirmed resolution matches, lead with it.
- Recommendations must be concrete and safe. Never recommend an action that could
  destroy data or affect production without explicit human review.
- When you were shown the offending code and the fix is a small, certain edit,
  include a `patch`: a unified diff with correct @@ hunk headers, touching ONLY
  files whose contents or diff appear above. Quote the existing line exactly as
  given. If you are guessing at the surrounding lines, omit the patch entirely —
  a patch that does not apply is worse than none."""

# Schema-constrained decoding: with this set, a compliant provider cannot emit
# invalid JSON at all.
ANALYZE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["category", "severity", "confidence", "summary", "root_cause"],
    "properties": {
        "category": {
            "type": "string",
            "enum": [
                "BUILD", "TEST", "DEPENDENCY", "DATABASE", "NETWORK", "DOCKER",
                "DEPLOYMENT", "CONFIGURATION", "AUTHENTICATION", "PERMISSION",
                "INFRASTRUCTURE", "RESOURCE", "UNKNOWN",
            ],
        },
        "subcategory": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "confidence": {"type": "number"},
        "summary": {"type": "string"},
        "root_cause": {"type": "string"},
        "explanation": {"type": "string"},
        "is_transient": {"type": "boolean"},
        "retry_recommended": {"type": "boolean"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "content"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "log_line", "changed_file", "historical_failure",
                            "metric", "config", "commit", "doc",
                        ],
                    },
                    "content": {"type": "string"},
                    "source_ref": {"type": "string"},
                    "line_number": {"type": "integer"},
                    "weight": {"type": "number"},
                },
            },
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "action_type"],
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "rationale": {"type": "string"},
                    "action_type": {
                        "type": "string",
                        "enum": [
                            "investigate", "retry_job", "retry_pipeline", "create_issue",
                            "edit_file", "update_config", "update_dependency",
                            "create_merge_request", "rollback_deployment", "manual",
                        ],
                    },
                    "confidence": {"type": "number"},
                    "affected_files": {"type": "array", "items": {"type": "string"}},
                    # A unified diff the user can read and apply. Constrained
                    # hard on the way back out (see recommender.sanitize): a
                    # patch touching a file the model was never shown is a
                    # hallucination with a plausible shape, and the most
                    # damaging thing this system could produce.
                    "patch": {"type": "string"},
                },
            },
        },
    },
}


def render_analyze_prompt(request: Any, classification: Any, rag: Any, excerpt: str) -> str:
    template = _env.get_template("analyze.j2")

    return template.render(
        project=request.project,
        pipeline=request.pipeline,
        job=request.job,
        failure=request.failure,
        classification={
            "category": classification.category,
            "subcategory": classification.subcategory,
            "confidence": classification.confidence,
            "source": classification.source,
            "matched_rules": classification.matched_rules,
        },
        log={
            "excerpt": excerpt,
            "start_line": getattr(request, "excerpt_start_line", 1),
            "end_line": getattr(request, "excerpt_end_line", excerpt.count("\n") + 1),
            "total_lines": getattr(request, "total_lines", excerpt.count("\n") + 1),
        },
        stack_trace=request.stack_trace,
        source_context=[w.model_dump() for w in request.source_context],
        rag={
            "signature_history": getattr(rag, "signature_history", None) if rag else None,
            "similar_failures": [f.model_dump() for f in rag.similar_failures] if rag else [],
            "knowledge_chunks": getattr(rag, "knowledge_chunks", []) if rag else [],
        },
    )
