"""Orchestrates the analysis pipeline.

    redact → classify → retrieve → (short-circuit?) → budget → LLM → calibrate → sanitise

The ordering is the design. Redaction runs before anything that could transmit
text. Classification runs before retrieval because the category sharpens the
embedding. Retrieval runs before the LLM because a confirmed historical fix can
make the call unnecessary. The budget is checked last, immediately before the
only step that costs money.
"""

import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.budget import assert_within_budget
from app.core.errors import InvalidLLMResponse
from app.core.logging import log
from app.models.requests import AnalyzeRequest
from app.models.responses import AnalyzeResponse, Evidence, Recommendation, Usage
from app.providers.registry import build_provider_with_fallback
from app.services import recommender, redactor
from app.services.classifier.hybrid import Classification, HybridClassifier
from app.services.prompts import ANALYZE_SCHEMA, SYSTEM_PROMPT, render_analyze_prompt
from app.services.rag import RetrievedContext, retrieve_safely

MAX_EVIDENCE = 8
SHORT_CIRCUIT_MIN_CONFIDENCE = 0.85


async def analyze(request: AnalyzeRequest, session: AsyncSession | None = None) -> AnalyzeResponse:
    started = time.perf_counter()
    config = settings()

    # 0. Defence in depth. Laravel should have sent redacted text; verify anyway.
    #    A second pass over already-clean text is a no-op and costs nothing.
    safe = redactor.redact(request.log_excerpt, strict=config.redaction_strict)
    excerpt = safe.text

    # 1. CLASSIFY on the error block: surrounding context dilutes the signal.
    basis = request.error_block or excerpt
    classification = HybridClassifier().classify(basis, request.failure.ecosystem)

    # 2. RETRIEVE
    rag: RetrievedContext | None = None

    if request.use_rag and session is not None:
        rag = await retrieve_safely(
            session,
            team_id=request.team_id,
            project_id=request.project.id,
            error_message=basis[:600],
            category=classification.category,
            ecosystem=request.failure.ecosystem,
            job_name=request.job.name,
            signature_hash=request.failure.signature_hash,
        )

    # 3. SHORT-CIRCUIT. A known signature with a confirmed fix and a confident
    #    classifier needs no model call: the cheapest, fastest and most
    #    trustworthy path there is. Take it whenever it is available — the only
    #    escape is an explicit force_llm, which is what "Re-analyze" sends when a
    #    user disputes the stored resolution.
    known = rag.signature_history if rag else None

    if (known and known.get("is_known")
            and classification.confidence >= SHORT_CIRCUIT_MIN_CONFIDENCE
            and not request.force_llm):
        log.info("analysis.short_circuit", signature=(known.get("hash") or "")[:12],
                 occurrences=known.get("occurrence_count"))

        return _from_known_signature(request, classification, known, rag, started)

    if not request.use_llm:
        return _from_classification_only(request, classification, rag, started)

    # 4. LLM. Budget checked here and nowhere else: this is the only step that
    #    spends money, and a ceiling enforced after the fact is a report.
    if session is not None:
        await assert_within_budget(session, request.team_id)

    provider = build_provider_with_fallback(request.llm_override)
    prompt = render_analyze_prompt(request, classification, rag, excerpt)

    response = await provider.complete(
        system=SYSTEM_PROMPT,
        prompt=prompt,
        json_schema=ANALYZE_SCHEMA,
        temperature=config.llm_temperature,
        timeout=config.llm_timeout_seconds,
    )

    data = response.parsed

    if not data:
        raise InvalidLLMResponse("The model returned an empty structured response.")

    is_default_branch = request.pipeline.ref == request.project.default_branch

    return AnalyzeResponse(
        service_version=config.service_version,
        category=data.get("category") or classification.category,
        subcategory=data.get("subcategory") or classification.subcategory,
        severity=data.get("severity", "medium"),
        confidence=_calibrate(data.get("confidence", 0.5), classification, rag),
        summary=str(data.get("summary", ""))[:500],
        root_cause=str(data.get("root_cause", "")),
        explanation=data.get("explanation"),
        is_transient=bool(data.get("is_transient", False)),
        retry_recommended=bool(data.get("retry_recommended", False)),
        evidence=[Evidence(**e) for e in _valid_evidence(data.get("evidence", []))][:MAX_EVIDENCE],
        recommendations=[
            Recommendation(**r)
            for r in recommender.sanitize(
                data.get("recommendations", []),
                is_default_branch=is_default_branch,
                known_files=_shown_files(request),
            )
        ],
        similar_failures=rag.similar_failures if rag else [],
        classification_source=classification.source,
        classification_confidence=classification.confidence,
        used_rag=bool(rag and rag.used),
        usage=Usage(
            provider=response.provider,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=provider.cost(response.prompt_tokens, response.completion_tokens),
            latency_ms=int((time.perf_counter() - started) * 1000),
        ),
    )


def _shown_files(request: AnalyzeRequest) -> set[str]:
    """Paths the model actually saw the contents of.

    Only files whose diff or source appeared in the prompt. A changed file with
    no patch is deliberately excluded: the model was told it changed, not what
    changed in it, so a patch for it would be invention.
    """
    return {
        *(f.path for f in request.pipeline.changed_files if f.patch),
        *(w.path for w in request.source_context),
    }


def _valid_evidence(items: list[Any]) -> list[dict]:
    """Drop malformed entries rather than failing the whole analysis.

    Losing one citation is far better than losing the root cause.
    """
    valid = []

    for item in items:
        if not isinstance(item, dict) or not item.get("type") or not item.get("content"):
            continue

        valid.append({
            "type": item["type"],
            "content": str(item["content"])[:2000],
            "source_ref": item.get("source_ref"),
            "line_number": item.get("line_number"),
            "related_failure_uuid": item.get("related_failure_uuid"),
            "weight": _clamp(item.get("weight", 0.5)),
        })

    return valid


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _calibrate(raw: Any, classification: Classification, rag: RetrievedContext | None) -> float:
    """Adjust the model's stated confidence against independent signals.

    An LLM's self-reported confidence is not calibrated. Corroboration from the
    rule engine and from history is evidence the model does not get to grade
    itself on.
    """
    score = _clamp(raw)

    if classification.source == "hybrid" and classification.confidence >= 0.85:
        score = min(0.97, score + 0.05)

    if rag and rag.signature_history and rag.signature_history.get("is_known"):
        score = min(0.98, score + 0.08)

    if rag and not rag.used:
        # Retrieval ran and found nothing — first time we have seen anything
        # like this. (rag is None means it never ran, which is not evidence.)
        score = min(score, 0.85)

    if classification.category == "UNKNOWN":
        score = min(score, 0.70)

    return round(score, 3)


def _from_known_signature(
    request: AnalyzeRequest, classification: Classification,
    known: dict, rag: RetrievedContext | None, started: float,
) -> AnalyzeResponse:
    config = settings()
    occurrences = known.get("occurrence_count") or 0

    return AnalyzeResponse(
        service_version=config.service_version,
        category=known.get("category") or classification.category,
        subcategory=known.get("subcategory") or classification.subcategory,
        severity="high" if request.pipeline.ref == request.project.default_branch else "medium",
        confidence=0.95,
        summary=(known.get("known_root_cause") or "A known failure signature recurred.")[:500],
        root_cause=known.get("known_root_cause") or "",
        explanation=(
            f"This exact error signature has been seen {occurrences} time(s) in this "
            f"workspace and carries a confirmed resolution, so no model was consulted."
        ),
        is_transient=False,
        retry_recommended=False,
        evidence=[
            Evidence(
                type="historical_failure",
                content=f"Confirmed resolution: {known.get('known_resolution')}",
                source_ref=f"signature:{(known.get('hash') or '')[:12]}",
                weight=0.95,
            )
        ],
        recommendations=[
            Recommendation(
                title=(known.get("known_resolution") or "Apply the known resolution")[:200],
                description="This resolved the same signature previously.",
                action_type="investigate",
                risk="low",
                confidence=0.95,
            )
        ],
        similar_failures=rag.similar_failures if rag else [],
        classification_source=classification.source,
        classification_confidence=classification.confidence,
        used_rag=True,
        # No model was called: zero cost, and the usage row proves it.
        usage=Usage(provider="none", model="known_signature", cache_hit=True,
                    latency_ms=int((time.perf_counter() - started) * 1000)),
    )


def _from_classification_only(
    request: AnalyzeRequest, classification: Classification,
    rag: RetrievedContext | None, started: float,
) -> AnalyzeResponse:
    config = settings()

    return AnalyzeResponse(
        service_version=config.service_version,
        category=classification.category,
        subcategory=classification.subcategory,
        severity="medium",
        confidence=classification.confidence,
        summary=f"{classification.category} failure detected by rule classification.",
        root_cause="No LLM analysis was requested for this failure.",
        similar_failures=rag.similar_failures if rag else [],
        classification_source=classification.source,
        classification_confidence=classification.confidence,
        used_rag=bool(rag and rag.used),
        usage=Usage(provider="none", model="rules",
                    latency_ms=int((time.perf_counter() - started) * 1000)),
    )
