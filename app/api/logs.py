from fastapi import APIRouter

from app.config import settings
from app.models.requests import ProcessLogRequest
from app.models.responses import ProcessLogResponse
from app.services import log_processor, redactor, signature
from app.services.classifier.hybrid import HybridClassifier

router = APIRouter(tags=["logs"])


@router.post("/logs/process", response_model=ProcessLogResponse)
async def process_log(request: ProcessLogRequest) -> ProcessLogResponse:
    """Redact, clean, extract, and produce a failure signature.

    The order is not negotiable: redaction runs first because every later stage
    copies text around.
    """
    config = settings()
    original_length = len(request.raw_log)

    # 1. REDACT
    redacted = redactor.redact(request.raw_log, strict=request.strict_redaction)

    # 2. CLEAN
    cleaned = log_processor.clean(redacted.text)

    # 3. EXTRACT
    extraction = log_processor.extract(
        cleaned,
        context=config.excerpt_context_lines,
        max_chars=config.max_log_chars,
    )

    # 4. SIGNATURE — from the error block, not the whole excerpt. Including the
    # surrounding context would make every occurrence unique and defeat dedup.
    basis = extraction.error_block or extraction.error_message or cleaned[-2000:]
    signature_hash, normalized = signature.signature_hash(basis, ecosystem=extraction.ecosystem)

    # 5. CLASSIFY on the error block, not the whole excerpt: surrounding context
    # dilutes the signal and pulls in unrelated rule matches.
    classification = HybridClassifier().classify(basis, extraction.ecosystem)

    return ProcessLogResponse(
        excerpt=extraction.excerpt,
        excerpt_start_line=extraction.excerpt_start_line,
        excerpt_end_line=extraction.excerpt_end_line,
        error_block=extraction.error_block,
        stack_trace=extraction.stack_trace,
        error_message=extraction.error_message,
        ecosystem=extraction.ecosystem,
        exit_code=extraction.exit_code or request.exit_code,
        matched_lines=extraction.matched_lines,
        signature_hash=signature_hash,
        normalized_error=normalized,
        category=classification.category,
        subcategory=classification.subcategory,
        classification_confidence=classification.confidence,
        classification_source=classification.source,
        is_redacted=redacted.count > 0,
        redaction_count=redacted.count,
        redaction_types=redacted.types,
        original_chars=original_length,
        excerpt_chars=len(extraction.excerpt),
        reduction_ratio=round(1 - len(extraction.excerpt) / max(original_length, 1), 4),
    )
