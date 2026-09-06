from fastapi import APIRouter

from app.models.requests import ClassifyRequest
from app.models.responses import ClassifyResponse
from app.services.classifier.hybrid import HybridClassifier

router = APIRouter(tags=["classify"])


@router.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    result = HybridClassifier().classify(request.text, request.ecosystem)

    return ClassifyResponse(
        category=result.category,
        subcategory=result.subcategory,
        confidence=result.confidence,
        source=result.source,
        matched_rules=result.matched_rules,
        probabilities=result.probabilities,
    )
