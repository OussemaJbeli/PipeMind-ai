from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models.requests import AnalyzeRequest
from app.models.responses import AnalyzeResponse
from app.services.analyzer import analyze as run_analysis

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest, session: AsyncSession = Depends(get_session)
) -> AnalyzeResponse:
    """Full failure analysis: classify, retrieve, and explain."""
    return await run_analysis(request, session)
