from fastapi import APIRouter

from app.api import analyze, classify, health, logs, providers, vectors

router = APIRouter()
router.include_router(health.router)
router.include_router(logs.router)
router.include_router(classify.router)
router.include_router(analyze.router)
router.include_router(vectors.router)
router.include_router(providers.router)
