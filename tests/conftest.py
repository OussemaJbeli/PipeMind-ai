import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-PipeMind-Token": settings().service_token}
