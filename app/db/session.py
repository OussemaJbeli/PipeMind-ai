"""Async database access.

The AI service reads Laravel's schema; it does not own it. Everything here is
read-only except embedding writes.
"""

from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Built lazily.

    Creating the engine at import time would make the whole service refuse to
    boot without a reachable database — and classification, redaction and the
    stub provider have no need of one.
    """
    return create_async_engine(
        settings().database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,   # WSL drops idle connections; without this the first query after a pause fails
        echo=False,
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_sessionmaker()() as session:
        yield session


async def ping() -> bool:
    from sqlalchemy import text

    try:
        async with get_sessionmaker()() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
