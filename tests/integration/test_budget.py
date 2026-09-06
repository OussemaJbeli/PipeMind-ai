"""Budget enforcement, against the live schema.

Every case runs inside a transaction that is rolled back, so the dev database
is left exactly as it was found.
"""

import pytest
from sqlalchemy import text

from app.core.budget import assert_within_budget
from app.core.errors import BudgetExceeded
from app.db.session import get_sessionmaker, ping

pytestmark = pytest.mark.skipif(
    not __import__("asyncio").run(ping()),
    reason="no database reachable",
)

TEAM_ID = 1


@pytest.mark.asyncio
async def test_a_team_under_its_ceiling_is_allowed():
    async with get_sessionmaker()() as session:
        await assert_within_budget(session, TEAM_ID)


@pytest.mark.asyncio
async def test_a_zero_budget_stops_the_call_before_it_costs_anything():
    async with get_sessionmaker()() as session:
        await session.begin()
        await session.execute(
            text("UPDATE teams SET monthly_ai_budget_usd = 0.01 WHERE id = :id"),
            {"id": TEAM_ID},
        )
        await session.execute(text("""
            INSERT INTO ai_requests (team_id, operation, provider, model, cost_usd, status, created_at)
            VALUES (:id, 'analyze', 'test', 'test', 5.00, 'success', now())
        """), {"id": TEAM_ID})

        with pytest.raises(BudgetExceeded) as caught:
            await assert_within_budget(session, TEAM_ID)

        # The message has to tell the user what to do about it.
        assert "Workspace" in str(caught.value)

        await session.rollback()


@pytest.mark.asyncio
async def test_failed_requests_do_not_consume_the_budget():
    """A provider error we never got tokens back from must not eat the ceiling."""
    async with get_sessionmaker()() as session:
        await session.begin()
        await session.execute(
            text("UPDATE teams SET monthly_ai_budget_usd = 1.00 WHERE id = :id"),
            {"id": TEAM_ID},
        )
        await session.execute(text("""
            INSERT INTO ai_requests (team_id, operation, provider, model, cost_usd, status, created_at)
            VALUES (:id, 'analyze', 'test', 'test', 99.00, 'error', now())
        """), {"id": TEAM_ID})

        await assert_within_budget(session, TEAM_ID)

        await session.rollback()


@pytest.mark.asyncio
async def test_a_team_with_no_ceiling_is_not_blocked():
    async with get_sessionmaker()() as session:
        await assert_within_budget(session, 999_999)
