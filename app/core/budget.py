"""Monthly cost ceiling.

Checked before the call, not after: a budget that only notices once the money is
spent is a report, not a limit.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import BudgetExceeded
from app.core.logging import log

WARN_AT = 0.80


async def assert_within_budget(session: AsyncSession, team_id: int) -> None:
    row = await session.execute(text("""
        SELECT
            (SELECT COALESCE(SUM(cost_usd), 0) FROM ai_requests
              WHERE team_id = :tid AND status = 'success'
                AND created_at >= date_trunc('month', now())) AS spent,
            (SELECT monthly_ai_budget_usd FROM teams WHERE id = :tid) AS budget
    """), {"tid": team_id})

    spent, budget = row.one()

    if not budget:
        return

    spent, budget = float(spent), float(budget)

    if spent >= budget:
        raise BudgetExceeded(
            f"Monthly AI budget reached (${spent:.2f} of ${budget:.2f}). "
            f"Raise it in Workspace → Settings, or switch this project to a local model."
        )

    if spent >= budget * WARN_AT:
        log.warning("budget.approaching", team_id=team_id, spent=spent, budget=budget,
                    pct=round(spent / budget * 100, 1))
