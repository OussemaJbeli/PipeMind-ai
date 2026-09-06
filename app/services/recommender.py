"""Recommendation sanitisation.

The model proposes; this constrains. Risk determines whether the policy engine
executes something automatically, so a model that could label a production
rollback "low risk" could talk its way past the safety layer. Risk is a property
of the action type, assigned here.
"""

from typing import Any

ACTION_RISK: dict[str, str] = {
    "investigate": "low",
    "retry_job": "low",
    "retry_pipeline": "low",
    "create_issue": "low",
    "edit_file": "medium",
    "update_dependency": "medium",
    "create_merge_request": "medium",
    "manual": "medium",
    "update_config": "high",
    "rollback_deployment": "critical",
}

_ESCALATION = {"medium": "high", "high": "critical"}

MAX_RECOMMENDATIONS = 5


def sanitize(recommendations: list[dict[str, Any]], *, is_default_branch: bool) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []

    for raw in recommendations[:MAX_RECOMMENDATIONS]:
        action = raw.get("action_type", "manual")

        if action not in ACTION_RISK:
            action = "manual"

        # Whatever the model claimed, risk comes from the action.
        risk = ACTION_RISK[action]

        # Anything touching the default branch escalates one level.
        if is_default_branch:
            risk = _ESCALATION.get(risk, risk)

        cleaned.append({
            **raw,
            "action_type": action,
            "risk": risk,
            "confidence": _clamp(raw.get("confidence", 0.5)),
            "affected_files": list(raw.get("affected_files") or [])[:20],
        })

    return cleaned


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5
