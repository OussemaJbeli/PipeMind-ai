"""Recommendation sanitisation.

The model proposes; this constrains. Risk determines whether the policy engine
executes something automatically, so a model that could label a production
rollback "low risk" could talk its way past the safety layer. Risk is a property
of the action type, assigned here.
"""

import re
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

_RISK_ORDER = ["low", "medium", "high", "critical"]

# File changes the model proposes as a diff. Laravel has exactly one way to
# apply a file change — branch, commit, open a merge request — so a diff that
# survives validation is a merge-request proposal whatever the model called it.
# Without this promotion the best recommendations PipeMind produces are
# unexecutable: they arrive as `edit_file`, which has no executor, so an
# applyable patch can never become a pull request.
_PATCHABLE = {"edit_file", "update_config", "update_dependency", "manual"}

MAX_RECOMMENDATIONS = 5


def sanitize(
    recommendations: list[dict[str, Any]],
    *,
    is_default_branch: bool,
    known_files: set[str] | None = None,
) -> list[dict[str, Any]]:
    """`known_files` are the paths the model was actually shown.

    A patch touching anything else is discarded. A hallucinated diff has exactly
    the same shape as a real one, and is the most damaging thing this system
    could hand a user — so the check is structural, not a matter of trusting the
    model to have been careful.
    """
    cleaned: list[dict[str, Any]] = []

    for raw in recommendations[:MAX_RECOMMENDATIONS]:
        action = raw.get("action_type", "manual")

        if action not in ACTION_RISK:
            action = "manual"

        # Whatever the model claimed, risk comes from the action.
        risk = ACTION_RISK[action]

        patch = validate_patch(raw.get("patch"), known_files)

        if patch and action in _PATCHABLE:
            action = "create_merge_request"
            # The higher of the two risks, never the promoted action's own.
            # update_config is HIGH and create_merge_request is MEDIUM, so
            # taking the new action's risk would quietly relax the gate — and
            # the whole point of assigning risk in code is that nothing can.
            risk = max(risk, ACTION_RISK[action], key=_RISK_ORDER.index)

        # Anything touching the default branch escalates one level.
        if is_default_branch:
            risk = _ESCALATION.get(risk, risk)

        cleaned.append({
            **raw,
            "action_type": action,
            "risk": risk,
            "confidence": _clamp(raw.get("confidence", 0.5)),
            "affected_files": list(raw.get("affected_files") or [])[:20],
            "patch": patch,
        })

    return cleaned


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


# A unified-diff file header: "--- a/path" / "+++ b/path".
_DIFF_FILE = re.compile(r"^(?:---|\+\+\+)\s+(?:[ab]/)?(\S+)", re.MULTILINE)
_HUNK = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@", re.MULTILINE)

MAX_PATCH_CHARS = 8_000


def validate_patch(patch: Any, known_files: set[str] | None) -> str | None:
    """Returns the patch only if it is structurally sound and in scope.

    Four ways to fail, all of which produce a plausible-looking diff:

    - not a string, or empty
    - no @@ hunk header, so nothing could apply it
    - longer than a human will read, which means the model rewrote a file
    - touches a path that never appeared in the prompt

    The last is the one that matters. Everything else is a broken patch the user
    would notice; that one is a confident edit to a file the model invented.
    """
    if not isinstance(patch, str) or not patch.strip():
        return None

    if len(patch) > MAX_PATCH_CHARS:
        return None

    if not _HUNK.search(patch):
        return None

    referenced = {
        path for path in _DIFF_FILE.findall(patch)
        if path not in {"/dev/null"}
    }

    if not referenced:
        return None

    if known_files is not None:
        # Compared on the trailing path, because a diff header may carry a
        # repository-relative path where the prompt carried a fuller one.
        allowed = {f.lstrip("./") for f in known_files}

        for path in referenced:
            candidate = path.lstrip("./")

            if not any(
                candidate == a or candidate.endswith("/" + a) or a.endswith("/" + candidate)
                for a in allowed
            ):
                return None

    return patch
