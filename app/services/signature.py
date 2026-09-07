"""Failure signatures.

The load-bearing column: one hash gives deduplication, clustering, the LLM
response cache key, and the similarity join — all at once.

Tuning: if occurrence_count is always 1 the normalisation is too weak; if
distinct bugs share a signature it is too strong. Check with
    SELECT occurrence_count, count(*) FROM failure_signatures GROUP BY 1;
A healthy distribution is a long tail of 1s with a real head of repeat offenders.
"""

import hashlib
import re
from re import Pattern

# Order matters: broad patterns last, or they swallow the specific ones.
NORMALIZERS: list[tuple[Pattern[str], str]] = [
    (re.compile(r"\[REDACTED[_A-Z]*\]"), "<secret>"),
    (re.compile(r"\b[0-9a-f]{40}\b", re.IGNORECASE), "<sha>"),
    (re.compile(r"\bsha(?:1|256|512)[:\-][0-9a-f]{32,128}\b", re.IGNORECASE), "<digest>"),
    (
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
        ),
        "<uuid>",
    ),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*"), "<timestamp>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<ip>"),
    (re.compile(r"\b[a-f0-9]{12}\b", re.IGNORECASE), "<container_id>"),
    (re.compile(r"(?:/[\w.\-]+){2,}"), "<path>"),
    # Line numbers, in both forms. Must follow the path rule and precede the
    # generic <num> rule, which only catches 3+ digits and would otherwise leave
    # ":42" alone while normalising ":915" — the same bug on two different lines
    # would then produce two different signatures.
    (re.compile(r"(<path>):\d+"), r"\1:<line>"),
    (
        re.compile(
            r"(\.(?:php|py|js|ts|jsx|tsx|go|rb|java|kt|rs|cs|vue)):\d+(?::\d+)?", re.IGNORECASE
        ),
        r"\1:<line>",
    ),
    (re.compile(r"\b\d+(?:\.\d+){2,}\b"), "<version>"),
    (re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE), "<hex>"),
    (
        re.compile(r"\b\d+(?:\.\d+)?(?:ms|s|m|h|MB|GB|KB|MiB|GiB|KiB)\b", re.IGNORECASE),
        "<quantity>",
    ),
    # Run-scoped identifiers, at ANY digit count. The generic <num> rule below
    # only matches 3+ digits — a deliberate floor, so ":42" line numbers and small
    # diagnostic codes survive. But that floor let short volatile values through:
    # "pid=4821" normalised while "pid=99" and "pid=7" did not, so the same
    # recurring failure produced a new signature on most runs. Deduplication, the
    # analysis cache, occurrence_count and the known-signature short-circuit all
    # hang off this hash — a leak here disables every one of them, silently.
    (
        re.compile(
            r"\b(pid|ppid|tid|thread|worker|job|build|run|attempt|retry|try|seq)"
            r"\s*[:=#]?\s*\d+\b",
            re.IGNORECASE,
        ),
        r"\1=<num>",
    ),
    (re.compile(r"\b\d{3,}\b"), "<num>"),
]

WHITESPACE = re.compile(r"\s+")

# Numbers that ARE the diagnosis and must survive normalisation.
#
# Without this, SQLSTATE[2002] (cannot connect) and SQLSTATE[1045] (access
# denied) collapse to the same signature — two completely different failures
# with completely different fixes sharing one cache entry and one history.
# Same for HTTP 401 vs 404 vs 500, and exit 1 vs exit 137 (OOM-kill).
MEANINGFUL_NUMBER = re.compile(
    r"\[\d{3,5}\]"  # [2002], [1045]
    r"|\b(?:HTTP|status|code|errno)\s*[:=]?\s*\d{3}\b"  # HTTP 401
    r"|\bexit(?:ed)?(?: with)?(?: code| status)?\s*\d{1,3}\b",  # exit code 137
    re.IGNORECASE,
)

_SENTINEL = "\x00PM{}\x00"


def normalize(error_text: str, *, max_chars: int = 2000) -> str:
    text = error_text.strip()[:max_chars]

    # Park the meaningful codes so the generic number rule cannot reach them.
    protected: list[str] = []

    def _park(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return _SENTINEL.format(len(protected) - 1)

    text = MEANINGFUL_NUMBER.sub(_park, text)

    for pattern, token in NORMALIZERS:
        text = pattern.sub(token, text)

    for index, original in enumerate(protected):
        text = text.replace(_SENTINEL.format(index), original)

    return WHITESPACE.sub(" ", text).strip().lower()


def signature_hash(error_text: str, *, ecosystem: str | None = None) -> tuple[str, str]:
    """Return (sha256, normalized_text).

    The ecosystem is part of the hash: an identical "connection refused" from a
    Docker daemon and from Postgres are different problems with different fixes.
    """
    normalized = normalize(error_text)
    payload = f"{ecosystem or 'unknown'}::{normalized}"

    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), normalized
