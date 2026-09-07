"""Signature invariants.

Tests the behaviour, not the implementation: the normalisation rules will change
as real logs arrive, but these properties must hold whatever they become.
"""

import pytest

from app.services.signature import normalize, signature_hash


def test_same_error_different_run_produces_the_same_hash() -> None:
    a = "SQLSTATE[HY000] [2002] Connection refused at /app/src/Db.php:42 (2026-08-30T14:00:00Z)"
    b = "SQLSTATE[HY000] [2002] Connection refused at /srv/lib/Db.php:915 (2026-08-31T09:14:22Z)"

    assert signature_hash(a, ecosystem="db")[0] == signature_hash(b, ecosystem="db")[0]


def test_different_error_codes_do_not_collide() -> None:
    # 2002 is "cannot connect", 1045 is "access denied" — different failures
    # with different fixes. Collapsing them would share one cache entry.
    connect = "SQLSTATE[HY000] [2002] Connection refused"
    denied = "SQLSTATE[HY000] [1045] Access denied for user"

    assert signature_hash(connect, ecosystem="db")[0] != signature_hash(denied, ecosystem="db")[0]


def test_http_statuses_do_not_collide() -> None:
    assert (
        signature_hash("Request failed with HTTP 401", ecosystem="node")[0]
        != signature_hash("Request failed with HTTP 500", ecosystem="node")[0]
    )


def test_exit_codes_do_not_collide() -> None:
    # 137 is an OOM-kill; 1 is an ordinary failure. Collapsing them loses the
    # diagnosis entirely.
    assert (
        signature_hash("Process exited with code 1", ecosystem="shell")[0]
        != signature_hash("Process exited with code 137", ecosystem="shell")[0]
    )


def test_ecosystem_separates_identical_text() -> None:
    # The same words from a Docker daemon and from Postgres are different
    # problems with different fixes.
    assert (
        signature_hash("connection refused", ecosystem="db")[0]
        != signature_hash("connection refused", ecosystem="docker")[0]
    )


def test_volatile_values_are_normalised() -> None:
    normalized = normalize(
        "failed at 2026-08-30T14:00:00Z on 10.0.0.14:5432 "
        "sha a82c91f3b4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9 in /a/b/c.php:42"
    )

    for token in ("<timestamp>", "<ip>", "<sha>", "<path>", "<line>"):
        assert token in normalized, f"{token} missing from {normalized}"


def test_redaction_markers_normalise_consistently() -> None:
    # Two runs leaking different secrets are still the same failure.
    a = normalize("auth failed for [REDACTED_GITHUB_TOKEN]")
    b = normalize("auth failed for [REDACTED_AWS_KEY]")

    assert a == b


def test_hash_is_stable_and_hex() -> None:
    first, _ = signature_hash("boom", ecosystem="test")
    second, _ = signature_hash("boom", ecosystem="test")

    assert first == second
    assert len(first) == 64
    assert all(c in "0123456789abcdef" for c in first)


class TestVolatileIdentifiers:
    """Run-scoped numbers must never reach the hash.

    The generic <num> rule matches 3+ digits only — a deliberate floor so that
    ":42" line numbers and small diagnostic codes survive. That floor let short
    volatile values through: "pid=4821" normalised while "pid=99" and "pid=7" did
    not, so the same recurring failure produced a fresh signature on most runs.

    Deduplication, the analysis cache, occurrence_count and the known-signature
    short-circuit all hang off this hash. A leak here disables every one of them
    and reports nothing.
    """

    @pytest.mark.parametrize("keyword", [
        "pid", "ppid", "tid", "thread", "worker", "job", "build", "run",
        "attempt", "retry", "try", "seq",
    ])
    def test_short_and_long_values_collapse_together(self, keyword):
        hashes = {
            signature_hash(f"connection refused {keyword}={value}")[0]
            for value in ("7", "99", "4821", "1000000")
        }

        assert len(hashes) == 1, f"{keyword} leaks into the signature"

    def test_separator_style_does_not_matter(self):
        hashes = {
            signature_hash(f"timeout in worker{sep}12")[0]
            for sep in ("=", ":", " ", "#", " = ")
        }

        assert len(hashes) == 1

    def test_a_full_run_line_is_stable_across_runs(self):
        hashes = {
            signature_hash(
                f"SQLSTATE[HY000] [2002] Connection refused at {ts} pid={pid} attempt {n}"
            )[0]
            for ts, pid, n in [
                ("2026-08-30T14:30:11Z", "4821", "1"),
                ("2026-09-01T09:12:44Z", "99", "2"),
                ("2025-01-15T23:59:59Z", "7", "13"),
            ]
        }

        assert len(hashes) == 1


class TestMeaningfulNumbersSurvive:
    """The numbers that ARE the diagnosis must keep separating failures.

    Over-normalising is the opposite failure and just as bad: SQLSTATE[2002]
    (cannot connect) and SQLSTATE[1045] (access denied) sharing one signature
    means one cache entry, one history and one wrong answer for two unrelated
    problems.
    """

    @pytest.mark.parametrize(("left", "right"), [
        ("SQLSTATE[2002] Connection refused", "SQLSTATE[1045] Access denied"),
        ("Job failed: exit code 1", "Job failed: exit code 137"),
        ("HTTP 401 Unauthorized", "HTTP 404 Not Found"),
        ("error TS2345: argument type", "error TS2551: property missing"),
    ])
    def test_distinct_diagnoses_keep_distinct_signatures(self, left, right):
        assert signature_hash(left)[0] != signature_hash(right)[0]
