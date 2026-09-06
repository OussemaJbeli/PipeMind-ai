from app.services import log_processor as lp
from app.services import signature as sig

DB_LOG = """\
Running with runner 16.11.0
section_start:1756563004:prepare_executor
Using Docker executor with image php:8.3-cli
section_end:1756563008:prepare_executor
2026-08-30 14:30:00 $ composer install --no-interaction
Downloading vendor packages
 45% [############        ]
2026-08-30 14:30:40 $ php artisan test
   PASS  Tests\\Integration\\HealthTest
   FAIL  Tests\\Integration\\DatabaseTest
   SQLSTATE[HY000] [2002] Connection refused
  at vendor/laravel/framework/src/Connectors/Connector.php:70
  Tests:    1 failed, 1 passed
ERROR: Job failed: exit code 1
"""


def test_clean_strips_decoration_not_content() -> None:
    cleaned = lp.clean(DB_LOG)

    assert "section_start" not in cleaned
    assert "2026-08-30 14:30:00 " not in cleaned  # per-line prefix removed
    assert "45% [####" not in cleaned  # progress noise removed
    assert "SQLSTATE[HY000] [2002] Connection refused" in cleaned


def test_extract_finds_the_root_cause() -> None:
    result = lp.extract(lp.clean(DB_LOG))

    assert "SQLSTATE[HY000] [2002] Connection refused" in result.excerpt
    assert result.ecosystem == "db"
    assert result.exit_code == 1
    assert result.matched_lines


def test_extract_windows_the_first_error_not_the_last() -> None:
    """A cause at line 10 produces symptoms at line 900.

    Leading with the tail yields a confident, useless analysis of the symptom.
    """
    lines = ["ECONNREFUSED 127.0.0.1:5432"] + [f"noise {i}" for i in range(500)]
    lines += ["Process exited with code 1"]

    result = lp.extract("\n".join(lines))

    assert result.excerpt_start_line == 1
    assert "ECONNREFUSED" in result.excerpt


def test_extract_falls_back_to_the_tail_when_nothing_matches() -> None:
    result = lp.extract("\n".join(f"step {i} completed" for i in range(300)))

    assert result.ecosystem is None
    assert "step 299 completed" in result.excerpt


def test_resource_failures_outrank_their_symptoms() -> None:
    """OOM explains everything downstream of it."""
    log = "FATAL ERROR: JavaScript heap out of memory\nFAIL build.test.ts\nTests: 1 failed"

    assert lp.extract(log).ecosystem == "resource"


class TestSignature:
    def test_same_error_different_run_same_hash(self) -> None:
        a = "SQLSTATE[HY000] [2002] Connection refused at /app/src/Db.php:42 (2026-08-30T14:00:00Z)"
        b = "SQLSTATE[HY000] [2002] Connection refused at /srv/lib/Db.php:87 (2026-08-31T09:14:22Z)"

        assert sig.signature_hash(a, ecosystem="db")[0] == sig.signature_hash(b, ecosystem="db")[0]

    def test_different_errors_different_hash(self) -> None:
        a = "SQLSTATE[HY000] [2002] Connection refused"
        b = "SQLSTATE[42S02] Base table or view not found"

        assert sig.signature_hash(a, ecosystem="db")[0] != sig.signature_hash(b, ecosystem="db")[0]

    def test_ecosystem_separates_identical_text(self) -> None:
        """'connection refused' from Docker and from Postgres are different bugs."""
        error = "connection refused"

        assert (
            sig.signature_hash(error, ecosystem="db")[0]
            != sig.signature_hash(error, ecosystem="docker")[0]
        )

    def test_small_integers_survive(self) -> None:
        """137 is an OOM kill, 1 is an ordinary failure. Collapsing loses the diagnosis."""
        assert (
            sig.signature_hash("process exited with code 1")[0]
            != sig.signature_hash("process exited with code 137")[0]
        )
