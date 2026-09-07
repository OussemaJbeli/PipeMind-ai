"""Log-processing benchmark.

Two numbers matter, and they are not equally important:

  reduction        the easy metric — how much smaller the excerpt is
  root-cause kept  the one that decides correctness

An excerpt 99.9% smaller that drops the actual error has made the system worse,
not cheaper. Treat retention as a hard gate and reduction as a nice-to-have.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import log_processor, redactor, signature

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# (file, phrase that must survive)
CASES = [
    ("failed-db-test.log", "SQLSTATE[HY000] [2002] Connection refused"),
    ("failed-npm-dependency.log", "ERESOLVE unable to resolve dependency tree"),
]


def signature_arms() -> None:
    """Signature stability and separation.

    Reduction is the easy metric. This is the one deduplication depends on: if a
    run-scoped value reaches the hash, the same recurring failure gets a fresh
    signature every time and occurrence_count, the analysis cache and the
    known-signature short-circuit all quietly stop working.
    """
    error = "SQLSTATE[HY000] [2002] Connection refused"

    def sig(raw: str) -> str:
        redaction = redactor.redact(raw)
        extraction = log_processor.extract(log_processor.clean(redaction.text))
        digest, _ = signature.signature_hash(
            extraction.error_block or extraction.error_message or "",
            ecosystem=extraction.ecosystem,
        )
        return digest

    print("\nSignature stability — same error, growing noise:\n")
    baseline = None
    for noise in (0, 50, 1_000, 10_000, 50_000):
        digest = sig(synthetic(noise, error))
        baseline = baseline or digest
        verdict = "STABLE" if digest == baseline else "DRIFTED"
        print(f"  {noise:>6} noise lines   {digest[:16]}…   {verdict}")

    print("\nVolatile identifiers — same failure, different run:\n")
    runs = [
        ("2026-08-30T14:30:11Z", "4821", "1"),
        ("2026-09-01T09:12:44Z", "99", "2"),
        ("2025-01-15T23:59:59Z", "7", "13"),
    ]
    digests = {
        sig(synthetic(20, f"{error} at {ts} pid={pid} attempt {n}"))
        for ts, pid, n in runs
    }
    print(f"  distinct across {len(runs)} runs: {len(digests)}   "
          f"{'GOOD' if len(digests) == 1 else 'LEAKING — dedup is broken'}")

    print("\nSeparation — distinct diagnoses must NOT collide:\n")
    distinct = [
        error,
        "Error: connect ECONNREFUSED 127.0.0.1:5432",
        "npm ERR! ERESOLVE unable to resolve dependency tree",
        "FATAL ERROR: Reached heap limit Allocation failed",
        "SQLSTATE[HY000] [1045] Access denied for user",
    ]
    seen = {sig(synthetic(20, e)): e for e in distinct}
    for digest, text in seen.items():
        print(f"  {digest[:16]}…   {text[:52]}")
    print(f"\n  unique: {len(seen)}/{len(distinct)}")


def synthetic(noise_lines: int, error: str) -> str:
    return "\n".join(
        [
            f"\x1b[32m2026-08-30T14:30:{i % 60:02d}Z npm install package-{i}\x1b[0m"
            for i in range(noise_lines)
        ]
        + [error]
        + [f"cleanup {i}" for i in range(200)]
        + ["ERROR: Job failed: exit code 1"]
    )


def run(name: str, raw: str, must_keep: str) -> dict:
    started = time.perf_counter()

    redaction = redactor.redact(raw)
    cleaned = log_processor.clean(redaction.text)
    extraction = log_processor.extract(cleaned)
    sig, _ = signature.signature_hash(
        extraction.error_block or extraction.error_message or "",
        ecosystem=extraction.ecosystem,
    )

    elapsed = (time.perf_counter() - started) * 1000

    return {
        "name": name,
        "in_lines": raw.count("\n") + 1,
        "in_chars": len(raw),
        "out_lines": extraction.excerpt.count("\n") + 1,
        "out_chars": len(extraction.excerpt),
        "reduction": 1 - len(extraction.excerpt) / max(len(raw), 1),
        "kept": must_keep in extraction.excerpt,
        "ecosystem": extraction.ecosystem or "-",
        "redactions": redaction.count,
        "ms": elapsed,
        "sig": sig[:12],
    }


def main() -> None:
    rows = [
        run(name, (FIXTURES / name).read_text(), keep)
        for name, keep in CASES
        if (FIXTURES / name).exists()
    ]

    for lines in (1_000, 10_000, 50_000):
        rows.append(
            run(
                f"synthetic-{lines // 1000}k",
                synthetic(lines, "SQLSTATE[HY000] [2002] Connection refused"),
                "SQLSTATE[HY000] [2002] Connection refused",
            )
        )

    header = f"| {'log':<26} | {'in':>9} | {'out':>7} | {'cut':>6} | {'kept':<4} | {'eco':<8} | {'ms':>7} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")

    for r in rows:
        print(
            f"| {r['name']:<26} | {r['in_lines']:>6} ln | {r['out_lines']:>4} ln "
            f"| {r['reduction'] * 100:>5.1f}% | {'yes' if r['kept'] else 'NO':<4} "
            f"| {r['ecosystem']:<8} | {r['ms']:>6.1f} |"
        )

    missed = [r["name"] for r in rows if not r["kept"]]
    print()
    print(f"root-cause retention: {len(rows) - len(missed)}/{len(rows)}")

    signature_arms()

    if missed:
        print(f"LOST THE ROOT CAUSE IN: {', '.join(missed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
