"""Turn a 48,000-line CI log into the ~40 lines that matter.

Order is redact → clean → extract. Redaction happens in redactor.py and must
already have run before anything here touches the text.
"""

import bisect
import re
from dataclasses import dataclass, field
from re import Pattern

ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
GITLAB_SECTION = re.compile(r"section_(?:start|end):\d+:\S+\r?", re.MULTILINE)
GITHUB_CMD = re.compile(
    r"^##\[(?:group|endgroup|command|section|debug|add-matcher)\].*$", re.MULTILINE
)

# Only the per-line PREFIX. A timestamp inside an error message ("token expired
# at 2026-08-30T14:00:00Z") is evidence; the runner's prefix is noise, and
# leaving it in wastes roughly a fifth of the token budget.
TIMESTAMP_PREFIX = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\s+)"
    r"|^(?:\[\d{2}:\d{2}:\d{2}\]\s+)"
    r"|^(?:\d{2}:\d{2}:\d{2}\.\d+\s+)",
    re.MULTILINE,
)

CARRIAGE_PROGRESS = re.compile(r"^.*\r(?!\n)", re.MULTILINE)
DOWNLOAD_NOISE = re.compile(
    r"^\s*(?:Downloading|Fetching|Receiving|Resolving|Unpacking|Extracting|"
    r"Pulling|Downloaded|Get:|Selecting|Preparing to unpack|"
    r"\d+%\s*[\[\|#=.]+|\s*[\d.]+ [KMG]iB/s).*$",
    re.MULTILINE | re.IGNORECASE,
)
BLANK_RUNS = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    """Strip presentation noise. Content is never altered — only decoration."""
    text = ANSI.sub("", text)
    text = CARRIAGE_PROGRESS.sub("", text)
    text = GITLAB_SECTION.sub("", text)
    text = GITHUB_CMD.sub("", text)
    text = TIMESTAMP_PREFIX.sub("", text)
    text = DOWNLOAD_NOISE.sub("", text)
    text = BLANK_RUNS.sub("\n\n", text)

    return text.strip()


# Ordered by specificity AND by explanatory power. The first family that matches
# wins, which is why RESOURCE-style failures and language runtimes come before
# the generic "ERROR" catch-all.
ERROR_MARKERS: list[tuple[str, Pattern[str]]] = [
    # ORDERED BY EXPLANATORY POWER, not by specificity.
    #
    # The first family that matches claims the log, and that decides both the
    # ecosystem and — through it — the signature hash. So the families that
    # EXPLAIN other failures must come first: an OOM-kill explains the test
    # failure beneath it, a refused database connection explains the assertion
    # error, and a DNS failure explains the failed install. A failing test
    # explains nothing on its own, so it comes near the end.
    (
        "resource",
        re.compile(
            r"(OOMKilled|out of memory|Cannot allocate memory|JavaScript heap out of memory"
            r"|java\.lang\.OutOfMemoryError|no space left on device|ENOSPC)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "db",
        re.compile(
            r"(SQLSTATE|ECONNREFUSED|connection refused|could not connect to server"
            r"|Access denied for user|too many connections|deadlock detected)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "network",
        re.compile(
            r"(getaddrinfo (?:ENOTFOUND|EAI_AGAIN)|Temporary failure in name resolution"
            r"|Could not resolve host|ETIMEDOUT|ECONNRESET|certificate has expired"
            r"|x509: certificate)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "k8s",
        re.compile(
            r"(ImagePullBackOff|CrashLoopBackOff|FailedScheduling|readiness probe failed)",
            re.MULTILINE,
        ),
    ),
    (
        "docker",
        re.compile(
            r"^\s*(?:ERROR \[|failed to solve|manifest unknown|pull access denied"
            r"|Cannot connect to the Docker daemon|denied: requested access)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "node",
        re.compile(
            r"^\s*(?:npm ERR!|yarn error|ERR_\w+|(?:Type|Reference|Range|Syntax)Error:"
            r"|Cannot find module)",
            re.MULTILINE,
        ),
    ),
    (
        "php",
        re.compile(
            r"^\s*(?:PHP )?(?:Fatal error|Parse error|Uncaught \w+Exception)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "python",
        re.compile(
            r"^\s*(?:Traceback \(most recent call last\):|\w*Error: |\w*Exception: |E\s{3}\w+Error)",
            re.MULTILINE,
        ),
    ),
    ("java", re.compile(r"^\s*(?:Exception in thread|Caused by:|\[ERROR\])", re.MULTILINE)),
    # Go's own markers only. A bare "FAIL <word>" also matches PHPUnit and Pest
    # output, which would hand every PHP test log the "go" ecosystem — and a
    # different signature for the same error.
    (
        "go",
        re.compile(
            r"^\s*(?:panic: |--- FAIL: |FAIL\s+\S+\s+[\d.]+m?s\b|# \S+\n.*\.go:\d+:)", re.MULTILINE
        ),
    ),
    (
        "test",
        re.compile(
            r"^\s*(?:FAIL|FAILED|✕|✗|●.*›|Tests:.*failed|Failures:|AssertionError"
            r"|Expected .*(?:but )?[Rr]eceived|⨯)",
            re.MULTILINE,
        ),
    ),
    (
        "shell",
        re.compile(
            r"(command not found|Permission denied|No such file or directory|Killed"
            r"|Segmentation fault)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "generic",
        re.compile(r"^\s*(?:ERROR|FATAL|CRITICAL|error:|fatal:)\b", re.IGNORECASE | re.MULTILINE),
    ),
]

STACK_TRACE: dict[str, Pattern[str]] = {
    "python": re.compile(
        r"Traceback \(most recent call last\):[\s\S]{0,4000}?^\w*(?:Error|Exception)[^\n]*",
        re.MULTILINE,
    ),
    "java": re.compile(
        r"^(?:Exception in thread|Caused by:)[\s\S]{0,4000}?(?=\n\S|\Z)", re.MULTILINE
    ),
    "php": re.compile(r"^#0 [\s\S]{0,4000}?(?=\n\n|\Z)", re.MULTILINE),
    "node": re.compile(r"^\s+at [\w.$<>\[\] ]+ \([^)]+\)(?:\n\s+at .+)*", re.MULTILINE),
    "go": re.compile(r"^panic: [\s\S]{0,4000}?(?=\n\n|\Z)", re.MULTILINE),
}

EXIT_CODE = re.compile(
    r"(?:exit(?:ed)?(?: with)?(?: code| status)?|ERROR: Job failed: exit code)\s*[:=]?\s*(\d{1,3})",
    re.IGNORECASE,
)


@dataclass
class Extraction:
    excerpt: str
    excerpt_start_line: int
    excerpt_end_line: int
    error_block: str | None = None
    stack_trace: str | None = None
    ecosystem: str | None = None
    exit_code: int | None = None
    error_message: str | None = None
    matched_lines: list[int] = field(default_factory=list)
    total_lines: int = 0


def extract(cleaned: str, *, context: int = 15, max_chars: int = 60_000) -> Extraction:
    lines = cleaned.split("\n")
    total = len(lines)

    ecosystem: str | None = None
    hits: list[int] = []

    # Offsets of every line start, so a match position converts to a line number
    # by binary search. Running the pattern once over the whole text and mapping
    # offsets is far cheaper than running it per line: a 50k-line log costs one
    # scan instead of 50,000.
    line_starts = [0]
    for index, char in enumerate(cleaned):
        if char == "\n":
            line_starts.append(index + 1)

    for name, pattern in ERROR_MARKERS:
        matches = list(pattern.finditer(cleaned))

        if not matches:
            continue

        hits = sorted({bisect.bisect_right(line_starts, m.start()) - 1 for m in matches})
        ecosystem = name
        break

    if not hits:
        # No marker matched. The error is almost always at the end.
        start = max(0, total - 120)
        excerpt = "\n".join(lines[start:])

        return Extraction(
            excerpt=excerpt[-max_chars:],
            excerpt_start_line=start + 1,
            excerpt_end_line=total,
            exit_code=_exit_code(cleaned),
            error_message=_last_meaningful(lines),
            total_lines=total,
        )

    # Window the FIRST error. A dependency conflict at line 400 produces a build
    # failure at line 8,000 and a "job failed" at 8,100 — leading with the tail
    # gets a confident, useless analysis of the symptom.
    first, last = hits[0], hits[-1]
    start = max(0, first - context)
    end = min(total, last + context + 1)

    if end - start > 400:
        # A huge span means repetitive errors (400 failing assertions). Keep the
        # head of the span plus the tail of the log rather than all of it.
        end = min(total, first + 200)
        excerpt = "\n".join(lines[start:end]) + "\n…\n" + "\n".join(lines[-40:])
    else:
        excerpt = "\n".join(lines[start:end])

    trace = None
    if ecosystem in STACK_TRACE:
        match = STACK_TRACE[ecosystem].search(cleaned)
        trace = match.group(0)[:4000] if match else None

    return Extraction(
        excerpt=excerpt[:max_chars],
        excerpt_start_line=start + 1,
        excerpt_end_line=end,
        error_block="\n".join(lines[first : min(total, first + 30)]),
        stack_trace=trace,
        ecosystem=ecosystem,
        exit_code=_exit_code(cleaned),
        error_message=lines[first].strip()[:500] or None,
        # What the UI highlights and scrolls to.
        matched_lines=[h + 1 for h in hits[:50]],
        total_lines=total,
    )


def _exit_code(text: str) -> int | None:
    codes = EXIT_CODE.findall(text)

    return int(codes[-1]) if codes else None


def _last_meaningful(lines: list[str]) -> str | None:
    """The most informative trailing line, preferring substance over brevity."""
    fallback: str | None = None

    for line in reversed(lines):
        stripped = line.strip()

        if not stripped:
            continue

        # Shell echoes are noise, never the message.
        if stripped.startswith(("$", "+", "#")):
            continue

        if len(stripped) > 10:
            return stripped[:500]

        # Remember the first short line seen, in case nothing longer exists:
        # returning something beats returning nothing when the whole tail is
        # short lines.
        fallback = fallback or stripped[:500]

    return fallback
