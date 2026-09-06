"""Deterministic failure classification.

~2 ms, no model, no training data, fully explainable. This is the baseline that
anything more sophisticated has to beat — and on a fresh install it is the only
classifier there is, so it has to be good enough to ship alone.

Two ordering principles:

  1. Rules are ordered by EXPLANATORY POWER, not alphabetically. RESOURCE runs
     first because OOM explains every symptom downstream of it; TEST runs last
     because a failing assertion explains nothing on its own.
  2. Within a category, specific patterns precede general ones.
"""

import re
from dataclasses import dataclass, field
from re import Pattern


@dataclass(frozen=True)
class Rule:
    category: str
    subcategory: str
    pattern: Pattern[str]
    weight: float = 1.0


def _p(expr: str) -> Pattern[str]:
    return re.compile(expr, re.IGNORECASE | re.MULTILINE)


RULES: list[Rule] = [
    # --- RESOURCE: first, because it explains other symptoms -----------------
    Rule(
        "RESOURCE",
        "OutOfMemory",
        _p(
            r"OOMKilled|out of memory|Cannot allocate memory"
            r"|JavaScript heap out of memory|java\.lang\.OutOfMemoryError"
            r"|exit code 137|Killed\s+process"
        ),
        1.0,
    ),
    Rule(
        "RESOURCE",
        "DiskFull",
        _p(r"no space left on device|ENOSPC|disk quota exceeded|write error: No space"),
        1.0,
    ),
    Rule("RESOURCE", "FileDescriptors", _p(r"EMFILE|too many open files"), 1.0),
    # --- DATABASE ------------------------------------------------------------
    Rule(
        "DATABASE",
        "ConnectionRefused",
        _p(
            r"SQLSTATE\[HY000\]\s*\[2002\]"
            r"|could not connect to server"
            r"|connection to server at .* failed"
            r"|ECONNREFUSED[^\n]*:(?:5432|3306|27017|1433|6379)"
            r"|Connection refused[^\n]*(?:postgres|mysql|mariadb|mongo|redis)"
        ),
        1.0,
    ),
    Rule(
        "DATABASE",
        "AuthFailed",
        _p(
            r"password authentication failed"
            r"|Access denied for user"
            r"|SQLSTATE\[28000\]"
            r"|authentication failed for user"
        ),
        1.0,
    ),
    Rule(
        "DATABASE",
        "MigrationError",
        _p(
            r"Migration .* failed"
            r"|relation \"?\w+\"? already exists"
            r"|SQLSTATE\[42S02\]"
            r"|relation \"?\w+\"? does not exist"
            r"|Base table or view not found"
        ),
        0.95,
    ),
    Rule("DATABASE", "Deadlock", _p(r"deadlock detected|Lock wait timeout exceeded"), 1.0),
    Rule("DATABASE", "Timeout", _p(r"statement timeout|canceling statement due to"), 0.9),
    # --- DEPENDENCY ----------------------------------------------------------
    Rule(
        "DEPENDENCY",
        "Conflict",
        _p(
            r"ERESOLVE"
            r"|unable to resolve dependency tree"
            r"|Your requirements could not be resolved"
            r"|version solving failed"
            r"|conflicting peer dependency"
            r"|peer \S+@\"[^\"]+\" from"
        ),
        1.0,
    ),
    Rule(
        "DEPENDENCY",
        "NotFound",
        _p(
            r"Cannot find module"
            r"|Module not found"
            r"|Could not find a version that satisfies"
            r"|404 Not Found[^\n]*(?:npm|registry|packagist|pypi)"
            r"|E404|npm error 404"
            r"|package \S+ is not installed"
        ),
        0.95,
    ),
    Rule(
        "DEPENDENCY",
        "LockfileDrift",
        _p(
            r"lock file .* out of date"
            r"|composer\.lock is not up to date"
            r"|npm ci can only install .* package-lock\.json"
            r"|lockfile .* needs to be updated"
        ),
        1.0,
    ),
    Rule(
        "DEPENDENCY",
        "RegistryError",
        _p(r"(?:npm|yarn|pip|composer)[^\n]*(?:ETIMEDOUT|ECONNRESET)[^\n]*registr"),
        0.85,
    ),
    # --- DOCKER --------------------------------------------------------------
    Rule(
        "DOCKER",
        "RegistryAuth",
        _p(
            r"pull access denied"
            r"|unauthorized: authentication required"
            r"|denied: requested access to the resource is denied"
            r"|manifest unknown"
        ),
        1.0,
    ),
    Rule(
        "DOCKER",
        "DaemonUnavailable",
        _p(r"Cannot connect to the Docker daemon|docker: error during connect"),
        1.0,
    ),
    Rule(
        "DOCKER",
        "BuildFailed",
        _p(
            r"failed to solve"
            r"|ERROR \[\d+/\d+\]"
            r"|The command '/bin/sh -c .*' returned a non-zero code"
            r"|did not complete successfully: exit code"
        ),
        0.9,
    ),
    # --- NETWORK -------------------------------------------------------------
    Rule(
        "NETWORK",
        "DNSFailure",
        _p(
            r"getaddrinfo (?:ENOTFOUND|EAI_AGAIN)"
            r"|Temporary failure in name resolution"
            r"|Could not resolve host"
            r"|curl: \(6\)"
        ),
        1.0,
    ),
    Rule(
        "NETWORK",
        "TLSError",
        _p(
            r"certificate (?:has expired|verify failed)"
            r"|x509: certificate"
            r"|SSL_ERROR|unable to get local issuer certificate"
        ),
        1.0,
    ),
    Rule("NETWORK", "ConnectionReset", _p(r"ECONNRESET|Connection reset by peer|broken pipe"), 0.8),
    Rule(
        "NETWORK",
        "Timeout",
        _p(
            r"ETIMEDOUT|Connection timed out|Operation timed out"
            r"|context deadline exceeded|curl: \(28\)"
        ),
        0.85,
    ),
    # --- AUTHENTICATION ------------------------------------------------------
    Rule(
        "AUTHENTICATION",
        "ExpiredToken",
        _p(r"token (?:has )?expired|credentials have expired|JWT expired"),
        1.0,
    ),
    Rule(
        "AUTHENTICATION",
        "InvalidToken",
        _p(
            r"401 Unauthorized|HTTP 401|invalid[_ ]token|Bad credentials"
            r"|authentication (?:failed|required)"
        ),
        0.9,
    ),
    # --- PERMISSION ----------------------------------------------------------
    # 403 is PERMISSION (identity known, access refused); 401 is AUTHENTICATION
    # (identity not established). Different fixes, so they must stay separate.
    Rule(
        "PERMISSION", "FilePermission", _p(r"EACCES|Permission denied|Operation not permitted"), 0.9
    ),
    Rule(
        "PERMISSION",
        "RepoPermission",
        _p(
            r"403 Forbidden|insufficient (?:scope|permission)"
            r"|You are not allowed to push"
        ),
        0.9,
    ),
    # --- CONFIGURATION -------------------------------------------------------
    Rule(
        "CONFIGURATION",
        "MissingEnvVar",
        _p(
            r"(?:environment )?variable \S+ (?:is )?not (?:set|defined)"
            r"|\$\{?\w+\}?: unbound variable"
            r"|required environment variable"
            r"|missing required (?:env|environment)"
        ),
        1.0,
    ),
    Rule(
        "CONFIGURATION",
        "InvalidYAML",
        _p(
            r"yaml: line \d+"
            r"|mapping values are not allowed"
            r"|did not find expected key"
            r"|Invalid (?:CI|workflow) config"
        ),
        1.0,
    ),
    Rule(
        "CONFIGURATION",
        "WrongPath",
        _p(r"No such file or directory|ENOENT: no such file|cannot stat"),
        0.7,
    ),
    # --- DEPLOYMENT ----------------------------------------------------------
    Rule(
        "DEPLOYMENT",
        "HealthCheckFailed",
        _p(
            r"readiness probe failed|liveness probe failed|health check failed"
            r"|CrashLoopBackOff|ImagePullBackOff"
        ),
        1.0,
    ),
    Rule(
        "DEPLOYMENT",
        "RolloutTimeout",
        _p(r"rollout .* timed out|exceeded its progress deadline"),
        1.0,
    ),
    # --- INFRASTRUCTURE ------------------------------------------------------
    Rule(
        "INFRASTRUCTURE",
        "RunnerUnavailable",
        _p(
            r"no (?:active )?runner|This job is stuck"
            r"|job was cancelled by the system|runner system failure"
            r"|The runner has (?:disappeared|crashed)"
        ),
        1.0,
    ),
    Rule(
        "INFRASTRUCTURE",
        "QuotaExceeded",
        _p(r"quota exceeded|rate limit exceeded|429 Too Many Requests|API rate limit"),
        0.9,
    ),
    # --- BUILD ---------------------------------------------------------------
    Rule("BUILD", "TypeCheck", _p(r"error TS\d+:|Type error:|mypy: error"), 1.0),
    Rule(
        "BUILD",
        "Linting",
        _p(
            r"eslint[^\n]*\d+ problems?"
            r"|PHP_CodeSniffer[^\n]*ERRORS?"
            r"|ruff[^\n]*Found \d+ error"
            r"|✖ \d+ problems"
        ),
        0.95,
    ),
    Rule(
        "BUILD",
        "Compilation",
        _p(
            r"compilation (?:failed|error)"
            r"|^\s*syntax error"
            r"|PHP Parse error"
            r"|cannot find symbol"
            r"|Module build failed"
        ),
        0.9,
    ),
    # --- TEST: last, because a failing test is usually a symptom -------------
    Rule(
        "TEST",
        "Assertion",
        _p(
            r"AssertionError"
            r"|Failed asserting that"
            r"|Expected .*(?:but )?(?:received|got|actual)"
            r"|Expected status \d+, Received status \d+"
            r"|expect\(.*\)\.to"
        ),
        0.8,
    ),
    Rule(
        "TEST",
        "E2ETest",
        _p(
            r"(?:cypress|playwright|selenium|puppeteer)[^\n]*(?:failed|error)"
            r"|Timed out retrying after \d+ms"
        ),
        0.85,
    ),
    Rule(
        "TEST",
        "UnitTest",
        _p(
            r"^(?:FAIL|FAILED|✕|✗)\s"
            r"|Tests?:[^\n]*\d+ failed"
            r"|Failures: [1-9]"
            r"|\d+ failing"
            r"|^not ok \d+"
            r"|^# fail [1-9]"
        ),
        0.7,
    ),
]


# Categories that are almost always a SYMPTOM of something else. A database
# outage makes tests fail; so does an OOM. When a cause category has also
# matched, these must not dilute its confidence — they corroborate the story
# rather than competing with it.
SYMPTOM_CATEGORIES = frozenset({"TEST", "BUILD"})


@dataclass
class RuleResult:
    category: str
    subcategory: str | None
    confidence: float
    matched_rules: list[str] = field(default_factory=list)


class RuleClassifier:
    """Stateless. Safe to construct per request."""

    @staticmethod
    def warm() -> None:
        """Compile every pattern at boot rather than on the first real request."""
        for rule in RULES:
            rule.pattern.search("")

    def classify(self, text: str, ecosystem: str | None = None) -> RuleResult:
        if not text:
            return RuleResult("UNKNOWN", None, 0.0)

        scores: dict[tuple[str, str], float] = {}
        matched: list[str] = []
        first_seen: dict[tuple[str, str], int] = {}

        for index, rule in enumerate(RULES):
            hits = len(rule.pattern.findall(text))

            if not hits:
                continue

            key = (rule.category, rule.subcategory)

            # Diminishing returns: ten matches is not ten times the evidence of
            # one. Repetition mostly means a loop, not stronger proof.
            scores[key] = scores.get(key, 0.0) + rule.weight * min(1.0 + (hits - 1) * 0.1, 1.5)
            first_seen.setdefault(key, index)
            matched.append(f"{rule.category}/{rule.subcategory}")

        if not scores:
            return RuleResult("UNKNOWN", None, 0.0)

        # Aggregate to the CATEGORY before measuring dominance. Two TEST rules
        # both firing is corroboration, not ambiguity — scoring them as rivals
        # would make a clear assertion failure look like a coin toss.
        category_scores: dict[str, float] = {}
        category_first_seen: dict[str, int] = {}

        for (category, _), value in scores.items():
            category_scores[category] = category_scores.get(category, 0.0) + value

        for (category, _), index in first_seen.items():
            category_first_seen[category] = min(category_first_seen.get(category, index), index)

        # Ties break by declaration order, which is ordered by explanatory power:
        # an OOM that also produced a failing test is a RESOURCE failure.
        best_category = max(
            category_scores,
            key=lambda c: (category_scores[c], -category_first_seen[c]),
        )

        # Subcategory is the strongest rule within the winning category.
        best_key = max(
            (key for key in scores if key[0] == best_category),
            key=lambda k: (scores[k], -first_seen[k]),
        )

        best_score = category_scores[best_category]

        # Confidence is DOMINANCE, not raw score: a category that fires alone is
        # far more trustworthy than one tied with three others.
        #
        # Symptoms are excluded from the denominator when a cause won. A log
        # containing "connection refused" AND "1 test failed" is a confident
        # DATABASE failure, not a 50/50 toss-up — treating it as ambiguous would
        # escalate an obvious case to the LLM for nothing.
        competitors = {
            category: value
            for category, value in category_scores.items()
            if category not in SYMPTOM_CATEGORIES or best_category in SYMPTOM_CATEGORIES
        }

        total = sum(competitors.values()) or best_score
        dominance = best_score / total
        strength = min(best_score, 1.0)
        confidence = min(0.98, dominance * strength)

        return RuleResult(
            category=best_key[0],
            subcategory=best_key[1],
            confidence=round(confidence, 3),
            matched_rules=sorted(set(matched)),
        )
