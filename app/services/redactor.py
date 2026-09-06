"""Secret redaction.

Runs FIRST, before cleaning or extraction: every later stage copies text around,
and a secret that survives stage one is in four places by stage four.

Maintenance rules:
  1. Order matters. Specific patterns (AKIA...) run before generic ones (40-char
     base64), or the generic rule eats the specific match and the type label
     is lost.
  2. Entropy detection is a net, not a scalpel. It needs the exclusion list, or
     it eats commit SHAs and image digests — which are the most useful tokens
     in the whole log.
  3. Every new rule needs a positive test with a real-shaped example AND a
     negative test proving it does not eat legitimate content.
  4. Fail closed. Any exception aborts the whole call.
"""

import math
import re
from dataclasses import dataclass, field
from re import Pattern

from app.core.errors import RedactionFailed


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: Pattern[str]
    replacement: str
    # Cheap literal gate. Running 23 regexes over a multi-megabyte log costs
    # ~1.1 s, and most of them cannot match at all — a substring check on a
    # lowercased copy is a C-level scan and skips them outright.
    # Empty means "always run" (patterns with no distinctive literal).
    prefilter: tuple[str, ...] = ()


def _c(p: str) -> Pattern[str]:
    return re.compile(p, re.IGNORECASE | re.MULTILINE)


RULES: list[Rule] = [
    # --- private keys and certs (multiline: must run before anything else) ---
    Rule(
        "private_key",
        re.compile(
            r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]+?-----END[A-Z ]*PRIVATE KEY-----",
            re.IGNORECASE,
        ),
        "[REDACTED_PRIVATE_KEY_BLOCK]",
        ("private key",),
    ),
    Rule(
        "ssh_key",
        _c(r"\bssh-(?:rsa|ed25519|dss)\s+[A-Za-z0-9+/=]{40,}"),
        "[REDACTED_SSH_KEY]",
        ("ssh-",),
    ),
    # --- cloud provider keys (most specific first) ---
    Rule(
        "aws_access_key",
        _c(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"),
        "[REDACTED_AWS_KEY]",
        ("akia", "asia", "abia", "acca"),
    ),
    Rule("gcp_key", _c(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED_GCP_KEY]", ("aiza",)),
    Rule(
        "gcp_service_acct",
        _c(r'"private_key"\s*:\s*"-----BEGIN[^"]+"'),
        '"private_key": "[REDACTED_PRIVATE_KEY]"',
        ("private_key",),
    ),
    # --- vcs / registry / ci tokens ---
    Rule(
        "github_token",
        _c(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"),
        "[REDACTED_GITHUB_TOKEN]",
        ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
    ),
    Rule(
        "github_fine",
        _c(r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b"),
        "[REDACTED_GITHUB_PAT]",
        ("github_pat_",),
    ),
    Rule(
        "gitlab_token",
        _c(r"\b(?:glpat|glptt|gldt|glrt|glsoat|glimt|glagent)-[A-Za-z0-9_\-]{20,}\b"),
        "[REDACTED_GITLAB_TOKEN]",
        ("glpat-", "glptt-", "gldt-", "glrt-", "glsoat-", "glimt-", "glagent-"),
    ),
    Rule(
        "slack_token", _c(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "[REDACTED_SLACK_TOKEN]", ("xox",)
    ),
    Rule(
        "slack_webhook",
        _c(r"https://hooks\.slack\.com/services/\S+"),
        "[REDACTED_SLACK_WEBHOOK]",
        ("hooks.slack.com",),
    ),
    Rule("npm_token", _c(r"\bnpm_[A-Za-z0-9]{36}\b"), "[REDACTED_NPM_TOKEN]", ("npm_",)),
    Rule(
        "pypi_token",
        _c(r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{20,}\b"),
        "[REDACTED_PYPI_TOKEN]",
        ("pypi-",),
    ),
    Rule(
        "docker_auth",
        _c(r'"auth"\s*:\s*"[A-Za-z0-9+/=]{20,}"'),
        '"auth": "[REDACTED]"',
        ('"auth"',),
    ),
    Rule(
        "anthropic_key", _c(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_API_KEY]", ("sk-ant-",)
    ),
    Rule(
        "stripe_key",
        _c(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{20,}\b"),
        "[REDACTED_STRIPE_KEY]",
        ("sk_live_", "sk_test_", "rk_live_", "rk_test_", "pk_live_", "pk_test_"),
    ),
    Rule("openai_key", _c(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_API_KEY]", ("sk-",)),
    # --- jwt / bearer / basic ---
    Rule(
        "jwt",
        _c(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
        "[REDACTED_JWT]",
        ("eyj",),
    ),
    Rule(
        "bearer",
        _c(r"\b(Bearer|Token|Authorization:)\s+\S{12,}"),
        r"\1 [REDACTED]",
        ("bearer ", "token ", "authorization:"),
    ),
    Rule("basic_auth_url", _c(r"://([^:/\s]+):([^@/\s]+)@"), "://[REDACTED]:[REDACTED]@", ("://",)),
    # --- connection strings ---
    Rule(
        "db_url",
        _c(
            r"\b(postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|mssql)://[^:\s]+:[^@\s]+@[^\s]+"
        ),
        r"\1://[REDACTED_CONNECTION_STRING]",
        ("://",),
    ),
    # --- env-var assignments: the single most common leak in CI logs ---
    Rule(
        "env_secret",
        _c(
            r"\b([A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN|APIKEY|API_KEY|"
            r"PRIVATE_KEY|ACCESS_KEY|CREDENTIAL|AUTH|SIGNING_KEY|"
            r"ENCRYPTION_KEY|DSN)[A-Z0-9_]*)\s*[:=]\s*['\"]?([^\s'\"]+)"
        ),
        r"\1=[REDACTED]",
        (
            "password",
            "passwd",
            "secret",
            "token",
            "apikey",
            "api_key",
            "private_key",
            "access_key",
            "credential",
            "auth",
            "signing_key",
            "encryption_key",
            "dsn",
        ),
    ),
    # --- aws secret: 40 chars of base64 ---
    # A git SHA is ALSO exactly 40 characters, so the naive pattern eats every
    # commit hash in the log — destroying the most useful token there is. The
    # lookahead excludes all-hex runs, which AWS secrets effectively never are.
    # No prefilter: there is no distinctive literal to gate on.
    Rule(
        "aws_secret",
        _c(
            r"(?<![A-Za-z0-9/+=])(?![0-9a-fA-F]{40}(?![A-Za-z0-9/+=]))[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"
        ),
        "[REDACTED_AWS_SECRET]",
    ),
    # --- PII ---
    Rule(
        "email",
        _c(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
        ("@",),
    ),
]

# Commit authorship is legitimate signal, and these hosts are not secrets.
EMAIL_ALLOWLIST = re.compile(
    r"@(?:users\.noreply\.github\.com|noreply\.gitlab\.com|example\.(?:com|org|test))$",
    re.IGNORECASE,
)

HIGH_ENTROPY_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9+/=_\-])[A-Za-z0-9+/=_\-]{32,128}(?![A-Za-z0-9+/=_\-])"
)

# Things that look random but are not secrets. Redacting these destroys the
# signal the analysis actually needs.
ENTROPY_EXCLUDE = re.compile(
    r"^(?:"
    r"[0-9a-f]{7,40}"  # git SHAs
    r"|sha(?:1|256|512)[:\-][0-9a-f]{32,128}"  # image digests
    r"|[0-9a-f]{32}"  # md5 / cache keys
    r"|[A-Za-z0-9_\-]+\.(?:js|ts|css|map|so|jar|whl|tar|gz|min)"  # hashed assets
    r"|\[REDACTED[_A-Z]*\]"  # our own markers
    r")$",
    re.IGNORECASE,
)

ENTROPY_THRESHOLD = 4.2


@dataclass
class RedactionResult:
    text: str
    count: int = 0
    types: list[str] = field(default_factory=list)


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0

    length = len(value)
    counts = {ch: value.count(ch) for ch in set(value)}

    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def redact(text: str, *, strict: bool = True) -> RedactionResult:
    """Remove secrets. Fails closed — never returns partially-redacted text."""
    if not text:
        return RedactionResult(text="", count=0, types=[])

    try:
        found: dict[str, int] = {}
        out = text
        haystack = text.lower()

        for rule in RULES:
            if rule.prefilter and not any(literal in haystack for literal in rule.prefilter):
                continue

            if rule.name == "email":

                def _email(match: re.Match[str]) -> str:
                    if EMAIL_ALLOWLIST.search(match.group(0)):
                        return match.group(0)

                    found["email"] = found.get("email", 0) + 1
                    return "[REDACTED_EMAIL]"

                out = rule.pattern.sub(_email, out)
                continue

            out, hits = rule.pattern.subn(rule.replacement, out)

            if hits:
                found[rule.name] = found.get(rule.name, 0) + hits

        if strict:

            def _entropy(match: re.Match[str]) -> str:
                token = match.group(0)

                if ENTROPY_EXCLUDE.match(token):
                    return token

                if _shannon_entropy(token) >= ENTROPY_THRESHOLD:
                    found["high_entropy"] = found.get("high_entropy", 0) + 1
                    return "[REDACTED_HIGH_ENTROPY]"

                return token

            out = HIGH_ENTROPY_CANDIDATE.sub(_entropy, out)

        return RedactionResult(text=out, count=sum(found.values()), types=sorted(found))

    except Exception as exc:
        # Deliberately broad: a partially-redacted result must never escape.
        raise RedactionFailed(f"redaction aborted: {type(exc).__name__}") from exc
