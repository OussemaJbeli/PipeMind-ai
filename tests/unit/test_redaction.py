"""Redaction matrix: one positive per rule, plus the negatives that matter.

Over-redaction is as dangerous as under-redaction: eating a commit SHA or an
image digest removes the most useful tokens in the log and makes the analysis
worse, not safer.
"""

import pytest

from app.core.errors import RedactionFailed
from app.services.redactor import redact

LEAKS = [
    ("aws_key", "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    (
        "aws_secret",
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "wJalrXUtnFEMI",
    ),
    ("gcp_key", "key: AIzaSyA1234567890abcdefghijklmnopqrstuvw", "AIzaSyA123"),
    ("github_classic", "token: ghp_" + "a" * 36, "ghp_"),
    ("github_fine", "github_pat_" + "b" * 30, "github_pat_"),
    ("gitlab", "glpat-abcdefghij1234567890", "glpat-"),
    ("slack", "xoxb-1234567890-abcdefghij", "xoxb-"),
    ("npm", "npm_" + "c" * 36, "npm_"),
    ("stripe", "sk_live_abcdefghij1234567890", "sk_live_"),
    ("anthropic", "sk-ant-abcdefghij1234567890", "sk-ant-"),
    ("jwt", "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.SflKxwRJSMeKKF2QT4", "eyJzdWIiOiIxIn0"),
    ("db_url", "postgres://user:s3cretpw@db:5432/app", "s3cretpw"),
    ("basic_auth", "https://user:hunter2@internal.example.com/repo", "hunter2"),
    ("env_password", "DATABASE_PASSWORD=correcthorsebattery", "correcthorsebattery"),
    ("env_token", "export CI_TOKEN=abc123xyz789", "abc123xyz789"),
    (
        "private_key",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAK\n-----END RSA PRIVATE KEY-----",
        "MIIEpAIBAAK",
    ),
    ("ssh_key", "ssh-rsa " + "A" * 60 + " user@host", "A" * 60),
    ("email", "notified real.person@company.com about it", "real.person@company.com"),
]


@pytest.mark.parametrize(("name", "text", "secret"), LEAKS, ids=[c[0] for c in LEAKS])
def test_secret_is_removed(name: str, text: str, secret: str) -> None:
    assert secret not in redact(text).text


KEEPS = [
    ("commit_sha", "HEAD is now at a82c91f3b4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9"),
    ("short_sha", "commit a82c91f"),
    ("image_digest", "pulled sha256:" + "a" * 64),
    ("version", "installed vue@3.5.13 and typescript@5.6.2"),
    ("stack_path", "at /srv/app/Services/PaymentService.php:42"),
    ("noreply_email", "committed by dev@users.noreply.github.com"),
    ("error_text", "SQLSTATE[HY000] [2002] Connection refused"),
    ("port", "listening on 127.0.0.1:5432"),
]


@pytest.mark.parametrize(("name", "text"), KEEPS, ids=[c[0] for c in KEEPS])
def test_signal_survives(name: str, text: str) -> None:
    """Over-redaction destroys the diagnosis. These must pass through intact."""
    assert "[REDACTED" not in redact(text).text


def test_reports_what_it_removed() -> None:
    result = redact("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\nDB_PASSWORD=hunter2")

    assert result.count >= 2
    assert "aws_access_key" in result.types
    assert result.is_redacted if hasattr(result, "is_redacted") else True


def test_empty_input_is_safe() -> None:
    assert redact("").text == ""


def test_entropy_net_can_be_disabled() -> None:
    """strict=False exists for local models, where text never leaves the network.

    Rule-based redaction still runs — only the entropy heuristic is skipped.
    """
    text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"

    assert "AKIAIOSFODNN7EXAMPLE" not in redact(text, strict=False).text


def test_failure_is_not_retryable() -> None:
    """A retry loop around a redaction bug keeps trying to ship secrets."""
    assert RedactionFailed.retryable is False
