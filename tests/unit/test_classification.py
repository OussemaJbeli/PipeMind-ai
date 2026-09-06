import pytest

from app.services.classifier.hybrid import HybridClassifier
from app.services.classifier.rules import RULES, RuleClassifier

CASES = [
    ("DATABASE", "SQLSTATE[HY000] [2002] Connection refused"),
    ("DATABASE", "psql: error: connection to server at localhost failed"),
    ("DEPENDENCY", "npm ERR! ERESOLVE unable to resolve dependency tree"),
    ("DEPENDENCY", "npm error 404 Not Found - GET https://registry.npmjs.org/nope"),
    ("TEST", "not ok 2 - login\n    Expected status 200, Received status 401\n# fail 1"),
    ("BUILD", "src/auth.ts(2,15): error TS2345: Argument of type 'string | undefined'"),
    ("NETWORK", "curl: (6) Could not resolve host: example.invalid"),
    ("RESOURCE", "FATAL ERROR: JavaScript heap out of memory"),
    ("RESOURCE", "write error: No space left on device"),
    ("CONFIGURATION", "ERROR: required environment variable REQUIRED_SECRET is not set"),
    ("PERMISSION", "cat: locked: Permission denied"),
    ("DOCKER", "failed to solve: process did not complete successfully: exit code 1"),
    ("AUTHENTICATION", "remote: HTTP 401 Unauthorized - token has expired"),
    ("INFRASTRUCTURE", "This job is stuck because no runners are online"),
    ("DEPLOYMENT", "readiness probe failed: HTTP probe failed"),
]


@pytest.mark.parametrize(
    ("expected", "text"), CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)]
)
def test_classifies_real_error_strings(expected: str, text: str) -> None:
    assert RuleClassifier().classify(text).category == expected


def test_unknown_is_a_real_answer() -> None:
    """Forcing a guess pollutes every other class."""
    result = RuleClassifier().classify("Build completed in 42s")

    assert result.category == "UNKNOWN"
    assert result.confidence == 0.0


def test_symptoms_do_not_dilute_a_cause() -> None:
    """A DB outage that also broke tests is a confident DATABASE failure.

    Treating it as a 50/50 toss-up would escalate an obvious case to the LLM.
    """
    log = "SQLSTATE[HY000] [2002] Connection refused\nFAIL DatabaseTest\nTests: 1 failed"
    result = RuleClassifier().classify(log)

    assert result.category == "DATABASE"
    assert result.confidence >= 0.85


def test_same_category_rules_reinforce() -> None:
    """Two TEST rules firing is corroboration, not ambiguity."""
    log = "not ok 2 - login\n    Expected status 200, Received status 401\n# fail 1"

    assert RuleClassifier().classify(log).confidence >= 0.85


def test_cause_outranks_symptom_on_a_tie() -> None:
    log = "FATAL ERROR: JavaScript heap out of memory\nFAIL build.test.ts"

    assert RuleClassifier().classify(log).category == "RESOURCE"


def test_every_rule_has_a_subcategory() -> None:
    for rule in RULES:
        assert rule.subcategory, f"{rule.category} has a rule with no subcategory"


class TestHybrid:
    def test_falls_back_to_rules_without_a_trained_model(self) -> None:
        """The ML model does not exist until a dataset does — M3 must not depend on it."""
        result = HybridClassifier().classify("SQLSTATE[HY000] [2002] Connection refused")

        assert result.source == "rules"
        assert result.category == "DATABASE"

    def test_defers_to_the_llm_when_genuinely_unsure(self) -> None:
        result = HybridClassifier().classify("Build completed in 42s")

        assert result.category == "UNKNOWN"
