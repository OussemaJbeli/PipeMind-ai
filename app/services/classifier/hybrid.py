"""Cheapest sufficient answer wins.

The LLM is a fallback, not the default. `classification_source` is persisted on
every analysis so the question "how often did we actually need the model?" is
answerable with data rather than opinion — and that number belongs in the report.
"""

from dataclasses import dataclass, field

from app.services.classifier.ml import MLClassifier
from app.services.classifier.rules import RuleClassifier


@dataclass
class Classification:
    category: str
    subcategory: str | None = None
    confidence: float = 0.0
    source: str = "rules"
    matched_rules: list[str] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)


class HybridClassifier:
    HIGH = 0.85
    LOW = 0.50

    def __init__(self) -> None:
        self.rules = RuleClassifier()
        self.ml = MLClassifier()

    def classify(self, text: str, ecosystem: str | None = None) -> Classification:
        rule = self.rules.classify(text, ecosystem)

        # A confident rule hit is final: deterministic, explainable and free.
        if rule.confidence >= self.HIGH:
            return Classification(
                rule.category, rule.subcategory, rule.confidence, "rules", rule.matched_rules
            )

        ml = self.ml.classify(text)

        if ml is None:
            return Classification(
                rule.category, rule.subcategory, rule.confidence, "rules", rule.matched_rules
            )

        # Agreement between two independent methods is worth more than either
        # alone, so the combined confidence exceeds both.
        if ml.category == rule.category and rule.confidence > 0:
            return Classification(
                rule.category,
                rule.subcategory,
                min(0.97, max(rule.confidence, ml.confidence) + 0.10),
                "hybrid",
                rule.matched_rules,
                ml.probabilities,
            )

        if ml.confidence >= self.HIGH:
            return Classification(ml.category, None, ml.confidence, "ml", [], ml.probabilities)

        # Both unsure. Hand it to the LLM, which sees full context the
        # classifiers never get: changed files, previous status, history.
        if max(rule.confidence, ml.confidence) < self.LOW:
            return Classification(
                "UNKNOWN",
                None,
                max(rule.confidence, ml.confidence),
                "llm",
                rule.matched_rules,
                ml.probabilities,
            )

        return (
            Classification(
                rule.category,
                rule.subcategory,
                rule.confidence,
                "rules",
                rule.matched_rules,
                ml.probabilities,
            )
            if rule.confidence >= ml.confidence
            else Classification(ml.category, None, ml.confidence, "ml", [], ml.probabilities)
        )
