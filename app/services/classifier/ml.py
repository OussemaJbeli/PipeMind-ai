"""Optional TF-IDF classifier.

Absent until `scripts/train.py` has run, which needs a labelled dataset — and the
dataset comes from analysis_feedback.correct_category, i.e. from people using the
product. Until then this returns None and the hybrid falls back to rules, which
is why M3 does not depend on it.
"""

from dataclasses import dataclass
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "classifier_v1.joblib"


@dataclass
class MLResult:
    category: str
    confidence: float
    probabilities: dict[str, float]


class MLClassifier:
    def __init__(self) -> None:
        self._pipeline = None
        self._unavailable = False

    def available(self) -> bool:
        return MODEL_PATH.exists() and not self._unavailable

    def classify(self, text: str) -> MLResult | None:
        if not self.available():
            return None

        try:
            if self._pipeline is None:
                import joblib  # imported lazily: the ml extra is optional

                self._pipeline = joblib.load(MODEL_PATH)

            probabilities = self._pipeline.predict_proba([text])[0]
            classes = self._pipeline.classes_
        except Exception:
            # A missing extra or a stale artefact must degrade to rules, never
            # take the analysis down with it.
            self._unavailable = True
            return None

        ranked = sorted(
            zip(classes, probabilities, strict=False), key=lambda kv: kv[1], reverse=True
        )

        return MLResult(
            category=str(ranked[0][0]),
            confidence=float(ranked[0][1]),
            probabilities={str(c): round(float(p), 4) for c, p in ranked[:5]},
        )
