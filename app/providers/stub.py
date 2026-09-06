"""Deterministic fake LLM.

Lets the entire pipeline — persistence, caching, the frontend, the E2E suite —
be built and tested with no key, no network and no cost.

Its output is LABELLED as stubbed, in the root cause itself, so a stubbed
analysis can never be mistaken for a real one in a screenshot or a demo. A
silent fake is worse than no fake.
"""

import json
import time

from app.providers.base import LLMResponse
from app.services.classifier.hybrid import HybridClassifier

STUB_MARKER = "[STUB — no LLM configured]"


class StubProvider:
    name = "stub"

    def __init__(self, model: str = "stub-v1") -> None:
        self._model = model

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: dict | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 90,
    ) -> LLMResponse:
        started = time.perf_counter()

        # Classify from the prompt body so the output tracks the real input
        # rather than being a fixed blob — the frontend then exercises every
        # category, severity and confidence path.
        classification = HybridClassifier().classify(prompt)

        matched = ", ".join(classification.matched_rules) or "no specific pattern"

        payload = {
            "category": classification.category,
            "subcategory": classification.subcategory,
            "severity": "high" if classification.confidence > 0.8 else "medium",
            "confidence": round(min(0.93, max(0.45, classification.confidence)), 3),
            "summary": f"{STUB_MARKER} {classification.category} failure detected.",
            "root_cause": (
                f"{STUB_MARKER} The rule classifier matched {matched}. "
                "Configure an AI provider to get a real root-cause analysis."
            ),
            "explanation": None,
            "is_transient": classification.category in {"NETWORK", "INFRASTRUCTURE"},
            "retry_recommended": classification.category in {"NETWORK", "INFRASTRUCTURE"},
            "evidence": [
                {
                    "type": "log_line",
                    "content": f"Matched by the rule classifier: {matched}",
                    "source_ref": "classifier",
                    "weight": classification.confidence,
                }
            ],
            "recommendations": [
                {
                    "title": "Configure an AI provider for a real analysis",
                    "description": (
                        "Add a Gemini API key or a local Ollama model under "
                        "Workspace → AI Providers."
                    ),
                    "action_type": "investigate",
                    "risk": "low",
                    "confidence": 1.0,
                    "affected_files": [],
                }
            ],
        }

        return LLMResponse(
            text=json.dumps(payload),
            parsed=payload,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=180,
            model=self._model,
            provider=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason="stop",
        )

    async def health(self) -> bool:
        return True

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0
