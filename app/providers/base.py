"""Provider abstraction.

Nothing outside app/providers/ may import a vendor SDK. The analyzer calls
`complete()` and never learns which model answered.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    text: str
    parsed: dict[str, Any] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: dict | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 90,
    ) -> LLMResponse: ...

    async def health(self) -> bool: ...

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float: ...
