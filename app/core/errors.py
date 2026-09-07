"""Typed errors that map onto the same JSON shape Laravel returns.

Laravel forwards `error_code` straight through to the browser, and the UI
switches on it — a failed analysis is not a failed pipeline, and the two must
stay distinguishable all the way to the screen.
"""


class PipeMindAIError(Exception):
    code = "AI_ERROR"
    status = 500
    retryable = False


class LLMUnavailable(PipeMindAIError):
    code, status, retryable = "AI_PROVIDER_ERROR", 503, True


class LLMTimeout(PipeMindAIError):
    code, status, retryable = "AI_PROVIDER_TIMEOUT", 504, True


class LLMRateLimited(PipeMindAIError):
    code, status, retryable = "AI_PROVIDER_RATE_LIMITED", 429, True


class InvalidLLMResponse(PipeMindAIError):
    """The model returned something that will not fit the contract."""

    code, status, retryable = "AI_INVALID_RESPONSE", 502, True


class LLMAuthenticationFailed(PipeMindAIError):
    """The provider rejected the credentials.

    Deliberately NOT retryable, and deliberately not `LLMUnavailable`: a wrong
    API key is a configuration mistake only a human can fix. Retrying it — at the
    tenacity layer, then the fallback provider, then the queue — turns one typo
    into nine useless round trips and buries the one message that would have
    told the user what to do.
    """

    code, status, retryable = "AI_PROVIDER_UNAUTHORIZED", 401, False


class BudgetExceeded(PipeMindAIError):
    code, status, retryable = "AI_BUDGET_EXCEEDED", 402, False


class RedactionFailed(PipeMindAIError):
    """Fail closed: if redaction cannot complete, nothing leaves the building.

    Deliberately NOT retryable. A retry loop around a redaction bug is a loop
    that keeps trying to ship secrets to a third party.
    """

    code, status, retryable = "AI_REDACTION_FAILED", 500, False
