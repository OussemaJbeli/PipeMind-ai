import pytest

from app.core.errors import LLMAuthenticationFailed, LLMRateLimited, LLMUnavailable
from app.providers.gemini import GeminiProvider
from app.providers.registry import build_provider
from app.providers.stub import StubProvider


@pytest.mark.asyncio
async def test_stub_verifies_without_credentials():
    check = await StubProvider().verify()

    assert check.ok
    assert "canned" in check.message


@pytest.mark.asyncio
async def test_ollama_reports_an_unreachable_server_rather_than_hanging():
    from app.providers.ollama import OllamaProvider

    check = await OllamaProvider(base_url="http://127.0.0.1:9", model="llama3.1").verify()

    assert not check.ok
    assert "127.0.0.1:9" in check.message


class TestGeminiErrorTranslation:
    def test_a_malformed_key_is_diagnosed_as_a_bad_paste(self):
        """Google's 401 blames the credential *type*, which is misleading: the
        same reason code comes back for a valid-format key with one stray
        character. Chasing the type wastes the user's time."""
        error = GeminiProvider._translate(
            Exception("401 UNAUTHENTICATED ... reason: ACCESS_TOKEN_TYPE_UNSUPPORTED")
        )

        assert isinstance(error, LLMAuthenticationFailed)
        assert "aistudio.google.com/apikey" in str(error)
        # Points at a truncated paste, which is what actually causes this.
        # Google returns ACCESS_TOKEN_TYPE_UNSUPPORTED for a malformed key of a
        # valid format, so claiming "wrong credential type" sends the user off to
        # regenerate a key that was never the problem.
        assert "copied whole" in str(error)

    def test_a_rejected_key_is_not_retryable(self):
        """Retrying a typo through tenacity, the fallback provider and the queue
        turns one mistake into nine calls and buries the useful message."""
        error = GeminiProvider._translate(Exception("400 API key not valid"))

        assert isinstance(error, LLMAuthenticationFailed)
        assert error.retryable is False
        assert error.code == "AI_PROVIDER_UNAUTHORIZED"

    def test_rate_limits_stay_retryable(self):
        error = GeminiProvider._translate(Exception("429 RESOURCE_EXHAUSTED"))

        assert isinstance(error, LLMRateLimited)
        assert error.retryable is True

    def test_an_unknown_failure_stays_a_generic_outage(self):
        assert isinstance(GeminiProvider._translate(Exception("kaboom")), LLMUnavailable)


# The exact text Google returns. Kept verbatim: a paraphrase is what let the
# original bug through — the synthetic message in an earlier version of this test
# omitted "GenerateContent", which is the very substring that caused the
# misclassification.
REAL_GEMINI_401 = (
    "401 UNAUTHENTICATED. {'error': {'code': 401, 'message': 'Request had invalid "
    "authentication credentials. Expected OAuth 2 access token, login cookie or other "
    "valid authentication credential.', 'status': 'UNAUTHENTICATED', 'details': "
    "[{'@type': 'type.googleapis.com/google.rpc.ErrorInfo', 'reason': "
    "'ACCESS_TOKEN_TYPE_UNSUPPORTED', 'metadata': {'service': "
    "'generativelanguage.googleapis.com', 'method': "
    "'google.ai.generativelanguage.v1beta.GenerativeService.GenerateContent'}}]}}"
)


class TestSubstringMisclassification:
    def test_generate_content_is_not_a_rate_limit(self):
        """'rate' is a substring of 'generate'.

        Gemini names the method GenerateContent in every error it returns, so a
        bare `"rate" in message` test classified every failure as a rate limit —
        which is retryable, so a wrong API key was retried three times with
        backoff and then surfaced as the wrong problem entirely.
        """
        error = GeminiProvider._translate(Exception(REAL_GEMINI_401))

        assert isinstance(error, LLMAuthenticationFailed)
        assert not isinstance(error, LLMRateLimited)
        assert error.retryable is False

    def test_the_real_401_still_yields_the_actionable_message(self):
        error = GeminiProvider._translate(Exception(REAL_GEMINI_401))

        assert "aistudio.google.com/apikey" in str(error)

    def test_a_genuine_rate_limit_is_still_recognised(self):
        error = GeminiProvider._translate(
            Exception("429 RESOURCE_EXHAUSTED. Quota exceeded for GenerateContent.")
        )

        assert isinstance(error, LLMRateLimited)
        assert error.retryable is True

    def test_the_openai_provider_agrees_on_ordering(self):
        from app.providers.openai_compatible import OpenAICompatibleProvider

        error = OpenAICompatibleProvider._translate(
            Exception("Error code: 401 - Incorrect API key provided")
        )

        assert isinstance(error, LLMAuthenticationFailed)
        assert error.retryable is False


class TestCredentialFallback:
    def test_analysis_may_fall_back_to_the_server_key(self, monkeypatch):
        from app.config import settings

        settings.cache_clear()
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaTestKeyForFallbackChecking000000000")

        provider = build_provider({"provider": "gemini"})

        assert provider._api_key.startswith("AIza")
        settings.cache_clear()

    def test_credential_testing_may_not(self, monkeypatch):
        """Borrowing the server's key here would report 'your key works' about a
        key the user never entered — worse than any error message."""
        from app.config import settings

        settings.cache_clear()
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaTestKeyForFallbackChecking000000000")

        with pytest.raises(LLMUnavailable, match="No Gemini API key"):
            build_provider({"provider": "gemini"}, allow_env_fallback=False)

        settings.cache_clear()


class TestModelAvailability:
    """ListModels is not evidence a model can be called.

    Google lists models that `generateContent` then rejects with 404 "no longer
    available to new users". A verify() that trusted the list reported
    "Connected" for a model that failed on its first real analysis — the exact
    false confidence the endpoint exists to prevent.
    """

    def test_the_replacement_model_is_extracted_from_googles_message(self):
        message = (
            "404 NOT_FOUND. This model models/gemini-2.5-flash is no longer available "
            "to new users. Please update your code to use models/gemini-3.6-flash for "
            "the latest features."
        )

        assert GeminiProvider._suggested(message) == "Google suggests 'gemini-3.6-flash'."

    def test_an_unparseable_message_still_gives_direction(self):
        assert "Pick another model" in GeminiProvider._suggested("404 gone")
