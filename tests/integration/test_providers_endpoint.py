def test_testing_a_provider_requires_the_service_token(client):
    assert client.post("/v1/providers/test", json={"provider": "stub"}).status_code == 401


def test_a_working_provider_reports_ok(client, auth):
    body = client.post("/v1/providers/test", json={"provider": "stub"}, headers=auth).json()

    assert body["ok"] is True
    assert body["provider"] == "stub"


def test_bad_credentials_are_a_200_not_a_500(client, auth):
    """'These credentials are wrong' is a valid answer to 'do these work?'.
    A 500 would make the wizard show a crash instead of the reason."""
    response = client.post("/v1/providers/test", json={"provider": "gemini"}, headers=auth)

    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["message"]


def test_an_unknown_provider_is_reported_not_raised(client, auth):
    body = client.post("/v1/providers/test", json={"provider": "nonsense"}, headers=auth).json()

    assert body["ok"] is False
    assert "nonsense" in body["message"]


def test_the_test_payload_matches_the_analysis_override_shape(client, auth):
    """Testing a different shape from the one analysis sends proves nothing."""
    from app.models.requests import AnalyzeRequest, TestProviderRequest

    override_keys = set(TestProviderRequest.model_fields)

    assert override_keys == {"provider", "api_key", "model", "base_url"}
    assert "llm_override" in AnalyzeRequest.model_fields
