from fastapi.testclient import TestClient

from omniground.api.app import create_app


def test_health_endpoint_with_structured_modules(tmp_path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """default_model: remote\nmodels:\n  remote:\n    backend: openai\n    mode: api\n    base_url: https://example.invalid/v1\n    model_name: example\n    api_key_env: EXAMPLE_API_KEY\n""",
        encoding="utf-8",
    )

    with TestClient(create_app(config_path=str(config_path))) as client:
        assert client.get("/health").json() == {"status": "ok"}
