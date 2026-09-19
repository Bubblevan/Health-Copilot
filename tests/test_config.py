from health_ai_copilot.config import load_openai_config


def test_live_config_accepts_quoted_health_copilot_model_id(monkeypatch) -> None:
    monkeypatch.setenv("HEALTH_COPILOT_API_KEY", '"secret"')
    monkeypatch.delenv("HEALTH_COPILOT_MODEL", raising=False)
    monkeypatch.setenv("HEALTH_COPILOT_MODEL_ID", '"deepseek-flash"')
    monkeypatch.setenv("HEALTH_COPILOT_BASE_URL", '"https://api.example.test"')

    config = load_openai_config()

    assert config.api_key == "secret"
    assert config.model == "deepseek-flash"
    assert config.base_url == "https://api.example.test"
