import json

from terminalai.config import AppConfig


def test_app_config_loads_openai_and_model_reasoning_from_file(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "terminalai.config.json"
    config_path.write_text(
        json.dumps(
            {
                "openai": {
                    "api_key": "test-key",
                    "api_url": "https://api.openai.com/v1/responses",
                },
                "default_model": "gpt-5.2-codex",
                "models": {"gpt-5.2-codex": {"reasoning_effort": "medium"}},
                "log_dir": "test-logs",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("TERMINALAI_CONFIG_FILE", str(config_path))

    config = AppConfig.from_env()

    assert config.api_key == "test-key"
    assert config.model == "gpt-5.2-codex"
    assert config.reasoning_effort == "medium"
    assert config.log_dir == "test-logs"


def test_env_overrides_reasoning_effort(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "terminalai.config.json"
    config_path.write_text(
        json.dumps(
            {
                "default_model": "gpt-5.2-codex",
                "models": {"gpt-5.2-codex": {"reasoning_effort": "low"}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("TERMINALAI_CONFIG_FILE", str(config_path))
    monkeypatch.setenv("TERMINALAI_REASONING_EFFORT", "high")

    config = AppConfig.from_env()

    assert config.reasoning_effort == "high"


def test_reasoning_defaults_to_medium_for_gpt5_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("TERMINALAI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("TERMINALAI_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("TERMINALAI_MODEL", "gpt-5.2")

    config = AppConfig.from_env()

    assert config.reasoning_effort == "medium"


def test_top_level_api_key_is_supported_in_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "terminalai.config.json"
    config_path.write_text(
        json.dumps({"api_key": "top-level-key", "default_model": "gpt-5.2"}),
        encoding="utf-8",
    )

    monkeypatch.setenv("TERMINALAI_CONFIG_FILE", str(config_path))
    monkeypatch.delenv("TERMINALAI_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TERMINALAI_API_KEY", raising=False)

    config = AppConfig.from_env()

    assert config.api_key == "top-level-key"
