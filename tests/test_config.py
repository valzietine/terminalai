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
    assert config.allow_user_feedback_pause is False
    assert config.confirm_before_complete is False


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


def test_system_prompt_supports_file_and_env_override(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "terminalai.config.json"
    config_path.write_text(
        json.dumps({"system_prompt": "prompt from file", "default_model": "gpt-5.2"}),
        encoding="utf-8",
    )

    monkeypatch.setenv("TERMINALAI_CONFIG_FILE", str(config_path))
    monkeypatch.delenv("TERMINALAI_SYSTEM_PROMPT", raising=False)

    file_config = AppConfig.from_env()
    assert file_config.system_prompt == "prompt from file"

    monkeypatch.setenv("TERMINALAI_SYSTEM_PROMPT", "prompt from env")

    env_config = AppConfig.from_env()
    assert env_config.system_prompt == "prompt from env"


def test_allow_user_feedback_pause_loads_from_file_and_env(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "terminalai.config.json"
    config_path.write_text(
        json.dumps({"allow_user_feedback_pause": True, "default_model": "gpt-5.2"}),
        encoding="utf-8",
    )

    monkeypatch.setenv("TERMINALAI_CONFIG_FILE", str(config_path))
    monkeypatch.delenv("TERMINALAI_ALLOW_USER_FEEDBACK_PAUSE", raising=False)

    file_config = AppConfig.from_env()
    assert file_config.allow_user_feedback_pause is True

    monkeypatch.setenv("TERMINALAI_ALLOW_USER_FEEDBACK_PAUSE", "false")

    env_config = AppConfig.from_env()
    assert env_config.allow_user_feedback_pause is False




def test_confirm_before_complete_loads_from_file_and_env(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "terminalai.config.json"
    config_path.write_text(
        json.dumps({"confirm_before_complete": True, "default_model": "gpt-5.2"}),
        encoding="utf-8",
    )

    monkeypatch.setenv("TERMINALAI_CONFIG_FILE", str(config_path))
    monkeypatch.delenv("TERMINALAI_CONFIRM_BEFORE_COMPLETE", raising=False)

    file_config = AppConfig.from_env()
    assert file_config.confirm_before_complete is True

    monkeypatch.setenv("TERMINALAI_CONFIRM_BEFORE_COMPLETE", "false")

    env_config = AppConfig.from_env()
    assert env_config.confirm_before_complete is False

def test_runtime_options_load_from_file_and_env(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "terminalai.config.json"
    config_path.write_text(
        json.dumps(
            {
                "default_model": "gpt-5.2",
                "shell": "cmd",
                "max_steps": 11,
                "cwd": "./test-dir",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("TERMINALAI_CONFIG_FILE", str(config_path))
    monkeypatch.delenv("TERMINALAI_SHELL", raising=False)
    monkeypatch.delenv("TERMINALAI_MAX_STEPS", raising=False)
    monkeypatch.delenv("TERMINALAI_CWD", raising=False)

    file_config = AppConfig.from_env()
    assert file_config.shell == "cmd"
    assert file_config.max_steps == 11
    assert file_config.working_directory == "./test-dir"

    monkeypatch.setenv("TERMINALAI_SHELL", "powershell")
    monkeypatch.setenv("TERMINALAI_MAX_STEPS", "7")
    monkeypatch.setenv("TERMINALAI_CWD", "~/project")

    env_config = AppConfig.from_env()
    assert env_config.shell == "powershell"
    assert env_config.max_steps == 7
    assert env_config.working_directory == "~/project"


def test_local_config_auto_loaded_without_env_override(tmp_path, monkeypatch) -> None:
    (tmp_path / "terminalai.config.json").write_text(
        json.dumps(
            {
                "default_model": "gpt-5.2",
                "openai": {
                    "api_url": "https://example.invalid/base",
                    "api_key": "base-key",
                },
                "max_steps": 20,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "terminalai.config.local.json").write_text(
        json.dumps(
            {
                "openai": {"api_key": "local-key"},
                "max_steps": 7,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TERMINALAI_CONFIG_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.api_key == "local-key"
    assert config.api_url == "https://example.invalid/base"
    assert config.max_steps == 7


def test_explicit_config_file_disables_local_auto_merge(tmp_path, monkeypatch) -> None:
    explicit_path = tmp_path / "custom.config.json"
    explicit_path.write_text(
        json.dumps({"default_model": "gpt-5.2", "max_steps": 3}),
        encoding="utf-8",
    )
    (tmp_path / "terminalai.config.local.json").write_text(
        json.dumps({"max_steps": 99}),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TERMINALAI_CONFIG_FILE", str(explicit_path))

    config = AppConfig.from_env()

    assert config.max_steps == 3
