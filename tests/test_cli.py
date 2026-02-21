from __future__ import annotations

from pathlib import Path

import pytest

from terminalai import cli
from terminalai.config import AppConfig


def _fake_config() -> AppConfig:
    return AppConfig(
        api_key=None,
        model="gpt-5.2",
        reasoning_effort="medium",
        safety_enabled=True,
        allow_unsafe=False,
        api_url="https://api.openai.com/v1/responses",
        log_dir="logs",
        system_prompt="prompt",
        allow_user_feedback_pause=False,
        shell="powershell",
        max_steps=20,
        working_directory=None,
    )


def test_parser_defaults() -> None:
    args = cli.build_parser().parse_args([])
    assert args.goal is None


def test_main_rejects_invalid_cwd_from_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_argv = ["terminalai", "list files"]
    monkeypatch.setattr("sys.argv", fake_argv)

    class FakeAdapter:
        name = "fake"

    def fake_config_with_missing_cwd() -> AppConfig:
        config = _fake_config()
        config.working_directory = "./definitely-missing-dir"
        return config

    monkeypatch.setattr(cli, "create_shell_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(
        cli,
        "AppConfig",
        type("FakeConfig", (), {"from_env": staticmethod(fake_config_with_missing_cwd)}),
    )

    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "Invalid configured cwd directory" in out


def test_main_passes_resolved_cwd_to_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_argv = ["terminalai", "list files"]
    monkeypatch.setattr("sys.argv", fake_argv)

    class FakeAdapter:
        name = "fake"

    captured: dict[str, object] = {}

    class FakeLoop:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self, _goal: str) -> list[object]:
            return []

    def fake_config_with_cwd() -> AppConfig:
        config = _fake_config()
        config.working_directory = str(tmp_path)
        return config

    monkeypatch.setattr(cli, "create_shell_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(cli, "AgentLoop", FakeLoop)
    monkeypatch.setattr(
        cli,
        "AppConfig",
        type("FakeConfig", (), {"from_env": staticmethod(fake_config_with_cwd)}),
    )

    assert cli.main() == 0
    assert captured["working_directory"] == str(tmp_path.resolve())


def test_build_runtime_context_contains_shell_and_directory() -> None:
    context = cli.build_runtime_context("powershell", "/tmp/work")

    assert "Runtime environment context:" in context
    assert "shell: powershell" in context
    assert "starting_working_directory: /tmp/work" in context
