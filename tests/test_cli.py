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
    )


def test_parser_defaults() -> None:
    args = cli.build_parser().parse_args([])
    assert args.shell == "powershell"
    assert args.goal is None
    assert args.model is None
    assert args.max_steps == 20
    assert args.cwd is None


def test_parser_accepts_cwd() -> None:
    args = cli.build_parser().parse_args(["--cwd", "./tmp", "list files"])
    assert args.cwd == "./tmp"
    assert args.goal == "list files"


def test_main_rejects_invalid_cwd_via_argv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_argv = ["terminalai", "--cwd", "./definitely-missing-dir", "list files"]
    monkeypatch.setattr("sys.argv", fake_argv)

    class FakeAdapter:
        name = "fake"

    monkeypatch.setattr(cli, "create_shell_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(
        cli,
        "AppConfig",
        type("FakeConfig", (), {"from_env": staticmethod(_fake_config)}),
    )

    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "Invalid --cwd directory" in out


def test_main_passes_resolved_cwd_to_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_argv = ["terminalai", "--cwd", str(tmp_path), "list files"]
    monkeypatch.setattr("sys.argv", fake_argv)

    class FakeAdapter:
        name = "fake"

    captured: dict[str, object] = {}

    class FakeLoop:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self, _goal: str) -> list[object]:
            return []

    monkeypatch.setattr(cli, "create_shell_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(cli, "AgentLoop", FakeLoop)
    monkeypatch.setattr(
        cli,
        "AppConfig",
        type("FakeConfig", (), {"from_env": staticmethod(_fake_config)}),
    )

    assert cli.main() == 0
    assert captured["working_directory"] == str(tmp_path.resolve())
