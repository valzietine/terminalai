from __future__ import annotations

from pathlib import Path

import pytest

from terminalai import cli
from terminalai.agent.models import SessionTurn
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
        confirm_before_complete=False,
        completion_prompt_enabled=True,
        completion_prompt_text="Task finished. Do you want to keep going with new instructions?",
        shell="powershell",
        max_steps=20,
        working_directory=None,
    )


def test_parser_defaults() -> None:
    args = cli.build_parser().parse_args([])
    assert args.goal is None
    assert args.working_directory is None


def test_parser_accepts_cwd_override() -> None:
    args = cli.build_parser().parse_args(["--cwd", "./sandbox", "list files"])

    assert args.working_directory == "./sandbox"
    assert args.goal == "list files"


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
        config.completion_prompt_enabled = False
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
    assert captured["safety_enabled"] is True
    assert captured["allow_unsafe"] is False


def test_main_cwd_cli_override_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    fake_argv = ["terminalai", "--cwd", str(override_dir), "list files"]
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
        config.working_directory = "./ignored-from-config"
        config.completion_prompt_enabled = False
        return config

    monkeypatch.setattr(cli, "create_shell_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(cli, "AgentLoop", FakeLoop)
    monkeypatch.setattr(
        cli,
        "AppConfig",
        type("FakeConfig", (), {"from_env": staticmethod(fake_config_with_cwd)}),
    )

    assert cli.main() == 0
    assert captured["working_directory"] == str(override_dir.resolve())


def test_build_runtime_context_contains_shell_and_directory() -> None:
    context = cli.build_runtime_context(
        "powershell",
        "/tmp/work",
        safety_enabled=True,
        allow_unsafe=False,
    )

    assert "Runtime environment context:" in context
    assert "shell: powershell" in context
    assert "starting_working_directory: /tmp/work" in context
    assert "safety_enabled: True" in context
    assert "allow_unsafe: False" in context


def test_main_collects_feedback_and_prints_resumed_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_argv = ["terminalai", "need input"]
    monkeypatch.setattr("sys.argv", fake_argv)

    class FakeAdapter:
        name = "fake"

    class FakeLoop:
        def __init__(self, **kwargs: object) -> None:
            self.request_user_feedback = kwargs["request_user_feedback"]

        def run(self, _goal: str) -> list[SessionTurn]:
            response = self.request_user_feedback("Which environment should I target?")
            assert response == "Use staging"
            return [
                SessionTurn(
                    input="need input",
                    command="",
                    output="",
                    next_action_hint="Which environment should I target?",
                    awaiting_user_feedback=True,
                ),
                SessionTurn(
                    input="need input",
                    command="echo resumed",
                    output="ok",
                ),
            ]

    monkeypatch.setattr(cli, "create_shell_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(cli, "AgentLoop", FakeLoop)

    def _config_without_prompt() -> AppConfig:
        config = _fake_config()
        config.completion_prompt_enabled = False
        return config

    monkeypatch.setattr(
        cli,
        "AppConfig",
        type("FakeConfig", (), {"from_env": staticmethod(_config_without_prompt)}),
    )

    answers = iter(["Use staging"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    assert cli.main() == 0

    out = capsys.readouterr().out
    assert "model paused and needs user input" in out
    assert "question: Which environment should I target?" in out
    assert "[2] $ echo resumed" in out


def test_completion_prompt_only_shows_after_run_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_argv = ["terminalai", "need input"]
    monkeypatch.setattr("sys.argv", fake_argv)

    class FakeAdapter:
        name = "fake"

    class FakeLoop:
        def __init__(self, **kwargs: object) -> None:
            self.request_user_feedback = kwargs["request_user_feedback"]

        def run(self, _goal: str) -> list[SessionTurn]:
            self.request_user_feedback("Which environment should I target?")
            return [
                SessionTurn(
                    input="need input",
                    command="",
                    output="",
                    next_action_hint="Which environment should I target?",
                    awaiting_user_feedback=True,
                ),
                SessionTurn(input="need input", command="echo resumed", output="ok"),
            ]

    monkeypatch.setattr(cli, "create_shell_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(cli, "AgentLoop", FakeLoop)
    monkeypatch.setattr(
        cli,
        "AppConfig",
        type("FakeConfig", (), {"from_env": staticmethod(_fake_config)}),
    )

    prompts: list[str] = []
    answers = iter(["Use staging", "n"])

    def _fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", _fake_input)

    assert cli.main() == 0
    assert prompts[0] == "Your response: "
    assert prompts[1].startswith("Task finished.")


def test_confirm_command_execution_prompts_and_accepts_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    assert cli._confirm_command_execution("rm -rf ./tmp") is True


def test_main_prompts_for_continuation_and_accepts_new_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_argv = ["terminalai", "first task"]
    monkeypatch.setattr("sys.argv", fake_argv)

    class FakeAdapter:
        name = "fake"

    goals: list[str] = []

    class FakeLoop:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, goal: str) -> list[SessionTurn]:
            goals.append(goal)
            return [SessionTurn(input=goal, command="echo done", output="ok")]

    monkeypatch.setattr(cli, "create_shell_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(cli, "AgentLoop", FakeLoop)
    monkeypatch.setattr(
        cli,
        "AppConfig",
        type("FakeConfig", (), {"from_env": staticmethod(_fake_config)}),
    )

    answers = iter(["y", "second task", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    assert cli.main() == 0
    assert goals == ["first task", "second task"]


def test_main_skips_continuation_prompt_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_argv = ["terminalai", "first task"]
    monkeypatch.setattr("sys.argv", fake_argv)

    class FakeAdapter:
        name = "fake"

    goals: list[str] = []

    class FakeLoop:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, goal: str) -> list[SessionTurn]:
            goals.append(goal)
            return [SessionTurn(input=goal, command="echo done", output="ok")]

    def _config_without_prompt() -> AppConfig:
        config = _fake_config()
        config.completion_prompt_enabled = False
        return config

    monkeypatch.setattr(cli, "create_shell_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(cli, "AgentLoop", FakeLoop)
    monkeypatch.setattr(
        cli,
        "AppConfig",
        type("FakeConfig", (), {"from_env": staticmethod(_config_without_prompt)}),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt="": (_ for _ in ()).throw(AssertionError("input should not be called")),
    )

    assert cli.main() == 0
    assert goals == ["first task"]
