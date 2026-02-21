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
        safety_mode="strict",
        api_url="https://api.openai.com/v1/responses",
        log_dir="logs",
        system_prompt="prompt",
        allow_user_feedback_pause=False,
        confirm_before_complete=False,
        continuation_prompt_enabled=True,
        auto_progress_turns=True,
        readable_cli_output=True,
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
    assert captured["safety_mode"] == "strict"
    assert captured["continuation_prompt_enabled"] is True
    assert captured["auto_progress_turns"] is True


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
        safety_mode="strict",
    )

    assert "Runtime environment context:" in context
    assert "shell: powershell" in context
    assert "starting_working_directory: /tmp/work" in context
    assert "safety_mode: strict" in context


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
    monkeypatch.setattr(
        cli,
        "AppConfig",
        type("FakeConfig", (), {"from_env": staticmethod(_fake_config)}),
    )

    answers = iter(["Use staging"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    assert cli.main() == 0

    out = capsys.readouterr().out
    assert "=== Turn 1 (needs input) ===" in out
    assert "[question]" in out
    assert "Which environment should I target?" in out
    assert "=== Turn 2 (running) ===" in out
    assert "[command]" in out
    assert "echo resumed" in out


def test_confirm_command_execution_prompts_and_accepts_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    assert cli._confirm_command_execution("rm -rf ./tmp") is True


def test_request_turn_progress_allows_quit_and_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["", "check only src/", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    assert cli._request_turn_progress(1) == (True, None)
    assert cli._request_turn_progress(2) == (True, "check only src/")
    assert cli._request_turn_progress(3) == (False, None)


def test_render_turn_trims_trailing_output_whitespace() -> None:
    rendered = cli._render_turn(
        SessionTurn(
            input="goal",
            command="echo hi",
            output="line 1\nline 2\n\n",
            next_action_hint="Done",
            turn_complete=True,
        ),
        3,
    )

    assert "=== Turn 3 (completed) ===" in rendered
    assert "[output]" in rendered
    assert "line 1\nline 2" in rendered
    assert not rendered.endswith("\n")


def test_main_uses_legacy_output_when_readable_cli_disabled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_argv = ["terminalai", "legacy output"]
    monkeypatch.setattr("sys.argv", fake_argv)

    class FakeAdapter:
        name = "fake"

    class FakeLoop:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _goal: str) -> list[SessionTurn]:
            return [
                SessionTurn(
                    input="legacy output",
                    command="echo resumed",
                    output="ok\n",
                    next_action_hint="next",
                )
            ]

    def fake_config() -> AppConfig:
        config = _fake_config()
        config.readable_cli_output = False
        return config

    monkeypatch.setattr(cli, "create_shell_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(cli, "AgentLoop", FakeLoop)
    monkeypatch.setattr(
        cli,
        "AppConfig",
        type("FakeConfig", (), {"from_env": staticmethod(fake_config)}),
    )

    assert cli.main() == 0

    out = capsys.readouterr().out
    assert "[1] $ echo resumed" in out
    assert "hint: next" in out
    assert "=== Turn" not in out


def test_should_offer_continuation_requires_completion_and_flag() -> None:
    turn = SessionTurn(
        input="goal",
        command="",
        output="",
        turn_complete=True,
        overarching_goal_complete=True,
        continuation_prompt_added=True,
    )

    assert cli._should_offer_continuation(turn, True) is True
    assert cli._should_offer_continuation(turn, False) is False


def test_main_accepts_continuation_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_argv = ["terminalai", "first goal"]
    monkeypatch.setattr("sys.argv", fake_argv)

    class FakeAdapter:
        name = "fake"

    goals: list[str] = []

    class FakeLoop:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, goal: str) -> list[SessionTurn]:
            goals.append(goal)
            if len(goals) == 1:
                return [
                    SessionTurn(
                        input=goal,
                        command="",
                        output="",
                        next_action_hint="done",
                        turn_complete=True,
                        overarching_goal_complete=True,
                        continuation_prompt_added=True,
                    )
                ]
            return [
                SessionTurn(
                    input=goal,
                    command="echo second",
                    output="ok",
                    turn_complete=True,
                )
            ]

    monkeypatch.setattr(cli, "create_shell_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(cli, "AgentLoop", FakeLoop)
    monkeypatch.setattr(
        cli,
        "AppConfig",
        type("FakeConfig", (), {"from_env": staticmethod(_fake_config)}),
    )

    answers = iter(["y", "second goal"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    assert cli.main() == 0
    assert goals == ["first goal", "second goal"]
