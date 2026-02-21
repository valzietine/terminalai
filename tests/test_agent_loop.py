from __future__ import annotations

import json
from dataclasses import dataclass

from terminalai.agent.loop import AgentLoop
from terminalai.llm.client import ModelDecision


@dataclass
class FakeResult:
    stdout: str
    stderr: str
    returncode: int
    duration_seconds: float = 0.1
    blocked: bool = False
    block_reason: str | None = None


class FakeShell:
    name = "fake"

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.working_directories: list[str | None] = []
        self.confirmed_calls: list[bool] = []

    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        confirmed: bool = False,
    ) -> FakeResult:
        self.commands.append(command)
        self.working_directories.append(cwd)
        self.confirmed_calls.append(confirmed)
        if self.is_destructive_command(command) and not confirmed:
            return FakeResult(
                stdout="",
                stderr="destructive command requires explicit confirmation",
                returncode=126,
            )
        return FakeResult(stdout="ok", stderr="", returncode=0)

    def is_destructive_command(self, command: str) -> bool:
        return command.startswith("rm -rf")


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[list[dict[str, object]]] = []

    def next_command(self, goal: str, session_context: list[dict[str, object]]) -> ModelDecision:
        self.calls += 1
        self.contexts.append(session_context)
        if self.calls == 1:
            assert goal == "list files"
            assert session_context[-1]["type"] == "safety_policy"
            return ModelDecision(command="echo hi", notes="continue", complete=False)
        return ModelDecision(command=None, notes="done", complete=True)


def test_agent_loop_executes_and_logs(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(
        client=FakeClient(),
        shell=shell,
        log_dir=tmp_path,
        max_steps=3,
        working_directory="/tmp/workspace",
    )

    turns = loop.run("list files")

    assert len(turns) == 1
    assert shell.working_directories == ["/tmp/workspace"]
    assert turns[0].command == "echo hi"
    log_files = list(tmp_path.glob("session-*.log"))
    assert len(log_files) == 1
    lines = log_files[0].read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[0])
    assert payload["command"] == "echo hi"


class FakeQuestionClient:
    def __init__(self) -> None:
        self.calls = 0

    def next_command(self, goal: str, session_context: list[dict[str, object]]) -> ModelDecision:
        self.calls += 1
        assert goal == "need input"
        if self.calls == 1:
            assert session_context[-1]["type"] == "safety_policy"
            return ModelDecision(
                command=None,
                notes=None,
                complete=False,
                ask_user=True,
                user_question="Which environment should I target?",
            )

        user_feedback_events = [
            event for event in session_context if event.get("type") == "user_feedback"
        ]
        assert user_feedback_events[-1]["goal"] == "need input"
        assert user_feedback_events[-1]["question"] == "Which environment should I target?"
        assert user_feedback_events[-1]["response"] == "Use staging"
        if self.calls == 2:
            return ModelDecision(command="echo resumed", notes="continue", complete=False)
        return ModelDecision(command=None, notes="done", complete=True)


def test_agent_loop_can_pause_for_user_feedback(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(client=FakeQuestionClient(), shell=shell, log_dir=tmp_path, max_steps=1)

    turns = loop.run("need input")

    assert shell.commands == []
    assert len(turns) == 1
    assert turns[0].awaiting_user_feedback is True
    assert turns[0].next_action_hint == "Which environment should I target?"


def test_agent_loop_resumes_when_feedback_is_collected(tmp_path) -> None:
    shell = FakeShell()
    prompts: list[str] = []

    def request_user_feedback(question: str) -> str:
        prompts.append(question)
        return "Use staging"

    loop = AgentLoop(
        client=FakeQuestionClient(),
        shell=shell,
        log_dir=tmp_path,
        max_steps=3,
        request_user_feedback=request_user_feedback,
    )

    turns = loop.run("need input")

    assert prompts == ["Which environment should I target?"]
    assert [turn.awaiting_user_feedback for turn in turns] == [True, False]
    assert turns[1].command == "echo resumed"
    assert shell.commands == ["echo resumed"]


def test_agent_loop_confirms_completion_before_ending(tmp_path) -> None:
    shell = FakeShell()
    prompts: list[str | None] = []

    def confirm_completion(notes: str | None) -> tuple[bool, str | None]:
        prompts.append(notes)
        return True, None

    loop = AgentLoop(
        client=FakeClient(),
        shell=shell,
        log_dir=tmp_path,
        confirm_before_complete=True,
        confirm_completion=confirm_completion,
    )

    turns = loop.run("list files")

    assert len(turns) == 1
    assert prompts == ["done"]


class FakeCompletionFeedbackClient:
    def __init__(self) -> None:
        self.calls = 0

    def next_command(self, goal: str, session_context: list[dict[str, object]]) -> ModelDecision:
        self.calls += 1
        if self.calls == 1:
            return ModelDecision(command=None, notes="done", complete=True)
        assert session_context[0]["next_action_hint"] == (
            "User declined completion: Please show what changed and explain it."
        )
        return ModelDecision(command="echo follow-up", notes=None, complete=False)


def test_agent_loop_resumes_after_user_declines_completion(tmp_path) -> None:
    shell = FakeShell()

    def confirm_completion(_notes: str | None) -> tuple[bool, str | None]:
        return False, "Please show what changed and explain it."

    loop = AgentLoop(
        client=FakeCompletionFeedbackClient(),
        shell=shell,
        log_dir=tmp_path,
        max_steps=2,
        confirm_before_complete=True,
        confirm_completion=confirm_completion,
    )

    turns = loop.run("list files")

    assert len(turns) == 2
    assert turns[0].command == ""
    assert turns[1].command == "echo follow-up"
    assert shell.commands == ["echo follow-up"]


class DestructiveCommandClient:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[list[dict[str, object]]] = []

    def next_command(self, goal: str, session_context: list[dict[str, object]]) -> ModelDecision:
        self.calls += 1
        self.contexts.append(session_context)
        assert goal == "delete temp files"
        if self.calls == 1:
            assert session_context[-1]["type"] == "safety_policy"
            return ModelDecision(command="rm -rf ./tmp", notes="cleanup", complete=False)

        return ModelDecision(command=None, notes="done", complete=True)


def test_agent_loop_prompts_and_skips_destructive_command_when_user_declines(tmp_path) -> None:
    shell = FakeShell()
    client = DestructiveCommandClient()
    prompts: list[str] = []

    def confirm_command_execution(command: str) -> bool:
        prompts.append(command)
        return False

    loop = AgentLoop(
        client=client,
        shell=shell,
        log_dir=tmp_path,
        max_steps=2,
        safety_enabled=True,
        allow_unsafe=False,
        confirm_command_execution=confirm_command_execution,
    )

    turns = loop.run("delete temp files")

    assert prompts == ["rm -rf ./tmp"]
    assert shell.commands == []
    assert "User declined destructive command execution" in turns[0].output
    command_declined_events = [
        event for event in client.contexts[1] if event.get("type") == "command_declined"
    ]
    assert len(command_declined_events) == 1
    assert command_declined_events[0]["command"] == "rm -rf ./tmp"
    assert command_declined_events[0]["reason"] == "user_declined"
    assert command_declined_events[0]["safety_enabled"] is True
    assert command_declined_events[0]["allow_unsafe"] is False


def test_agent_loop_confirms_and_runs_destructive_command_when_user_accepts(tmp_path) -> None:
    shell = FakeShell()

    loop = AgentLoop(
        client=DestructiveCommandClient(),
        shell=shell,
        log_dir=tmp_path,
        max_steps=1,
        safety_enabled=True,
        allow_unsafe=False,
        confirm_command_execution=lambda _command: True,
    )

    turns = loop.run("delete temp files")

    assert shell.confirmed_calls == [True]
    assert "returncode=0" in turns[0].output


def test_agent_loop_allows_destructive_commands_when_allow_unsafe_enabled(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(
        client=DestructiveCommandClient(),
        shell=shell,
        log_dir=tmp_path,
        max_steps=1,
        safety_enabled=True,
        allow_unsafe=True,
    )

    turns = loop.run("delete temp files")

    assert shell.confirmed_calls == [True]
    assert "returncode=0" in turns[0].output


class GuardrailBlockedClient:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[list[dict[str, object]]] = []

    def next_command(self, goal: str, session_context: list[dict[str, object]]) -> ModelDecision:
        self.calls += 1
        self.contexts.append(session_context)
        if self.calls == 1:
            return ModelDecision(command="echo restricted", notes="try", complete=False)
        return ModelDecision(command=None, notes="done", complete=True)


class GuardrailBlockedShell(FakeShell):
    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        confirmed: bool = False,
    ) -> FakeResult:
        self.commands.append(command)
        self.working_directories.append(cwd)
        self.confirmed_calls.append(confirmed)
        return FakeResult(
            stdout="",
            stderr="command blocked by denylist policy",
            returncode=126,
            blocked=True,
            block_reason="command blocked by denylist policy",
        )


def test_agent_loop_emits_blocked_command_context_event(tmp_path) -> None:
    shell = GuardrailBlockedShell()
    client = GuardrailBlockedClient()
    loop = AgentLoop(client=client, shell=shell, log_dir=tmp_path, max_steps=2)

    turns = loop.run("list files")

    assert len(turns) == 1
    blocked_events = [
        event for event in client.contexts[1] if event.get("type") == "command_blocked"
    ]
    assert len(blocked_events) == 1
    assert blocked_events[0]["command"] == "echo restricted"
    assert blocked_events[0]["reason"] == "denylist"
    assert blocked_events[0]["safety_enabled"] is True
    assert blocked_events[0]["allow_unsafe"] is False


def test_agent_loop_emits_executed_command_context_event(tmp_path) -> None:
    shell = FakeShell()
    client = FakeClient()
    loop = AgentLoop(client=client, shell=shell, log_dir=tmp_path, max_steps=2)

    turns = loop.run("list files")

    assert len(turns) == 1
    executed_events = [
        event for event in client.contexts[1] if event.get("type") == "command_executed"
    ]
    assert len(executed_events) == 1
    assert executed_events[0]["command"] == "echo hi"
    assert executed_events[0]["returncode"] == 0
    assert executed_events[0]["safety_enabled"] is True
    assert executed_events[0]["allow_unsafe"] is False
