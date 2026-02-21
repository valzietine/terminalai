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


class FakeShell:
    name = "fake"

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.working_directories: list[str | None] = []

    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        confirmed: bool = False,
    ) -> FakeResult:  # noqa: ARG002
        self.commands.append(command)
        self.working_directories.append(cwd)
        return FakeResult(stdout="ok", stderr="", returncode=0)


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def next_command(self, goal: str, session_context: list[dict[str, object]]) -> ModelDecision:
        self.calls += 1
        if self.calls == 1:
            assert goal == "list files"
            assert session_context == []
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
    def next_command(self, goal: str, session_context: list[dict[str, object]]) -> ModelDecision:
        assert goal == "need input"
        assert session_context == []
        return ModelDecision(
            command=None,
            notes=None,
            complete=False,
            ask_user=True,
            user_question="Which environment should I target?",
        )


def test_agent_loop_can_pause_for_user_feedback(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(client=FakeQuestionClient(), shell=shell, log_dir=tmp_path)

    turns = loop.run("need input")

    assert shell.commands == []
    assert len(turns) == 1
    assert turns[0].awaiting_user_feedback is True
    assert turns[0].next_action_hint == "Which environment should I target?"
