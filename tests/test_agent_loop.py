from __future__ import annotations

import json
from dataclasses import dataclass

from terminalai.agent.loop import CONTINUATION_PROMPT_TEXT, AgentLoop
from terminalai.agent.models import SessionTurn
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
            step_budget = [event for event in session_context if event.get("type") == "step_budget"]
            assert len(step_budget) == 1
            assert step_budget[0]["current_step"] == 1
            assert step_budget[0]["max_steps"] >= 1
            assert step_budget[0]["steps_remaining"] == step_budget[0]["max_steps"] - 1
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

    assert len(turns) == 2
    assert shell.working_directories == ["/tmp/workspace"]
    assert turns[0].command == "echo hi"
    assert turns[1].command == ""
    log_files = list(tmp_path.glob("session-*.log"))
    assert len(log_files) == 1
    lines = log_files[0].read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[0])
    assert payload["log_version"] == 2
    assert payload["goal"] == "list files"
    assert payload["model"] is None
    assert payload["shell"] == "fake"
    assert payload["working_directory"] == "/tmp/workspace"
    assert payload["step_index"] == 1
    assert payload["command"] == "echo hi"
    assert payload["returncode"] == 0
    assert payload["duration"] == 0.1
    assert payload["awaiting_user_feedback"] is False
    assert payload["complete_signal"] is False


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

    log_files = list(tmp_path.glob("session-*.log"))
    payload = json.loads(log_files[0].read_text(encoding="utf-8").strip())
    assert payload["awaiting_user_feedback"] is True
    assert payload["returncode"] is None
    assert payload["duration"] is None
    assert payload["complete_signal"] is False


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
    assert [turn.awaiting_user_feedback for turn in turns] == [True, False, False]
    assert turns[1].command == "echo resumed"
    assert turns[2].command == ""
    assert shell.commands == ["echo resumed"]



class CompletionOnlyClient:
    def next_command(self, goal: str, session_context: list[dict[str, object]]) -> ModelDecision:
        assert goal == "summarize"
        assert session_context[-1]["type"] == "safety_policy"
        return ModelDecision(command=None, notes="all done", complete=True)


def test_agent_loop_logs_explicit_completion_only_turn(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(client=CompletionOnlyClient(), shell=shell, log_dir=tmp_path, max_steps=1)

    turns = loop.run("summarize")

    assert shell.commands == []
    assert len(turns) == 1
    assert turns[0].command == ""
    assert turns[0].turn_complete is True
    assert turns[0].next_action_hint is not None
    assert "all done" in turns[0].next_action_hint

    payload = json.loads(list(tmp_path.glob("session-*.log"))[0].read_text(encoding="utf-8"))
    assert payload["command"] == ""
    assert payload["complete_signal"] is True
    assert payload["turn_complete"] is True


def test_agent_loop_completion_log_metadata_matches_terminal_turn(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(client=FakeClient(), shell=shell, log_dir=tmp_path, max_steps=3)

    turns = loop.run("list files")

    payloads = [
        json.loads(line)
        for line in list(tmp_path.glob("session-*.log"))[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payloads[-1]["command"] == turns[-1].command == ""
    assert payloads[-1]["next_action_hint"] == "done"
    assert turns[-1].next_action_hint is not None
    assert turns[-1].next_action_hint.startswith(payloads[-1]["next_action_hint"])
    assert payloads[-1]["complete_signal"] is True
    assert payloads[-1]["overarching_goal_complete"] is True
    assert turns[-1].overarching_goal_complete is True
    assert CONTINUATION_PROMPT_TEXT in (turns[-1].next_action_hint or "")


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

    assert len(turns) == 2
    assert turns[-1].command == ""
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

    assert len(turns) == 3
    assert turns[0].command == ""
    assert turns[1].command == "echo follow-up"
    assert "Step budget exhausted" in turns[2].output
    assert shell.commands == ["echo follow-up"]

    log_files = list(tmp_path.glob("session-*.log"))
    payloads = [
        json.loads(line)
        for line in log_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payloads[0]["complete_signal"] is True
    assert payloads[1]["complete_signal"] is False
    assert payloads[2]["step_index"] == 2


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
        safety_mode="strict",
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
    assert command_declined_events[0]["safety_mode"] == "strict"


def test_agent_loop_confirms_and_runs_destructive_command_when_user_accepts(tmp_path) -> None:
    shell = FakeShell()

    loop = AgentLoop(
        client=DestructiveCommandClient(),
        shell=shell,
        log_dir=tmp_path,
        max_steps=1,
        safety_mode="strict",
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
        safety_mode="allow_unsafe",
    )

    turns = loop.run("delete temp files")

    assert shell.confirmed_calls == [True]
    assert "returncode=0" in turns[0].output




def test_agent_loop_does_not_auto_confirm_when_safety_mode_off(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(
        client=DestructiveCommandClient(),
        shell=shell,
        log_dir=tmp_path,
        max_steps=1,
        safety_mode="off",
    )

    turns = loop.run("delete temp files")

    assert shell.confirmed_calls == [False]
    assert "returncode=126" in turns[0].output

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

    assert len(turns) == 2
    blocked_events = [
        event for event in client.contexts[1] if event.get("type") == "command_blocked"
    ]
    assert len(blocked_events) == 1
    assert blocked_events[0]["command"] == "echo restricted"
    assert blocked_events[0]["reason"] == "denylist"
    assert blocked_events[0]["safety_mode"] == "strict"


def test_agent_loop_emits_executed_command_context_event(tmp_path) -> None:
    shell = FakeShell()
    client = FakeClient()
    loop = AgentLoop(client=client, shell=shell, log_dir=tmp_path, max_steps=2)

    turns = loop.run("list files")

    assert len(turns) == 2
    executed_events = [
        event for event in client.contexts[1] if event.get("type") == "command_executed"
    ]
    assert len(executed_events) == 1
    assert executed_events[0]["command"] == "echo hi"
    assert executed_events[0]["returncode"] == 0
    assert executed_events[0]["safety_mode"] == "strict"


def test_agent_loop_updates_step_budget_context_each_iteration(tmp_path) -> None:
    shell = FakeShell()
    client = FakeClient()
    loop = AgentLoop(client=client, shell=shell, log_dir=tmp_path, max_steps=3)

    loop.run("list files")

    first_budget_events = [
        event for event in client.contexts[0] if event.get("type") == "step_budget"
    ]
    second_budget_events = [
        event for event in client.contexts[1] if event.get("type") == "step_budget"
    ]

    assert first_budget_events[0]["current_step"] == 1
    assert first_budget_events[0]["steps_remaining"] == 2
    assert second_budget_events[0]["current_step"] == 2
    assert second_budget_events[0]["steps_remaining"] == 1


class NeverCompleteClient:
    def __init__(self) -> None:
        self.calls = 0

    def next_command(self, goal: str, session_context: list[dict[str, object]]) -> ModelDecision:
        self.calls += 1
        return ModelDecision(command="echo still-working", notes="continue", complete=False)


def test_agent_loop_notifies_when_step_budget_is_exhausted(tmp_path) -> None:
    shell = FakeShell()
    client = NeverCompleteClient()
    loop = AgentLoop(client=client, shell=shell, log_dir=tmp_path, max_steps=1)

    turns = loop.run("keep working")

    assert client.calls == 1
    assert shell.commands == ["echo still-working"]
    assert len(turns) == 2
    assert turns[-1].command == ""
    assert "Step budget exhausted" in turns[-1].output
    assert turns[-1].next_action_hint == (
        "I reached the maximum number of steps and could not finish. "
        "Please continue in a new run or raise max_steps."
    )


def test_agent_loop_appends_continuation_prompt_once_on_goal_completion(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(client=FakeClient(), shell=shell, log_dir=tmp_path, max_steps=3)

    turns = loop.run("list files")

    assert len(turns) == 2
    assert turns[-1].command == ""
    assert turns[-1].overarching_goal_complete is True
    assert turns[-1].next_action_hint is not None
    assert turns[-1].next_action_hint.count(CONTINUATION_PROMPT_TEXT) == 1
    assert turns[0].next_action_hint == "continue"


def test_agent_loop_does_not_append_continuation_prompt_for_non_final_turns(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(client=NeverCompleteClient(), shell=shell, log_dir=tmp_path, max_steps=1)

    turns = loop.run("keep working")

    assert len(turns) == 2
    assert all(not turn.overarching_goal_complete for turn in turns)
    for turn in turns:
        assert (
            turn.next_action_hint is None
            or CONTINUATION_PROMPT_TEXT not in turn.next_action_hint
        )


def test_continuation_prompt_not_repeated_when_final_message_retried(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(client=FakeClient(), shell=shell, log_dir=tmp_path, max_steps=3)

    turns = loop.run("list files")
    loop._append_continuation_prompt(turns)

    assert turns[-1].next_action_hint is not None
    assert turns[-1].next_action_hint.count(CONTINUATION_PROMPT_TEXT) == 1


def test_agent_loop_can_disable_continuation_prompt(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(
        client=FakeClient(),
        shell=shell,
        log_dir=tmp_path,
        max_steps=3,
        continuation_prompt_enabled=False,
    )

    turns = loop.run("list files")

    assert turns[0].next_action_hint == "continue"
    assert turns[-1].next_action_hint == "done"
    assert turns[0].overarching_goal_complete is False
    assert turns[-1].overarching_goal_complete is True


def test_agent_loop_pauses_between_turns_when_auto_progress_disabled(tmp_path) -> None:
    shell = FakeShell()
    client = FakeClient()
    prompts: list[int] = []

    def request_turn_progress(step_number: int) -> tuple[bool, str | None]:
        prompts.append(step_number)
        if step_number == 1:
            return True, "Only inspect top-level files first"
        return True, None

    loop = AgentLoop(
        client=client,
        shell=shell,
        log_dir=tmp_path,
        max_steps=2,
        auto_progress_turns=False,
        request_turn_progress=request_turn_progress,
    )

    turns = loop.run("list files")

    assert len(turns) == 2
    assert prompts == [1, 2]
    turn_instruction_events = [
        event for event in client.contexts[0] if event.get("type") == "user_turn_instruction"
    ]
    assert len(turn_instruction_events) == 1
    assert turn_instruction_events[0]["instruction"] == "Only inspect top-level files first"


def test_agent_loop_stops_when_user_declines_to_progress(tmp_path) -> None:
    shell = FakeShell()

    loop = AgentLoop(
        client=FakeClient(),
        shell=shell,
        log_dir=tmp_path,
        max_steps=2,
        auto_progress_turns=False,
        request_turn_progress=lambda _step: (False, None),
    )

    turns = loop.run("list files")

    assert turns == []
    assert shell.commands == []


class MutationThenCompletionClient:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[list[dict[str, object]]] = []

    def next_command(self, goal: str, session_context: list[dict[str, object]]) -> ModelDecision:
        self.calls += 1
        self.contexts.append(session_context)
        if self.calls == 1:
            return ModelDecision(
                command="echo mutate",
                notes="apply",
                complete=False,
                phase="mutation",
                expected_outcome="file updated",
                verification_command="pytest -q",
                risk_level="medium",
            )
        if self.calls == 2:
            return ModelDecision(
                command=None,
                notes="done",
                complete=True,
                phase="completion",
                expected_outcome="all complete",
            )
        if self.calls == 3:
            return ModelDecision(
                command="pytest -q",
                notes="verify",
                complete=False,
                phase="verification",
                expected_outcome="tests pass",
                verification_command="pytest -q",
                risk_level="low",
            )
        return ModelDecision(
            command=None,
            notes="done",
            complete=True,
            phase="completion",
            expected_outcome="all complete",
        )


def test_agent_loop_requires_verification_after_mutation_before_completion(tmp_path) -> None:
    shell = FakeShell()
    client = MutationThenCompletionClient()
    loop = AgentLoop(client=client, shell=shell, log_dir=tmp_path, max_steps=4)

    turns = loop.run("update files")

    assert shell.commands == ["echo mutate", "pytest -q"]
    assert turns[1].command == ""
    assert turns[1].phase == "completion"
    assert "Run a verification phase" in (turns[1].next_action_hint or "")
    blocked_events = [
        event for event in client.contexts[2] if event.get("type") == "phase_transition_blocked"
    ]
    assert len(blocked_events) == 1
    assert blocked_events[0]["required_phase"] == "verification"


class MutationVerificationCompletionClient:
    def __init__(self) -> None:
        self.calls = 0

    def next_command(self, goal: str, session_context: list[dict[str, object]]) -> ModelDecision:
        self.calls += 1
        if self.calls == 1:
            return ModelDecision(
                command="echo mutate",
                complete=False,
                phase="mutation",
                expected_outcome="changed",
            )
        if self.calls == 2:
            return ModelDecision(
                command="pytest -q",
                complete=False,
                phase="verification",
                expected_outcome="tests pass",
                verification_command="pytest -q",
            )
        return ModelDecision(
            command=None,
            complete=True,
            phase="completion",
            expected_outcome="done",
        )


def test_agent_loop_allows_completion_after_verification_phase(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(
        client=MutationVerificationCompletionClient(),
        shell=shell,
        log_dir=tmp_path,
        max_steps=4,
    )

    turns = loop.run("update files")

    assert shell.commands == ["echo mutate", "pytest -q"]
    assert len(turns) == 3
    assert turns[-1].command == ""
    assert turns[-1].overarching_goal_complete is True


def test_agent_loop_logs_phase_metadata(tmp_path) -> None:
    shell = FakeShell()
    loop = AgentLoop(client=FakeClient(), shell=shell, log_dir=tmp_path, max_steps=2)

    loop.run("list files")

    log_file = list(tmp_path.glob("session-*.log"))[0]
    payload = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
    assert payload["phase"] == "analysis"
    assert payload["expected_outcome"] is None
    assert payload["verification_command"] is None
    assert payload["risk_level"] is None

def test_agent_loop_includes_prior_turns_in_session_context(tmp_path) -> None:
    shell = FakeShell()

    class PriorAwareClient:
        def next_command(
            self, goal: str, session_context: list[dict[str, object]]
        ) -> ModelDecision:
            assert goal == "second goal"
            commands = [
                entry.get("command")
                for entry in session_context
                if isinstance(entry, dict) and isinstance(entry.get("command"), str)
            ]
            assert "echo from first goal" in commands
            return ModelDecision(command=None, notes="done", complete=True)

    loop = AgentLoop(client=PriorAwareClient(), shell=shell, log_dir=tmp_path, max_steps=1)
    prior_turns = [
        SessionTurn(
            input="first goal",
            command="echo from first goal",
            output="ok",
            next_action_hint="done",
            turn_complete=True,
        )
    ]

    turns = loop.run("second goal", prior_turns=prior_turns)

    assert len(turns) == 1
    assert turns[0].turn_complete is True
