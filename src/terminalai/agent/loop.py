"""Minimal orchestration loop for direct terminal execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from terminalai.agent.models import AgentStep, ExecutionResult, SessionTurn
from terminalai.llm.client import LLMClient
from terminalai.shell import ShellAdapter


class AgentLoop:
    """Runs the command-propose/execute/report cycle until completion."""

    def __init__(
        self,
        *,
        client: LLMClient,
        shell: ShellAdapter,
        log_dir: str | Path,
        max_steps: int = 20,
        working_directory: str | None = None,
        confirm_before_complete: bool = False,
        safety_enabled: bool = True,
        allow_unsafe: bool = False,
        confirm_command_execution: Callable[[str], bool] | None = None,
        confirm_completion: Callable[[str | None], tuple[bool, str | None]] | None = None,
        request_user_feedback: Callable[[str], str] | None = None,
    ) -> None:
        self.client = client
        self.shell = shell
        self.log_dir = Path(log_dir)
        self.max_steps = max_steps
        self.working_directory = working_directory
        self.confirm_before_complete = confirm_before_complete
        self.safety_enabled = safety_enabled
        self.allow_unsafe = allow_unsafe
        self.confirm_command_execution = confirm_command_execution
        self.confirm_completion = confirm_completion
        self.request_user_feedback = request_user_feedback

    def run(self, goal: str) -> list[SessionTurn]:
        turns: list[SessionTurn] = []
        context_events: list[dict[str, object]] = []
        safety_policy_context = self._safety_policy_context()

        for _ in range(self.max_steps):
            context = [asdict(turn) for turn in turns] + context_events + [safety_policy_context]
            decision = self.client.next_command(goal=goal, session_context=context)
            if decision.ask_user and decision.user_question:
                turn = SessionTurn(
                    input=goal,
                    command="",
                    output="",
                    next_action_hint=decision.user_question,
                    awaiting_user_feedback=True,
                )
                turns.append(turn)
                self._append_log(turn)
                if not self.request_user_feedback:
                    break

                feedback = self.request_user_feedback(decision.user_question).strip()
                if not feedback:
                    break

                self._append_context_event(
                    context_events,
                    event_type="user_feedback",
                    goal=goal,
                    question=decision.user_question,
                    response=feedback,
                )
                continue
            if decision.complete or not decision.command:
                if self.confirm_before_complete and self.confirm_completion:
                    should_end, feedback = self.confirm_completion(decision.notes)
                    if not should_end:
                        feedback_text = (
                            feedback.strip()
                            if isinstance(feedback, str) and feedback.strip()
                            else "User asked to continue instead of ending."
                        )
                        turn = SessionTurn(
                            input=goal,
                            command="",
                            output="",
                            next_action_hint=f"User declined completion: {feedback_text}",
                        )
                        turns.append(turn)
                        self._append_log(turn)
                        continue
                break

            step = AgentStep(goal=goal, proposed_command=decision.command)
            destructive = self.shell.is_destructive_command(step.proposed_command)
            confirmed, declined = self._resolve_command_confirmation(
                command=step.proposed_command,
                destructive=destructive,
            )
            if declined:
                self._append_context_event(
                    context_events,
                    event_type="command_declined",
                    goal=goal,
                    command=step.proposed_command,
                    reason="user_declined",
                    safety_enabled=self.safety_enabled,
                    allow_unsafe=self.allow_unsafe,
                )
                turn = SessionTurn(
                    input=goal,
                    command=step.proposed_command,
                    output=(
                        "returncode=130\n"
                        "duration=0.0000s\n"
                        "stdout:\n\n"
                        "stderr:\n"
                        "User declined destructive command execution."
                    ),
                    next_action_hint=(
                        "User declined command execution; choose a safer alternative."
                    ),
                )
                turns.append(turn)
                self._append_log(turn)
                continue

            command_result = self.shell.execute(
                step.proposed_command,
                cwd=self.working_directory,
                confirmed=confirmed,
            )
            if command_result.blocked:
                self._append_context_event(
                    context_events,
                    event_type="command_blocked",
                    goal=goal,
                    command=step.proposed_command,
                    reason=self._normalize_block_reason(command_result),
                    safety_enabled=self.safety_enabled,
                    allow_unsafe=self.allow_unsafe,
                    returncode=command_result.returncode,
                )
            else:
                self._append_context_event(
                    context_events,
                    event_type="command_executed",
                    goal=goal,
                    command=step.proposed_command,
                    safety_enabled=self.safety_enabled,
                    allow_unsafe=self.allow_unsafe,
                    returncode=command_result.returncode,
                )
            result = ExecutionResult(
                stdout=command_result.stdout,
                stderr=command_result.stderr,
                returncode=command_result.returncode,
                duration=command_result.duration_seconds,
            )
            output = self._format_result(result)
            turn = SessionTurn(
                input=goal,
                command=step.proposed_command,
                output=output,
                next_action_hint=decision.notes,
            )
            turns.append(turn)
            self._append_log(turn)

        return turns

    @staticmethod
    def _append_context_event(
        context_events: list[dict[str, object]],
        *,
        event_type: str,
        goal: str,
        timestamp: str | None = None,
        **fields: object,
    ) -> None:
        context_events.append(
            {
                "type": event_type,
                "goal": goal,
                "timestamp": timestamp or datetime.now(UTC).isoformat(),
                **fields,
            }
        )

    @staticmethod
    def _normalize_block_reason(command_result: object) -> str:
        reason_source = ""
        if isinstance(getattr(command_result, "block_reason", None), str):
            reason_source = str(getattr(command_result, "block_reason")).strip().lower()
        if not reason_source and isinstance(getattr(command_result, "stderr", None), str):
            reason_source = str(getattr(command_result, "stderr")).strip().lower()

        if "denylist" in reason_source:
            return "denylist"
        if "allowlist" in reason_source:
            return "allowlist"
        if "confirmation" in reason_source or "requires explicit confirmation" in reason_source:
            return "requires_confirmation"
        return "unknown"

    @staticmethod
    def _format_result(result: ExecutionResult) -> str:
        return (
            f"returncode={result.returncode}\n"
            f"duration={result.duration:.4f}s\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def _append_log(self, turn: SessionTurn) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        day_file = self.log_dir / f"session-{datetime.now(UTC).date().isoformat()}.log"
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "command": turn.command,
            "output": turn.output,
            "next_action_hint": turn.next_action_hint,
        }
        with day_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _safety_policy_context(self) -> dict[str, object]:
        mode = "disabled"
        if self.safety_enabled and self.allow_unsafe:
            mode = "allow_unsafe"
        elif self.safety_enabled:
            mode = "strict"
        return {
            "type": "safety_policy",
            "safety_enabled": self.safety_enabled,
            "allow_unsafe": self.allow_unsafe,
            "mode": mode,
            "instructions": (
                "Avoid destructive commands unless explicitly requested by the user."
                if self.safety_enabled and not self.allow_unsafe
                else "Destructive commands may run when required by the goal."
            ),
        }

    def _resolve_command_confirmation(
        self, *, command: str, destructive: bool
    ) -> tuple[bool, bool]:
        if not destructive:
            return False, False
        if not self.safety_enabled:
            return True, False
        if self.allow_unsafe:
            return True, False
        if not self.confirm_command_execution:
            return False, False
        if self.confirm_command_execution(command):
            return True, False
        return False, True
