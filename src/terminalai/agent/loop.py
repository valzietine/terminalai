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
        confirm_completion: Callable[[str | None], tuple[bool, str | None]] | None = None,
        request_user_feedback: Callable[[str], str] | None = None,
    ) -> None:
        self.client = client
        self.shell = shell
        self.log_dir = Path(log_dir)
        self.max_steps = max_steps
        self.working_directory = working_directory
        self.confirm_before_complete = confirm_before_complete
        self.confirm_completion = confirm_completion
        self.request_user_feedback = request_user_feedback

    def run(self, goal: str) -> list[SessionTurn]:
        turns: list[SessionTurn] = []
        extra_context: list[dict[str, object]] = []

        for _ in range(self.max_steps):
            context = [asdict(turn) for turn in turns] + extra_context
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

                extra_context.append(
                    {
                        "type": "user_feedback",
                        "goal": goal,
                        "question": decision.user_question,
                        "response": feedback,
                    }
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
            command_result = self.shell.execute(
                step.proposed_command,
                cwd=self.working_directory,
                confirmed=True,
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
