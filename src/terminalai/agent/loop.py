"""Minimal orchestration loop for direct terminal execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from terminalai.agent.models import AgentStep, ExecutionResult, SessionTurn
from terminalai.config import SafetyMode
from terminalai.llm.client import LLMClient
from terminalai.shell import ShellAdapter

ContextEventField = str | int | bool
ContextEvent = dict[str, ContextEventField]
ConfirmCommandExecution = Callable[[str], bool]
ConfirmCompletion = Callable[[str | None], tuple[bool, str | None]]
RequestUserFeedback = Callable[[str], str]
RequestTurnProgress = Callable[[int], tuple[bool, str | None]]

CONTINUATION_PROMPT_TEXT = (
    "Would you like to keep going with new instructions while we retain this context?"
)


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
        safety_mode: SafetyMode = "strict",
        confirm_command_execution: ConfirmCommandExecution | None = None,
        confirm_completion: ConfirmCompletion | None = None,
        request_user_feedback: RequestUserFeedback | None = None,
        continuation_prompt_enabled: bool = True,
        auto_progress_turns: bool = True,
        request_turn_progress: RequestTurnProgress | None = None,
    ) -> None:
        self.client = client
        self.shell = shell
        self.log_dir = Path(log_dir)
        self.max_steps = max_steps
        self.working_directory = working_directory
        self.confirm_before_complete = confirm_before_complete
        self.safety_mode = safety_mode
        self.confirm_command_execution = confirm_command_execution
        self.confirm_completion = confirm_completion
        self.request_user_feedback = request_user_feedback
        self.continuation_prompt_enabled = continuation_prompt_enabled
        self.auto_progress_turns = auto_progress_turns
        self.request_turn_progress = request_turn_progress
        self._session_turn_history: list[SessionTurn] = []

    def run(self, goal: str) -> list[SessionTurn]:
        turns: list[SessionTurn] = []
        context_events: list[ContextEvent] = []
        safety_policy_context = self._safety_policy_context()

        exhausted_step_budget = True
        for step_index in range(self.max_steps):
            if not self.auto_progress_turns and self.request_turn_progress:
                should_continue, instruction = self.request_turn_progress(step_index + 1)
                if isinstance(instruction, str) and instruction.strip():
                    self._append_context_event(
                        context_events,
                        event_type="user_turn_instruction",
                        goal=goal,
                        step=step_index + 1,
                        instruction=instruction.strip(),
                    )
                if not should_continue:
                    exhausted_step_budget = False
                    break

            step_context = self._step_budget_context(step_index)
            context = (
                [self._serialize_turn(turn) for turn in self._session_turn_history]
                + [self._serialize_turn(turn) for turn in turns]
                + [dict(event) for event in context_events]
                + [step_context, safety_policy_context]
            )
            decision = self.client.next_command(goal=goal, session_context=context)
            if decision.ask_user and decision.user_question:
                turn = SessionTurn(
                    input=goal,
                    command="",
                    output="",
                    next_action_hint=decision.user_question,
                    awaiting_user_feedback=True,
                    turn_complete=True,
                )
                turns.append(turn)
                self._append_log(
                    turn,
                    goal=goal,
                    step_index=step_index + 1,
                    complete_signal=decision.complete,
                )
                if not self.request_user_feedback:
                    exhausted_step_budget = False
                    break

                feedback = self.request_user_feedback(decision.user_question).strip()
                if not feedback:
                    exhausted_step_budget = False
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
                    should_end, completion_feedback = self.confirm_completion(decision.notes)
                    if not should_end:
                        feedback_text = (
                            completion_feedback.strip()
                            if isinstance(completion_feedback, str)
                            and completion_feedback.strip()
                            else "User asked to continue instead of ending."
                        )
                        turn = SessionTurn(
                            input=goal,
                            command="",
                            output="",
                            next_action_hint=f"User declined completion: {feedback_text}",
                            turn_complete=True,
                            subtask_complete=True,
                        )
                        turns.append(turn)
                        self._append_log(
                            turn,
                            goal=goal,
                            step_index=step_index + 1,
                            complete_signal=decision.complete,
                        )
                        continue
                completion_hint = decision.notes
                if not completion_hint:
                    completion_hint = (
                        "Model marked the task complete."
                        if decision.complete
                        else "Model did not provide a command; ending this run."
                    )
                turn = SessionTurn(
                    input=goal,
                    command="",
                    output="",
                    next_action_hint=completion_hint,
                    turn_complete=True,
                    overarching_goal_complete=bool(decision.complete),
                )
                if decision.complete:
                    self._append_continuation_prompt([turn])
                turns.append(turn)
                self._append_log(
                    turn,
                    goal=goal,
                    step_index=step_index + 1,
                    complete_signal=decision.complete,
                )
                exhausted_step_budget = False
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
                    safety_mode=self.safety_mode,
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
                    turn_complete=True,
                )
                turns.append(turn)
                self._append_log(
                    turn,
                    goal=goal,
                    step_index=step_index + 1,
                    returncode=130,
                    duration=0.0,
                    complete_signal=decision.complete,
                )
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
                    safety_mode=self.safety_mode,
                    returncode=command_result.returncode,
                )
            else:
                self._append_context_event(
                    context_events,
                    event_type="command_executed",
                    goal=goal,
                    command=step.proposed_command,
                    safety_mode=self.safety_mode,
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
                turn_complete=True,
            )
            turns.append(turn)
            self._append_log(
                turn,
                goal=goal,
                step_index=step_index + 1,
                returncode=result.returncode,
                duration=result.duration,
                complete_signal=decision.complete,
            )

        if exhausted_step_budget and self.max_steps > 0:
            self._append_context_event(
                context_events,
                event_type="step_budget_exhausted",
                goal=goal,
                current_step=self.max_steps,
                max_steps=self.max_steps,
                steps_remaining=0,
            )
            turn = SessionTurn(
                input=goal,
                command="",
                output=(
                    "Step budget exhausted before the model marked the task complete. "
                    f"Reached step {self.max_steps}/{self.max_steps}."
                ),
                next_action_hint=(
                    "I reached the maximum number of steps and could not finish. "
                    "Please continue in a new run or raise max_steps."
                ),
                turn_complete=True,
            )
            turns.append(turn)
            self._append_log(
                turn,
                goal=goal,
                step_index=self.max_steps,
                complete_signal=False,
            )

        if turns and turns[-1].overarching_goal_complete:
            self._append_context_event(
                context_events,
                event_type="goal_completion",
                goal=goal,
                overarching_goal_complete=True,
            )

        self._session_turn_history.extend(turns)

        return turns

    @staticmethod
    def _serialize_turn(turn: SessionTurn) -> dict[str, object]:
        return {
            "input": turn.input,
            "command": turn.command,
            "output": turn.output,
            "next_action_hint": turn.next_action_hint,
            "awaiting_user_feedback": turn.awaiting_user_feedback,
            "turn_complete": turn.turn_complete,
            "subtask_complete": turn.subtask_complete,
            "overarching_goal_complete": turn.overarching_goal_complete,
        }

    @staticmethod
    def _append_context_event(
        context_events: list[ContextEvent],
        *,
        event_type: str,
        goal: str,
        timestamp: str | None = None,
        **fields: ContextEventField,
    ) -> None:
        context_events.append(
            {
                "type": event_type,
                "goal": goal,
                "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
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

    def _append_log(
        self,
        turn: SessionTurn,
        *,
        goal: str,
        step_index: int,
        returncode: int | None = None,
        duration: float | None = None,
        complete_signal: bool = False,
    ) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        day_file = self.log_dir / f"session-{datetime.now(timezone.utc).date().isoformat()}.log"
        entry = {
            "log_version": 2,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "goal": goal,
            "model": getattr(self.client, "model", None),
            "shell": getattr(self.shell, "name", self.shell.__class__.__name__),
            "working_directory": self.working_directory,
            "step_index": step_index,
            "command": turn.command,
            "output": turn.output,
            "next_action_hint": turn.next_action_hint,
            "returncode": returncode,
            "duration": duration,
            "awaiting_user_feedback": turn.awaiting_user_feedback,
            "turn_complete": turn.turn_complete,
            "subtask_complete": turn.subtask_complete,
            "overarching_goal_complete": turn.overarching_goal_complete,
            "continuation_prompt_added": turn.continuation_prompt_added,
            "complete_signal": complete_signal,
        }
        with day_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _safety_policy_context(self) -> dict[str, object]:
        mode = self.safety_mode
        return {
            "type": "safety_policy",
            "mode": mode,
            "instructions": (
                "Avoid destructive commands unless explicitly requested by the user."
                if mode == "strict"
                else (
                    "Destructive commands may run when required by the goal."
                    if mode == "allow_unsafe"
                    else "Safety policy disabled; shell guardrails still apply."
                )
            ),
        }

    def _resolve_command_confirmation(
        self, *, command: str, destructive: bool
    ) -> tuple[bool, bool]:
        if not destructive:
            return False, False
        if self.safety_mode == "allow_unsafe":
            return True, False
        if self.safety_mode == "off":
            return False, False
        if not self.confirm_command_execution:
            return False, False
        if self.confirm_command_execution(command):
            return True, False
        return False, True

    def _append_continuation_prompt(self, turns: list[SessionTurn]) -> None:
        if not self.continuation_prompt_enabled:
            return
        if not turns:
            return

        final_turn = turns[-1]
        if final_turn.continuation_prompt_added:
            return

        final_turn.overarching_goal_complete = True
        existing_hint = final_turn.next_action_hint or ""
        if CONTINUATION_PROMPT_TEXT not in existing_hint:
            final_turn.next_action_hint = (
                f"{existing_hint}\n\n{CONTINUATION_PROMPT_TEXT}"
                if existing_hint
                else CONTINUATION_PROMPT_TEXT
            )
        final_turn.continuation_prompt_added = True

    def _step_budget_context(self, step_index: int) -> dict[str, object]:
        current_step = step_index + 1
        return {
            "type": "step_budget",
            "current_step": current_step,
            "max_steps": self.max_steps,
            "steps_remaining": self.max_steps - current_step,
            "instructions": (
                "Plan execution with the remaining step budget in mind and avoid unnecessary"
                " commands."
            ),
        }
