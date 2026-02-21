"""Minimal orchestration loop for direct terminal execution."""

from __future__ import annotations

import json
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
    ) -> None:
        self.client = client
        self.shell = shell
        self.log_dir = Path(log_dir)
        self.max_steps = max_steps
        self.working_directory = working_directory

    def run(self, goal: str) -> list[SessionTurn]:
        turns: list[SessionTurn] = []

        for _ in range(self.max_steps):
            context = [asdict(turn) for turn in turns]
            decision = self.client.next_command(goal=goal, session_context=context)
            if decision.complete or not decision.command:
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
