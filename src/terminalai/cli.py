"""Command-line interface for terminalai."""

from __future__ import annotations

import argparse
import logging
import os
import platform
from pathlib import Path
from typing import cast

from .agent.loop import AgentLoop
from .agent.models import SessionTurn
from .config import AppConfig, SafetyMode
from .llm.client import LLMClient
from .shell import create_shell_adapter

LOGGER = logging.getLogger(__name__)


class CLIArgs(argparse.Namespace):
    goal: str | None
    working_directory: str | None


def build_runtime_context(
    shell_name: str,
    working_directory: str | None,
    *,
    safety_mode: SafetyMode,
) -> str:
    """Build startup orientation context for the model."""
    effective_cwd = working_directory or str(Path.cwd())
    return "\n".join(
        [
            "Runtime environment context:",
            f"- operating_system: {platform.system()} {platform.release()}",
            f"- platform: {platform.platform()}",
            f"- architecture: {platform.machine()}",
            f"- os_name: {os.name}",
            f"- shell: {shell_name}",
            f"- starting_working_directory: {effective_cwd}",
            f"- safety_mode: {safety_mode}",
            "Use this context to orient command choices to this machine.",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="terminalai", description="Terminal AI assistant")
    parser.add_argument(
        "--cwd",
        dest="working_directory",
        help=(
            "Override the starting working directory for command execution. "
            "Takes precedence over config/env cwd values."
        ),
    )
    parser.add_argument("goal", nargs="?", help="Goal for the model-driven terminal session")
    return parser


def main() -> int:
    parser = build_parser()
    args = cast(CLIArgs, parser.parse_args())
    config = AppConfig.from_env()
    adapter = create_shell_adapter(config.shell)

    goal = args.goal or input("Goal: ").strip()
    if not goal:
        print("No goal provided.")
        return 1

    configured_working_directory = (
        args.working_directory if args.working_directory is not None else config.working_directory
    )
    working_directory: str | None = None
    if configured_working_directory is not None:
        resolved_working_directory = Path(configured_working_directory).expanduser().resolve()
        if not resolved_working_directory.exists() or not resolved_working_directory.is_dir():
            print(f"Invalid configured cwd directory: {configured_working_directory}")
            return 1
        working_directory = str(resolved_working_directory)

    client = LLMClient(
        api_key=config.api_key,
        model=config.model,
        system_prompt=config.system_prompt,
        runtime_context=build_runtime_context(
            config.shell,
            working_directory,
            safety_mode=config.safety_mode,
        ),
        reasoning_effort=config.reasoning_effort,
        api_url=config.api_url,
        allow_user_feedback_pause=config.allow_user_feedback_pause,
    )
    loop = AgentLoop(
        client=client,
        shell=adapter,
        log_dir=config.log_dir,
        max_steps=config.max_steps,
        working_directory=working_directory,
        confirm_before_complete=config.confirm_before_complete,
        continuation_prompt_enabled=config.continuation_prompt_enabled,
        auto_progress_turns=config.auto_progress_turns,
        safety_mode=config.safety_mode,
        confirm_command_execution=_confirm_command_execution,
        confirm_completion=_confirm_completion,
        request_user_feedback=_request_user_feedback,
        request_turn_progress=_request_turn_progress,
    )

    current_goal = goal
    while True:
        turns = loop.run(current_goal)
        if not turns:
            print("Session ended without command execution.")
            return 0

        for idx, turn in enumerate(turns, start=1):
            if config.readable_cli_output:
                print(_render_turn(turn, idx))
                continue
            _print_turn_legacy(turn, idx)

        if not _should_offer_continuation(turns[-1], config.continuation_prompt_enabled):
            break

        if not _confirm_continuation():
            break

        next_goal = input("New instruction: ").strip()
        if not next_goal:
            print("No instruction provided. Ending session.")
            break
        current_goal = next_goal

    LOGGER.debug("shell_adapter_selected", extra={"shell": adapter.name})
    return 0


def _request_turn_progress(step_number: int) -> tuple[bool, str | None]:
    message = (
        f"Step {step_number}: press Enter to run the next model turn, "
        "type your instruction to guide this turn, or enter q to stop: "
    )
    response = input(message).strip()
    if response.lower() in {"q", "quit", "exit"}:
        return False, None
    return True, response or None


def _should_offer_continuation(final_turn: SessionTurn, continuation_prompt_enabled: bool) -> bool:
    return (
        continuation_prompt_enabled
        and final_turn.overarching_goal_complete
        and final_turn.continuation_prompt_added
    )


def _confirm_continuation() -> bool:
    choice = input("Continue with new instructions? [y/N]: ").strip().lower()
    return choice in {"y", "yes"}

def _confirm_command_execution(command: str) -> bool:
    print("\n=== DESTRUCTIVE COMMAND CONFIRMATION ===")
    print(f"Command: {command}")
    print("=======================================")
    choice = input("Run this destructive command and continue? [y/N]: ").strip().lower()
    return choice in {"y", "yes"}


def _confirm_completion(model_notes: str | None) -> tuple[bool, str | None]:
    if model_notes:
        print(f"model completion note: {model_notes}")

    choice = input("Model marked the task complete. End session? [Y/n]: ").strip().lower()
    if choice in {"", "y", "yes"}:
        return True, None

    follow_up = input(
        "What should happen next? (objective changes, debrief questions, etc.): "
    ).strip()
    return False, follow_up


def _request_user_feedback(question: str) -> str:
    print("\n=== MODEL PAUSED: INPUT REQUIRED ===")
    print(f"Question: {question}")
    return input("Your response: ")


def _render_status(*, awaiting_user_feedback: bool, turn_complete: bool) -> str:
    if awaiting_user_feedback:
        return "needs input"
    if turn_complete:
        return "completed"
    return "running"


def _render_turn(turn: SessionTurn, idx: int) -> str:
    status = _render_status(
        awaiting_user_feedback=turn.awaiting_user_feedback,
        turn_complete=turn.turn_complete,
    )
    lines = [f"=== Turn {idx} ({status}) ==="]

    if turn.command:
        lines.append("[command]")
        lines.append(turn.command)

    output = turn.output.rstrip()
    if output:
        lines.append("[output]")
        lines.append(output)

    if turn.awaiting_user_feedback:
        lines.append("[question]")
        lines.append(turn.next_action_hint or "(no question provided)")
    elif turn.next_action_hint:
        lines.append("[hint]")
        lines.append(turn.next_action_hint)

    return "\n".join(lines)


def _print_turn_legacy(turn: SessionTurn, idx: int) -> None:
    if turn.awaiting_user_feedback:
        print(f"[{idx}] model paused and needs user input")
        if turn.next_action_hint:
            print(f"question: {turn.next_action_hint}")
        return

    print(f"[{idx}] $ {turn.command}")
    print(turn.output)
    if turn.next_action_hint:
        print(f"hint: {turn.next_action_hint}")


if __name__ == "__main__":
    raise SystemExit(main())
