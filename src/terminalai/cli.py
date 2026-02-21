"""Command-line interface for terminalai."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

from .agent.loop import CONTINUATION_PROMPT_TEXT, AgentLoop
from .agent.models import SessionTurn
from .config import AppConfig
from .llm.client import LLMClient
from .shell import create_shell_adapter

LOGGER = logging.getLogger(__name__)


class CLIArgs(argparse.Namespace):
    goal: str | None
    working_directory: str | None


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

    LOGGER.info(
        "terminalai_session_start",
        extra={
            "model": config.model,
            "api_url": config.api_url,
            "shell": adapter.name,
            "working_directory": working_directory,
            "max_steps": config.max_steps,
            "safety_mode": config.safety_mode,
        },
    )

    client = LLMClient(
        api_key=config.api_key,
        model=config.model,
        system_prompt=config.system_prompt,
        max_context_chars=config.max_context_chars,
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
        continuation_prompt_enabled=config.continuation_prompt_enabled,
        auto_progress_turns=config.auto_progress_turns,
        safety_mode=config.safety_mode,
        confirm_command_execution=_confirm_command_execution,
        request_user_feedback=_request_user_feedback,
        request_turn_progress=_request_turn_progress,
    )

    session_turns: list[SessionTurn] = []
    current_goal = goal
    turn_offset = 0

    while True:
        turns = loop.run(current_goal, prior_turns=session_turns)
        if not turns and not session_turns:
            print("Session ended without command execution.")
            return 0

        for idx, turn in enumerate(turns, start=turn_offset + 1):
            if config.readable_cli_output:
                print(_render_turn(turn, idx))
                continue
            _print_turn_legacy(turn, idx)

        if not turns:
            break

        session_turns.extend(turns)
        turn_offset = len(session_turns)
        final_turn = turns[-1]
        if not _should_prompt_for_continuation(final_turn):
            break

        should_continue = _request_continuation()
        if not should_continue:
            break

        current_goal = _request_new_instruction()
        if not current_goal:
            print("No additional instruction provided. Ending session.")
            break

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


def _should_prompt_for_continuation(turn: SessionTurn) -> bool:
    return turn.overarching_goal_complete and turn.continuation_prompt_added


def _request_continuation() -> bool:
    response = input(f"{CONTINUATION_PROMPT_TEXT} [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def _request_new_instruction() -> str:
    return input("New instruction: ").strip()


def _display_hint(turn: SessionTurn) -> str | None:
    hint = turn.next_action_hint
    if not hint:
        return None
    if not turn.continuation_prompt_added:
        return hint

    sanitized_hint = hint.replace(CONTINUATION_PROMPT_TEXT, "").strip()
    return sanitized_hint or None

def _confirm_command_execution(command: str) -> bool:
    print("\n=== DESTRUCTIVE COMMAND CONFIRMATION ===")
    print(f"Command: {command}")
    print("=======================================")
    choice = input("Run this destructive command and continue? [y/N]: ").strip().lower()
    return choice in {"y", "yes"}


def _request_user_feedback(question: str) -> str:
    print("\n=== MODEL PAUSED: INPUT REQUIRED ===")
    print(f"Question: {question}")
    return input("Your response: ")


def _render_status(*, awaiting_user_feedback: bool, run_terminal: bool) -> str:
    if awaiting_user_feedback:
        return "needs input"
    if run_terminal:
        return "completed"
    return "running"


def _render_turn(turn: SessionTurn, idx: int) -> str:
    status = _render_status(
        awaiting_user_feedback=turn.awaiting_user_feedback,
        run_terminal=turn.overarching_goal_complete,
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
    else:
        display_hint = _display_hint(turn)
        if not display_hint:
            return "\n".join(lines)
        lines.append("[hint]")
        lines.append(display_hint)

    return "\n".join(lines)


def _print_turn_legacy(turn: SessionTurn, idx: int) -> None:
    if turn.awaiting_user_feedback:
        print(f"[{idx}] model paused and needs user input")
        if turn.next_action_hint:
            print(f"question: {turn.next_action_hint}")
        return

    print(f"[{idx}] $ {turn.command}")
    print(turn.output)
    display_hint = _display_hint(turn)
    if display_hint:
        print(f"hint: {display_hint}")


if __name__ == "__main__":
    raise SystemExit(main())
