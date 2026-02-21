"""Command-line interface for terminalai."""

from __future__ import annotations

import argparse
import logging
import os
import platform
from pathlib import Path

from .agent.loop import AgentLoop
from .config import AppConfig
from .llm.client import LLMClient
from .shell import create_shell_adapter

LOGGER = logging.getLogger(__name__)


def build_runtime_context(
    shell_name: str,
    working_directory: str | None,
    *,
    safety_enabled: bool,
    allow_unsafe: bool,
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
            f"- safety_enabled: {safety_enabled}",
            f"- allow_unsafe: {allow_unsafe}",
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
    args = parser.parse_args()
    config = AppConfig.from_env()
    adapter = create_shell_adapter(config.shell)

    goal = args.goal or input("Goal: ").strip()
    if not goal:
        print("No goal provided.")
        return 1

    configured_working_directory = args.working_directory or config.working_directory
    working_directory = (
        str(Path(configured_working_directory).expanduser().resolve())
        if configured_working_directory
        else None
    )
    if working_directory is not None:
        candidate = Path(working_directory)
        if not candidate.exists() or not candidate.is_dir():
            print(f"Invalid configured cwd directory: {configured_working_directory}")
            return 1

    client = LLMClient(
        api_key=config.api_key,
        model=config.model,
        system_prompt=config.system_prompt,
        runtime_context=build_runtime_context(
            config.shell,
            working_directory,
            safety_enabled=config.safety_enabled,
            allow_unsafe=config.allow_unsafe,
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
        safety_enabled=config.safety_enabled,
        allow_unsafe=config.allow_unsafe,
        confirm_command_execution=_confirm_command_execution,
        confirm_completion=_confirm_completion,
        request_user_feedback=_request_user_feedback,
    )

    turns = loop.run(goal)
    if not turns:
        print("Session ended without command execution.")
        return 0

    for idx, turn in enumerate(turns, start=1):
        if turn.awaiting_user_feedback:
            print(f"[{idx}] model paused and needs user input")
            if turn.next_action_hint:
                print(f"question: {turn.next_action_hint}")
            continue
        print(f"[{idx}] $ {turn.command}")
        print(turn.output)
        if turn.next_action_hint:
            print(f"hint: {turn.next_action_hint}")

    LOGGER.debug("shell_adapter_selected", extra={"shell": adapter.name})
    return 0



def _confirm_command_execution(command: str) -> bool:
    print("Destructive command proposed by model:")
    print(f"command: {command}")
    choice = input("Run this command and continue? [y/N]: ").strip().lower()
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
    print("model paused and needs user input")
    print(f"question: {question}")
    return input("Your response: ")


if __name__ == "__main__":
    raise SystemExit(main())
