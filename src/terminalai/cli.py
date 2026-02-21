"""Command-line interface for terminalai."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .agent.loop import AgentLoop
from .config import AppConfig
from .llm.client import LLMClient
from .shell import create_shell_adapter

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="terminalai", description="Terminal AI assistant")
    parser.add_argument("goal", nargs="?", help="Goal for the model-driven terminal session")
    parser.add_argument(
        "--shell",
        choices=["cmd", "powershell"],
        default="powershell",
        help="Shell to use for command execution",
    )
    parser.add_argument("--model", help="Override model name from environment config")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum model-execution iterations",
    )
    parser.add_argument(
        "--cwd",
        help="Starting working directory for executed commands (defaults to current directory)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = AppConfig.from_env()
    adapter = create_shell_adapter(args.shell)

    goal = args.goal or input("Goal: ").strip()
    if not goal:
        print("No goal provided.")
        return 1

    working_directory = str(Path(args.cwd).expanduser().resolve()) if args.cwd else None
    if working_directory is not None:
        candidate = Path(working_directory)
        if not candidate.exists() or not candidate.is_dir():
            print(f"Invalid --cwd directory: {args.cwd}")
            return 1

    selected_model = args.model or config.model
    client = LLMClient(
        api_key=config.api_key,
        model=selected_model,
        system_prompt=config.system_prompt,
        reasoning_effort=config.reasoning_effort,
        api_url=config.api_url,
    )
    loop = AgentLoop(
        client=client,
        shell=adapter,
        log_dir=config.log_dir,
        max_steps=args.max_steps,
        working_directory=working_directory,
    )

    turns = loop.run(goal)
    if not turns:
        print("Session ended without command execution.")
        return 0

    for idx, turn in enumerate(turns, start=1):
        print(f"[{idx}] $ {turn.command}")
        print(turn.output)
        if turn.next_action_hint:
            print(f"hint: {turn.next_action_hint}")

    LOGGER.debug("shell_adapter_selected", extra={"shell": adapter.name})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
