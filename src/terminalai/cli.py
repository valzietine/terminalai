"""Command-line interface for terminalai."""

from __future__ import annotations

import argparse

from .config import AppConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="terminalai", description="Terminal AI assistant")
    parser.add_argument("--shell", default="bash", help="Shell to use for command execution")
    parser.add_argument("--model", help="Override model name from environment config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan without performing any actions",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = AppConfig.from_env()

    selected_model = args.model or config.model
    print("terminalai execution path is not implemented yet.")
    print(f"shell={args.shell} model={selected_model} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
