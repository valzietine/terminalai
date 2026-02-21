"""Base shell adapter primitives with safety guardrails."""

from __future__ import annotations

import abc
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

PolicyHook = Callable[[str, str], bool]

_DESTRUCTIVE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\brm\s+-rf\b",
        r"\bdel\s+(/s\s+)?(/q\s+)?",
        r"\bformat\b",
        r"\bremove-item\b",
        r"\bdrop\s+table\b",
    )
]

_SECRET_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(--?(?:password|token|secret|api[-_]?key)\s+)([^\s]+)",
        r"((?:password|token|secret|api[-_]?key)\s*=\s*)([^\s]+)",
    )
]


@dataclass(slots=True)
class CommandResult:
    """Result of a command execution."""

    command: str
    shell: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_seconds: float = 0.0
    executed: bool = True
    blocked: bool = False
    block_reason: str | None = None


class ShellAdapter(abc.ABC):
    """Abstract adapter for shell-specific command execution."""

    def __init__(
        self,
        *,
        allowlist_hook: PolicyHook | None = None,
        denylist_hook: PolicyHook | None = None,
        confirmation_mode: bool = True,
    ) -> None:
        self.allowlist_hook = allowlist_hook
        self.denylist_hook = denylist_hook
        self.confirmation_mode = confirmation_mode

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Friendly shell adapter name."""

    @abc.abstractmethod
    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        dry_run: bool = False,
        confirmed: bool = False,
    ) -> CommandResult:
        """Execute a shell command and return a normalized result."""

    def enforce_guardrails(
        self,
        command: str,
        *,
        dry_run: bool,
        confirmed: bool,
    ) -> str | None:
        """Run policy checks and return a block reason when rejected."""
        if self.denylist_hook and self.denylist_hook(command, self.name):
            return "command blocked by denylist policy"
        if self.allowlist_hook and not self.allowlist_hook(command, self.name):
            return "command rejected by allowlist policy"

        if self.confirmation_mode and self._is_destructive(command) and not (confirmed or dry_run):
            return "destructive command requires explicit confirmation"
        return None

    def _is_destructive(self, command: str) -> bool:
        return any(pattern.search(command) for pattern in _DESTRUCTIVE_PATTERNS)

    def is_destructive_command(self, command: str) -> bool:
        """Return true when a command matches destructive command heuristics."""
        return self._is_destructive(command)

    def log_request(self, command: str, *, timeout: float | None, dry_run: bool) -> None:
        LOGGER.info(
            "command_request",
            extra={
                "shell": self.name,
                "command": self._sanitize_command(command),
                "timeout": timeout,
                "dry_run": dry_run,
            },
        )

    def log_result(self, result: CommandResult) -> None:
        LOGGER.info(
            "command_result",
            extra={
                "shell": result.shell,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "duration_seconds": round(result.duration_seconds, 4),
                "executed": result.executed,
                "blocked": result.blocked,
                "stdout_length": len(result.stdout),
                "stderr_length": len(result.stderr),
            },
        )

    @staticmethod
    def monotonic_now() -> float:
        return time.monotonic()

    def _sanitize_command(self, command: str) -> str:
        sanitized = command
        for pattern in _SECRET_PATTERNS:
            sanitized = pattern.sub(r"\1***", sanitized)
        return sanitized
