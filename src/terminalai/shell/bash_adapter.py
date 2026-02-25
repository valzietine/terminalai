"""Bash shell adapter implementation."""

from __future__ import annotations

import locale
import os
import re
import shutil
import subprocess

from .base import CommandResult, PolicyHook, ShellAdapter

_ALREADY_ELEVATED_PATTERN = re.compile(r"^\s*(sudo|doas|su\b)", re.IGNORECASE)


class BashAdapter(ShellAdapter):
    """Adapter for command execution via ``bash``/``sh``."""

    def __init__(
        self,
        executable: str | None = None,
        *,
        allowlist_hook: PolicyHook | None = None,
        denylist_hook: PolicyHook | None = None,
        confirmation_mode: bool = True,
        fallback_to_sh: bool = True,
        elevate_process: bool = False,
    ) -> None:
        super().__init__(
            allowlist_hook=allowlist_hook,
            denylist_hook=denylist_hook,
            confirmation_mode=confirmation_mode,
        )
        self.executable = executable or _default_executable(fallback_to_sh=fallback_to_sh)
        self.elevate_process = elevate_process

    @property
    def name(self) -> str:
        return "bash"

    @property
    def elevation_enabled(self) -> bool:
        return self.elevate_process

    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        dry_run: bool = False,
        confirmed: bool = False,
    ) -> CommandResult:
        self.log_request(command, timeout=timeout, dry_run=dry_run)
        blocked_reason = self.enforce_guardrails(command, dry_run=dry_run, confirmed=confirmed)
        if blocked_reason:
            result = CommandResult(
                command=command,
                shell=self.name,
                returncode=126,
                stdout="",
                stderr=blocked_reason,
                executed=False,
                blocked=True,
                block_reason=blocked_reason,
                elevation_requested=self.elevate_process,
            )
            self.log_result(result)
            return result

        if dry_run:
            result = CommandResult(
                command=command,
                shell=self.name,
                returncode=0,
                stdout="dry-run: command not executed",
                stderr="",
                executed=False,
                elevation_requested=self.elevate_process,
            )
            self.log_result(result)
            return result

        started = self.monotonic_now()
        run_command = command
        elevated = False
        elevation_error: str | None = None
        if self.elevate_process:
            if os.name == "nt":
                elevation_error = (
                    "Elevation requested with bash on Windows; sudo strategy is unsupported. "
                    "Running without elevation."
                )
            elif _ALREADY_ELEVATED_PATTERN.match(command):
                pass
            elif shutil.which("sudo"):
                run_command = f"sudo -- {command}"
                elevated = True
            else:
                elevation_error = (
                    "Elevation requested but sudo is not available in PATH; "
                    "running without elevation."
                )

        try:
            process = subprocess.run(
                [self.executable, "-lc", run_command],
                capture_output=True,
                cwd=cwd,
                timeout=timeout,
                check=False,
                text=False,
            )
            stderr = _normalize_output(process.stderr)
            if elevation_error:
                stderr = f"{elevation_error}\n{stderr}" if stderr else elevation_error
            result = CommandResult(
                command=command,
                shell=self.name,
                returncode=process.returncode,
                stdout=_normalize_output(process.stdout),
                stderr=stderr,
                duration_seconds=self.monotonic_now() - started,
                elevated=elevated,
                elevation_requested=self.elevate_process,
                elevation_error=elevation_error,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = _normalize_output(exc.stderr)
            if elevation_error:
                stderr = f"{elevation_error}\n{stderr}" if stderr else elevation_error
            result = CommandResult(
                command=command,
                shell=self.name,
                returncode=124,
                stdout=_normalize_output(exc.stdout),
                stderr=stderr,
                timed_out=True,
                duration_seconds=self.monotonic_now() - started,
                elevated=elevated,
                elevation_requested=self.elevate_process,
                elevation_error=elevation_error,
            )

        self.log_result(result)
        return result


def _default_executable(*, fallback_to_sh: bool) -> str:
    if shutil.which("bash"):
        return "bash"
    if fallback_to_sh and shutil.which("sh"):
        return "sh"
    return "bash"


def _normalize_output(payload: bytes | str | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload

    for encoding in ("utf-8", "utf-8-sig", "utf-16", locale.getpreferredencoding(False), "cp1252"):
        try:
            return payload.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="replace")
