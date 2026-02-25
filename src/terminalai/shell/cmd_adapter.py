"""Windows Command Prompt adapter."""

from __future__ import annotations

import locale
import os
import subprocess

from .base import CommandResult, PolicyHook, ShellAdapter


class CmdAdapter(ShellAdapter):
    """Adapter for command execution via ``cmd.exe``."""

    def __init__(
        self,
        executable: str = "cmd.exe",
        *,
        allowlist_hook: PolicyHook | None = None,
        denylist_hook: PolicyHook | None = None,
        confirmation_mode: bool = True,
        elevate_process: bool = False,
    ) -> None:
        super().__init__(
            allowlist_hook=allowlist_hook,
            denylist_hook=denylist_hook,
            confirmation_mode=confirmation_mode,
        )
        self.executable = executable
        self.elevate_process = elevate_process

    @property
    def name(self) -> str:
        return "cmd"

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
        elevation_error: str | None = None
        run_args = [self.executable, "/d", "/s", "/c", command]
        elevated = False
        if self.elevate_process:
            if os.name != "nt":
                elevation_error = (
                    "Elevation requested but cmd elevation is only supported on Windows; "
                    "running without elevation."
                )
            else:
                escaped = command.replace("'", "''")
                run_args = [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    (
                        "Start-Process -FilePath 'cmd.exe' -Verb RunAs "
                        f"-ArgumentList '/d','/s','/c','{escaped}' -Wait"
                    ),
                ]
                elevated = True

        try:
            process = subprocess.run(
                run_args,
                capture_output=True,
                cwd=cwd,
                timeout=timeout,
                check=False,
                text=False,
            )
            stderr = _normalize_output(process.stderr)
            if elevation_error:
                stderr = f"{elevation_error}\n{stderr}" if stderr else elevation_error
            stdout = _normalize_output(process.stdout)
            if elevated:
                note = (
                    "Command was launched through a UAC elevation boundary; "
                    "child process output may be unavailable."
                )
                stdout = f"{stdout}\n{note}" if stdout else note
            result = CommandResult(
                command=command,
                shell=self.name,
                returncode=process.returncode,
                stdout=stdout,
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
        except FileNotFoundError:
            missing = (
                "powershell executable not found; cannot launch elevated command. "
                "Running without elevation requires cmd.exe and powershell on PATH."
                if self.elevate_process
                else f"{self.name} executable not found: {self.executable}"
            )
            result = CommandResult(
                command=command,
                shell=self.name,
                returncode=127,
                stdout="",
                stderr=missing,
                executed=False,
                duration_seconds=self.monotonic_now() - started,
                elevated=False,
                elevation_requested=self.elevate_process,
                elevation_error=missing if self.elevate_process else None,
            )

        self.log_result(result)
        return result


def _normalize_output(payload: bytes | str | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload

    for encoding in ("utf-8", "utf-8-sig", locale.getpreferredencoding(False), "cp1252"):
        try:
            return payload.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="replace")
