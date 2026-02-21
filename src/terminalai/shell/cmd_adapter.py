"""Windows Command Prompt adapter."""

from __future__ import annotations

import locale
import subprocess

from .base import CommandResult, ShellAdapter


class CmdAdapter(ShellAdapter):
    """Adapter for command execution via ``cmd.exe``."""

    def __init__(self, executable: str = "cmd.exe", **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.executable = executable

    @property
    def name(self) -> str:
        return "cmd"

    def execute(
        self,
        command: str,
        *,
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
            )
            self.log_result(result)
            return result

        started = self.monotonic_now()
        try:
            process = subprocess.run(
                [self.executable, "/d", "/s", "/c", command],
                capture_output=True,
                timeout=timeout,
                check=False,
                text=False,
            )
            result = CommandResult(
                command=command,
                shell=self.name,
                returncode=process.returncode,
                stdout=_normalize_output(process.stdout),
                stderr=_normalize_output(process.stderr),
                duration_seconds=self.monotonic_now() - started,
            )
        except subprocess.TimeoutExpired as exc:
            result = CommandResult(
                command=command,
                shell=self.name,
                returncode=124,
                stdout=_normalize_output(exc.stdout),
                stderr=_normalize_output(exc.stderr),
                timed_out=True,
                duration_seconds=self.monotonic_now() - started,
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
