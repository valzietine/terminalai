"""Bash shell adapter implementation."""

from __future__ import annotations

import locale
import shutil
import subprocess

from .base import CommandResult, PolicyHook, ShellAdapter


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
    ) -> None:
        super().__init__(
            allowlist_hook=allowlist_hook,
            denylist_hook=denylist_hook,
            confirmation_mode=confirmation_mode,
        )
        self.executable = executable or _default_executable(fallback_to_sh=fallback_to_sh)

    @property
    def name(self) -> str:
        return "bash"

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
                [self.executable, "-lc", command],
                capture_output=True,
                cwd=cwd,
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
