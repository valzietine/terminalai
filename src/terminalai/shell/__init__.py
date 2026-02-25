"""Shell adapter implementations."""

from .base import CommandResult, ShellAdapter
from .bash_adapter import BashAdapter
from .cmd_adapter import CmdAdapter
from .powershell_adapter import PowerShellAdapter


def create_shell_adapter(shell_name: str, *, elevate_process: bool = False) -> ShellAdapter:
    normalized = shell_name.strip().lower()
    if normalized == "cmd":
        return CmdAdapter(elevate_process=elevate_process)
    if normalized in {"bash", "sh", "shell"}:
        return BashAdapter(
            executable="sh" if normalized == "sh" else None,
            elevate_process=elevate_process,
        )
    if normalized == "powershell":
        return PowerShellAdapter(elevate_process=elevate_process)
    msg = f"Unsupported shell adapter: {shell_name}"
    raise ValueError(msg)


__all__ = [
    "BashAdapter",
    "CmdAdapter",
    "CommandResult",
    "PowerShellAdapter",
    "ShellAdapter",
    "create_shell_adapter",
]
