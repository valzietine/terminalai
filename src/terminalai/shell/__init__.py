"""Shell adapter implementations."""

from .base import CommandResult, ShellAdapter
from .bash_adapter import BashAdapter
from .cmd_adapter import CmdAdapter
from .powershell_adapter import PowerShellAdapter


def create_shell_adapter(shell_name: str) -> ShellAdapter:
    normalized = shell_name.strip().lower()
    if normalized == "cmd":
        return CmdAdapter()
    if normalized in {"bash", "sh", "shell"}:
        return BashAdapter(executable="sh" if normalized == "sh" else None)
    if normalized == "powershell":
        return PowerShellAdapter()
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
