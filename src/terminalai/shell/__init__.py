"""Shell adapter implementations."""

from .base import CommandResult, ShellAdapter
from .cmd_adapter import CmdAdapter
from .powershell_adapter import PowerShellAdapter


def create_shell_adapter(shell_name: str) -> ShellAdapter:
    normalized = shell_name.strip().lower()
    if normalized == "cmd":
        return CmdAdapter()
    if normalized == "powershell":
        return PowerShellAdapter()
    msg = f"Unsupported shell adapter: {shell_name}"
    raise ValueError(msg)


__all__ = [
    "CmdAdapter",
    "CommandResult",
    "PowerShellAdapter",
    "ShellAdapter",
    "create_shell_adapter",
]
