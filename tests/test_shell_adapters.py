from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from terminalai.shell import CmdAdapter, PowerShellAdapter, create_shell_adapter


@pytest.mark.parametrize(
    ("factory_input", "expected_type"),
    [("cmd", CmdAdapter), ("powershell", PowerShellAdapter)],
)
def test_create_shell_adapter(factory_input: str, expected_type: type[object]) -> None:
    adapter = create_shell_adapter(factory_input)
    assert isinstance(adapter, expected_type)


def test_create_shell_adapter_invalid() -> None:
    with pytest.raises(ValueError):
        create_shell_adapter("bash")


def test_cmd_adapter_dry_run() -> None:
    result = CmdAdapter().execute("echo hello", dry_run=True)
    assert result.executed is False
    assert result.returncode == 0
    assert "dry-run" in result.stdout


def test_cmd_adapter_requires_confirmation() -> None:
    result = CmdAdapter().execute("rm -rf ./tmp")
    assert result.executed is False
    assert result.blocked is True
    assert "requires explicit confirmation" in result.stderr


def test_cmd_adapter_runs_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        assert args[0] == ["cmd.exe", "/d", "/s", "/c", "echo hello"]
        assert kwargs["timeout"] == 10
        assert kwargs["cwd"] == "C:/work"
        return SimpleNamespace(returncode=0, stdout=b"h\xe9llo", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CmdAdapter().execute("echo hello", cwd="C:/work", timeout=10, confirmed=True)
    assert result.returncode == 0
    assert result.stdout == "héllo"


def test_powershell_adapter_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1, output=b"", stderr=b"late")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = PowerShellAdapter(executable="pwsh").execute(
        "Write-Output hi", timeout=1, confirmed=True
    )
    assert result.timed_out is True
    assert result.returncode == 124
    assert result.stderr == "late"


def test_allowlist_hook_rejects() -> None:
    adapter = PowerShellAdapter(executable="pwsh", allowlist_hook=lambda _command, _shell: False)
    result = adapter.execute("Get-ChildItem", confirmed=True)
    assert result.executed is False
    assert result.blocked is True
    assert "allowlist" in result.stderr


def test_powershell_adapter_command_formatting(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        assert args[0] == [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Write-Output hi",
        ]
        assert kwargs["cwd"] == "C:/repo"
        return SimpleNamespace(returncode=0, stdout=b"hi", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = PowerShellAdapter(executable="pwsh").execute(
        "Write-Output hi",
        cwd="C:/repo",
        confirmed=True,
    )
    assert result.returncode == 0


def test_cmd_adapter_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1, output=b"", stderr=b"too slow")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CmdAdapter().execute("echo hello", timeout=1, confirmed=True)
    assert result.timed_out is True
    assert result.returncode == 124
    assert result.stderr == "too slow"


def test_cmd_adapter_nonzero_exit_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=42, stdout=b"", stderr=b"bad command")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CmdAdapter().execute("broken command", confirmed=True)
    assert result.returncode == 42
    assert result.stderr == "bad command"
    assert result.timed_out is False
