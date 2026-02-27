from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from terminalai.shell import BashAdapter, PowerShellAdapter, create_shell_adapter


@pytest.mark.parametrize(
    ("factory_input", "expected_type"),
    [
        ("powershell", PowerShellAdapter),
        ("bash", BashAdapter),
        ("sh", BashAdapter),
    ],
)
def test_create_shell_adapter(factory_input: str, expected_type: type[object]) -> None:
    adapter = create_shell_adapter(factory_input)
    assert isinstance(adapter, expected_type)


def test_create_shell_adapter_invalid() -> None:
    with pytest.raises(ValueError):
        create_shell_adapter("zsh")


def test_create_shell_adapter_rejects_cmd() -> None:
    with pytest.raises(ValueError, match="Unsupported shell adapter"):
        create_shell_adapter("cmd")


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


def test_powershell_adapter_missing_executable() -> None:
    missing_executable = "C:/missing/pwsh.exe"
    result = PowerShellAdapter(executable=missing_executable).execute(
        "Write-Output hi", confirmed=True
    )

    assert result.executed is False
    assert result.returncode == 127
    assert result.stderr == f"powershell executable not found: {missing_executable}"


def test_bash_adapter_runs_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        assert args[0] == ["bash", "-lc", "echo hi"]
        assert kwargs["cwd"] == "/repo"
        return SimpleNamespace(returncode=0, stdout=b"hi", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = BashAdapter(executable="bash").execute("echo hi", cwd="/repo", confirmed=True)
    assert result.returncode == 0


def test_bash_adapter_falls_back_to_sh(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(name: str) -> str | None:
        if name == "bash":
            return None
        if name == "sh":
            return "/bin/sh"
        return None

    monkeypatch.setattr("terminalai.shell.bash_adapter.shutil.which", fake_which)

    adapter = BashAdapter()

    assert adapter.executable == "sh"


def test_bash_adapter_allows_destructive_command_when_confirmation_mode_disabled() -> None:
    result = BashAdapter(executable="bash", confirmation_mode=False).execute(
        "rm -rf ./tmp", dry_run=True
    )

    assert result.executed is False
    assert result.blocked is False
    assert result.returncode == 0


def test_create_shell_adapter_threads_elevation_flag() -> None:
    assert create_shell_adapter("bash", elevate_process=True).elevation_enabled is True
    assert create_shell_adapter("powershell", elevate_process=False).elevation_enabled is False


def test_bash_adapter_prefixes_sudo_when_elevation_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **_kwargs: object) -> SimpleNamespace:
        assert args[0] == ["bash", "-lc", "sudo -- echo hello"]
        return SimpleNamespace(returncode=0, stdout=b"hi", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("terminalai.shell.bash_adapter.os.name", "posix")
    monkeypatch.setattr("terminalai.shell.bash_adapter.shutil.which", lambda name: "/usr/bin/sudo")

    result = BashAdapter(executable="bash", elevate_process=True).execute(
        "echo hello", confirmed=True
    )

    assert result.elevated is True
    assert result.elevation_requested is True


def test_bash_adapter_skips_sudo_if_command_already_elevated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **_kwargs: object) -> SimpleNamespace:
        assert args[0] == ["bash", "-lc", "sudo apt update"]
        return SimpleNamespace(returncode=0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("terminalai.shell.bash_adapter.os.name", "posix")

    result = BashAdapter(executable="bash", elevate_process=True).execute(
        "sudo apt update", confirmed=True
    )

    assert result.elevated is False


def test_bash_adapter_falls_back_when_sudo_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **_kwargs: object) -> SimpleNamespace:
        assert args[0] == ["bash", "-lc", "echo hello"]
        return SimpleNamespace(returncode=0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("terminalai.shell.bash_adapter.os.name", "posix")
    monkeypatch.setattr("terminalai.shell.bash_adapter.shutil.which", lambda _name: None)

    result = BashAdapter(executable="bash", elevate_process=True).execute(
        "echo hello", confirmed=True
    )

    assert result.elevated is False
    assert result.elevation_error is not None
    assert "sudo is not available" in result.stderr


def test_powershell_adapter_uses_runas_when_elevated(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **_kwargs: object) -> SimpleNamespace:
        cmd = args[0]
        assert cmd[0] == "pwsh"
        assert "Start-Process -FilePath 'pwsh' -Verb RunAs" in cmd[-1]
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("terminalai.shell.powershell_adapter.os.name", "nt")

    result = PowerShellAdapter(executable="pwsh", elevate_process=True).execute(
        "Write-Output hi", confirmed=True
    )

    assert result.elevated is True
    assert result.elevation_requested is True
