"""Tests for running the app directly via start.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_start_script_help() -> None:
    script = Path(__file__).resolve().parents[1] / "start.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Terminal AI assistant" in result.stdout
