"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _to_bool(value: str | None, default: bool = False) -> bool:
    """Convert common env var truthy/falsy values into booleans."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(slots=True)
class AppConfig:
    """Runtime settings loaded from environment variables."""

    api_key: str | None
    model: str
    safety_enabled: bool
    allow_unsafe: bool

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            api_key=os.getenv("TERMINALAI_API_KEY"),
            model=os.getenv("TERMINALAI_MODEL", "gpt-5.2-codex"),
            safety_enabled=_to_bool(os.getenv("TERMINALAI_SAFETY_ENABLED"), default=True),
            allow_unsafe=_to_bool(os.getenv("TERMINALAI_ALLOW_UNSAFE"), default=False),
        )
