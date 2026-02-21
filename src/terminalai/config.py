"""Environment-backed application configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SafetyMode = Literal["strict", "allow_unsafe", "off"]

DEFAULT_SYSTEM_PROMPT = " ".join(
    [
        "You are TerminalAI, an expert terminal orchestration assistant.",
        (
            "First orient yourself to the machine context and execution history,"
            " then choose the single best next shell command for the user's goal."
        ),
        "Prefer safe, reversible, and idempotent operations.",
        (
            "Avoid destructive commands unless they are explicitly requested"
            " and clearly justified by the goal."
        ),
        (
            "If the goal is complete, or no command should be run, set"
            " command to null and complete to true."
        ),
        (
            "Always return strict JSON with keys: command (string or null),"
            " notes (string or null), complete (boolean)."
        ),
    ]
)


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


def _to_bool_from_object(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _to_bool(value, default=default)
    return default


@dataclass(slots=True)
class AppConfig:
    """Runtime settings loaded from environment variables."""

    api_key: str | None
    model: str
    reasoning_effort: str | None
    safety_mode: SafetyMode
    api_url: str
    log_dir: str
    system_prompt: str
    allow_user_feedback_pause: bool
    confirm_before_complete: bool
    continuation_prompt_enabled: bool
    auto_progress_turns: bool
    readable_cli_output: bool
    shell: str
    max_steps: int
    max_context_chars: int
    working_directory: str | None

    @classmethod
    def from_env(cls) -> AppConfig:
        file_config = _load_preferred_file_config()
        openai_from_file = file_config.get("openai")
        openai_config = openai_from_file if isinstance(openai_from_file, dict) else {}
        models_from_file = file_config.get("models")
        model_config = models_from_file if isinstance(models_from_file, dict) else {}

        selected_model = os.getenv("TERMINALAI_MODEL") or str(
            file_config.get("default_model", "gpt-5.2")
        )
        selected_model_entry = model_config.get(selected_model)
        selected_model_config = (
            selected_model_entry if isinstance(selected_model_entry, dict) else {}
        )

        return cls(
            api_key=(
                os.getenv("TERMINALAI_OPENAI_API_KEY")
                or os.getenv("TERMINALAI_API_KEY")
                or _to_optional_string(openai_config.get("api_key"))
                or _to_optional_string(file_config.get("api_key"))
            ),
            model=selected_model,
            reasoning_effort=(
                os.getenv("TERMINALAI_REASONING_EFFORT")
                or _to_optional_string(selected_model_config.get("reasoning_effort"))
                or _default_reasoning_effort(selected_model)
            ),
            safety_mode=_resolve_safety_mode(file_config),
            api_url=(
                os.getenv("TERMINALAI_API_URL")
                or _to_optional_string(openai_config.get("api_url"))
                or "https://api.openai.com/v1/responses"
            ),
            log_dir=(
                os.getenv("TERMINALAI_LOG_DIR")
                or _to_optional_string(file_config.get("log_dir"))
                or "logs"
            ),
            system_prompt=(
                os.getenv("TERMINALAI_SYSTEM_PROMPT")
                or _to_optional_string(file_config.get("system_prompt"))
                or DEFAULT_SYSTEM_PROMPT
            ),
            allow_user_feedback_pause=_to_bool(
                os.getenv("TERMINALAI_ALLOW_USER_FEEDBACK_PAUSE"),
                default=_to_bool_from_object(
                    file_config.get("allow_user_feedback_pause"),
                    default=False,
                ),
            ),
            confirm_before_complete=_to_bool(
                os.getenv("TERMINALAI_CONFIRM_BEFORE_COMPLETE"),
                default=_to_bool_from_object(
                    file_config.get("confirm_before_complete"),
                    default=False,
                ),
            ),
            continuation_prompt_enabled=_to_bool(
                os.getenv("TERMINALAI_CONTINUATION_PROMPT_ENABLED"),
                default=_to_bool_from_object(
                    file_config.get("continuation_prompt_enabled"),
                    default=True,
                ),
            ),
            auto_progress_turns=_to_bool(
                os.getenv("TERMINALAI_AUTO_PROGRESS_TURNS"),
                default=_to_bool_from_object(
                    file_config.get("auto_progress_turns"),
                    default=True,
                ),
            ),
            readable_cli_output=_to_bool(
                os.getenv("TERMINALAI_READABLE_CLI_OUTPUT"),
                default=_to_bool_from_object(
                    file_config.get("readable_cli_output"),
                    default=True,
                ),
            ),
            shell=_resolve_shell(
                os.getenv("TERMINALAI_SHELL")
                or _to_optional_string(file_config.get("shell"))
            ),
            max_steps=_to_positive_int(
                os.getenv("TERMINALAI_MAX_STEPS")
                or file_config.get("max_steps"),
                default=20,
            ),
            max_context_chars=_to_positive_int(
                os.getenv("TERMINALAI_MAX_CONTEXT_CHARS")
                or file_config.get("max_context_chars"),
                default=12000,
            ),
            working_directory=(
                os.getenv("TERMINALAI_CWD")
                or _to_optional_string(file_config.get("cwd"))
            ),
        )


def _resolve_safety_mode(file_config: dict[str, object]) -> SafetyMode:
    env_mode = _to_safety_mode(os.getenv("TERMINALAI_SAFETY_MODE"))
    if env_mode is not None:
        return env_mode

    file_mode = _to_safety_mode(_to_optional_string(file_config.get("safety_mode")))
    if file_mode is not None:
        return file_mode

    return "strict"


def _to_safety_mode(value: str | None) -> SafetyMode | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    aliases: dict[str, SafetyMode] = {
        "strict": "strict",
        "allow_unsafe": "allow_unsafe",
        "unsafe": "allow_unsafe",
        "off": "off",
        "disabled": "off",
    }
    return aliases.get(normalized)

def _to_optional_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _load_file_config(path_value: str) -> dict[str, object]:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            parsed = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _load_preferred_file_config() -> dict[str, object]:
    explicit_path = os.getenv("TERMINALAI_CONFIG_FILE")
    if explicit_path:
        return _load_file_config(explicit_path)

    shared_config = _load_file_config("terminalai.config.json")
    local_override = _load_file_config("terminalai.config.local.json")
    return _merge_dicts(shared_config, local_override)


def _merge_dicts(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(base_value, value)
        else:
            merged[key] = value
    return merged


def _default_reasoning_effort(model: str) -> str | None:
    """Provide practical defaults for reasoning-capable model families."""
    normalized = model.strip().lower()
    if normalized.startswith("gpt-5") or normalized.startswith("o"):
        return "medium"
    return None


def _shell_value(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "cmd": "cmd",
        "powershell": "powershell",
        "pwsh": "powershell",
        "bash": "bash",
        "sh": "bash",
        "shell": "bash",
    }
    return aliases.get(normalized, "powershell")


def _default_shell_for_platform(os_name: str | None = None) -> str:
    platform_name = os.name if os_name is None else os_name
    return "powershell" if platform_name == "nt" else "bash"


def _resolve_shell(value: str | None) -> str:
    if value is None:
        return _default_shell_for_platform()
    return _shell_value(value)


def _to_positive_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value if value > 0 else default
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return default
        return parsed if parsed > 0 else default
    return default
