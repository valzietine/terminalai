# terminalai

This program enables an LLM from an API like OpenAI to interface with a user's terminal and run commands. The current MVP is direct-execution focused: the model proposes commands and the agent executes them immediately in the selected shell.

## MVP behavior (current)

- Minimal orchestration loop that cycles model suggestion -> command execution -> model feedback.
- Default CLI behavior is direct terminal orchestration.
- Lightweight session logs are written to `logs/` (or `TERMINALAI_LOG_DIR`) with command/output/timestamps.
- Command execution starts in the caller's current working directory at launch time unless `cwd` is configured; the model does not choose the starting directory.

## Safety posture

The MVP intentionally prioritizes direct execution and does not add command gating in the orchestration path. Additional safety controls (policy filters, guardrails, and confirmations) are planned for a later phase.

## Platforms

- Operating system: Windows 10 (others may come later)
- Shell adapters: Command Prompt and PowerShell

## Usage

1. Set your OpenAI API key:

   ```bash
   export TERMINALAI_OPENAI_API_KEY="your_openai_api_key"
   ```

   `TERMINALAI_API_KEY` is still supported as a legacy alias.

2. Run with a goal (installed CLI):

   ```bash
   terminalai "List files in the current directory and summarize what is here"
   ```

   Or run directly from source without installing the package:

   ```bash
   python start.py "List files in the current directory and summarize what is here"
   ```

3. Run without a goal to be prompted interactively:

   ```bash
   terminalai
   # or
   python start.py
   ```

### CLI input

The CLI now accepts only the optional goal argument. Runtime options are configured via `terminalai.config.json` (or environment variable overrides).

Example:

```bash
terminalai "Create a TODO.txt with three tasks"
```

### Understanding runtime output in the terminal

When `terminalai` executes commands, it prints one block per executed step. For example:

```text
[1] $ ls -la
returncode=0
duration=0.1090s
stdout:
total 75
...
stderr:

hint: Listing all files (including hidden) in the current directory so I can summarize what’s here.
```

How to read each field:

- `[1]`: step number in the current run (1-based index).
- `$ ls -la`: exact command the model asked to run in the selected shell.
- `returncode=0`: process exit code (`0` usually means success; non-zero usually means an error or partial failure).
- `duration=0.1090s`: wall-clock runtime for that command in seconds.
- `stdout:`: standard output stream (normal command output).
- `stderr:`: standard error stream (warnings/errors emitted by the command).
- `hint: ...`: model-provided reasoning note for what it plans next. This is not command output.

Notes:

- `stdout` and `stderr` are always printed, even when empty.
- A run can include multiple blocks (`[1]`, `[2]`, `[3]`, ...), one per command.
- The same `output` text and `hint` are also appended to JSONL session logs in `logs/session-YYYY-MM-DD.log` (or `TERMINALAI_LOG_DIR`).

This section documents the **current** output contract. If the CLI output format changes, update this section in the same change.

### Environment variables

- `TERMINALAI_OPENAI_API_KEY`: OpenAI API key for model calls.
- `TERMINALAI_API_KEY`: legacy alias for `TERMINALAI_OPENAI_API_KEY`.
- `TERMINALAI_MODEL`: default model name (default: `gpt-5.2`).
- `TERMINALAI_REASONING_EFFORT`: optional reasoning level override (for example `low`, `medium`, or `high`) for models that support reasoning. If unset, `terminalai` defaults to `medium` for reasoning-capable families like GPT-5.
- `TERMINALAI_API_URL`: responses endpoint URL.
- `TERMINALAI_CONFIG_FILE`: path to JSON config file (default: `terminalai.config.json`).
- `TERMINALAI_LOG_DIR`: directory for session logs (default: `logs`).
- `TERMINALAI_SYSTEM_PROMPT`: override the system prompt sent to the model.
- `TERMINALAI_ALLOW_USER_FEEDBACK_PAUSE`: when true, allows the model to pause and ask one critical question if blocked.
- `TERMINALAI_CONFIRM_BEFORE_COMPLETE`: when true, asks the user to confirm before ending after the model marks the task complete. If the user declines, the CLI captures follow-up objectives/questions and continues the run with that feedback.
- `TERMINALAI_SHELL`: shell adapter (`cmd` or `powershell`, default `powershell`).
- `TERMINALAI_MAX_STEPS`: maximum model-execution iterations (default `20`).
- `TERMINALAI_CWD`: starting working directory for command execution.
- `TERMINALAI_SAFETY_ENABLED`: parsed but currently not enforced in the MVP execution path.
- `TERMINALAI_ALLOW_UNSAFE`: parsed but currently not enforced in the MVP execution path.

### JSON config file

`terminalai` can load defaults from `terminalai.config.json` (or a custom path via `TERMINALAI_CONFIG_FILE`) with detailed OpenAI and per-model settings. You can set the OpenAI API key in either `openai.api_key` (recommended) or top-level `api_key`.

For local machine-specific overrides, create a personal config file named
`terminalai.config.local.json` (this file is intentionally gitignored). When
present in the project root, it is loaded automatically and merged on top of
`terminalai.config.json`.

```bash
cp terminalai.config.json terminalai.config.local.json
terminalai "your goal"
```

Use `terminalai.config.local.json` for machine-specific values (for example,
API keys and local shell/cwd preferences) while keeping
`terminalai.config.json` as the shared project baseline.

```json
{
  "openai": {
    "api_url": "https://api.openai.com/v1/responses",
    "api_key": "YOUR_OPENAI_API_KEY"
  },
  "default_model": "gpt-5.2",
  "models": {
    "gpt-5.2": {
      "reasoning_effort": "medium"
    },
    "gpt-5.2-codex": {
      "reasoning_effort": "medium"
    }
  },
  "confirm_before_complete": false,
  "shell": "powershell",
  "max_steps": 20,
  "cwd": null,
  "log_dir": "logs",
  "system_prompt": "You are TerminalAI, an expert terminal orchestration assistant..."
}
```

When both config file and environment variables are present, environment variables take precedence.


## Development workflow

Install dev dependencies:

```bash
pip install -e .[dev]
```

Run local checks before committing:

```bash
ruff check .      # lint
mypy              # typecheck (strict on shell modules)
pytest            # test
```

Tests now auto-add `src/` to `sys.path` via `tests/conftest.py`, so `pytest`
works in clean runners without requiring `PYTHONPATH=src`.
