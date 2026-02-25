# terminalai

This program enables an LLM from an API like OpenAI to interface with a user's terminal and run commands. The current MVP is direct-execution focused: the model proposes commands and the agent executes them immediately in the selected shell.

## MVP behavior (current)

- Minimal orchestration loop that cycles model suggestion -> command execution -> model feedback.
- Default CLI behavior is direct terminal orchestration.
- Lightweight session logs are written to `logs/` (or `TERMINALAI_LOG_DIR`) as JSONL entries with versioned command/output metadata for auditing and debugging.
- Command execution starts in the caller's current working directory at launch time unless `cwd` is configured; the model does not choose the starting directory.

## Safety posture

TerminalAI enforces shell guardrails by default. Commands matching destructive patterns (for example `rm -rf`, `del /s /q`, and similar) are sent to the shell adapter with `confirmed=false` unless unsafe execution is explicitly enabled.

- `safety_mode: "strict"` (default): guardrails are active; when a destructive command is proposed, the CLI asks for explicit user confirmation (`Run this command and continue? [y/N]`) before executing.
- `safety_mode: "allow_unsafe"`: terminalai sets `confirmed=true` for destructive commands, so they are auto-approved by terminalai and can run without an interactive prompt.
- `safety_mode: "off"` (or `"disabled"`): terminalai does not do safety confirmation or prompting, but it also does **not** auto-approve destructive commands (`confirmed=false`), so the selected shell integration (for example the built-in denylist/allowlist checks and confirmation-required checks in the shell adapter) still decides whether the command runs or is blocked.
- Practical distinction:
  - `allow_unsafe` = terminalai actively bypasses its destructive-command confirmation gate.
  - `off` = terminalai steps out of the way and defers destructive-command handling to shell adapter guardrails (denylist/allowlist/confirmation-required checks).
- Every command outcome is also fed back to the model as a structured `session_context` event (`command_executed`, `command_blocked`, or `command_declined`) including command text, `safety_mode`, and normalized reason codes for blocked/declined paths.

## Platforms

- Operating system support:
  - Windows 10/11: supported
  - Linux (tested on modern POSIX environments): supported
  - macOS: expected to work as a POSIX target, but not yet explicitly documented/tested in this MVP
- Shell adapters:
  - Windows: `powershell` (default), `cmd` (explicit opt-in)
  - Linux/POSIX: `bash` (default)
  - Optional cross-platform adapter alias: `pwsh` (PowerShell 7+, if installed)

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

4. Linux shell example (`bash`):

   ```bash
   export TERMINALAI_SHELL=bash
   terminalai "Show disk usage for the current directory"
   # or
   TERMINALAI_SHELL=bash python start.py "Show disk usage for the current directory"
   ```

### CLI input

The CLI accepts the optional goal argument and an optional `--cwd` override for
the starting working directory. If `--cwd` is not provided, `terminalai` uses
`cwd` from config (or `TERMINALAI_CWD`), and if that is also unset it defaults
to the directory where you launched the script.

Example:

```bash
terminalai "Create a TODO.txt with three tasks"

terminalai --cwd ~/projects/demo "Create a TODO.txt with three tasks"
```

### Understanding runtime output in the terminal

By default (`readable_cli_output: true`), `terminalai` prints each turn using labeled sections so in-progress and completed states are easier to scan. For example:

```text

=== Turn 1 (running) ===

[command]

ls -la

[output]

returncode=0
duration=0.1090s

stdout:
total 75
...

stderr:

[hint]

Listing all files (including hidden) in the current directory so I can summarize what’s here.
```

When the model pauses for a critical question, the turn is labeled clearly:

```text
=== Turn 2 (needs input) ===
[question]
Which environment should I target?
```

If you disable this with `readable_cli_output: false` (or `TERMINALAI_READABLE_CLI_OUTPUT=false`), the CLI falls back to the legacy plain output style (`[n] $ command`, then output, then `hint:`).

Notes:

- A run can include multiple blocks (`Turn 1`, `Turn 2`, ...), one per turn.
- `output` preserves multiline command output while trimming trailing whitespace.
- The same `output` text and `hint` are also appended to JSONL session logs in `logs/session-YYYY-MM-DD.log` (or `TERMINALAI_LOG_DIR`).
- Each log line is schema-versioned with `log_version` and includes: `timestamp`, `goal`, `model`, `shell`, `working_directory`, `step_index`, `command`, `output`, `next_action_hint`, `returncode`, `duration`, `awaiting_user_feedback`, and `complete_signal`.

When `TERMINALAI_ALLOW_USER_FEEDBACK_PAUSE=true`, the model can pause and ask one critical
question if it cannot proceed safely. In that case, the CLI prompts:

```text
=== MODEL PAUSED: INPUT REQUIRED ===
Question: <model question>
Your response: <you type here>
```

After you answer, the same run continues automatically. The follow-up model turn receives both
the original goal and your new response in `session_context`.

When `TERMINALAI_AUTO_PROGRESS_TURNS=false`, terminalai pauses before each model iteration.
Press Enter to proceed to the next turn, type a short instruction to steer that specific turn,
or enter `q` to stop the run. Per-turn instructions are injected into model context before the
next command proposal.

When the model marks the **overarching goal** complete, terminalai appends a continuation question to the final assistant hint:

```text
Would you like to keep going with new instructions while we retain this context?
```

This prompt appears exactly once per completed top-level run, and does not appear for intermediate turns, tool-step boundaries, or subtask/milestone updates while a run is still active.

This section documents the **current** output contract. If the CLI output format changes, update this section in the same change.

At startup, `terminalai` does not preload repository files (such as `README.md`) into model context. The model can choose to read files by issuing explicit shell commands.

### Environment variables

- `TERMINALAI_OPENAI_API_KEY`: OpenAI API key for model calls.
- `TERMINALAI_API_KEY`: legacy alias for `TERMINALAI_OPENAI_API_KEY`.
- `TERMINALAI_MODEL`: default model name (default: `gpt-5.2`).
- `TERMINALAI_REASONING_EFFORT`: optional reasoning level override (for example `low`, `medium`, or `high`) for models that support reasoning. If unset, `terminalai` defaults to `medium` for reasoning-capable families like GPT-5.
- `TERMINALAI_API_URL`: responses endpoint URL.
- `TERMINALAI_CONFIG_FILE`: path to JSON config file (default: `terminalai.config.json`).
- `TERMINALAI_LOG_DIR`: directory for session logs (default: `logs`).
- `TERMINALAI_ALLOW_USER_FEEDBACK_PAUSE`: when true, allows the model to pause and ask one critical question if blocked.
- `TERMINALAI_CONTINUATION_PROMPT_ENABLED`: enables/disables the post-completion continuation question appended after the overarching goal is complete (default: `true`).
- `TERMINALAI_AUTO_PROGRESS_TURNS`: when `true` (default), model turns run continuously; when `false`, the CLI pauses before each turn and waits for Enter/instructions.
- `TERMINALAI_READABLE_CLI_OUTPUT`: when `true` (default), print structured turn sections (`[command]`, `[output]`, `[hint]/[question]`); when `false`, use legacy plain output.
- `TERMINALAI_SHELL`: shell adapter (`cmd`, `powershell`, `bash`; aliases `pwsh`, `sh`, `shell`). If unset, defaults are platform-aware: `powershell` on Windows and `bash` on POSIX systems.
- `TERMINALAI_MAX_STEPS`: maximum model-execution iterations (default `20`).
- `TERMINALAI_MAX_CONTEXT_CHARS`: maximum number of serialized `session_context` characters included per model request (default `12000`).
- `TERMINALAI_CWD`: starting working directory for command execution.
- `TERMINALAI_SAFETY_MODE`: canonical destructive-command safety mode. Supported values: `strict`, `allow_unsafe`, `off` (alias: `disabled`).

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
`terminalai.config.json` as the shared project baseline. In the shared baseline,
`shell` can be set to `null` to defer shell selection to environment variables
or platform defaults at runtime.

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
  "continuation_prompt_enabled": true,
  "auto_progress_turns": true,
  "readable_cli_output": true,
  "safety_mode": "strict",
  "shell": null,
  "max_steps": 20,
  "max_context_chars": 12000,
  "cwd": null,
  "log_dir": "logs"
}
```

The model system prompt is hardcoded by the application and is not user-configurable. Shell-specific guidance and optional feature instructions (such as user-feedback pause behavior) are dynamically included based on the runtime context and active configuration.

When both config file and environment variables are present, environment variables take precedence.

#### Linux distro notes

- `bash` should be installed and available in `PATH` (it is present by default on most mainstream distributions).
- `pwsh` is optional on Linux and only needed if you explicitly select `TERMINALAI_SHELL=pwsh`/`powershell`.
- `cmd` is Windows-specific and is not expected to be available on Linux distributions.


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
