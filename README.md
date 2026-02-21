# terminalai

This program enables an LLM from an API like OpenAI to interface with a user's terminal and run commands. The current MVP is direct-execution focused: the model proposes commands and the agent executes them immediately in the selected shell.

## MVP behavior (current)

- Minimal orchestration loop that cycles model suggestion -> command execution -> model feedback.
- Default CLI behavior is direct terminal orchestration.
- Lightweight session logs are written to `logs/` (or `TERMINALAI_LOG_DIR`) with command/output/timestamps.
- Command execution starts in the caller's current working directory at launch time unless `--cwd` is provided; the model does not choose the starting directory.

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

### Common CLI options

- `--shell {cmd,powershell}`: choose shell adapter (default: `powershell`)
- `--model <name>`: override model from environment
- `--max-steps <n>`: cap model-execution iterations (default: `20`)
- `--cwd <path>`: set the starting working directory for command execution (must exist and be a directory)

Example:

```bash
terminalai --shell cmd --model gpt-5.2-codex --max-steps 10 "Create a TODO.txt with three tasks"
```

### Environment variables

- `TERMINALAI_OPENAI_API_KEY`: OpenAI API key for model calls.
- `TERMINALAI_API_KEY`: legacy alias for `TERMINALAI_OPENAI_API_KEY`.
- `TERMINALAI_MODEL`: default model name (default: `gpt-5.2`).
- `TERMINALAI_REASONING_EFFORT`: optional reasoning level override (for example `low`, `medium`, or `high`) for models that support reasoning. If unset, `terminalai` defaults to `medium` for reasoning-capable families like GPT-5.
- `TERMINALAI_API_URL`: responses endpoint URL.
- `TERMINALAI_CONFIG_FILE`: path to JSON config file (default: `terminalai.config.json`).
- `TERMINALAI_LOG_DIR`: directory for session logs (default: `logs`).
- `TERMINALAI_SAFETY_ENABLED`: parsed but currently not enforced in the MVP execution path.
- `TERMINALAI_ALLOW_UNSAFE`: parsed but currently not enforced in the MVP execution path.

### JSON config file

`terminalai` can load defaults from `terminalai.config.json` (or a custom path via `TERMINALAI_CONFIG_FILE`) with detailed OpenAI and per-model settings. You can set the OpenAI API key in either `openai.api_key` (recommended) or top-level `api_key`.

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
  "log_dir": "logs"
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
