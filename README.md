# terminalai

This program enables an LLM from an API like OpenAI to interface with a user's terminal and run commands. The current MVP is direct-execution focused: the model proposes commands and the agent executes them immediately in the selected shell.

## MVP behavior (current)

- Minimal orchestration loop that cycles model suggestion -> command execution -> model feedback.
- Default CLI behavior is direct terminal orchestration.
- Lightweight session logs are written to `logs/` (or `TERMINALAI_LOG_DIR`) with command/output/timestamps.

## Safety posture

The MVP intentionally prioritizes direct execution and does not add command gating in the orchestration path. Additional safety controls (policy filters, guardrails, and confirmations) are planned for a later phase.

## Platforms

- Operating system: Windows 10 (others may come later)
- Shell adapters: Command Prompt and PowerShell

## Usage

1. Set your API key:

   ```bash
   export TERMINALAI_API_KEY="your_api_key"
   ```

2. Run with a goal:

   ```bash
   terminalai "List files in the current directory and summarize what is here"
   ```

3. Or run without a goal to be prompted interactively:

   ```bash
   terminalai
   ```

### Common CLI options

- `--shell {cmd,powershell}`: choose shell adapter (default: `powershell`)
- `--model <name>`: override model from environment
- `--max-steps <n>`: cap model-execution iterations (default: `20`)

Example:

```bash
terminalai --shell cmd --model gpt-5.2-codex --max-steps 10 "Create a TODO.txt with three tasks"
```

### Environment variables

- `TERMINALAI_API_KEY`: API key for model calls.
- `TERMINALAI_MODEL`: default model name (default: `gpt-5.2-codex`).
- `TERMINALAI_API_URL`: responses endpoint URL.
- `TERMINALAI_LOG_DIR`: directory for session logs (default: `logs`).
- `TERMINALAI_SAFETY_ENABLED`: parsed but currently not enforced in the MVP execution path.
- `TERMINALAI_ALLOW_UNSAFE`: parsed but currently not enforced in the MVP execution path.


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
