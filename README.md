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
