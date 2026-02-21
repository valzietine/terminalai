# Contribution Guidelines

Thank you for contributing to this repository.

## Scope
These instructions apply to the entire repository unless a deeper `AGENTS.md` overrides them.

## General expectations
- Keep changes focused and minimal.
- Prefer small, reviewable commits with clear messages.
- Update relevant documentation when behavior changes.

## Code quality
- Follow existing project structure and naming patterns.
- Avoid unrelated refactors in the same change.
- Add or update tests when changing runtime behavior.

## Config template contribution policy
- When introducing a new configurable runtime feature, update `terminalai.config.json` in the same change.
- Add an explicit toggle option for the feature (for example a boolean on/off flag, or `null` only when runtime auto-detection is intentional).
- Choose a safe documented default that preserves expected behavior for existing users.
- Document the new toggle in the README configuration section, including related environment-variable override behavior.
- Include tests whenever runtime behavior changes, and ensure config parsing covers the new option.

## Validation policy
- Run tests and linters for every code change.
- Do **not** run tests or linters when changes are limited to comments or documentation.

## Pull requests
- Use a concise, descriptive PR title.
- Summarize what changed and why.
- Include verification steps and outcomes.
