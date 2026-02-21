"""Data models used by the minimal agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DecisionPhase = Literal["analysis", "mutation", "verification", "completion"]
RiskLevel = Literal["low", "medium", "high"]


@dataclass(slots=True)
class AgentStep:
    """A model-proposed command for the current goal."""

    goal: str
    proposed_command: str
    phase: DecisionPhase = "analysis"
    expected_outcome: str | None = None
    verification_command: str | None = None
    risk_level: RiskLevel | None = None


@dataclass(slots=True)
class ExecutionResult:
    """Normalized shell execution output returned to the model."""

    stdout: str
    stderr: str
    returncode: int
    duration: float


@dataclass(slots=True)
class SessionTurn:
    """Captured request/response data for a single orchestration turn."""

    input: str
    command: str
    output: str
    next_action_hint: str | None = None
    awaiting_user_feedback: bool = False
    turn_complete: bool = False
    subtask_complete: bool = False
    overarching_goal_complete: bool = False
    continuation_prompt_added: bool = False
    phase: DecisionPhase = "analysis"
    expected_outcome: str | None = None
    verification_command: str | None = None
    risk_level: RiskLevel | None = None
