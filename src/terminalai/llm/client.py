"""Thin model client that requests the next shell command."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import cast
from urllib import request
from urllib.error import HTTPError, URLError

from terminalai.agent.models import DecisionPhase, RiskLevel

VALID_PHASES: set[DecisionPhase] = {"analysis", "mutation", "verification", "completion"}
VALID_RISK_LEVELS: set[RiskLevel] = {"low", "medium", "high"}
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelDecision:
    """Structured model response used by the agent loop."""

    command: str | None
    notes: str | None = None
    complete: bool = False
    ask_user: bool = False
    user_question: str | None = None
    phase: DecisionPhase = "analysis"
    expected_outcome: str | None = None
    verification_command: str | None = None
    risk_level: RiskLevel | None = None


class LLMClient:
    """Small HTTP client for command-oriented model calls."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        system_prompt: str,
        max_context_chars: int = 12000,
        reasoning_effort: str | None = None,
        api_url: str = "https://api.openai.com/v1/responses",
        timeout: float = 60.0,
        allow_user_feedback_pause: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.max_context_chars = max_context_chars
        self.reasoning_effort = reasoning_effort
        self.api_url = api_url
        self.timeout = timeout
        self.allow_user_feedback_pause = allow_user_feedback_pause

    def next_command(self, goal: str, session_context: list[dict[str, object]]) -> ModelDecision:
        payload = self._build_payload(goal, session_context)
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        LOGGER.debug(
            "llm_request_prepared",
            extra={
                "api_url": self.api_url,
                "model": self.model,
                "payload_bytes": len(body),
                "context_events": len(session_context),
                "reasoning_effort": self.reasoning_effort,
                "allow_user_feedback_pause": self.allow_user_feedback_pause,
            },
        )

        req = request.Request(self.api_url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                raw_response = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            body_excerpt = self._read_error_body_excerpt(exc)
            LOGGER.error(
                "llm_request_http_error",
                extra={
                    "api_url": self.api_url,
                    "model": self.model,
                    "http_status": exc.code,
                    "reason": exc.reason,
                    "response_excerpt": body_excerpt,
                },
            )
            details = f"Model request failed with HTTP {exc.code}: {exc.reason}"
            if body_excerpt:
                details = f"{details}. Response body: {body_excerpt}"
            return self._safe_failure_decision(
                details
            )
        except URLError as exc:
            LOGGER.error(
                "llm_request_transport_error",
                extra={
                    "api_url": self.api_url,
                    "model": self.model,
                    "reason": str(exc.reason),
                },
            )
            return self._safe_failure_decision(f"Model request transport error: {exc.reason}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            LOGGER.error(
                "llm_response_parse_error",
                extra={
                    "api_url": self.api_url,
                    "model": self.model,
                    "error": str(exc),
                },
            )
            return self._safe_failure_decision(f"Model response parsing error: {exc}")

        raw = self._coerce_object_dict(raw_response)
        if raw is None:
            return self._safe_failure_decision(
                "Model response parsing error: expected top-level object"
            )

        try:
            parsed = self._extract_output_json(raw)
        except json.JSONDecodeError as exc:
            return self._safe_failure_decision(f"Model structured output parsing error: {exc}")
        return self._to_model_decision(parsed)

    def _build_payload(
        self, goal: str, session_context: list[dict[str, object]]
    ) -> dict[str, object]:
        schema_properties: dict[str, object] = {
            "command": {"type": ["string", "null"]},
            "notes": {"type": ["string", "null"]},
            "complete": {"type": "boolean"},
            "phase": {"type": "string", "enum": sorted(VALID_PHASES)},
            "expected_outcome": {"type": ["string", "null"]},
            "verification_command": {"type": ["string", "null"]},
            "risk_level": {"type": ["string", "null"], "enum": [*sorted(VALID_RISK_LEVELS), None]},
        }
        required_keys = [
            "command",
            "complete",
            "notes",
            "phase",
            "expected_outcome",
            "verification_command",
            "risk_level",
        ]
        if self.allow_user_feedback_pause:
            schema_properties["ask_user"] = {"type": "boolean"}
            schema_properties["user_question"] = {"type": ["string", "null"]}
            required_keys.extend(["ask_user", "user_question"])

        input_messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        if self.allow_user_feedback_pause:
            input_messages.append(
                {
                    "role": "system",
                    "content": (
                        "You may set ask_user=true and provide user_question only when a"
                        " critical missing fact blocks safe progress. Ask only one concise"
                        " question, set command to null, and set complete to false."
                    ),
                }
            )
        input_messages.append(
                {
                    "role": "user",
                    "content": self._build_user_message(
                        goal,
                        session_context,
                        max_context_chars=self.max_context_chars,
                    ),
                }
        )

        payload: dict[str, object] = {
            "model": self.model,
            "input": input_messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "terminal_step",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": schema_properties,
                        "required": required_keys,
                        "additionalProperties": False,
                    },
                }
            },
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        return payload

    @staticmethod
    def _build_user_message(
        goal: str,
        session_context: list[dict[str, object]],
        *,
        max_context_chars: int,
    ) -> str:
        """Build a clear prompt containing goal and serialized context."""
        context_json = LLMClient._serialize_context_with_limit(
            session_context,
            max_context_chars=max_context_chars,
        )
        return (
            "User goal:\n"
            f"{goal}\n\n"
            "Session context (ordered oldest to newest):\n"
            f"{context_json}\n\n"
            "Use both the goal and context to decide the next safest, most useful step."
            " Set notes to a concise hint that explains what is happening now, what"
            " just happened, and what I will do next for the immediate step; avoid"
            " discussing unrelated internal mistakes."
        )

    @staticmethod
    def _serialize_context_with_limit(
        session_context: list[dict[str, object]],
        *,
        max_context_chars: int,
    ) -> str:
        if max_context_chars <= 0 or not session_context:
            return "[]"

        selected: list[dict[str, object]] = []
        for event in reversed(session_context):
            candidate = [event, *selected]
            serialized = json.dumps(candidate, indent=2, ensure_ascii=False)
            if len(serialized) > max_context_chars:
                break
            selected = candidate

        return json.dumps(selected, indent=2, ensure_ascii=False)

    @staticmethod
    def _coerce_object_dict(value: object) -> dict[str, object] | None:
        if not isinstance(value, dict):
            return None
        return {str(key): raw_value for key, raw_value in value.items()}

    @classmethod
    def _extract_output_json(cls, payload: dict[str, object]) -> dict[str, object]:
        output_items = payload.get("output")
        if not isinstance(output_items, list):
            return {"command": None, "notes": "No structured output returned", "complete": True}

        for item in output_items:
            item_object = cls._coerce_object_dict(item)
            if item_object is None:
                continue
            content_items = item_object.get("content")
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                content_object = cls._coerce_object_dict(content)
                if content_object is None:
                    continue
                content_type = content_object.get("type")
                content_text = content_object.get("text")
                if content_type == "output_text" and isinstance(content_text, str):
                    parsed = cls._coerce_object_dict(json.loads(content_text))
                    if parsed is not None:
                        return parsed
                    break
        return {"command": None, "notes": "No structured output returned", "complete": True}

    @staticmethod
    def _safe_failure_decision(notes: str) -> ModelDecision:
        return ModelDecision(command=None, notes=notes, complete=False)

    @staticmethod
    def _read_error_body_excerpt(exc: HTTPError, *, max_chars: int = 500) -> str | None:
        if exc.fp is None:
            return None
        try:
            raw = exc.read()
        except OSError:
            return None

        if not raw:
            return None

        excerpt = raw.decode("utf-8", errors="replace").replace("\n", " ").strip()
        if len(excerpt) > max_chars:
            return f"{excerpt[:max_chars]}..."
        return excerpt

    @staticmethod
    def _to_model_decision(parsed: dict[str, object]) -> ModelDecision:
        command = parsed.get("command")
        notes = parsed.get("notes")
        complete = parsed.get("complete", False)
        ask_user = parsed.get("ask_user", False)
        user_question = parsed.get("user_question")
        phase = parsed.get("phase", "analysis")
        expected_outcome = parsed.get("expected_outcome")
        verification_command = parsed.get("verification_command")
        risk_level = parsed.get("risk_level")

        if command is not None and not isinstance(command, str):
            command = None
        if notes is not None and not isinstance(notes, str):
            notes = "Model returned a non-string notes value"
        if not isinstance(complete, bool):
            complete = True
            notes = "Model returned a non-boolean complete value"
        if not isinstance(ask_user, bool):
            ask_user = False
        if user_question is not None and not isinstance(user_question, str):
            user_question = None
        if not isinstance(phase, str) or phase not in VALID_PHASES:
            phase = "analysis"
        normalized_phase = cast(DecisionPhase, phase)
        if expected_outcome is not None and not isinstance(expected_outcome, str):
            expected_outcome = None
        if verification_command is not None and not isinstance(verification_command, str):
            verification_command = None
        if not isinstance(risk_level, str) or risk_level not in VALID_RISK_LEVELS:
            risk_level = None
        normalized_risk_level = cast(RiskLevel | None, risk_level)

        return ModelDecision(
            command=command,
            notes=notes,
            complete=complete,
            ask_user=ask_user,
            user_question=user_question,
            phase=normalized_phase,
            expected_outcome=expected_outcome,
            verification_command=verification_command,
            risk_level=normalized_risk_level,
        )
