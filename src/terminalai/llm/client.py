"""Thin model client that requests the next shell command."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import request


@dataclass(slots=True)
class ModelDecision:
    """Structured model response used by the agent loop."""

    command: str | None
    notes: str | None = None
    complete: bool = False
    ask_user: bool = False
    user_question: str | None = None


class LLMClient:
    """Small HTTP client for command-oriented model calls."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        system_prompt: str,
        reasoning_effort: str | None = None,
        api_url: str = "https://api.openai.com/v1/responses",
        timeout: float = 60.0,
        allow_user_feedback_pause: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
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

        req = request.Request(self.api_url, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            raw = json.loads(resp.read().decode("utf-8"))

        parsed = self._extract_output_json(raw)
        return ModelDecision(
            command=parsed.get("command"),
            notes=parsed.get("notes"),
            complete=bool(parsed.get("complete", False)),
            ask_user=bool(parsed.get("ask_user", False)),
            user_question=parsed.get("user_question"),
        )

    def _build_payload(
        self, goal: str, session_context: list[dict[str, object]]
    ) -> dict[str, object]:
        schema_properties: dict[str, object] = {
            "command": {"type": ["string", "null"]},
            "notes": {"type": ["string", "null"]},
            "complete": {"type": "boolean"},
        }
        required_keys = ["command", "complete", "notes"]
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
                "content": json.dumps({"goal": goal, "context": session_context}),
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
    def _extract_output_json(payload: dict[str, object]) -> dict[str, object]:
        for item in payload.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text" and content.get("text"):
                    return json.loads(str(content["text"]))
        return {"command": None, "notes": "No structured output returned", "complete": True}
