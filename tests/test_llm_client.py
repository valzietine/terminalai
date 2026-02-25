import io
from urllib.error import HTTPError

from terminalai.llm.client import LLMClient


def test_extract_output_json() -> None:
    payload = {
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"command":"echo hi","notes":"next","complete":false}',
                    }
                ]
            }
        ]
    }

    parsed = LLMClient._extract_output_json(payload)

    assert parsed["command"] == "echo hi"
    assert parsed["complete"] is False


def test_payload_includes_reasoning_effort_when_configured() -> None:
    client = LLMClient(
        api_key=None,
        model="gpt-5.2-codex",
        reasoning_effort="medium",
    )

    payload = client._build_payload("test goal", [])

    assert payload["reasoning"] == {"effort": "medium"}


def test_payload_omits_reasoning_effort_when_unset() -> None:
    client = LLMClient(api_key=None, model="gpt-4.1-mini")

    payload = client._build_payload("test goal", [])

    assert "reasoning" not in payload


def test_payload_uses_hardcoded_system_prompt() -> None:
    client = LLMClient(api_key=None, model="gpt-5.2")

    payload = client._build_payload("test goal", [])
    system_prompt = payload["input"][0]["content"]

    assert "expert terminal orchestration assistant" in system_prompt
    assert "do not wrap" in system_prompt
    assert "powershell -Command" in system_prompt
    assert "runtime_context.shell_adapter" in system_prompt


def test_payload_includes_shell_specific_guidance_for_cmd_only() -> None:
    client = LLMClient(api_key=None, model="gpt-5.2")

    payload = client._build_payload(
        "test goal",
        [{"type": "runtime_context", "shell_adapter": "cmd"}],
    )
    system_prompt = payload["input"][0]["content"]

    assert "cmd uses double quotes and caret escaping" in system_prompt
    assert "never backslash-escaped quotes like \\\"" in system_prompt
    assert "powershell prefers single quotes for literals" not in system_prompt
    assert "bash uses POSIX quoting" not in system_prompt


def test_payload_includes_shell_specific_guidance_for_powershell_only() -> None:
    client = LLMClient(api_key=None, model="gpt-5.2")

    payload = client._build_payload(
        "test goal",
        [{"type": "runtime_context", "shell_adapter": "powershell"}],
    )
    system_prompt = payload["input"][0]["content"]

    assert "powershell prefers single quotes for literals" in system_prompt
    assert "here-strings for longer scripts" in system_prompt
    assert "cmd uses double quotes and caret escaping" not in system_prompt
    assert "bash uses POSIX quoting" not in system_prompt


def test_payload_includes_shell_specific_guidance_for_bash_only() -> None:
    client = LLMClient(api_key=None, model="gpt-5.2")

    payload = client._build_payload(
        "test goal",
        [{"type": "runtime_context", "shell_adapter": "bash"}],
    )
    system_prompt = payload["input"][0]["content"]

    assert "bash uses POSIX quoting" in system_prompt
    assert "cmd uses double quotes and caret escaping" not in system_prompt
    assert "powershell prefers single quotes for literals" not in system_prompt


def test_payload_exposes_user_feedback_pause_controls_when_enabled() -> None:
    client = LLMClient(
        api_key=None,
        model="gpt-5.2",
        allow_user_feedback_pause=True,
    )

    payload = client._build_payload("test goal", [])

    schema = payload["text"]["format"]["schema"]
    assert "ask_user" in schema["properties"]
    assert "user_question" in schema["properties"]
    assert "ask_user" in schema["required"]
    assert "Include ask_user (boolean) and user_question" in payload["input"][0]["content"]


def test_payload_hides_user_feedback_pause_controls_when_disabled() -> None:
    client = LLMClient(api_key=None, model="gpt-5.2")

    payload = client._build_payload("test goal", [])

    schema = payload["text"]["format"]["schema"]
    assert "ask_user" not in schema["properties"]
    assert "user_question" not in schema["properties"]
    assert "Include ask_user (boolean) and user_question" not in payload["input"][0]["content"]
    assert len(payload["input"]) == 2


def test_payload_user_message_formats_goal_and_context() -> None:
    client = LLMClient(api_key=None, model="gpt-5.2")

    payload = client._build_payload(
        "find large files",
        [{"type": "step_budget", "current_step": 1, "max_steps": 20}],
    )

    user_message = payload["input"][-1]["content"]
    assert "User goal:" in user_message
    assert "find large files" in user_message
    assert "Session context (ordered oldest to newest):" in user_message
    assert '"type": "step_budget"' in user_message
    assert "what is happening now" in user_message
    assert "what just happened" in user_message
    assert "what I will do next" in user_message
    assert "Read runtime_context.shell_adapter first" in user_message


def test_payload_trims_session_context_to_max_chars() -> None:
    client = LLMClient(
        api_key=None,
        model="gpt-5.2",
        max_context_chars=120,
    )

    payload = client._build_payload(
        "test",
        [
            {"type": "first", "message": "a" * 120},
            {"type": "second", "message": "b" * 40},
        ],
    )

    user_message = payload["input"][-1]["content"]
    assert '"type": "first"' not in user_message
    assert '"type": "second"' in user_message


def test_payload_empty_context_when_limit_non_positive() -> None:
    client = LLMClient(
        api_key=None,
        model="gpt-5.2",
        max_context_chars=0,
    )

    payload = client._build_payload("test", [{"type": "step_budget", "current_step": 1}])

    user_message = payload["input"][-1]["content"]
    assert "Session context (ordered oldest to newest):\n[]" in user_message


def test_payload_includes_phase_metadata_schema() -> None:
    client = LLMClient(api_key=None, model="gpt-5.2")

    payload = client._build_payload("test goal", [])

    schema = payload["text"]["format"]["schema"]
    assert schema["properties"]["phase"]["enum"] == [
        "analysis",
        "completion",
        "mutation",
        "verification",
    ]
    assert "phase" in schema["required"]
    assert "expected_outcome" in schema["required"]
    assert "verification_command" in schema["required"]
    assert "risk_level" in schema["properties"]
    assert "risk_level" in schema["required"]
    assert set(schema["required"]) == set(schema["properties"].keys())


def test_to_model_decision_coerces_invalid_phase_metadata() -> None:
    decision = LLMClient._to_model_decision(
        {
            "command": "echo hi",
            "notes": "next",
            "complete": False,
            "phase": "invalid",
            "expected_outcome": 42,
            "verification_command": ["pytest"],
            "risk_level": "critical",
        }
    )

    assert decision.phase == "analysis"
    assert decision.expected_outcome is None
    assert decision.verification_command is None
    assert decision.risk_level is None


def test_to_model_decision_preserves_valid_phase_metadata() -> None:
    decision = LLMClient._to_model_decision(
        {
            "command": "pytest -q",
            "notes": "verify",
            "complete": False,
            "phase": "verification",
            "expected_outcome": "tests pass",
            "verification_command": "pytest -q",
            "risk_level": "low",
        }
    )

    assert decision.phase == "verification"
    assert decision.expected_outcome == "tests pass"
    assert decision.verification_command == "pytest -q"
    assert decision.risk_level == "low"


def test_next_command_returns_safe_decision_on_http_error(monkeypatch) -> None:
    client = LLMClient(api_key=None, model="gpt-5.2")

    def fake_urlopen(*_args, **_kwargs):
        raise HTTPError(
            url="https://example.com",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("terminalai.llm.client.request.urlopen", fake_urlopen)

    decision = client.next_command("test", [])

    assert decision.command is None
    assert decision.complete is False
    assert "HTTP 503" in (decision.notes or "")


def test_next_command_returns_safe_decision_on_invalid_json_response(monkeypatch) -> None:
    client = LLMClient(api_key=None, model="gpt-5.2")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return b"not-json"

    monkeypatch.setattr("terminalai.llm.client.request.urlopen", lambda *_a, **_k: FakeResponse())

    decision = client.next_command("test", [])

    assert decision.command is None
    assert decision.complete is False
    assert "parsing error" in (decision.notes or "")


def test_next_command_returns_safe_decision_when_missing_output_text(monkeypatch) -> None:
    client = LLMClient(api_key=None, model="gpt-5.2")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return b'{"output":[{"content":[{"type":"reasoning","text":"ignored"}]}]}'

    monkeypatch.setattr("terminalai.llm.client.request.urlopen", lambda *_a, **_k: FakeResponse())

    decision = client.next_command("test", [])

    assert decision.command is None
    assert decision.complete is False
    assert decision.notes == "No structured output returned"


def test_extract_output_json_marks_incomplete_when_output_is_missing() -> None:
    parsed = LLMClient._extract_output_json({"id": "resp_123"})

    assert parsed["command"] is None
    assert parsed["complete"] is False
    assert parsed["notes"] == "No structured output returned"


def test_next_command_http_error_includes_response_excerpt(monkeypatch) -> None:
    client = LLMClient(api_key=None, model="gpt-5.2")

    class FakeHTTPError(HTTPError):
        def __init__(self):
            super().__init__(
                url="https://example.com",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=io.BytesIO(b'{"error":{"message":"invalid schema"}}'),
            )

    def fake_urlopen(*_args, **_kwargs):
        raise FakeHTTPError()

    monkeypatch.setattr("terminalai.llm.client.request.urlopen", fake_urlopen)

    decision = client.next_command("test", [])

    assert "HTTP 400" in (decision.notes or "")
    assert "invalid schema" in (decision.notes or "")


def test_next_command_returns_safe_decision_on_timeout(monkeypatch) -> None:
    client = LLMClient(api_key=None, model="gpt-5.2", timeout=12.5)

    def fake_urlopen(*_args, **_kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("terminalai.llm.client.request.urlopen", fake_urlopen)

    decision = client.next_command("test", [])

    assert decision.command is None
    assert decision.complete is False
    assert decision.notes == "Model request timed out after 12.5s"
