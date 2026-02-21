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
        system_prompt="custom prompt",
        reasoning_effort="medium",
    )

    payload = client._build_payload("test goal", [])

    assert payload["reasoning"] == {"effort": "medium"}


def test_payload_omits_reasoning_effort_when_unset() -> None:
    client = LLMClient(api_key=None, model="gpt-4.1-mini", system_prompt="custom prompt")

    payload = client._build_payload("test goal", [])

    assert "reasoning" not in payload


def test_payload_uses_custom_system_prompt() -> None:
    client = LLMClient(api_key=None, model="gpt-5.2", system_prompt="be careful")

    payload = client._build_payload("test goal", [])

    assert payload["input"][0]["content"] == "be careful"


def test_payload_includes_runtime_context_when_provided() -> None:
    client = LLMClient(
        api_key=None,
        model="gpt-5.2",
        system_prompt="be careful",
        runtime_context="Runtime environment context",
    )

    payload = client._build_payload("test goal", [])

    assert payload["input"][1]["content"] == "Runtime environment context"


def test_payload_exposes_user_feedback_pause_controls_when_enabled() -> None:
    client = LLMClient(
        api_key=None,
        model="gpt-5.2",
        system_prompt="be careful",
        allow_user_feedback_pause=True,
    )

    payload = client._build_payload("test goal", [])

    schema = payload["text"]["format"]["schema"]
    assert "ask_user" in schema["properties"]
    assert "user_question" in schema["properties"]
    assert "ask_user" in schema["required"]
    assert "critical missing fact" in payload["input"][1]["content"]


def test_payload_hides_user_feedback_pause_controls_when_disabled() -> None:
    client = LLMClient(api_key=None, model="gpt-5.2", system_prompt="be careful")

    payload = client._build_payload("test goal", [])

    schema = payload["text"]["format"]["schema"]
    assert "ask_user" not in schema["properties"]
    assert "user_question" not in schema["properties"]
    assert len(payload["input"]) == 2


def test_payload_user_message_formats_goal_and_context() -> None:
    client = LLMClient(api_key=None, model="gpt-5.2", system_prompt="be careful")

    payload = client._build_payload(
        "find large files",
        [{"type": "step_budget", "current_step": 1, "max_steps": 20}],
    )

    user_message = payload["input"][-1]["content"]
    assert "User goal:" in user_message
    assert "find large files" in user_message
    assert "Session context (ordered oldest to newest):" in user_message
    assert '"type": "step_budget"' in user_message



def test_payload_includes_phase_metadata_schema() -> None:
    client = LLMClient(api_key=None, model="gpt-5.2", system_prompt="be careful")

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
    client = LLMClient(api_key=None, model="gpt-5.2", system_prompt="be careful")

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
    assert decision.complete is True
    assert "HTTP 503" in (decision.notes or "")


def test_next_command_returns_safe_decision_on_invalid_json_response(monkeypatch) -> None:
    client = LLMClient(api_key=None, model="gpt-5.2", system_prompt="be careful")

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
    assert decision.complete is True
    assert "parsing error" in (decision.notes or "")


def test_next_command_returns_safe_decision_when_missing_output_text(monkeypatch) -> None:
    client = LLMClient(api_key=None, model="gpt-5.2", system_prompt="be careful")

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
    assert decision.complete is True
    assert decision.notes == "No structured output returned"
