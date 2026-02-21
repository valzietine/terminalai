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
