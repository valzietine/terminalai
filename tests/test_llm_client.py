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
