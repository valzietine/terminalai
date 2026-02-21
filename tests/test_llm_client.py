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
