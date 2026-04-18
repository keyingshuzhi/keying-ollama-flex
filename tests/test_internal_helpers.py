from ollama_flex._internal.history import trim_messages
from ollama_flex._internal.messages import build_messages, tool_message


def test_trim_messages_keeps_leading_system_messages():
    messages = [
        {"role": "system", "content": "s1"},
        {"role": "system", "content": "s2"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]

    trimmed = trim_messages(messages, 3)
    assert trimmed == [
        {"role": "system", "content": "s1"},
        {"role": "system", "content": "s2"},
        {"role": "user", "content": "u2"},
    ]


def test_build_messages_requires_messages_or_prompt():
    try:
        build_messages(None, None, None)
    except ValueError:
        pass
    else:
        raise AssertionError("build_messages should raise ValueError when both messages and prompt are missing")


def test_tool_message_contains_compatibility_fields():
    message = tool_message("math.add", {"result": 3})
    assert message["role"] == "tool"
    assert message["tool_name"] == "math.add"
    assert message["name"] == "math.add"
    assert "result" in message["content"]
