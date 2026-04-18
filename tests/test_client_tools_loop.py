import asyncio

from ollama_flex import OllamaToolkit, ToolRegistry


class DummyClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        if kwargs.get("stream"):
            if self.calls == 1:
                return iter(
                    [
                        {"message": {"role": "assistant", "thinking": "思考中"}},
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "add",
                                            "arguments": '{"a": 2, "b": 3}',
                                        }
                                    }
                                ],
                            },
                            "done": True,
                        },
                    ]
                )

            return iter(
                [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "答案是 5",
                        },
                        "done": True,
                    }
                ]
            )

        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            }
                        }
                    ],
                }
            }

        return {
            "message": {
                "role": "assistant",
                "content": "答案是 5",
            }
        }


class DummyAsyncClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "mul_async",
                                "arguments": {"a": 6, "b": 7},
                            }
                        }
                    ],
                }
            }
        return {"message": {"role": "assistant", "content": "42"}}


def test_chat_with_tools_runs_registered_function():
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        return a + b

    kit = OllamaToolkit(client=DummyClient(), async_client=DummyAsyncClient())
    result = kit.chat_with_tools(registry=registry, prompt="2+3=?")

    assert result.tool_calls
    assert result.tool_calls[0].output == 5
    assert result.response["message"]["content"] == "答案是 5"

    tool_messages = [msg for msg in result.messages if isinstance(msg, dict) and msg.get("role") == "tool"]
    assert tool_messages
    assert tool_messages[0]["tool_name"] == "add"
    assert tool_messages[0]["name"] == "add"


def test_chat_with_tools_stream_collects_thinking_and_tool_calls():
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        return a + b

    kit = OllamaToolkit(client=DummyClient(), async_client=DummyAsyncClient())
    result = kit.chat_with_tools(registry=registry, prompt="2+3=?", stream=True, max_rounds=3)

    assert result.tool_calls
    assert result.response["message"]["content"] == "答案是 5"
    assert any(isinstance(m, dict) and m.get("role") == "tool" for m in result.messages)


def test_achat_with_tools_supports_async_executor():
    registry = ToolRegistry()

    @registry.register(name="mul_async")
    async def mul_async(a: int, b: int) -> int:
        return a * b

    kit = OllamaToolkit(client=DummyClient(), async_client=DummyAsyncClient())
    result = asyncio.run(kit.achat_with_tools(registry=registry, prompt="6*7=?"))

    assert result.tool_calls
    assert result.tool_calls[0].output == 42
    assert result.response["message"]["content"] == "42"
