import asyncio

from pydantic import BaseModel

from ollama_flex import OllamaToolkit


class Person(BaseModel):
    name: str
    age: int


class DummyClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {"message": {"role": "assistant", "content": "not json"}}
        return {"message": {"role": "assistant", "content": '{"name":"Alice","age":18}'}}


class DummyAsyncClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {"message": {"role": "assistant", "content": "not json"}}
        return {"message": {"role": "assistant", "content": '{"name":"Bob","age":20}'}}


def test_chat_structured_with_retry():
    kit = OllamaToolkit(client=DummyClient(), async_client=DummyAsyncClient())
    result = kit.chat_structured(Person, prompt="give json", retries=1)

    assert result.name == "Alice"
    assert result.age == 18


def test_achat_structured_with_retry():
    kit = OllamaToolkit(client=DummyClient(), async_client=DummyAsyncClient())
    result = asyncio.run(kit.achat_structured(Person, prompt="give json", retries=1))

    assert result.name == "Bob"
    assert result.age == 20
