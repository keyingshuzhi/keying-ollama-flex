from pydantic import BaseModel

from ollama_flex import OllamaToolkit


class Friend(BaseModel):
    name: str
    age: int
    is_available: bool


class FriendList(BaseModel):
    friends: list[Friend]


kit = OllamaToolkit(default_model="deepseek-r1:7b")
result = kit.chat_structured(
    FriendList,
    prompt=(
        "I have two friends. The first is Ollama 22 years old busy saving the world, "
        "and the second is Alonso 23 years old and wants to hang out. "
        "Return a list of friends in JSON format"
    ),
    options={"temperature": 0},
    retries=2,
)

print(result)
