from __future__ import annotations

from types import SimpleNamespace

from ollama_flex import OllamaRecipes, OllamaToolkit


class _Size:
    def __init__(self, real: int) -> None:
        self.real = real


class DummyClient:
    def __init__(self) -> None:
        self.chat_calls: list[dict] = []

    def close(self) -> None:
        return None

    def list(self):
        model = SimpleNamespace(
            model="deepseek-r1:7b",
            size=_Size(1024 * 1024 * 512),
            details=SimpleNamespace(
                format="gguf",
                family="deepseek",
                parameter_size="7B",
                quantization_level="Q4_K_M",
            ),
        )
        return SimpleNamespace(models=[model])

    def ps(self):
        proc = SimpleNamespace(
            model="llama3.2",
            digest="sha256:abc",
            expires_at="2099-01-01T00:00:00Z",
            size=123,
            size_vram=45,
            details={"family": "llama"},
        )
        return SimpleNamespace(models=[proc])

    def pull(self, model: str, *, insecure: bool = False, stream: bool = False, **kwargs):
        assert model == "llama3.2"
        assert stream is True
        return iter(
            [
                {"status": "pulling manifest"},
                {"status": "pulling manifest"},
                {"digest": "sha256:abc", "total": 10, "completed": 5},
                {"digest": "sha256:abc", "total": 10, "completed": 10},
            ]
        )

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        content = kwargs["messages"][-1]["content"]
        return {"message": {"role": "assistant", "content": f"echo:{content}"}}


class DummyAsyncClient:
    async def close(self) -> None:
        return None


def _make_recipes() -> tuple[OllamaRecipes, DummyClient]:
    client = DummyClient()
    toolkit = OllamaToolkit(client=client, async_client=DummyAsyncClient())
    return OllamaRecipes(toolkit), client


def test_list_models_text():
    recipes, _ = _make_recipes()
    text = recipes.list_models_text()

    assert "Name: deepseek-r1:7b" in text
    assert "Size (MB): 512.00" in text
    assert "Format: gguf" in text
    assert "Quantization Level: Q4_K_M" in text


def test_running_models_text():
    recipes, _ = _make_recipes()
    text = recipes.running_models_text()

    assert "Model: llama3.2" in text
    assert "Digest: sha256:abc" in text
    assert "Size vram: 45" in text


def test_pull_model_with_progress_without_tqdm():
    recipes, _ = _make_recipes()
    logs: list[str] = []

    report = recipes.pull_model_with_progress(
        "llama3.2",
        enable_progress_bar=False,
        output_fn=logs.append,
    )

    assert report.model == "llama3.2"
    assert report.statuses == ["pulling manifest"]
    assert report.digests == ["sha256:abc"]
    assert report.events == 4
    assert logs == ["pulling manifest"]


def test_warmup_model():
    recipes, _ = _make_recipes()
    text = recipes.warmup_model("llama3.2", prompt="hello")
    assert text == "echo:hello"


def test_chat_with_history_cli():
    recipes, client = _make_recipes()
    outputs: list[str] = []
    user_inputs = iter(["first", "exit"])

    history = recipes.chat_with_history_cli(
        model="llama3.2",
        input_fn=lambda _: next(user_inputs),
        output_fn=outputs.append,
    )

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "first"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "echo:first"
    assert outputs[-1] == "Goodbye!"
    assert len(client.chat_calls) == 1

