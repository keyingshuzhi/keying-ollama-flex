from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ._internal.extractors import extract_response_content
from .client import OllamaToolkit


OutputFn = Callable[[str], None]
InputFn = Callable[[str], str]


@dataclass(slots=True)
class PullReport:
    """Summary for pull progress workflow."""

    model: str
    statuses: list[str] = field(default_factory=list)
    digests: list[str] = field(default_factory=list)
    events: int = 0


class OllamaRecipes:
    """Script-level reusable recipes built on top of OllamaToolkit."""

    def __init__(self, toolkit: OllamaToolkit) -> None:
        self.toolkit = toolkit

    def pull_model_with_progress(
        self,
        model: str,
        *,
        insecure: bool = False,
        enable_progress_bar: bool = True,
        output_fn: OutputFn = print,
        **kwargs: Any,
    ) -> PullReport:
        """Pull a model and show digest-level progress, similar to studying scripts."""

        progress_bar_factory = _resolve_tqdm() if enable_progress_bar else None
        current_digest = ""
        seen_status: set[str] = set()
        statuses: list[str] = []
        bars: dict[str, Any] = {}
        digests_seen: set[str] = set()
        events = 0

        try:
            stream = self.toolkit.pull_model(model, insecure=insecure, stream=True, **kwargs)
            for progress in stream:
                events += 1
                digest = str(_pick(progress, "digest", "") or "")

                if digest and digest not in digests_seen:
                    digests_seen.add(digest)

                if digest != current_digest and current_digest in bars:
                    bars[current_digest].close()

                if not digest:
                    status = str(_pick(progress, "status", "") or "").strip()
                    if status and status not in seen_status:
                        seen_status.add(status)
                        statuses.append(status)
                        output_fn(status)
                    continue

                total = _as_int(_pick(progress, "total"))
                if progress_bar_factory is not None and digest not in bars and total > 0:
                    suffix = digest[7:19] if len(digest) >= 19 else digest
                    bars[digest] = progress_bar_factory(
                        total=total,
                        desc=f"pulling {suffix}",
                        unit="B",
                        unit_scale=True,
                    )

                completed = _as_int(_pick(progress, "completed"))
                if digest in bars and completed > 0:
                    bar = bars[digest]
                    delta = completed - int(getattr(bar, "n", 0))
                    if delta > 0:
                        bar.update(delta)

                current_digest = digest
        finally:
            for bar in bars.values():
                bar.close()

        return PullReport(
            model=model,
            statuses=statuses,
            digests=sorted(digests_seen),
            events=events,
        )

    def list_models_text(self) -> str:
        """Format local model list as readable plain text."""
        response = self.toolkit.list_models()
        models = list(_pick(response, "models", []) or [])
        if not models:
            return "No local models found."

        lines: list[str] = []
        for model in models:
            name = str(_pick(model, "model", ""))
            lines.append(f"Name: {name}")

            size_bytes = _size_to_int(_pick(model, "size"))
            lines.append(f"  Size (MB): {size_bytes / 1024 / 1024:.2f}")

            details = _pick(model, "details")
            if details is not None:
                fmt = _pick(details, "format")
                family = _pick(details, "family")
                parameter_size = _pick(details, "parameter_size")
                quantization_level = _pick(details, "quantization_level")
                if fmt is not None:
                    lines.append(f"  Format: {fmt}")
                if family is not None:
                    lines.append(f"  Family: {family}")
                if parameter_size is not None:
                    lines.append(f"  Parameter Size: {parameter_size}")
                if quantization_level is not None:
                    lines.append(f"  Quantization Level: {quantization_level}")

            lines.append("")

        return "\n".join(lines).rstrip()

    def running_models_text(self) -> str:
        """Format currently running model process info as readable text."""
        response = self.toolkit.running_models()
        models = list(_pick(response, "models", []) or [])
        if not models:
            return "No running models."

        lines: list[str] = []
        for model in models:
            lines.append(f"Model: {_pick(model, 'model')}")
            lines.append(f"  Digest: {_pick(model, 'digest')}")
            lines.append(f"  Expires at: {_pick(model, 'expires_at')}")
            lines.append(f"  Size: {_pick(model, 'size')}")
            lines.append(f"  Size vram: {_pick(model, 'size_vram')}")
            lines.append(f"  Details: {_pick(model, 'details')}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def warmup_model(self, model: str, *, prompt: str = "Why is the sky blue?", **kwargs: Any) -> str:
        """Run a warm-up chat call to ensure the model is loaded."""
        response = self.toolkit.chat(model=model, prompt=prompt, **kwargs)
        return extract_response_content(response)

    def chat_with_history_cli(
        self,
        *,
        model: str | None = None,
        messages: Sequence[dict[str, Any]] | None = None,
        prompt_text: str = "Chat with history: ",
        exit_words: Iterable[str] = ("exit", "quit"),
        goodbye_text: str = "Goodbye!",
        input_fn: InputFn = input,
        output_fn: OutputFn = print,
        **chat_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Interactive history chat loop (CLI-friendly)."""

        history = list(messages or [])
        exits = {word.lower() for word in exit_words}

        while True:
            user_input = input_fn(prompt_text)
            if user_input.strip().lower() in exits:
                output_fn(goodbye_text)
                break

            response = self.toolkit.chat(
                messages=[*history, {"role": "user", "content": user_input}],
                model=model,
                **chat_kwargs,
            )
            assistant = extract_response_content(response)
            output_fn(assistant)
            output_fn("")

            history.extend(
                [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": assistant},
                ]
            )

        return history


def _pick(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _size_to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)

    real_value = getattr(value, "real", None)
    if real_value is not None:
        try:
            return int(real_value)
        except (TypeError, ValueError):
            return 0

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _resolve_tqdm() -> Any | None:
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm

