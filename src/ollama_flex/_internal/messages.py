from __future__ import annotations

import json
from typing import Any, Sequence

from ..types import Message


def user_message(content: str, images: Sequence[str | bytes] | None = None) -> Message:
    message: Message = {"role": "user", "content": content}
    if images:
        message["images"] = list(images)
    return message


def system_message(content: str) -> Message:
    return {"role": "system", "content": content}


def tool_message(tool_name: str, content: Any) -> Message:
    return {
        "role": "tool",
        "tool_name": tool_name,
        "name": tool_name,
        "content": stringify_output(content),
    }


def build_messages(
    messages: Sequence[Any] | None,
    prompt: str | None,
    images: Sequence[str | bytes] | None,
) -> list[Any]:
    payload_messages = list(messages) if messages is not None else []
    if prompt is not None:
        payload_messages.append(user_message(prompt, images=images))
    if not payload_messages:
        raise ValueError("messages or prompt must be provided")
    return payload_messages


def stringify_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False)
    except TypeError:
        return str(output)

