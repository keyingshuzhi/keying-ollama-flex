from __future__ import annotations

from typing import Any


def extract_response_message(response: Any) -> Any:
    if isinstance(response, dict):
        return response.get("message", {})
    return getattr(response, "message", {})


def extract_response_content(response: Any) -> str:
    return extract_message_content(extract_response_message(response))


def extract_message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def extract_message_thinking(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("thinking") or "")
    return str(getattr(message, "thinking", "") or "")


def extract_tool_calls(message: Any) -> list[Any]:
    if isinstance(message, dict):
        return list(message.get("tool_calls") or [])
    return list(getattr(message, "tool_calls", None) or [])


def extract_done(response_or_chunk: Any) -> bool | None:
    if isinstance(response_or_chunk, dict):
        return response_or_chunk.get("done")
    return getattr(response_or_chunk, "done", None)


def extract_generate_text(response_or_chunk: Any) -> str:
    if isinstance(response_or_chunk, dict):
        return str(response_or_chunk.get("response") or "")
    return str(getattr(response_or_chunk, "response", "") or "")


def extract_generate_thinking(response_or_chunk: Any) -> str:
    if isinstance(response_or_chunk, dict):
        return str(response_or_chunk.get("thinking") or "")
    return str(getattr(response_or_chunk, "thinking", "") or "")


def extract_logprobs(response_or_chunk: Any) -> Any:
    if isinstance(response_or_chunk, dict):
        return response_or_chunk.get("logprobs")
    return getattr(response_or_chunk, "logprobs", None)

