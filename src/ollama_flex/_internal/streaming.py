from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from ..types import Message
from .extractors import (
    extract_done,
    extract_message_content,
    extract_message_thinking,
    extract_response_message,
    extract_tool_calls,
)


def collect_streamed_chat_message(stream: Iterator[Any]) -> tuple[Message, Any]:
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[Any] = []
    last_chunk: Any = None

    for chunk in stream:
        last_chunk = chunk
        message = extract_response_message(chunk)
        text = extract_message_content(message)
        thinking = extract_message_thinking(message)

        if text:
            content_parts.append(text)
        if thinking:
            thinking_parts.append(thinking)

        calls = extract_tool_calls(message)
        if calls:
            tool_calls.extend(calls)

    merged_message: Message = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if thinking_parts:
        merged_message["thinking"] = "".join(thinking_parts)
    if tool_calls:
        merged_message["tool_calls"] = tool_calls

    merged_response: dict[str, Any] = {"message": merged_message}
    if last_chunk is not None:
        done = extract_done(last_chunk)
        if done is not None:
            merged_response["done"] = done

    return merged_message, merged_response


async def acollect_streamed_chat_message(stream: AsyncIterator[Any]) -> tuple[Message, Any]:
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[Any] = []
    last_chunk: Any = None

    async for chunk in stream:
        last_chunk = chunk
        message = extract_response_message(chunk)
        text = extract_message_content(message)
        thinking = extract_message_thinking(message)

        if text:
            content_parts.append(text)
        if thinking:
            thinking_parts.append(thinking)

        calls = extract_tool_calls(message)
        if calls:
            tool_calls.extend(calls)

    merged_message: Message = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if thinking_parts:
        merged_message["thinking"] = "".join(thinking_parts)
    if tool_calls:
        merged_message["tool_calls"] = tool_calls

    merged_response: dict[str, Any] = {"message": merged_message}
    if last_chunk is not None:
        done = extract_done(last_chunk)
        if done is not None:
            merged_response["done"] = done

    return merged_message, merged_response

