from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, Sequence

from ._internal.extractors import extract_response_content, extract_response_message
from ._internal.history import leading_system_messages, trim_messages
from .client import OllamaToolkit
from .tools import ToolRegistry
from .types import ChatStreamEvent, Message, ToolChatResult


class ChatSession:
    """Stateful chat session with optional history window control."""

    def __init__(
        self,
        toolkit: OllamaToolkit,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        messages: Sequence[Any] | None = None,
        max_messages: int | None = None,
    ) -> None:
        self.toolkit = toolkit
        self.model = model
        self.max_messages = max_messages
        self.messages: list[Any] = list(messages or [])

        if system_prompt:
            self.messages.insert(0, {"role": "system", "content": system_prompt})

        self._apply_max_messages()

    def ask(
        self,
        content: str,
        *,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        user_message = self.toolkit.user_message(content, images=images)
        response = self.toolkit.chat(
            messages=[*self.messages, user_message],
            model=model or self.model,
            **kwargs,
        )
        assistant_message = extract_response_message(response)

        self.messages.extend([user_message, assistant_message])
        self._apply_max_messages()
        return response

    async def aask(
        self,
        content: str,
        *,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        user_message = self.toolkit.user_message(content, images=images)
        response = await self.toolkit.achat(
            messages=[*self.messages, user_message],
            model=model or self.model,
            **kwargs,
        )
        assistant_message = extract_response_message(response)

        self.messages.extend([user_message, assistant_message])
        self._apply_max_messages()
        return response

    def ask_text(self, content: str, **kwargs: Any) -> str:
        response = self.ask(content, **kwargs)
        return extract_response_content(response)

    async def aask_text(self, content: str, **kwargs: Any) -> str:
        response = await self.aask(content, **kwargs)
        return extract_response_content(response)

    def ask_structured(self, schema: type[Any], content: str, **kwargs: Any) -> Any:
        user_message = self.toolkit.user_message(content, images=kwargs.pop("images", None))
        result = self.toolkit.chat_structured(
            schema,
            messages=[*self.messages, user_message],
            model=kwargs.pop("model", self.model),
            **kwargs,
        )
        self.messages.append(user_message)
        self.messages.append({"role": "assistant", "content": result.model_dump_json(ensure_ascii=False)})
        self._apply_max_messages()
        return result

    async def aask_structured(self, schema: type[Any], content: str, **kwargs: Any) -> Any:
        user_message = self.toolkit.user_message(content, images=kwargs.pop("images", None))
        result = await self.toolkit.achat_structured(
            schema,
            messages=[*self.messages, user_message],
            model=kwargs.pop("model", self.model),
            **kwargs,
        )
        self.messages.append(user_message)
        self.messages.append({"role": "assistant", "content": result.model_dump_json(ensure_ascii=False)})
        self._apply_max_messages()
        return result

    def ask_stream_events(
        self,
        content: str,
        *,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatStreamEvent]:
        user_message = self.toolkit.user_message(content, images=images)

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[Any] = []

        for event in self.toolkit.stream_chat_events(
            messages=[*self.messages, user_message],
            model=model or self.model,
            **kwargs,
        ):
            if event.content:
                content_parts.append(event.content)
            if event.thinking:
                thinking_parts.append(event.thinking)
            if event.tool_calls:
                tool_calls.extend(event.tool_calls)
            yield event

        assistant_message: Message = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if thinking_parts:
            assistant_message["thinking"] = "".join(thinking_parts)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        self.messages.extend([user_message, assistant_message])
        self._apply_max_messages()

    async def aask_stream_events(
        self,
        content: str,
        *,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatStreamEvent]:
        user_message = self.toolkit.user_message(content, images=images)

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[Any] = []

        async for event in self.toolkit.astream_chat_events(
            messages=[*self.messages, user_message],
            model=model or self.model,
            **kwargs,
        ):
            if event.content:
                content_parts.append(event.content)
            if event.thinking:
                thinking_parts.append(event.thinking)
            if event.tool_calls:
                tool_calls.extend(event.tool_calls)
            yield event

        assistant_message: Message = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if thinking_parts:
            assistant_message["thinking"] = "".join(thinking_parts)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        self.messages.extend([user_message, assistant_message])
        self._apply_max_messages()

    def ask_stream(
        self,
        content: str,
        *,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        for event in self.ask_stream_events(content, images=images, model=model, **kwargs):
            if event.content:
                yield event.content

    async def aask_stream(
        self,
        content: str,
        *,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        async for event in self.aask_stream_events(content, images=images, model=model, **kwargs):
            if event.content:
                yield event.content

    def ask_with_tools(
        self,
        content: str,
        *,
        registry: ToolRegistry,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> ToolChatResult:
        user_message = self.toolkit.user_message(content, images=images)
        result = self.toolkit.chat_with_tools(
            registry=registry,
            messages=[*self.messages, user_message],
            model=model or self.model,
            **kwargs,
        )
        self.messages = list(result.messages)
        self._apply_max_messages()
        return result

    async def aask_with_tools(
        self,
        content: str,
        *,
        registry: ToolRegistry,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> ToolChatResult:
        user_message = self.toolkit.user_message(content, images=images)
        result = await self.toolkit.achat_with_tools(
            registry=registry,
            messages=[*self.messages, user_message],
            model=model or self.model,
            **kwargs,
        )
        self.messages = list(result.messages)
        self._apply_max_messages()
        return result

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self._apply_max_messages()

    def extend_messages(self, messages: Sequence[Message]) -> None:
        self.messages.extend(messages)
        self._apply_max_messages()

    def export_messages(self) -> list[Any]:
        return list(self.messages)

    def reset(self, *, keep_system: bool = True) -> None:
        if keep_system:
            self.messages = leading_system_messages(self.messages)
        else:
            self.messages.clear()

    def _apply_max_messages(self) -> None:
        self.messages = trim_messages(self.messages, self.max_messages)
