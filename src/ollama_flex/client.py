from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Literal, Mapping, Sequence, TypeVar

from ollama import AsyncClient, Client
from pydantic import BaseModel, ValidationError

from ._internal.extractors import (
    extract_done,
    extract_generate_text,
    extract_generate_thinking,
    extract_logprobs,
    extract_message_content,
    extract_message_thinking,
    extract_response_content,
    extract_response_message,
    extract_tool_calls,
)
from ._internal.messages import (
    build_messages as build_payload_messages,
    system_message as make_system_message,
    tool_message as make_tool_message,
    user_message as make_user_message,
)
from ._internal.streaming import acollect_streamed_chat_message, collect_streamed_chat_message
from .config import OllamaConfig
from .tools import ToolRegistry
from .types import (
    ChatStreamEvent,
    GenerateStreamEvent,
    Message,
    ThinkOption,
    ToolCallRecord,
    ToolChatResult,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class OllamaToolkit:
    """High-level reusable wrapper around Ollama's Python SDK."""

    def __init__(
        self,
        config: OllamaConfig | None = None,
        *,
        default_model: str | None = None,
        host: str | None = None,
        keep_alive: str | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        client: Client | None = None,
        async_client: AsyncClient | None = None,
    ) -> None:
        resolved_config = config or OllamaConfig.from_env()
        if default_model is not None:
            resolved_config = replace(resolved_config, default_model=default_model)
        if host is not None:
            resolved_config = replace(resolved_config, host=host)
        if keep_alive is not None:
            resolved_config = replace(resolved_config, keep_alive=keep_alive)

        self.config = resolved_config

        shared_client_kwargs = dict(client_kwargs or {})
        if self.config.host and "host" not in shared_client_kwargs:
            shared_client_kwargs["host"] = self.config.host

        self.client = client or Client(**shared_client_kwargs)
        self.async_client = async_client or AsyncClient(**shared_client_kwargs)

    def __enter__(self) -> "OllamaToolkit":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    async def __aenter__(self) -> "OllamaToolkit":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    def close(self) -> None:
        self.client.close()

    async def aclose(self) -> None:
        await self.async_client.close()

    @staticmethod
    def user_message(content: str, images: Sequence[str | bytes] | None = None) -> Message:
        return make_user_message(content, images=images)

    @staticmethod
    def system_message(content: str) -> Message:
        return make_system_message(content)

    @staticmethod
    def tool_message(tool_name: str, content: Any) -> Message:
        return make_tool_message(tool_name, content)

    def chat(
        self,
        *,
        messages: Sequence[Any] | None = None,
        prompt: str | None = None,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        stream: bool = False,
        tools: Sequence[Any] | None = None,
        think: ThinkOption | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        format: dict[str, Any] | str | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | float | None = None,
        **kwargs: Any,
    ) -> Any:
        payload_messages = build_payload_messages(messages, prompt, images)
        return self.client.chat(
            model=self._resolve_model(model),
            messages=payload_messages,
            stream=stream,
            tools=tools,
            think=think,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            format=format,
            options=options,
            keep_alive=keep_alive or self.config.keep_alive,
            **kwargs,
        )

    async def achat(
        self,
        *,
        messages: Sequence[Any] | None = None,
        prompt: str | None = None,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        stream: bool = False,
        tools: Sequence[Any] | None = None,
        think: ThinkOption | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        format: dict[str, Any] | str | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | float | None = None,
        **kwargs: Any,
    ) -> Any:
        payload_messages = build_payload_messages(messages, prompt, images)
        return await self.async_client.chat(
            model=self._resolve_model(model),
            messages=payload_messages,
            stream=stream,
            tools=tools,
            think=think,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            format=format,
            options=options,
            keep_alive=keep_alive or self.config.keep_alive,
            **kwargs,
        )

    def chat_text(self, **kwargs: Any) -> str:
        return extract_response_content(self.chat(**kwargs))

    async def achat_text(self, **kwargs: Any) -> str:
        return extract_response_content(await self.achat(**kwargs))

    def stream_chat_events(self, **kwargs: Any) -> Iterator[ChatStreamEvent]:
        stream = self.chat(stream=True, **kwargs)
        for chunk in stream:
            message = extract_response_message(chunk)
            yield ChatStreamEvent(
                content=extract_message_content(message),
                thinking=extract_message_thinking(message),
                tool_calls=extract_tool_calls(message),
                done=extract_done(chunk),
                raw=chunk,
            )

    async def astream_chat_events(self, **kwargs: Any) -> AsyncIterator[ChatStreamEvent]:
        stream = await self.achat(stream=True, **kwargs)
        async for chunk in stream:
            message = extract_response_message(chunk)
            yield ChatStreamEvent(
                content=extract_message_content(message),
                thinking=extract_message_thinking(message),
                tool_calls=extract_tool_calls(message),
                done=extract_done(chunk),
                raw=chunk,
            )

    def stream_chat_text(self, **kwargs: Any) -> Iterator[str]:
        for event in self.stream_chat_events(**kwargs):
            if event.content:
                yield event.content

    async def astream_chat_text(self, **kwargs: Any) -> AsyncIterator[str]:
        async for event in self.astream_chat_events(**kwargs):
            if event.content:
                yield event.content

    def stream_chat_thinking(self, **kwargs: Any) -> Iterator[str]:
        for event in self.stream_chat_events(**kwargs):
            if event.thinking:
                yield event.thinking

    async def astream_chat_thinking(self, **kwargs: Any) -> AsyncIterator[str]:
        async for event in self.astream_chat_events(**kwargs):
            if event.thinking:
                yield event.thinking

    def generate(
        self,
        *,
        prompt: str,
        model: str | None = None,
        suffix: str | None = None,
        system: str | None = None,
        template: str | None = None,
        context: Sequence[int] | None = None,
        stream: bool = False,
        think: ThinkOption | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        raw: bool | None = None,
        images: Sequence[str | bytes] | None = None,
        format: dict[str, Any] | str | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | float | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.client.generate(
            model=self._resolve_model(model),
            prompt=prompt,
            suffix=suffix,
            system=system,
            template=template,
            context=context,
            stream=stream,
            think=think,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            raw=raw,
            images=images,
            format=format,
            options=options,
            keep_alive=keep_alive or self.config.keep_alive,
            width=width,
            height=height,
            steps=steps,
            **kwargs,
        )

    async def agenerate(
        self,
        *,
        prompt: str,
        model: str | None = None,
        suffix: str | None = None,
        system: str | None = None,
        template: str | None = None,
        context: Sequence[int] | None = None,
        stream: bool = False,
        think: ThinkOption | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        raw: bool | None = None,
        images: Sequence[str | bytes] | None = None,
        format: dict[str, Any] | str | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | float | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self.async_client.generate(
            model=self._resolve_model(model),
            prompt=prompt,
            suffix=suffix,
            system=system,
            template=template,
            context=context,
            stream=stream,
            think=think,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            raw=raw,
            images=images,
            format=format,
            options=options,
            keep_alive=keep_alive or self.config.keep_alive,
            width=width,
            height=height,
            steps=steps,
            **kwargs,
        )

    def generate_text(self, **kwargs: Any) -> str:
        return extract_generate_text(self.generate(**kwargs))

    async def agenerate_text(self, **kwargs: Any) -> str:
        return extract_generate_text(await self.agenerate(**kwargs))

    def stream_generate_events(self, **kwargs: Any) -> Iterator[GenerateStreamEvent]:
        stream = self.generate(stream=True, **kwargs)
        for chunk in stream:
            yield GenerateStreamEvent(
                response=extract_generate_text(chunk),
                thinking=extract_generate_thinking(chunk),
                logprobs=extract_logprobs(chunk),
                done=extract_done(chunk),
                raw=chunk,
            )

    async def astream_generate_events(self, **kwargs: Any) -> AsyncIterator[GenerateStreamEvent]:
        stream = await self.agenerate(stream=True, **kwargs)
        async for chunk in stream:
            yield GenerateStreamEvent(
                response=extract_generate_text(chunk),
                thinking=extract_generate_thinking(chunk),
                logprobs=extract_logprobs(chunk),
                done=extract_done(chunk),
                raw=chunk,
            )

    def stream_generate_text(self, **kwargs: Any) -> Iterator[str]:
        for event in self.stream_generate_events(**kwargs):
            if event.response:
                yield event.response

    async def astream_generate_text(self, **kwargs: Any) -> AsyncIterator[str]:
        async for event in self.astream_generate_events(**kwargs):
            if event.response:
                yield event.response

    def stream_generate_thinking(self, **kwargs: Any) -> Iterator[str]:
        for event in self.stream_generate_events(**kwargs):
            if event.thinking:
                yield event.thinking

    async def astream_generate_thinking(self, **kwargs: Any) -> AsyncIterator[str]:
        async for event in self.astream_generate_events(**kwargs):
            if event.thinking:
                yield event.thinking

    def embed(
        self,
        *,
        input: str | Sequence[str],
        model: str | None = None,
        truncate: bool | None = None,
        dimensions: int | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | float | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.client.embed(
            model=self._resolve_model(model),
            input=input,
            truncate=truncate,
            dimensions=dimensions,
            options=options,
            keep_alive=keep_alive or self.config.keep_alive,
            **kwargs,
        )

    async def aembed(
        self,
        *,
        input: str | Sequence[str],
        model: str | None = None,
        truncate: bool | None = None,
        dimensions: int | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | float | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self.async_client.embed(
            model=self._resolve_model(model),
            input=input,
            truncate=truncate,
            dimensions=dimensions,
            options=options,
            keep_alive=keep_alive or self.config.keep_alive,
            **kwargs,
        )

    def chat_structured(
        self,
        schema: type[ModelT],
        *,
        messages: Sequence[Any] | None = None,
        prompt: str | None = None,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | float | None = None,
        retries: int = 1,
        retry_instruction: str | None = None,
        **kwargs: Any,
    ) -> ModelT:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        if "format" in kwargs:
            raise ValueError("chat_structured does not accept 'format' in kwargs")

        active_messages = build_payload_messages(messages, prompt, images)

        for attempt in range(retries + 1):
            response = self.chat(
                messages=active_messages,
                model=model,
                options=options,
                keep_alive=keep_alive,
                format=schema.model_json_schema(),
                **kwargs,
            )
            content = extract_response_content(response)

            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                if attempt >= retries:
                    raise

                active_messages.extend(
                    [
                        extract_response_message(response),
                        {
                            "role": "user",
                            "content": retry_instruction
                            or (
                                "Your previous response did not match the required JSON schema. "
                                "Return only valid JSON and do not include any extra text. "
                                f"Validation error: {exc.errors()}"
                            ),
                        },
                    ]
                )

        raise RuntimeError("unreachable")

    async def achat_structured(
        self,
        schema: type[ModelT],
        *,
        messages: Sequence[Any] | None = None,
        prompt: str | None = None,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | float | None = None,
        retries: int = 1,
        retry_instruction: str | None = None,
        **kwargs: Any,
    ) -> ModelT:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        if "format" in kwargs:
            raise ValueError("achat_structured does not accept 'format' in kwargs")

        active_messages = build_payload_messages(messages, prompt, images)

        for attempt in range(retries + 1):
            response = await self.achat(
                messages=active_messages,
                model=model,
                options=options,
                keep_alive=keep_alive,
                format=schema.model_json_schema(),
                **kwargs,
            )
            content = extract_response_content(response)

            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                if attempt >= retries:
                    raise

                active_messages.extend(
                    [
                        extract_response_message(response),
                        {
                            "role": "user",
                            "content": retry_instruction
                            or (
                                "Your previous response did not match the required JSON schema. "
                                "Return only valid JSON and do not include any extra text. "
                                f"Validation error: {exc.errors()}"
                            ),
                        },
                    ]
                )

        raise RuntimeError("unreachable")

    def chat_with_tools(
        self,
        *,
        registry: ToolRegistry,
        messages: Sequence[Any] | None = None,
        prompt: str | None = None,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        stream: bool = False,
        think: ThinkOption | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | float | None = None,
        max_rounds: int = 8,
        on_tool_error: Literal["raise", "message"] = "message",
        **kwargs: Any,
    ) -> ToolChatResult:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")

        history = build_payload_messages(messages, prompt, images)
        records: list[ToolCallRecord] = []

        for _ in range(max_rounds):
            if stream:
                message, response = collect_streamed_chat_message(
                    self.chat(
                        messages=history,
                        model=model,
                        stream=True,
                        tools=registry.tool_specs(),
                        think=think,
                        logprobs=logprobs,
                        top_logprobs=top_logprobs,
                        options=options,
                        keep_alive=keep_alive,
                        **kwargs,
                    )
                )
            else:
                response = self.chat(
                    messages=history,
                    model=model,
                    stream=False,
                    tools=registry.tool_specs(),
                    think=think,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    options=options,
                    keep_alive=keep_alive,
                    **kwargs,
                )
                message = extract_response_message(response)

            history.append(message)
            tool_calls = extract_tool_calls(message)
            if not tool_calls:
                return ToolChatResult(response=response, messages=history, tool_calls=records)

            for raw_tool_call in tool_calls:
                parsed = registry.parse_tool_call(raw_tool_call)

                output: Any = None
                error: str | None = None

                try:
                    output = registry.execute(parsed)
                except Exception as exc:
                    if on_tool_error == "raise":
                        raise
                    error = str(exc)
                    output = f"Tool execution error in '{parsed.name}': {exc}"

                records.append(
                    ToolCallRecord(
                        name=parsed.name,
                        arguments=parsed.arguments,
                        output=output,
                        error=error,
                    )
                )
                history.append(self.tool_message(parsed.name, output))

        raise RuntimeError("Tool call loop exceeded max_rounds without a final answer")

    async def achat_with_tools(
        self,
        *,
        registry: ToolRegistry,
        messages: Sequence[Any] | None = None,
        prompt: str | None = None,
        images: Sequence[str | bytes] | None = None,
        model: str | None = None,
        stream: bool = False,
        think: ThinkOption | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | float | None = None,
        max_rounds: int = 8,
        on_tool_error: Literal["raise", "message"] = "message",
        **kwargs: Any,
    ) -> ToolChatResult:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")

        history = build_payload_messages(messages, prompt, images)
        records: list[ToolCallRecord] = []

        for _ in range(max_rounds):
            if stream:
                stream_response = await self.achat(
                    messages=history,
                    model=model,
                    stream=True,
                    tools=registry.tool_specs(),
                    think=think,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    options=options,
                    keep_alive=keep_alive,
                    **kwargs,
                )
                message, response = await acollect_streamed_chat_message(stream_response)
            else:
                response = await self.achat(
                    messages=history,
                    model=model,
                    stream=False,
                    tools=registry.tool_specs(),
                    think=think,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    options=options,
                    keep_alive=keep_alive,
                    **kwargs,
                )
                message = extract_response_message(response)

            history.append(message)
            tool_calls = extract_tool_calls(message)
            if not tool_calls:
                return ToolChatResult(response=response, messages=history, tool_calls=records)

            for raw_tool_call in tool_calls:
                parsed = registry.parse_tool_call(raw_tool_call)

                output: Any = None
                error: str | None = None

                try:
                    output = await registry.execute_async(parsed)
                except Exception as exc:
                    if on_tool_error == "raise":
                        raise
                    error = str(exc)
                    output = f"Tool execution error in '{parsed.name}': {exc}"

                records.append(
                    ToolCallRecord(
                        name=parsed.name,
                        arguments=parsed.arguments,
                        output=output,
                        error=error,
                    )
                )
                history.append(self.tool_message(parsed.name, output))

        raise RuntimeError("Tool call loop exceeded max_rounds without a final answer")

    def list_models(self) -> Any:
        return self.client.list()

    async def alist_models(self) -> Any:
        return await self.async_client.list()

    def pull_model(self, model: str, *, insecure: bool = False, stream: bool = False, **kwargs: Any) -> Any:
        return self.client.pull(model, insecure=insecure, stream=stream, **kwargs)

    async def apull_model(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        return await self.async_client.pull(model, insecure=insecure, stream=stream, **kwargs)

    def push_model(self, model: str, *, insecure: bool = False, stream: bool = False, **kwargs: Any) -> Any:
        return self.client.push(model, insecure=insecure, stream=stream, **kwargs)

    async def apush_model(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        return await self.async_client.push(model, insecure=insecure, stream=stream, **kwargs)

    def create_model(
        self,
        *,
        model: str,
        from_model: str | None = None,
        quantize: str | None = None,
        files: dict[str, str] | None = None,
        adapters: dict[str, str] | None = None,
        template: str | None = None,
        license: str | list[str] | None = None,
        system: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        messages: Sequence[Any] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        return self.client.create(
            model=model,
            from_=from_model,
            quantize=quantize,
            files=files,
            adapters=adapters,
            template=template,
            license=license,
            system=system,
            parameters=parameters,
            messages=messages,
            stream=stream,
            **kwargs,
        )

    async def acreate_model(
        self,
        *,
        model: str,
        from_model: str | None = None,
        quantize: str | None = None,
        files: dict[str, str] | None = None,
        adapters: dict[str, str] | None = None,
        template: str | None = None,
        license: str | list[str] | None = None,
        system: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        messages: Sequence[Any] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        return await self.async_client.create(
            model=model,
            from_=from_model,
            quantize=quantize,
            files=files,
            adapters=adapters,
            template=template,
            license=license,
            system=system,
            parameters=parameters,
            messages=messages,
            stream=stream,
            **kwargs,
        )

    def copy_model(self, source: str, destination: str) -> Any:
        return self.client.copy(source, destination)

    async def acopy_model(self, source: str, destination: str) -> Any:
        return await self.async_client.copy(source, destination)

    def delete_model(self, model: str) -> Any:
        return self.client.delete(model)

    async def adelete_model(self, model: str) -> Any:
        return await self.async_client.delete(model)

    def show_model(self, model: str) -> Any:
        return self.client.show(model)

    async def ashow_model(self, model: str) -> Any:
        return await self.async_client.show(model)

    def running_models(self) -> Any:
        return self.client.ps()

    async def arunning_models(self) -> Any:
        return await self.async_client.ps()

    def create_blob(self, path: str | Path) -> str:
        return self.client.create_blob(path)

    async def acreate_blob(self, path: str | Path) -> str:
        return await self.async_client.create_blob(path)

    def web_search(self, query: str, *, max_results: int = 3) -> Any:
        return self.client.web_search(query=query, max_results=max_results)

    async def aweb_search(self, query: str, *, max_results: int = 3) -> Any:
        return await self.async_client.web_search(query=query, max_results=max_results)

    def web_fetch(self, url: str) -> Any:
        return self.client.web_fetch(url=url)

    async def aweb_fetch(self, url: str) -> Any:
        return await self.async_client.web_fetch(url=url)

    def _resolve_model(self, model: str | None) -> str:
        return model or self.config.default_model
