from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, TypeAlias


Message: TypeAlias = dict[str, Any]
ToolFunction: TypeAlias = Callable[..., Any]
AsyncToolFunction: TypeAlias = Callable[..., Awaitable[Any]]
ThinkLevel: TypeAlias = Literal["low", "medium", "high"]
ThinkOption: TypeAlias = bool | ThinkLevel


@dataclass(slots=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    output: Any = None
    error: str | None = None


@dataclass(slots=True)
class ToolChatResult:
    response: Any
    messages: list[Any] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    @property
    def final_message(self) -> Any:
        if not self.messages:
            return None
        return self.messages[-1]

    @property
    def final_content(self) -> str:
        message = self.final_message
        if message is None:
            return ""
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(getattr(message, "content", "") or "")

    @property
    def final_thinking(self) -> str:
        message = self.final_message
        if message is None:
            return ""
        if isinstance(message, dict):
            return str(message.get("thinking") or "")
        return str(getattr(message, "thinking", "") or "")


@dataclass(slots=True)
class ChatStreamEvent:
    content: str = ""
    thinking: str = ""
    tool_calls: list[Any] = field(default_factory=list)
    done: bool | None = None
    raw: Any = None


@dataclass(slots=True)
class GenerateStreamEvent:
    response: str = ""
    thinking: str = ""
    logprobs: Any = None
    done: bool | None = None
    raw: Any = None
