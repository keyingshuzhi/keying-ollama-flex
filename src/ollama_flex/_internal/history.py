from __future__ import annotations

from typing import Any, Sequence


def leading_system_messages(messages: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "system":
            break
        result.append(message)
    return result


def trim_messages(messages: Sequence[Any], max_messages: int | None) -> list[Any]:
    if max_messages is None:
        return list(messages)
    if max_messages < 1:
        return []
    if len(messages) <= max_messages:
        return list(messages)

    system_messages = leading_system_messages(messages)
    non_system = list(messages[len(system_messages) :])

    if len(system_messages) >= max_messages:
        return system_messages[-max_messages:]

    remain = max_messages - len(system_messages)
    return [*system_messages, *non_system[-remain:]]

