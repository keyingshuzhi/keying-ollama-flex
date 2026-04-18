from __future__ import annotations

import inspect
import json
from copy import deepcopy
from types import UnionType
from typing import Any, Literal, Mapping, Union, get_args, get_origin

from .errors import ToolArgumentsError, ToolNotFoundError
from .types import ParsedToolCall, ToolFunction


_PRIMITIVE_TYPE_MAP: dict[type[Any], str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


class ToolRegistry:
    """Register Python callables as Ollama function tools with execution helpers."""

    def __init__(self, *, coerce_arguments: bool = True) -> None:
        self._functions: dict[str, ToolFunction] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._signatures: dict[str, inspect.Signature] = {}
        self._annotations: dict[str, dict[str, Any]] = {}
        self._coerce_arguments = coerce_arguments

    def register(
        self,
        func: ToolFunction | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ):
        """Register a function. Can be used directly or as a decorator."""

        if func is None:
            return lambda wrapped: self.register(
                wrapped,
                name=name,
                description=description,
            )

        tool_name = name or func.__name__
        tool_schema = _build_tool_schema(func, tool_name, description)

        self._functions[tool_name] = func
        self._schemas[tool_name] = tool_schema
        self._signatures[tool_name] = inspect.signature(func)
        self._annotations[tool_name] = inspect.get_annotations(func, eval_str=True)
        return func

    def register_schema(
        self,
        schema: Mapping[str, Any],
        *,
        executor: ToolFunction | None = None,
        name: str | None = None,
    ) -> str:
        """Register a manually-defined Ollama tool schema, optionally with an executor."""

        normalized = _normalize_tool_schema(schema, name=name)
        tool_name = normalized["function"]["name"]
        self._schemas[tool_name] = normalized

        if executor is not None:
            self._functions[tool_name] = executor
            self._signatures[tool_name] = inspect.signature(executor)
            self._annotations[tool_name] = inspect.get_annotations(executor, eval_str=True)

        return tool_name

    def unregister(self, name: str) -> None:
        self._functions.pop(name, None)
        self._schemas.pop(name, None)
        self._signatures.pop(name, None)
        self._annotations.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._functions

    def names(self) -> list[str]:
        return sorted(self._schemas.keys())

    def tool_specs(self) -> list[dict[str, Any]]:
        return list(self._schemas.values())

    def parse_tool_call(self, raw_tool_call: Any) -> ParsedToolCall:
        function = (
            raw_tool_call.get("function")
            if isinstance(raw_tool_call, dict)
            else getattr(raw_tool_call, "function", None)
        )

        if function is None:
            raise ToolArgumentsError("Invalid tool call: missing function payload")

        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments", {})
        else:
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", {})

        if not name:
            raise ToolArgumentsError("Invalid tool call: missing function name")

        return ParsedToolCall(name=name, arguments=_decode_arguments(arguments))

    def execute(self, parsed_call: ParsedToolCall) -> Any:
        func = self._functions.get(parsed_call.name)
        if func is None:
            raise ToolNotFoundError(f"Tool '{parsed_call.name}' is not registered")

        kwargs = self._prepare_kwargs(parsed_call.name, parsed_call.arguments)
        result = func(**kwargs)

        if inspect.isawaitable(result):
            raise RuntimeError(
                f"Tool '{parsed_call.name}' returned an awaitable. "
                "Use 'achat_with_tools' for async tools."
            )

        return result

    async def execute_async(self, parsed_call: ParsedToolCall) -> Any:
        func = self._functions.get(parsed_call.name)
        if func is None:
            raise ToolNotFoundError(f"Tool '{parsed_call.name}' is not registered")

        kwargs = self._prepare_kwargs(parsed_call.name, parsed_call.arguments)
        result = func(**kwargs)

        if inspect.isawaitable(result):
            return await result
        return result

    def _prepare_kwargs(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._coerce_arguments:
            return dict(arguments)

        signature = self._signatures.get(tool_name)
        if signature is None:
            return dict(arguments)

        annotations = self._annotations.get(tool_name, {})
        coerced = dict(arguments)

        for param_name, param in signature.parameters.items():
            if param_name not in coerced:
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            annotation = annotations.get(param_name, param.annotation)
            coerced[param_name] = _coerce_value(coerced[param_name], annotation)

        return coerced


def _normalize_tool_schema(schema: Mapping[str, Any], *, name: str | None = None) -> dict[str, Any]:
    normalized = deepcopy(dict(schema))

    if normalized.get("type") != "function":
        raise ValueError("Tool schema must have type='function'")

    function = normalized.get("function")
    if not isinstance(function, dict):
        raise ValueError("Tool schema must contain a 'function' object")

    if name is not None:
        function["name"] = name

    tool_name = function.get("name")
    if not tool_name:
        raise ValueError("Tool schema function must contain a non-empty 'name'")

    parameters = function.get("parameters")
    if parameters is None:
        function["parameters"] = {"type": "object", "properties": {}, "required": []}

    return normalized


def _build_tool_schema(
    func: ToolFunction,
    tool_name: str,
    description: str | None,
) -> dict[str, Any]:
    signature = inspect.signature(func)
    annotations = inspect.get_annotations(func, eval_str=True)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError(
                f"Tool '{tool_name}' has variadic parameter '{param_name}', which is not supported"
            )

        annotation = annotations.get(param_name, Any)
        schema = _annotation_to_schema(annotation)

        if param.default is not inspect._empty:
            schema["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = schema

    final_description = description or _first_line_doc(func) or f"Run {tool_name}"

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": final_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _first_line_doc(func: ToolFunction) -> str | None:
    doc = inspect.getdoc(func)
    if not doc:
        return None
    return doc.splitlines()[0].strip() or None


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect._empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation in _PRIMITIVE_TYPE_MAP:
        return {"type": _PRIMITIVE_TYPE_MAP[annotation]}

    if origin is Literal and args:
        inferred_type = type(args[0]) if args else str
        schema_type = _PRIMITIVE_TYPE_MAP.get(inferred_type, "string")
        return {"type": schema_type, "enum": list(args)}

    if origin in (list, set, tuple):
        item_annotation = args[0] if args else Any
        return {"type": "array", "items": _annotation_to_schema(item_annotation)}

    if origin in (dict, Mapping):
        value_annotation = args[1] if len(args) > 1 else Any
        return {"type": "object", "additionalProperties": _annotation_to_schema(value_annotation)}

    if args:
        non_none_args = [arg for arg in args if arg is not type(None)]  # noqa: E721
        if len(non_none_args) == 1 and len(args) != len(non_none_args):
            schema = _annotation_to_schema(non_none_args[0])
            schema["nullable"] = True
            return schema

        return {"anyOf": [_annotation_to_schema(arg) for arg in non_none_args] or [{"type": "string"}]}

    return {"type": "string"}


def _decode_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments is None:
        return {}

    if isinstance(raw_arguments, dict):
        return raw_arguments

    if isinstance(raw_arguments, (bytes, bytearray)):
        raw_arguments = raw_arguments.decode("utf-8")

    if isinstance(raw_arguments, str):
        if not raw_arguments.strip():
            return {}
        try:
            decoded = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ToolArgumentsError(f"Tool arguments are not valid JSON: {exc}") from exc

        if not isinstance(decoded, dict):
            raise ToolArgumentsError("Tool arguments must decode to a JSON object")
        return decoded

    raise ToolArgumentsError(f"Unsupported tool arguments payload type: {type(raw_arguments)!r}")


def _coerce_value(value: Any, annotation: Any) -> Any:
    if annotation is inspect._empty or annotation is Any:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation is bool:
        return _coerce_bool(value)

    if annotation in (str, int, float):
        return _coerce_primitive(value, annotation)

    if origin is Literal and args:
        candidate_type = type(args[0])
        coerced = _coerce_value(value, candidate_type)
        return coerced if coerced in args else value

    if origin in (list, set, tuple):
        if not isinstance(value, (list, tuple, set)):
            return value
        item_type = args[0] if args else Any
        items = [_coerce_value(item, item_type) for item in value]
        if origin is tuple:
            return tuple(items)
        if origin is set:
            return set(items)
        return items

    if origin in (dict, Mapping):
        if not isinstance(value, dict):
            return value

        key_type = args[0] if len(args) > 0 else Any
        value_type = args[1] if len(args) > 1 else Any
        return {
            _coerce_value(k, key_type): _coerce_value(v, value_type)
            for k, v in value.items()
        }

    if origin in (Union, UnionType):
        non_none_args = [arg for arg in args if arg is not type(None)]  # noqa: E721
        for sub_annotation in non_none_args:
            try:
                coerced = _coerce_value(value, sub_annotation)
                if _value_matches_annotation(coerced, sub_annotation):
                    return coerced
            except Exception:
                continue
        return value

    return value


def _coerce_primitive(value: Any, target: type[Any]) -> Any:
    if isinstance(value, target):
        return value

    try:
        return target(value)
    except Exception:
        return value


def _coerce_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False

    if isinstance(value, (int, float)):
        return bool(value)

    return value


def _value_matches_annotation(value: Any, annotation: Any) -> bool:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation in (str, int, float, bool):
        return isinstance(value, annotation)

    if origin is Literal:
        return value in args

    if origin in (Union, UnionType):
        return any(_value_matches_annotation(value, arg) for arg in args if arg is not type(None))  # noqa: E721

    return True
