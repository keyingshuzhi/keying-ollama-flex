import asyncio

from ollama_flex import ToolRegistry


def test_register_and_schema():
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int = 1) -> int:
        """Add numbers."""
        return a + b

    specs = registry.tool_specs()
    assert len(specs) == 1

    tool = specs[0]["function"]
    assert tool["name"] == "add"
    assert tool["description"] == "Add numbers."

    params = tool["parameters"]
    assert params["required"] == ["a"]
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["default"] == 1


def test_parse_execute_and_type_coercion():
    registry = ToolRegistry()

    @registry.register
    def multiply(x: int, y: int) -> int:
        return x * y

    parsed = registry.parse_tool_call(
        {
            "function": {
                "name": "multiply",
                "arguments": '{"x": "6", "y": "7"}',
            }
        }
    )

    assert parsed.name == "multiply"
    assert parsed.arguments == {"x": "6", "y": "7"}
    assert registry.execute(parsed) == 42


def test_register_schema_with_executor():
    registry = ToolRegistry()

    schema = {
        "type": "function",
        "function": {
            "name": "browser.search",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                },
            },
        },
    }

    def search(query: str) -> str:
        return f"search:{query}"

    registry.register_schema(schema, executor=search)
    parsed = registry.parse_tool_call(
        {
            "function": {
                "name": "browser.search",
                "arguments": {"query": "ollama"},
            }
        }
    )

    assert registry.execute(parsed) == "search:ollama"


def test_execute_async_tool():
    registry = ToolRegistry()

    @registry.register
    async def add_async(a: int, b: int) -> int:
        return a + b

    parsed = registry.parse_tool_call(
        {
            "function": {
                "name": "add_async",
                "arguments": {"a": 2, "b": 3},
            }
        }
    )

    result = asyncio.run(registry.execute_async(parsed))
    assert result == 5
