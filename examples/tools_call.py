from ollama_flex import OllamaToolkit, ToolRegistry

kit = OllamaToolkit(default_model="llama3.2")
registry = ToolRegistry()


@registry.register
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@registry.register
def subtract_two_numbers(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


# 也支持手动 schema + 自定义执行器（含命名空间函数名）
registry.register_schema(
    {
        "type": "function",
        "function": {
            "name": "string.length",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
        },
    },
    executor=lambda text: len(text),
)

result = kit.chat_with_tools(
    registry=registry,
    prompt="What is three plus one?",
    max_rounds=6,
)

print("Final:", result.final_content)
for call in result.tool_calls:
    print(f"{call.name}({call.arguments}) -> {call.output}")
