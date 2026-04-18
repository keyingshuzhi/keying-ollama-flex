import asyncio

from ollama_flex import OllamaToolkit, ToolRegistry


async def main() -> None:
    kit = OllamaToolkit(default_model="gpt-oss:120b-cloud")
    registry = ToolRegistry()

    @registry.register(name="math.multiply")
    async def multiply(a: int, b: int) -> int:
        return a * b

    result = await kit.achat_with_tools(
        registry=registry,
        prompt="What is 6 * 7?",
        max_rounds=6,
    )

    print("Final:", result.final_content)
    for call in result.tool_calls:
        print(f"{call.name}({call.arguments}) -> {call.output}")


if __name__ == "__main__":
    asyncio.run(main())
