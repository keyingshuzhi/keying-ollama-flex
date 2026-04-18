import asyncio

from ollama_flex import OllamaToolkit


async def main() -> None:
    kit = OllamaToolkit(default_model="gpt-oss:120b-cloud")

    print("== one-shot async ==")
    text = await kit.achat_text(prompt="请用一句话解释什么是 RAG")
    print(text)

    print("\n== stream async ==")
    async for chunk in kit.astream_chat_text(prompt="请给出 3 个 RAG 落地建议"):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
