from ollama_flex import ChatSession, OllamaToolkit

session = ChatSession(
    OllamaToolkit(default_model="deepseek-r1:7b"),
    system_prompt="你是一个简洁的中文助手。",
    max_messages=12,
)

print(session.ask_text("请用一句话解释什么是大模型"))
print(session.ask_text("把上一句改成更口语化"))

print("\nstream:")
for chunk in session.ask_stream("再给一个更技术一点的版本"):
    print(chunk, end="", flush=True)
print()
