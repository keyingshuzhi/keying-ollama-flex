from ollama_flex import OllamaToolkit

kit = OllamaToolkit(default_model="gpt-oss:120b-cloud")

print("== one-shot text ==")
print(kit.chat_text(prompt="请用一句话介绍你自己"))

print("\n== stream events ==")
for event in kit.stream_chat_events(prompt="请列出本地部署大模型的三个优势"):
    if event.content:
        print(event.content, end="", flush=True)
print()
