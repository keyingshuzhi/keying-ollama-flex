from ollama_flex import OllamaToolkit

kit = OllamaToolkit(default_model="gpt-oss:20b")

print("== stream generate events ==")
for event in kit.stream_generate_events(
    prompt="用简洁中文解释为什么天空是蓝色的",
    think="low",
):
    if event.thinking:
        print("[thinking]", event.thinking, end="", flush=True)
    if event.response:
        print(event.response, end="", flush=True)

print()

