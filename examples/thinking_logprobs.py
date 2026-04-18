from ollama_flex import OllamaToolkit

kit = OllamaToolkit(default_model="gpt-oss:20b")

response = kit.chat(
    prompt="What is 10 + 23?",
    think="low",
    logprobs=True,
    top_logprobs=3,
)

print("Thinking:\n", response.message.thinking)
print("Answer:\n", response.message.content)
print("Logprobs count:", len(response.logprobs or []))
