from ollama_flex import OllamaRecipes, OllamaToolkit

kit = OllamaToolkit(default_model="deepseek-r1:7b")
recipes = OllamaRecipes(kit)

seed_messages = [
    {"role": "system", "content": "你是一个简洁专业的中文助手。"},
]

recipes.chat_with_history_cli(
    model="deepseek-r1:7b",
    messages=seed_messages,
    prompt_text="Chat with history: ",
    exit_words=("exit", "quit"),
)

