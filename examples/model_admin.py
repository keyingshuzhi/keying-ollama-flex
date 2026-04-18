from ollama_flex import OllamaRecipes, OllamaToolkit

kit = OllamaToolkit(default_model="llama3.2")
recipes = OllamaRecipes(kit)

print("== local models ==")
print(recipes.list_models_text())

print("\n== warmup llama3.2 ==")
print(recipes.warmup_model("llama3.2"))

print("\n== running models ==")
print(recipes.running_models_text())

