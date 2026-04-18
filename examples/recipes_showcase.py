from ollama_flex import OllamaRecipes, OllamaToolkit

kit = OllamaToolkit(default_model="llama3.2")
recipes = OllamaRecipes(kit)

report = recipes.pull_model_with_progress("llama3.2")
print("Pull statuses:", report.statuses)
print("Digests:", report.digests)

print("\n== local models ==")
print(recipes.list_models_text())

print("\n== running models ==")
print(recipes.running_models_text())

