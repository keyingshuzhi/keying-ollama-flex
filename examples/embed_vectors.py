from ollama_flex import OllamaToolkit

kit = OllamaToolkit(default_model="nomic-embed-text")

response = kit.embed(
    input=["你好，你是谁？", "RAG 是什么？"],
    dimensions=384,
)

embeddings = response["embeddings"] if isinstance(response, dict) else response.embeddings
print("Embedding count:", len(embeddings))
if embeddings:
    print("First vector dim:", len(embeddings[0]))

