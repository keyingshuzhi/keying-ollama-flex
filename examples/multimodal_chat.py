from pathlib import Path

from ollama_flex import OllamaToolkit

kit = OllamaToolkit(default_model="llama3.2-vision")

image_path = Path("data/images/刘亦菲.jpeg").resolve()
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

text = kit.chat_text(
    prompt="请解读这张图片",
    images=[str(image_path)],
)
print(text)

