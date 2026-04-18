from ollama_flex import OllamaToolkit

kit = OllamaToolkit(default_model="gpt-oss:20b")

search = kit.web_search("Ollama python examples", max_results=3)
print("Search result:")
print(search)

# 把 URL 换成你想抓取的页面
url = "https://ollama.com/"
page = kit.web_fetch(url)
print("\nFetch result:")
print(page)

