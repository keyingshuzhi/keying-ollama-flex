from ollama_flex import OllamaToolkit

kit = OllamaToolkit(default_model="codellama:7b-code")

prompt = """def remove_non_ascii(s: str) -> str:
    \"\"\"
"""

suffix = """
    return result
"""

result = kit.generate(
    prompt=prompt,
    suffix=suffix,
    options={
        "num_predict": 128,
        "temperature": 0,
        "top_p": 0.9,
        "stop": ["<EOT>"],
    },
)

print(result["response"] if isinstance(result, dict) else result.response)

