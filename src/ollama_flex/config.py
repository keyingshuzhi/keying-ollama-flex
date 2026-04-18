from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class OllamaConfig:
    """Runtime configuration for OllamaToolkit."""

    default_model: str = "llama3.2"
    host: str | None = None
    keep_alive: str | None = None

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        return cls(
            default_model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2"),
            host=os.getenv("OLLAMA_HOST"),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE"),
        )
