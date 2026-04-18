class OllamaFlexError(Exception):
    """Base error for this package."""


class ToolNotFoundError(OllamaFlexError):
    """Raised when model asks for a tool that is not registered."""


class ToolArgumentsError(OllamaFlexError):
    """Raised when tool call arguments cannot be decoded."""
