"""Model providers, and the router that picks between them."""

from insightsmith.llm.base import (
    Capabilities,
    Chunk,
    Completion,
    Message,
    Provider,
    ToolCall,
    Usage,
)
from insightsmith.llm.ollama import OllamaProvider
from insightsmith.llm.openai_compat import BACKENDS, OpenAICompatProvider
from insightsmith.llm.registry import build_provider, is_local_model, known_providers, split_model
from insightsmith.llm.router import Route, Router, Strategy, extract_json

__all__ = [
    "BACKENDS",
    "Capabilities",
    "Chunk",
    "Completion",
    "Message",
    "OllamaProvider",
    "OpenAICompatProvider",
    "Provider",
    "Route",
    "Router",
    "Strategy",
    "ToolCall",
    "Usage",
    "build_provider",
    "extract_json",
    "is_local_model",
    "known_providers",
    "split_model",
]
