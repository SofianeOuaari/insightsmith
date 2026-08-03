"""Model strings and provider construction.

Model strings are ``provider/model``. The split is on the *first* slash only, so
``openrouter/anthropic/claude-sonnet-4.5`` keeps its vendor prefix intact as part
of the model name.
"""

from __future__ import annotations

import os
from typing import Final

from insightsmith.errors import ProviderError
from insightsmith.llm.base import Provider
from insightsmith.llm.ollama import DEFAULT_HOST, OllamaProvider
from insightsmith.llm.openai_compat import BACKENDS, LOCAL_BACKENDS, OpenAICompatProvider

__all__ = ["build_provider", "is_local_model", "known_providers", "split_model"]

#: Providers that never send data off the machine.
LOCAL_PROVIDERS: Final = frozenset({"ollama"}) | LOCAL_BACKENDS


def split_model(reference: str) -> tuple[str, str]:
    """``"ollama/qwen3:8b"`` -> ``("ollama", "qwen3:8b")``.

    Raises:
        ProviderError: if the reference has no provider prefix.
    """
    provider, separator, model = reference.partition("/")
    if not separator or not provider or not model:
        raise ProviderError(
            f"{reference!r} is not a provider/model reference, e.g. 'ollama/qwen3:8b'"
        )
    return provider, model


def is_local_model(reference: str) -> bool:
    """Whether a model reference stays on this machine."""
    try:
        provider, _ = split_model(reference)
    except ProviderError:
        return False
    return provider in LOCAL_PROVIDERS


def known_providers() -> list[str]:
    return sorted({"ollama", *BACKENDS})


def build_provider(
    name: str,
    *,
    base_urls: dict[str, str] | None = None,
    environ: dict[str, str] | None = None,
) -> Provider:
    """Construct a provider by name.

    Raises:
        ProviderError: for an unknown provider, or a remote one with no API key.
    """
    env = os.environ if environ is None else environ
    overrides = base_urls or {}

    if name == "ollama":
        return OllamaProvider(host=overrides.get("ollama", env.get("OLLAMA_HOST", DEFAULT_HOST)))

    known = BACKENDS.get(name)
    if known is None and name not in overrides:
        raise ProviderError(f"unknown provider {name!r}. Known: {', '.join(known_providers())}")

    base_url = overrides.get(name) or (known[0] if known else "")
    key_variable = known[1] if known else ""
    api_key = env.get(key_variable) if key_variable else None
    if key_variable and not api_key and name not in LOCAL_BACKENDS:
        raise ProviderError(f"{name} needs an API key: set {key_variable}")

    return OpenAICompatProvider(name, base_url=base_url, api_key=api_key)
