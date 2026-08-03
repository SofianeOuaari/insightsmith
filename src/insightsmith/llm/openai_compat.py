"""One provider for every backend that speaks the OpenAI chat-completions wire.

OpenAI, OpenRouter, DeepInfra, Together, Groq, Fireworks, Mistral, Gemini's
compatibility endpoint, and local vLLM / llama.cpp / LM Studio servers differ
only by base URL and API key. Writing six SDKs would buy nothing, so this is a
table of base URLs and one class.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any, Final

import httpx

from insightsmith.errors import ProviderError
from insightsmith.llm.base import Capabilities, Chunk, Completion, Message, ToolCall, Usage

__all__ = ["BACKENDS", "OpenAICompatProvider"]

#: backend name -> (base URL, API key environment variable)
BACKENDS: Final[dict[str, tuple[str, str]]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "lmstudio": ("http://localhost:1234/v1", ""),
    "vllm": ("http://localhost:8000/v1", ""),
    "llamacpp": ("http://localhost:8080/v1", ""),
}

#: Backends that never leave the machine, so local_only permits them.
LOCAL_BACKENDS: Final = frozenset({"lmstudio", "vllm", "llamacpp"})

_DEFAULT_CONTEXT: Final = 8192
_TIMEOUT: Final = 120.0


class OpenAICompatProvider:
    """Any OpenAI-wire backend, selected by name from :data:`BACKENDS`."""

    def __init__(
        self,
        name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        known = BACKENDS.get(name)
        if base_url is None:
            if known is None:
                raise ProviderError(f"unknown backend {name!r}; pass base_url explicitly")
            base_url = known[0]
        self.name = name
        self.local = name in LOCAL_BACKENDS
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._headers = dict(headers or {})
        self._client = client
        self._capabilities: dict[str, Capabilities] = {}

    # -- wiring ---------------------------------------------------------- #

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=_TIMEOUT)
        return self._client

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        try:
            response = self.client.post(
                f"{self.base_url}{path}", json=payload, headers=self._request_headers()
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(f"{self.name}: HTTP {response.status_code}: {response.text[:200]}")
        return response

    # -- protocol -------------------------------------------------------- #

    def capabilities(self, model: str) -> Capabilities:
        """What we know without asking.

        OpenAI-compatible servers expose no capability endpoint, so this is a
        declared default. Register real values with :meth:`declare` when known.
        """
        if model in self._capabilities:
            return self._capabilities[model]
        return Capabilities(
            context_window=_DEFAULT_CONTEXT,
            tool_calling=True,
            json_mode=True,
            vision=False,
        )

    def declare(self, model: str, capabilities: Capabilities) -> None:
        """Record known capabilities for a model on this backend."""
        self._capabilities[model] = capabilities

    def list_models(self) -> list[str]:
        try:
            response = self.client.get(f"{self.base_url}/models", headers=self._request_headers())
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: could not list models: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(f"{self.name}: HTTP {response.status_code} listing models")
        return sorted(str(item.get("id", "")) for item in response.json().get("data", []))

    def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        **options: Any,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.as_wire() for m in messages],
            **options,
        }
        if tools:
            payload["tools"] = list(tools)
        body = self._post("/chat/completions", payload).json()
        return self._to_completion(body, model)

    def stream(self, model: str, messages: Sequence[Message], **options: Any) -> Iterator[Chunk]:
        payload = {
            "model": model,
            "messages": [m.as_wire() for m in messages],
            "stream": True,
            **options,
        }
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._request_headers(),
            ) as response:
                for line in response.iter_lines():
                    chunk = _parse_sse(line)
                    if chunk is not None:
                        yield chunk
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: stream failed: {exc}") from exc

    def _to_completion(self, body: dict[str, Any], model: str) -> Completion:
        choices = body.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.name}: response contained no choices")
        message = choices[0].get("message") or {}
        raw_usage = body.get("usage") or {}
        usage = Usage(
            input_tokens=int(raw_usage.get("prompt_tokens", 0)),
            output_tokens=int(raw_usage.get("completion_tokens", 0)),
        )
        return Completion(
            text=str(message.get("content") or ""),
            model=str(body.get("model") or model),
            tool_calls=_parse_tool_calls(message.get("tool_calls")),
            usage=usage.priced(self.capabilities(model)),
            finish_reason=str(choices[0].get("finish_reason") or "stop"),
        )


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for item in raw:
        function = item.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"_raw": arguments}
        calls.append(
            ToolCall(
                id=str(item.get("id", "")),
                name=str(function.get("name", "")),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return calls


def _parse_sse(line: str) -> Chunk | None:
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return Chunk(text="", done=True)
    try:
        payload = json.loads(data)
    except ValueError:
        return None
    choices = payload.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    return Chunk(text=str(delta.get("content") or ""), done=bool(choices[0].get("finish_reason")))
