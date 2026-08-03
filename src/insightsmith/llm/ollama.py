"""Ollama, written natively rather than through its OpenAI-compatible endpoint.

Local inference needs things the OpenAI shape cannot express: ``/api/show`` for
the architecture the KV-cache maths depends on, ``/api/ps`` for what is resident,
``/api/pull`` with progress, and ``keep_alive`` / ``num_ctx`` / ``num_gpu``
options. Those are the whole reason to run locally, so they get a real client.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any, Final

import httpx

from insightsmith.errors import ProviderError
from insightsmith.llm.base import Capabilities, Chunk, Completion, Message, ToolCall, Usage

__all__ = ["DEFAULT_HOST", "OllamaProvider"]

DEFAULT_HOST: Final = "http://localhost:11434"
_TIMEOUT: Final = 300.0
_DEFAULT_CONTEXT: Final = 4096


class OllamaProvider:
    """A local Ollama server."""

    name = "ollama"
    local = True

    def __init__(self, host: str = DEFAULT_HOST, *, client: httpx.Client | None = None) -> None:
        self.host = host.rstrip("/")
        self._client = client
        self._shown: dict[str, dict[str, Any]] = {}

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=_TIMEOUT)
        return self._client

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            response = self.client.request(method, f"{self.host}{path}", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"ollama: {self.host} is not reachable ({exc}). Is `ollama serve` running?"
            ) from exc
        if response.status_code >= 400:
            raise ProviderError(f"ollama: HTTP {response.status_code}: {response.text[:200]}")
        return response.json()

    # -- native endpoints ------------------------------------------------ #

    def show(self, model: str) -> dict[str, Any]:
        """``/api/show`` — architecture and capabilities, cached per model."""
        if model not in self._shown:
            body = self._request("POST", "/api/show", {"model": model})
            self._shown[model] = body if isinstance(body, dict) else {}
        return self._shown[model]

    def resident(self) -> list[str]:
        """``/api/ps`` — models currently loaded in memory."""
        body = self._request("GET", "/api/ps")
        return [str(m.get("name", "")) for m in (body or {}).get("models", [])]

    def pull(self, model: str) -> Iterator[str]:
        """``/api/pull``, yielding progress lines."""
        try:
            with self.client.stream(
                "POST", f"{self.host}/api/pull", json={"model": model}
            ) as response:
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        status = json.loads(line).get("status")
                    except ValueError:
                        continue
                    if status:
                        yield str(status)
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama: pull failed: {exc}") from exc

    # -- protocol -------------------------------------------------------- #

    def capabilities(self, model: str) -> Capabilities:
        """Read from ``/api/show`` rather than assumed.

        Ollama reports a ``capabilities`` array, so tool-calling support is a
        fact about the model rather than a guess — which is what lets the router
        pick the prompted-JSON path deliberately instead of failing at runtime.
        """
        body = self.show(model)
        info = body.get("model_info") or {}
        listed = {str(c).lower() for c in (body.get("capabilities") or [])}
        context = next(
            (int(v) for k, v in info.items() if k.endswith("context_length")), _DEFAULT_CONTEXT
        )
        return Capabilities(
            context_window=context,
            max_output=min(4096, context),
            tool_calling="tools" in listed,
            json_mode=True,  # Ollama honours format=json for every model
            vision="vision" in listed,
        )

    def list_models(self) -> list[str]:
        body = self._request("GET", "/api/tags")
        return sorted(str(m.get("name", "")) for m in (body or {}).get("models", []))

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
            "stream": False,
        }
        if tools:
            payload["tools"] = list(tools)
        if options.pop("json_mode", False):
            payload["format"] = "json"
        for passthrough in ("keep_alive",):
            if passthrough in options:
                payload[passthrough] = options.pop(passthrough)
        if options:
            payload["options"] = options

        body = self._request("POST", "/api/chat", payload)
        message = (body or {}).get("message") or {}
        return Completion(
            text=str(message.get("content") or ""),
            model=str(body.get("model") or model),
            tool_calls=_parse_tool_calls(message.get("tool_calls")),
            usage=Usage(
                input_tokens=int(body.get("prompt_eval_count", 0)),
                output_tokens=int(body.get("eval_count", 0)),
            ),
            finish_reason=str(body.get("done_reason") or "stop"),
        )

    def stream(self, model: str, messages: Sequence[Message], **options: Any) -> Iterator[Chunk]:
        payload = {
            "model": model,
            "messages": [m.as_wire() for m in messages],
            "stream": True,
        }
        if options:
            payload["options"] = options
        try:
            with self.client.stream("POST", f"{self.host}/api/chat", json=payload) as response:
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        body = json.loads(line)
                    except ValueError:
                        continue
                    yield Chunk(
                        text=str((body.get("message") or {}).get("content") or ""),
                        done=bool(body.get("done")),
                    )
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama: stream failed: {exc}") from exc


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        function = item.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"_raw": arguments}
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{index}"),
                name=str(function.get("name", "")),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return calls
