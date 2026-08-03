"""The provider contract.

Everything an agent needs from a model lives behind :class:`Provider`. The
router reads :class:`Capabilities` before dispatching, so an agent that needs
tool-calling is never handed a model that lacks it — it degrades to prompted
JSON instead. That path is not an edge case; it is the normal path for the small
local models this project is built around.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Capabilities",
    "Chunk",
    "Completion",
    "Message",
    "Provider",
    "ToolCall",
    "Usage",
]


@dataclass(slots=True)
class Capabilities:
    """What a model can do, and what it costs."""

    context_window: int
    max_output: int = 4096
    tool_calling: bool = False
    json_mode: bool = False
    vision: bool = False
    #: USD per million tokens. ``None`` for local models, which cost nothing.
    cost_in_per_mtok: float | None = None
    cost_out_per_mtok: float | None = None

    @property
    def is_free(self) -> bool:
        return self.cost_in_per_mtok is None and self.cost_out_per_mtok is None


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Message:
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    def as_wire(self) -> dict[str, Any]:
        """OpenAI chat-completions shape, which Ollama also accepts."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    def priced(self, capabilities: Capabilities) -> Usage:
        """Attach a cost, when the model has a published price."""
        if capabilities.cost_in_per_mtok is None or capabilities.cost_out_per_mtok is None:
            return self
        cost = (
            self.input_tokens * capabilities.cost_in_per_mtok
            + self.output_tokens * capabilities.cost_out_per_mtok
        ) / 1_000_000
        return Usage(self.input_tokens, self.output_tokens, cost)


@dataclass(slots=True)
class Completion:
    text: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"


@dataclass(slots=True)
class Chunk:
    text: str
    done: bool = False


@runtime_checkable
class Provider(Protocol):
    """A backend that can answer a chat request."""

    name: str
    #: False for anything that leaves the machine. The router refuses these
    #: outright when local_only is set.
    local: bool

    def capabilities(self, model: str) -> Capabilities: ...

    def list_models(self) -> list[str]: ...

    def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        **options: Any,
    ) -> Completion: ...

    def stream(
        self,
        model: str,
        messages: Sequence[Message],
        **options: Any,
    ) -> Iterator[Chunk]: ...
