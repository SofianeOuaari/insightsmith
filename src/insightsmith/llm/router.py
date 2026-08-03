"""Role to model, with a capability-driven fallback.

The router reads :class:`Capabilities` before dispatching. An agent that needs
structured output never gets handed a model that cannot do tool-calling and then
fails at runtime — it takes the prompted-JSON path instead, with a bounded
retry-on-parse-failure loop. For the small local models this project targets,
that path is the common case, not the exception.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from insightsmith.config import Config, load_config
from insightsmith.errors import BudgetError, ProviderError
from insightsmith.llm.base import Capabilities, Completion, Message, Provider
from insightsmith.llm.registry import build_provider, is_local_model, split_model

__all__ = ["MAX_JSON_RETRIES", "Route", "Router", "Strategy", "extract_json"]

#: How many times a prompted-JSON reply may fail to parse before giving up.
MAX_JSON_RETRIES: Final = 3
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class Strategy(str, Enum):
    """How structured output will be obtained from this model."""

    TOOL_CALLING = "tool_calling"
    JSON_MODE = "json_mode"
    PROMPTED_JSON = "prompted_json"


@dataclass(slots=True)
class Route:
    role: str
    reference: str
    provider: Provider
    model: str
    capabilities: Capabilities
    strategy: Strategy

    @property
    def local(self) -> bool:
        return bool(getattr(self.provider, "local", False))


@dataclass(slots=True)
class Router:
    """Resolves roles to providers and enforces the session budget."""

    config: Config = field(default_factory=load_config)
    spent_usd: float = 0.0
    _providers: dict[str, Provider] = field(default_factory=dict)
    _routes: dict[str, Route] = field(default_factory=dict)

    def provider_for(self, name: str) -> Provider:
        if name not in self._providers:
            self._providers[name] = build_provider(name, base_urls=self.config.base_urls)
        return self._providers[name]

    def route(self, role: str) -> Route:
        """Resolve a role, reading capabilities to choose a strategy.

        Raises:
            ProviderError: if the role is unconfigured, or local_only is set and
                the configured model is remote.
        """
        if role in self._routes:
            return self._routes[role]

        reference = self.config.model_for(role)
        if reference is None:
            raise ProviderError(
                f"no model configured for role {role!r}; set roles.{role} in config.toml"
            )
        if self.config.budget.local_only and not is_local_model(reference):
            raise ProviderError(
                f"local_only is set but role {role!r} uses remote model {reference!r}"
            )

        provider_name, model = split_model(reference)
        provider = self.provider_for(provider_name)
        capabilities = provider.capabilities(model)

        if capabilities.tool_calling:
            strategy = Strategy.TOOL_CALLING
        elif capabilities.json_mode:
            strategy = Strategy.JSON_MODE
        else:
            strategy = Strategy.PROMPTED_JSON

        route = Route(
            role=role,
            reference=reference,
            provider=provider,
            model=model,
            capabilities=capabilities,
            strategy=strategy,
        )
        self._routes[role] = route
        return route

    def complete(self, role: str, messages: Sequence[Message], **options: Any) -> Completion:
        """Plain completion for a role, with budget accounting."""
        route = self.route(role)
        completion = route.provider.chat(route.model, messages, **options)
        self._charge(completion)
        return completion

    def structured(
        self,
        role: str,
        messages: Sequence[Message],
        *,
        schema: dict[str, Any] | None = None,
        retries: int = MAX_JSON_RETRIES,
    ) -> dict[str, Any]:
        """Get a JSON object back, whatever the model is capable of.

        Tool-calling models are asked for a call. Everything else is asked in the
        prompt and re-asked, with the failure fed back, until it parses or the
        retry budget runs out.

        Raises:
            ProviderError: if no valid JSON arrived within ``retries`` attempts.
        """
        route = self.route(role)

        if route.strategy is Strategy.TOOL_CALLING and schema is not None:
            tool = {
                "type": "function",
                "function": {
                    "name": "respond",
                    "description": "Return the answer as structured data.",
                    "parameters": schema,
                },
            }
            completion = route.provider.chat(route.model, messages, tools=[tool])
            self._charge(completion)
            if completion.tool_calls:
                return completion.tool_calls[0].arguments
            # The model ignored the tool and answered in prose. Try to salvage it
            # rather than failing outright.
            salvaged = extract_json(completion.text)
            if salvaged is not None:
                return salvaged

        conversation = list(messages)
        if route.strategy is not Strategy.TOOL_CALLING:
            conversation = [_json_instruction(schema), *conversation]

        last_error = "no attempt made"
        for _ in range(max(1, retries)):
            options: dict[str, Any] = {}
            if route.strategy is Strategy.JSON_MODE:
                options["json_mode"] = True
            completion = route.provider.chat(route.model, conversation, **options)
            self._charge(completion)

            parsed = extract_json(completion.text)
            if parsed is not None:
                return parsed

            last_error = f"could not parse JSON from: {completion.text[:200]}"
            conversation = [
                *conversation,
                Message(role="assistant", content=completion.text),
                Message(
                    role="user",
                    content=(
                        "That was not valid JSON. Reply with a single JSON object and "
                        "nothing else — no prose, no code fence."
                    ),
                ),
            ]

        raise ProviderError(f"{route.reference} did not return valid JSON: {last_error}")

    def _charge(self, completion: Completion) -> None:
        cost = completion.usage.cost_usd
        if cost is None:
            return
        self.spent_usd += cost
        if self.spent_usd > self.config.budget.max_usd_per_session:
            raise BudgetError(
                f"session budget of ${self.config.budget.max_usd_per_session:.2f} exceeded "
                f"(${self.spent_usd:.2f} spent)"
            )


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model's reply.

    Small models wrap JSON in prose or a code fence however firmly they are told
    not to, so try the whole string, then any fenced block, then the outermost
    braces.
    """
    for candidate in _candidates(text):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _candidates(text: str) -> list[str]:
    stripped = text.strip()
    out = [stripped]
    out.extend(match.strip() for match in _FENCE.findall(stripped))
    start, end = stripped.find("{"), stripped.rfind("}")
    if 0 <= start < end:
        out.append(stripped[start : end + 1])
    return out


def _json_instruction(schema: dict[str, Any] | None) -> Message:
    body = "Reply with a single JSON object and nothing else."
    if schema is not None:
        body += f" It must match this JSON schema:\n{json.dumps(schema, indent=2)}"
    return Message(role="system", content=body)
