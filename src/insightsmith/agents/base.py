"""What every agent shares: a router, a role, and a card to reason from."""

from __future__ import annotations

from dataclasses import dataclass

from insightsmith.llm.router import Router
from insightsmith.profiling.card import DatasetCard

__all__ = ["Agent"]


@dataclass(slots=True)
class Agent:
    """Base for agents. Holds the router; subclasses add the task.

    Agents never receive a dataframe. They receive a :class:`DatasetCard`, which
    is what keeps token cost flat and raw records off the wire.
    """

    router: Router
    role: str = "planner"

    def ask(self, card: DatasetCard, prompt: str, schema: dict[str, object]) -> dict[str, object]:
        """Send the card plus a prompt, and get structured output back."""
        from insightsmith.llm.base import Message

        messages = [
            Message(role="system", content=self.system_prompt()),
            Message(role="user", content=f"Dataset card:\n{card.to_json(indent=None)}\n\n{prompt}"),
        ]
        return self.router.structured(self.role, messages, schema=schema)

    def system_prompt(self) -> str:
        return "You are a careful data analyst."
