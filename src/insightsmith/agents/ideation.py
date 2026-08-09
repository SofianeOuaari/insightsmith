"""Proposing analyses worth running.

The prompt is deliberately constrained: ranked objects with a fixed shape, and
any idea naming a column the card does not contain is dropped. That single check
removes most hallucination for the price of a set membership test (§6) — a model
that invents ``customer_lifetime_value`` cannot smuggle it past a column list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Final

from insightsmith.agents.base import Agent
from insightsmith.errors import ProviderError
from insightsmith.profiling.card import DatasetCard

__all__ = ["Idea", "IdeationAgent", "validate_ideas"]

MAX_IDEAS: Final = 8
_EFFORT = ("low", "medium", "high")
_ARTIFACTS = ("table", "chart", "number", "model")

IDEA_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "rationale": {"type": "string"},
                    "method": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "expected_artifact": {"type": "string", "enum": list(_ARTIFACTS)},
                    "effort": {"type": "string", "enum": list(_EFFORT)},
                },
                "required": [
                    "question",
                    "rationale",
                    "method",
                    "columns",
                    "expected_artifact",
                    "effort",
                ],
            },
        }
    },
    "required": ["ideas"],
}

_SYSTEM = """\
You are a careful data analyst proposing analyses for a dataset you can only see \
through its card. Rank ideas by what a decision-maker would find most useful.

Rules:
- Only reference columns that appear in the card. Never invent a column.
- Prefer questions the data can actually answer, given the row count and the \
quality notes.
- Say what method you would use, concretely.
- Reply with a single JSON object: {"ideas": [...]}.\
"""


@dataclass(slots=True)
class Idea:
    """One proposed analysis, already checked against the card."""

    question: str
    rationale: str = ""
    method: str = ""
    columns: list[str] = field(default_factory=list)
    expected_artifact: str = "table"
    effort: str = "medium"
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "question": self.question,
            "rationale": self.rationale,
            "method": self.method,
            "columns": self.columns,
            "expected_artifact": self.expected_artifact,
            "effort": self.effort,
        }


@dataclass(slots=True)
class IdeationAgent(Agent):
    """Turns a card into ranked, column-validated analysis ideas."""

    role: str = "planner"

    def system_prompt(self) -> str:
        return _SYSTEM

    def propose(self, card: DatasetCard, *, limit: int = MAX_IDEAS) -> list[Idea]:
        """Ask for ideas and return only those that survive validation.

        Raises:
            ProviderError: if nothing usable came back.
        """
        prompt = (
            f"Propose up to {limit} analyses for this dataset, best first. "
            "Use only the columns listed above."
        )
        payload = self.ask(card, prompt, IDEA_SCHEMA)
        ideas = validate_ideas(payload, card, limit=limit)
        if not ideas:
            raise ProviderError(
                "the model proposed no usable ideas — every suggestion referenced "
                "columns that are not in the dataset"
            )
        return ideas


def validate_ideas(
    payload: dict[str, Any] | list[Any], card: DatasetCard, *, limit: int = MAX_IDEAS
) -> list[Idea]:
    """Keep only well-formed ideas that reference real columns.

    Tolerates the shapes models actually return — a bare list, a ``{"ideas": …}``
    wrapper, or JSON encoded inside a string — because rejecting a good answer
    over its envelope is its own kind of failure.
    """
    raw = _unwrap(payload)
    known = card.column_names()
    out: list[Idea] = []

    for item in raw:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue

        columns = [str(c) for c in _as_list(item.get("columns"))]
        # The check that earns its keep: an invented column disqualifies the idea.
        if not columns or any(column not in known for column in columns):
            continue

        out.append(
            Idea(
                question=question,
                rationale=str(item.get("rationale") or "").strip(),
                method=str(item.get("method") or "").strip(),
                columns=columns,
                expected_artifact=_one_of(item.get("expected_artifact"), _ARTIFACTS, "table"),
                effort=_one_of(item.get("effort"), _EFFORT, "medium"),
                rank=len(out) + 1,
            )
        )
        if len(out) >= limit:
            break
    return out


def _unwrap(payload: dict[str, Any] | list[Any]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    candidate = payload.get("ideas", payload)
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except ValueError:
            return []
    if isinstance(candidate, dict):
        return [candidate]
    return candidate if isinstance(candidate, list) else []


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _one_of(value: Any, allowed: tuple[str, ...], fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else fallback
