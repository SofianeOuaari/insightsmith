"""The critic: everything measurable, plus the one thing that isn't.

§8 calls this the piece that separates the project from a toy, and most of what
it lists is arithmetic — :mod:`insightsmith.critique` computes it and never asks
anyone. What arithmetic cannot settle is whether the snippet answered *the
question that was asked*, because that compares an intent to a result. A model
is good at exactly that, and bad at pretending to a statistical judgement it
cannot make, so that comparison is all it is asked for.

The confidence score is computed from the caveats, never returned by the model.
A number a language model picks for its own certainty measures nothing.

The privacy rule holds here as everywhere: the critic sees the question, the
code, and the *shape* of the result. It never sees a result row, which matters
more here than elsewhere — a result can be `df.head(20)`, which is raw data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import polars as pl

from insightsmith.agents.base import Agent
from insightsmith.critique import (
    Caveat,
    Critique,
    Severity,
    confidence_for,
    review,
    verdict_for,
)
from insightsmith.profiling import Profile
from insightsmith.profiling.card import DatasetCard

__all__ = ["ANSWERED_SCHEMA", "CriticAgent"]

ANSWERED_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "answers_the_question": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["answers_the_question"],
}

_SYSTEM = """\
You judge one thing: does this result answer the question that was asked?

You are not reviewing style, efficiency, or whether the number is surprising. \
You are not judging whether the statistics are sound — that is measured \
elsewhere. Ask only: would someone who asked this question get an answer to \
*that* question from this result?

Answer false when the code computes something adjacent — a total when a rate \
was asked for, one group when a comparison was asked for, a column the question \
never mentioned. Answer true when the result answers the question, even if it \
is boring, small, or not what you expected.

You see the question, the code, and the shape of the result. You never see the \
rows, so do not reason about particular values.

Reply with a single JSON object: {"answers_the_question": true, "reason": "..."}.\
"""


@dataclass(slots=True)
class CriticAgent(Agent):
    """Measures what it can, asks about what it cannot, and scores the result."""

    role: str = "critic"
    #: Ask a model whether the question was answered. With this off the critique
    #: is purely computed — slower to be wrong, and free.
    consult_model: bool = True

    def system_prompt(self) -> str:
        return _SYSTEM

    def review(
        self,
        *,
        question: str,
        code: str,
        profile: Profile,
        card: DatasetCard | None = None,
        frame: pl.DataFrame | None = None,
        value: Any = None,
    ) -> Critique:
        """Everything that can be said about this answer, and how much survives."""
        caveats = review(question=question, code=code, profile=profile, frame=frame, value=value)
        answered, reason = self._answered(question, code, frame, value, card)
        if answered is False:
            caveats = [
                Caveat(
                    code="wrong-question",
                    severity=Severity.SERIOUS,
                    message=reason or "the code does not answer the question that was asked.",
                ),
                *caveats,
            ]
        return Critique(
            verdict=verdict_for(caveats, answered),
            caveats=caveats,
            confidence=confidence_for(caveats, answered),
            answered=answered,
            answered_reason=reason,
        )

    def _answered(
        self,
        question: str,
        code: str,
        frame: pl.DataFrame | None,
        value: Any,
        card: DatasetCard | None,
    ) -> tuple[bool | None, str]:
        """``None`` when no model was consulted — silence, not approval."""
        if not self.consult_model:
            return None, ""
        try:
            payload = self.ask(card, _prompt(question, code, frame, value), ANSWERED_SCHEMA)
        except Exception:
            # A critic that cannot be reached must not become a verdict of
            # "fine". The computed caveats still stand on their own.
            return None, ""
        answered = payload.get("answers_the_question")
        if not isinstance(answered, bool):
            return None, ""
        return answered, str(payload.get("reason") or "").strip()


def _prompt(question: str, code: str, frame: pl.DataFrame | None, value: Any) -> str:
    return (
        f"Question: {question}\n\n"
        f"--- code ---\n{code}\n\n"
        f"--- result ---\n{_shape(frame, value)}\n\n"
        "Does this result answer the question?"
    )


def _shape(frame: pl.DataFrame | None, value: Any) -> str:
    """The result's form, never its contents."""
    if frame is not None:
        columns = ", ".join(f"{name}: {dtype}" for name, dtype in frame.schema.items())
        return f"a table of {frame.height} row(s) with columns — {columns}"
    if value is None:
        return "nothing was assigned to `result`"
    return f"a single {type(value).__name__} value"
