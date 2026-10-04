"""What a conversation about one file remembers between turns.

§8's hand-rolled state, finally with something to hold. One question at a time
needs no state at all, which is why `ask` has none. A second question that says
"now break that down by region" needs to know what *that* was, and the only
honest answer is the turn before it.

The rule the card exists to enforce applies here twice over. A transcript is
sent on every turn, so a raw row leaked into it leaks on every turn thereafter.
So a turn records the question, the code, and the *shape* of what came back:
column names and a row count. Never the rows. A scalar is carried because a
scalar that reached a result is already an aggregate of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import polars as pl

from insightsmith.engine import Engine
from insightsmith.profiling import Profile
from insightsmith.profiling.card import DatasetCard

__all__ = ["MAX_TRANSCRIPT_BYTES", "MAX_TURNS", "SessionState", "Turn"]

#: Roughly a third of a card. Enough for a handful of turns beside it, and small
#: enough that a long session cannot push the card out of a 4k context window.
MAX_TRANSCRIPT_BYTES: Final = 1_800
#: Recency beats completeness: a follow-up refers to the last turn or two, and a
#: question from twenty turns ago is noise competing for the same tokens.
MAX_TURNS: Final = 6


@dataclass(slots=True)
class Turn:
    """One exchange, reduced to what a later turn needs to build on it."""

    question: str
    code: str
    kind: str = "none"
    #: Column names of the result, which is what a follow-up actually refers to.
    columns: list[str] = field(default_factory=list)
    rows: int = 0
    #: A scalar result. Already an aggregate, so carrying it leaks nothing a
    #: mean or a count would not.
    value: Any = None
    verdict: str = ""
    confidence: float | None = None
    narrative: str = ""

    @property
    def outcome(self) -> str:
        """The result, described rather than reproduced."""
        if self.kind == "frame":
            named = ", ".join(self.columns[:8])
            more = "" if len(self.columns) <= 8 else f", +{len(self.columns) - 8} more"
            return f"a table of {self.rows} row(s): [{named}{more}]"
        if self.kind == "value":
            return f"the single value {self.value!r}"
        return "no result"

    def render(self) -> str:
        """This turn as the model will read it."""
        lines = [f"Q: {self.question}", f"Code:\n{self.code}", f"Result: {self.outcome}"]
        if self.verdict and self.verdict != "sound":
            lines.append(f"Critic: {self.verdict}")
        return "\n".join(lines)


@dataclass(slots=True)
class SessionState:
    """One file, one conversation, for the life of the process."""

    source: Path
    card: DatasetCard
    profile: Profile
    frame: pl.DataFrame
    engine: Engine
    turns: list[Turn] = field(default_factory=list)
    #: Toggled from the REPL, so they live with the session rather than the loop.
    charts: bool = False
    show_code: bool = True
    critique: bool = True

    def record(self, turn: Turn) -> None:
        self.turns.append(turn)

    def undo(self) -> Turn | None:
        """Drop the last turn, for when a question was asked badly."""
        return self.turns.pop() if self.turns else None

    def forget(self) -> int:
        """Drop the transcript, keeping the file. Returns how many went."""
        count = len(self.turns)
        self.turns.clear()
        return count

    def transcript(self, *, budget: int = MAX_TRANSCRIPT_BYTES, limit: int = MAX_TURNS) -> str:
        """The recent turns that fit, oldest dropped first.

        Trimmed from the front rather than truncated at the end: half a snippet
        teaches a wrong API, and the turn a follow-up refers to is the last one.
        """
        if not self.turns:
            return ""
        kept: list[str] = []
        used = 0
        for turn in reversed(self.turns[-limit:]):
            rendered = turn.render()
            size = len(rendered.encode("utf-8")) + 2
            if kept and used + size > budget:
                break
            kept.append(rendered)
            used += size
        return "\n\n".join(reversed(kept))

    def context(self) -> str:
        """The transcript, framed so the model knows what it is reading."""
        body = self.transcript()
        if not body:
            return ""
        return (
            "Earlier in this session, against the same dataframe:\n\n"
            f"{body}\n\n"
            "The next question may refer to those results. Write fresh code that "
            "answers it from `df`; you cannot reuse variables from earlier turns.\n\n"
        )
