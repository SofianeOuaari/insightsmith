"""Saying what the numbers mean, after the numbers are in.

§8 puts the narrator after the chart, and the guide's own §14.6 says why: a
table is evidence, not an answer. "Avoid dumping raw output as the final answer,
synthesize it into plain-language sentences with the specific numbers embedded."
A column of eleven averages leaves the reader to do the comparison the question
already asked for.

This is the one agent that has to see values. Everywhere else the model reads a
dataset card precisely so raw records stay put, and a result is derived data
rather than a dataset, but it is not automatically safe: a snippet is free to
assign ``df.head(20)``, which is raw records wearing a different name. So the
result goes through the same masking the card uses and the same row cap, and
what reaches the model is what the reader is already looking at, no more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import polars as pl

from insightsmith.agents.base import Agent
from insightsmith.critique import Critique
from insightsmith.profiling.pii import mask_value

__all__ = ["NARRATIVE_SCHEMA", "NarratorAgent", "readable"]

#: Enough rows to see a shape, few enough to stay inside a small context.
MAX_ROWS: Final = 25
#: Precision past this is noise a reader cannot use.
DECIMALS: Final = 3

NARRATIVE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {"headline": {"type": "string"}, "detail": {"type": "string"}},
    "required": ["headline"],
}

_SYSTEM = """\
You read a result table and say what it means, for someone who asked the \
question and is looking at the answer.

Write two things:
- "headline": one sentence naming the finding, with the specific numbers in it. \
Not "there is variation between groups" but "Engineers sleep best at 8.4 and \
Sales Representatives worst at 4.0, a gap of 4.4 points".
- "detail": two or three sentences on what else the table shows. Where the \
spread is, which groups cluster, anything that looks like an outlier or a very \
small group.

Rules:
- Describe only what is in the table. Never explain *why*, never speculate \
about causes, never recommend anything. You cannot see the study design.
- Say "is associated with", never "causes".
- If caveats are listed, respect them. Do not call a mean typical when you are \
told the column is skewed.
- Round numbers the way a person would when speaking.
- Never call a value high or low without checking it against the others in the \
table. Second-highest is not low.
- A group whose standard deviation is zero or missing holds a single row. Say \
so; do not call it consistent.
- Reply with a single JSON object.\
"""


@dataclass(slots=True)
class NarratorAgent(Agent):
    """Turns a result into a sentence a reader can act on."""

    role: str = "cheap"
    max_rows: int = MAX_ROWS

    def system_prompt(self) -> str:
        return _SYSTEM

    def narrate(
        self,
        question: str,
        *,
        frame: pl.DataFrame | None = None,
        value: Any = None,
        critique: Critique | None = None,
    ) -> str:
        """One paragraph, or an empty string if there is nothing to say.

        Never raises. A narration is a courtesy on top of an answer the reader
        already has, so failing to produce one must not cost them the answer.
        """
        body = _describe(question, frame, value, critique, self.max_rows)
        if not body:
            return ""
        try:
            payload = self.ask(None, body, NARRATIVE_SCHEMA)
        except Exception:
            return ""
        headline = str(payload.get("headline") or "").strip()
        detail = str(payload.get("detail") or "").strip()
        if headline and detail and headline[-1] not in ".!?":
            headline += "."
        return f"{headline} {detail}".strip()


def readable(value: Any) -> str:
    """A number as a person would write it, not as a float repr.

    ``7.8936170212765955`` is fifteen digits of precision the data never had.
    """
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        rounded = round(value, DECIMALS)
        return f"{int(rounded):,}" if rounded == int(rounded) else f"{rounded:,}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _describe(
    question: str,
    frame: pl.DataFrame | None,
    value: Any,
    critique: Critique | None,
    max_rows: int,
) -> str:
    """The prompt: the question, the masked result, and what is wrong with it."""
    if frame is None and value is None:
        return ""

    parts = [f"Question: {question}", ""]
    if frame is not None:
        if frame.height == 0:
            return ""
        parts.append(f"Result: {frame.height} row(s), columns {', '.join(frame.columns)}")
        parts.append(" | ".join(frame.columns))
        for row in frame.head(max_rows).iter_rows(named=True):
            cells = [mask_value(readable(cell), column=name)[0] for name, cell in row.items()]
            parts.append(" | ".join(cells))
        if frame.height > max_rows:
            parts.append(f"... and {frame.height - max_rows} more rows")
    else:
        parts.append(f"Result: a single value, {readable(value)}")

    if critique is not None and critique.caveats:
        parts += ["", "Caveats already established about this result:"]
        parts += [f"- {caveat.message}" for caveat in critique.caveats]

    parts += ["", "Say what this shows."]
    return "\n".join(parts)
