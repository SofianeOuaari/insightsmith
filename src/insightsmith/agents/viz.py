"""Choosing how to draw an answer.

The model picks a *form* and which column fills which role. It does not write
plotting code — :mod:`insightsmith.viz.render` draws the spec. So a chart cannot
be malformed by a bad snippet, only mis-chosen, and a mis-choice is checkable:
every column named must exist in the result, exactly as ideation checks its own.

Where the model declines to answer usefully there is still a defensible default,
because the shape of a two-column result already implies a form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import polars as pl

from insightsmith.agents.base import Agent
from insightsmith.viz.render import ChartSpec, Form

__all__ = ["CHART_SCHEMA", "VizAgent", "default_spec", "validate_spec"]

CHART_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "form": {"type": "string", "enum": [f.value for f in Form]},
        "x": {"type": "string"},
        "y": {"type": "string"},
        "series": {"type": ["string", "null"]},
        "title": {"type": "string"},
    },
    "required": ["form", "x", "y"],
}

_SYSTEM = """\
You choose how to draw a result table. You do not write code.

Pick the form by the job the reader has to do:
- comparing magnitude across categories -> "bar" (horizontal; long labels fit)
- the same, few short labels -> "column"
- a trend along an ordered axis (dates, periods) -> "line"
- the relationship between two measures -> "scatter"

Rules:
- x, y and series must be column names from the result, spelled exactly.
- y must be numeric. x is the category or the time axis.
- Use "series" only when a column genuinely splits the data into distinct
  groups that must be told apart. Leave it null otherwise.
- Give a short, specific title that states the finding, not the mechanism.
- Reply with a single JSON object.\
"""


@dataclass(slots=True)
class VizAgent(Agent):
    """Picks a chart spec for a result frame."""

    role: str = "cheap"

    def system_prompt(self) -> str:
        return _SYSTEM

    def choose(self, frame: pl.DataFrame, question: str, card: Any = None) -> ChartSpec:
        """Ask for a spec, falling back to the shape of the data.

        Never raises for a bad answer: an unusable spec becomes the default,
        because refusing to draw anything is a worse outcome than drawing the
        obvious thing.
        """
        fallback = default_spec(frame, question)
        if fallback is None:
            raise ValueError(_undrawable(frame))

        schema = {name: str(dtype) for name, dtype in frame.schema.items()}
        prompt = (
            f"Question: {question}\n\n"
            f"Result columns: {schema}\n"
            f"Rows: {frame.height}\n\n"
            "Choose the chart."
        )
        try:
            payload = self.ask(card, prompt, CHART_SCHEMA)
        except Exception:
            return fallback
        return validate_spec(payload, frame) or fallback


def _undrawable(frame: pl.DataFrame) -> str:
    """Why this result cannot be drawn, in the reader's terms.

    Three different shapes reach here, and saying "no numeric column" about a
    frame holding one numeric column sends the reader looking for the wrong
    thing — a single correlation coefficient is undrawable because there is
    nothing to plot it *against*, not because it isn't a number.
    """
    if frame.height == 0:
        return "the result is empty"
    if not any(dtype.is_numeric() for dtype in frame.schema.values()):
        return "the result has no numeric column to plot"
    return (
        f"a chart needs a column to plot against, and the result is a single column "
        f"({frame.columns[0]!r}). Ask for the values behind the number"
    )


def validate_spec(payload: Any, frame: pl.DataFrame) -> ChartSpec | None:
    """Return a spec only if every part of it is real. Otherwise ``None``.

    ``payload`` is whatever the model sent, so it is genuinely ``Any`` — the
    isinstance check below is load-bearing, not defensive decoration.
    """
    if not isinstance(payload, dict):
        return None

    try:
        form = Form(str(payload.get("form", "")).strip().lower())
    except ValueError:
        return None

    x, y = str(payload.get("x") or ""), str(payload.get("y") or "")
    if x not in frame.columns or y not in frame.columns:
        return None
    # A column plotted against itself is a diagonal line that says nothing, and
    # it is what a model reaches for when a result has only one column to offer.
    if x == y:
        return None
    if not frame.schema[y].is_numeric():
        return None

    series = payload.get("series")
    name = str(series) if series else None
    if name is not None and (name not in frame.columns or name in {x, y}):
        name = None

    return ChartSpec(
        form=form,
        x=x,
        y=y,
        series=name,
        title=str(payload.get("title") or "").strip(),
    )


def default_spec(frame: pl.DataFrame, question: str = "") -> ChartSpec | None:
    """The form the data's own shape implies.

    A temporal or numeric x is a trend, so it gets a line; anything else is a
    comparison, so it gets ranked bars. Used when there is no model to ask and
    whenever the model's answer does not survive validation.
    """
    numeric = [n for n, d in frame.schema.items() if d.is_numeric()]
    if not numeric:
        return None

    y = numeric[-1]
    others = [n for n in frame.columns if n != y]
    if not others:
        return None
    x = others[0]

    dtype = frame.schema[x]
    if dtype.is_temporal():
        form = Form.LINE
    elif dtype.is_numeric():
        # A line asserts that the x axis is ordered. Two unrelated measures are
        # not, so they get a scatter; a numeric axis that is already sorted and
        # unique is an index or a period, and a line is right there.
        column = frame[x]
        ordered = column.n_unique() == column.len() and column.is_sorted()
        form = Form.LINE if ordered else Form.SCATTER
    else:
        form = Form.BAR
    if form is Form.BAR and frame.height <= 6 and all(len(str(v)) <= 8 for v in frame[x]):
        form = Form.COLUMN

    series = None
    remaining = [n for n in others[1:] if frame.schema[n] == pl.String]
    if remaining and 1 < frame[remaining[0]].n_unique() <= 8:
        series = remaining[0]

    return ChartSpec(form=form, x=x, y=y, series=series, title=question.strip())
