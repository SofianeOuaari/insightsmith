"""Turning a question into code, running it, and fixing it when it breaks.

Design doc §7 puts it plainly: failures are fuel. A traceback plus the offending
code goes back to the model, up to a bounded number of attempts, and then the
failure is surfaced honestly rather than dressed up. §8 wires the same loop into
the wider graph later.

The model sees the dataset card, never the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final

import polars as pl

from insightsmith.agents.base import Agent
from insightsmith.errors import ProviderError
from insightsmith.execution.gate import check
from insightsmith.execution.sandbox import DEFAULT_LIMITS, Limits, SandboxResult, run
from insightsmith.profiling.card import DatasetCard

__all__ = ["MAX_ATTEMPTS", "Answer", "Attempt", "CoderAgent", "extract_code"]

#: How many times the coder may be handed its own traceback (§7).
MAX_ATTEMPTS: Final = 3
_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)

CODE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {"code": {"type": "string"}, "explanation": {"type": "string"}},
    "required": ["code"],
}

_SYSTEM = """\
You write short **Polars** snippets to answer questions about a dataset you can \
only see through its card.

This is Polars, not pandas. The two APIs differ and pandas methods do not exist:
- `df.group_by("col").agg(pl.col("x").sum())`  — not `groupby()` / `.sum()`
- `df.filter(pl.col("x") > 1)`                 — not boolean indexing
- `df.select(pl.col("a"), pl.col("b"))`        — not `df[["a", "b"]]`
- `df.sort("x", descending=True)`              — not `sort_values(ascending=False)`
- `df.with_columns((pl.col("a") / pl.col("b")).alias("ratio"))`

Rules:
- A DataFrame named `df` already exists. Never read a file, never import os, sys, \
subprocess or pathlib.
- Assign your answer to a variable named `result`.
- `import polars as pl` is available. Use only columns that appear in the card, \
spelled exactly — including spaces and capitals.
- Keep it to a few lines. No printing, no plotting.
- Reply with a single JSON object: {"code": "...", "explanation": "..."}.\
"""


@dataclass(slots=True)
class Attempt:
    """One pass through write-run-check, kept so failures stay inspectable."""

    code: str
    ok: bool
    error: str = ""
    refused: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Answer:
    """The result, and everything it took to get there."""

    question: str
    code: str
    explanation: str = ""
    kind: str = "none"
    value: Any = None
    frame: pl.DataFrame | None = None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def display(self) -> str:
        if self.frame is not None:
            return f"{self.frame.height} rows x {self.frame.width} columns"
        return f"{self.value!r}"


@dataclass(slots=True)
class CoderAgent(Agent):
    """Question plus card in, executed answer out."""

    role: str = "coder"
    limits: Limits = DEFAULT_LIMITS

    def system_prompt(self) -> str:
        return _SYSTEM

    def answer(
        self,
        card: DatasetCard,
        frame: pl.DataFrame,
        question: str,
        *,
        attempts: int = MAX_ATTEMPTS,
        approve: bool = False,
        on_code: Any = None,
    ) -> Answer:
        """Write code for ``question``, run it, and retry on failure.

        ``on_code`` is called with each snippet before it runs; returning False
        aborts. That is the human-in-the-loop layer, off unless asked for.

        Raises:
            ProviderError: if no attempt produced a usable answer.
        """
        history: list[Attempt] = []
        prompt = (
            f"Question: {question}\n\n"
            "Write a Polars snippet that answers it, assigning to `result`."
        )

        for _ in range(max(1, attempts)):
            payload = self.ask(card, prompt, CODE_SCHEMA)
            code = extract_code(payload)
            explanation = str(payload.get("explanation") or "").strip()
            if not code:
                prompt = _retry_prompt(question, "", "the reply contained no code")
                history.append(Attempt(code="", ok=False, error="no code in the reply"))
                continue

            if approve and on_code is not None and on_code(code) is False:
                raise ProviderError("the proposed code was not approved")

            verdict = check(code)
            if not verdict.allowed:
                history.append(Attempt(code=code, ok=False, refused=list(verdict.reasons)))
                prompt = _retry_prompt(question, code, "; ".join(verdict.reasons))
                continue

            outcome = run(code, frame, limits=self.limits, gate=verdict)
            if outcome.ok:
                history.append(Attempt(code=code, ok=True))
                return Answer(
                    question=question,
                    code=code,
                    explanation=explanation,
                    kind=outcome.kind,
                    value=outcome.value,
                    frame=outcome.frame,
                    attempts=history,
                )

            error = _failure_text(outcome)
            history.append(Attempt(code=code, ok=False, error=error))
            prompt = _retry_prompt(question, code, error)

        raise ProviderError(
            f"could not answer after {len(history)} attempt(s). "
            f"Last failure: {history[-1].error or '; '.join(history[-1].refused)}"
            if history
            else "no attempt was made"
        )


def extract_code(payload: dict[str, Any]) -> str:
    """Pull the snippet out, tolerating a code fence the model was told not to use."""
    code = payload.get("code")
    if not isinstance(code, str):
        return ""
    fenced = _FENCE.search(code)
    return (fenced.group(1) if fenced else code).strip()


def _failure_text(outcome: SandboxResult) -> str:
    if outcome.timed_out:
        return "the snippet exceeded the time limit — it is probably looping"
    return (outcome.traceback or outcome.stderr or "unknown failure").strip()[-1500:]


def _retry_prompt(question: str, code: str, error: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Your previous snippet failed.\n\n"
        f"--- code ---\n{code}\n\n"
        f"--- failure ---\n{error}\n\n"
        "Fix it. Reply with the corrected snippet, still assigning to `result`. "
        "Remember this is Polars: group_by, filter(pl.col(...)), select, with_columns."
    )
