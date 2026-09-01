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
from typing import TYPE_CHECKING, Any, Final

import polars as pl

from insightsmith.agents.base import Agent
from insightsmith.critique import Critique
from insightsmith.errors import ProviderError
from insightsmith.execution.gate import check
from insightsmith.execution.sandbox import DEFAULT_LIMITS, Limits, SandboxResult, run
from insightsmith.knowledge import CODER_EXCLUDES, DEFAULT_BUDGET, reference
from insightsmith.profiling import Profile
from insightsmith.profiling.card import DatasetCard

if TYPE_CHECKING:  # pragma: no cover - import cycle: the critic imports Answer
    from insightsmith.agents.critic import CriticAgent

__all__ = ["MAX_ATTEMPTS", "Answer", "Attempt", "CoderAgent", "extract_code"]

#: How many times the coder may be handed its own traceback (§7).
MAX_ATTEMPTS: Final = 3
#: How much of a failure fits on a terminal line before it stops being readable.
_SUMMARY_CHARS: Final = 300
#: How much traceback goes back to the model on a retry.
_ERROR_CHARS: Final = 1500
#: Lines kept from the exception onward — the class, then whatever it added.
_SUMMARY_LINES: Final = 3
_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)
_SNIPPET_FRAME = re.compile(r'File "snippet\.py", line (\d+)')
_EXCEPTION = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt|Exit)\b")

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

#: Framing for the retrieved excerpts. The guide is written for an analyst with a
#: file in front of them; the coder has neither a file nor permission to open one,
#: so the excerpts are introduced as API reference rather than as instructions.
_REFERENCE = """\
Polars reference — excerpts from the bundled guide, closest match first. Use them \
for API names and syntax only: `df` is already in memory, so ignore any file \
reading, plotting or printing they happen to show.

{sections}"""


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
    #: What the critic made of it, when one was run.
    critique: Critique | None = None

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
    #: Retrieve Polars reference for each attempt. A small local model's memory of
    #: the Polars API is the weakest link in the chain, and a wrong method name is
    #: far cheaper to prevent than to discover in a traceback.
    guide: bool = True
    guide_budget: int = DEFAULT_BUDGET

    def system_prompt(self) -> str:
        return _SYSTEM

    def reference_for(self, question: str, *, failure: str = "") -> str:
        """Guide excerpts for a question, or nothing when they are switched off.

        ``failure`` is the traceback from the previous attempt, and outweighs the
        question: it names the mistake, where the question only names the goal.
        """
        if not self.guide or self.guide_budget <= 0:
            return ""
        found = reference(question, focus=failure, budget=self.guide_budget, exclude=CODER_EXCLUDES)
        return f"{_REFERENCE.format(sections=found)}\n\n" if found else ""

    def answer(
        self,
        card: DatasetCard,
        frame: pl.DataFrame,
        question: str,
        *,
        attempts: int = MAX_ATTEMPTS,
        approve: bool = False,
        on_code: Any = None,
        critic: CriticAgent | None = None,
        profile: Profile | None = None,
    ) -> Answer:
        """Write code for ``question``, run it, and retry on failure.

        ``on_code`` is called with each snippet before it runs; returning False
        aborts. That is the human-in-the-loop layer, off unless asked for.

        With ``critic`` and ``profile``, a snippet that runs is also reviewed
        (§8's critic → retry arrow). Only one finding sends it back: answering a
        different question than the one asked. Statistical caveats describe the
        *data*, and rewriting the snippet cannot make the data less skewed — those
        ride along on the answer instead.

        Raises:
            ProviderError: if no attempt produced a usable answer.
        """
        history: list[Attempt] = []
        prompt = self.reference_for(question) + _ask_prompt(question)

        for _ in range(max(1, attempts)):
            payload = self.ask(card, prompt, CODE_SCHEMA)
            code = extract_code(payload)
            explanation = str(payload.get("explanation") or "").strip()
            if not code:
                prompt = self._retry(question, "", "the reply contained no code")
                history.append(Attempt(code="", ok=False, error="no code in the reply"))
                continue

            if approve and on_code is not None and on_code(code) is False:
                raise ProviderError("the proposed code was not approved")

            verdict = check(code)
            if not verdict.allowed:
                history.append(Attempt(code=code, ok=False, refused=list(verdict.reasons)))
                prompt = self._retry(question, code, "; ".join(verdict.reasons))
                continue

            outcome = run(code, frame, limits=self.limits, gate=verdict)
            if outcome.ok and outcome.kind != "none":
                critique = None
                if critic is not None and profile is not None:
                    critique = critic.review(
                        question=question,
                        code=code,
                        profile=profile,
                        card=card,
                        frame=outcome.frame,
                        value=outcome.value,
                    )
                last = len(history) + 1 >= max(1, attempts)
                if critique is not None and critique.answered is False and not last:
                    reason = critique.answered_reason or "it answers a different question"
                    history.append(
                        Attempt(code=code, ok=False, error=f"the critic rejected it: {reason}")
                    )
                    prompt = self.reference_for(question, failure=reason) + _rejected_prompt(
                        question, code, reason
                    )
                    continue
                history.append(Attempt(code=code, ok=True))
                return Answer(
                    question=question,
                    code=code,
                    explanation=explanation,
                    kind=outcome.kind,
                    value=outcome.value,
                    frame=outcome.frame,
                    attempts=history,
                    critique=critique,
                )

            error = _failure_text(outcome)
            history.append(Attempt(code=code, ok=False, error=error))
            prompt = self._retry(question, code, error)

        if not history:
            raise ProviderError("no attempt was made")
        raise ProviderError(
            f"could not answer after {len(history)} attempt(s). "
            f"Last failure: {_summarise(history[-1])}"
        )

    def _retry(self, question: str, code: str, error: str) -> str:
        """Re-retrieve against the failure as well as the question.

        A traceback names the thing the model got wrong — ``no attribute
        'groupby'`` — which is a far sharper query than the question was.
        """
        return self.reference_for(question, failure=_exception_lines(error)) + _retry_prompt(
            question, code, error
        )


def _ask_prompt(question: str) -> str:
    return f"Question: {question}\n\nWrite a Polars snippet that answers it, assigning to `result`."


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
    if outcome.ok:
        # The process ran cleanly and computed nothing the caller can use. Small
        # models reach for this by wrapping the work in a function and assigning
        # `result` inside it, where the runner cannot see it.
        return (
            "the snippet ran but never assigned `result` at the top level. "
            "Assign it directly, not inside a function"
        )
    return _tail((outcome.traceback or outcome.stderr or "unknown failure").strip(), _ERROR_CHARS)


def _tail(text: str, limit: int) -> str:
    """Keep the end of a traceback, cutting on a line boundary.

    A cut mid-line leaves the remainder of an indented frame sitting at column
    zero, where everything downstream reads it as the exception rather than as
    the stack — which is how ``/insightsmith-6h_nejwd/runner.py", line 11`` ends
    up presented to a reader as the thing that went wrong.
    """
    if len(text) <= limit:
        return text
    kept: list[str] = []
    used = 0
    for line in reversed(text.splitlines()):
        used += len(line) + 1
        if used > limit:
            break
        kept.append(line)
    return "\n".join(reversed(kept)) if kept else text[-limit:]


def _summarise(attempt: Attempt) -> str:
    """One line for a human, from an attempt that keeps the whole traceback.

    ``Attempt.error`` holds the full traceback so a failure stays inspectable,
    but printing a polars-internal stack at a terminal buries the one line that
    says what went wrong under twenty that do not.
    """
    if attempt.refused:
        return "; ".join(attempt.refused)
    summary = " ".join(_exception_lines(attempt.error).split())
    if not summary:
        return "unknown failure"
    # The snippet's own frame stays: it is the one location in the stack the
    # reader can act on. Every polars frame above it is ours to hide.
    frames = _SNIPPET_FRAME.findall(attempt.error)
    where = f"snippet.py line {frames[-1]}: " if frames else ""
    return f"{where}{summary}"[:_SUMMARY_CHARS]


def _exception_lines(error: str) -> str:
    """The part of a traceback that names the mistake, for retrieval.

    Frames are paths and line numbers — noise against a Polars guide, and enough
    of it to drown the one line that matters. The model still sees the whole
    traceback; only the query is narrowed.

    Anchoring on the exception rather than taking the last few lines matters:
    polars 1.44 appends a context stack and a hint *after* it, which a fixed tail
    window silently cuts the exception class out of.
    """
    lines = [
        line
        for line in error.splitlines()
        if line.strip() and not line.startswith((" ", "\t")) and not line.startswith("Traceback")
    ]
    if not lines:
        return error
    anchors = [index for index, line in enumerate(lines) if _EXCEPTION.match(line)]
    start = anchors[-1] if anchors else max(0, len(lines) - _SUMMARY_LINES)
    return "\n".join(lines[start : start + _SUMMARY_LINES])


def _rejected_prompt(question: str, code: str, reason: str) -> str:
    """The snippet ran. It answered something else."""
    return (
        f"Question: {question}\n\n"
        f"Your previous snippet ran without error, but it does not answer the "
        f"question.\n\n--- code ---\n{code}\n\n--- why ---\n{reason}\n\n"
        "Write a snippet that answers the question as asked, still assigning to "
        "`result`."
    )


def _retry_prompt(question: str, code: str, error: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Your previous snippet failed.\n\n"
        f"--- code ---\n{code}\n\n"
        f"--- failure ---\n{error}\n\n"
        "Fix it. Reply with the corrected snippet, still assigning to `result`. "
        "Remember this is Polars: group_by, filter(pl.col(...)), select, with_columns."
    )
