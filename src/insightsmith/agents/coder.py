"""Turning a question into code, running it, and fixing it when it breaks.

Design doc §7 puts it plainly: failures are fuel. A traceback plus the offending
code goes back to the model, up to a bounded number of attempts, and then the
failure is surfaced honestly rather than dressed up. §8 wires the same loop into
the wider graph later.

The model sees the dataset card, never the data.
"""

from __future__ import annotations

import ast
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
#: Errors that mean the model named a column the data does not have.
_COLUMN_MISSING = re.compile(r"ColumnNotFound|not found|unable to find column", re.I)
#: Enough of the schema to correct a guess without spending the whole context.
_COLUMNS_SHOWN: Final = 60
#: `module 'polars' has no attribute 'sqrt'` — a numpy habit with a Polars answer.
_NO_MODULE_ATTRIBUTE = re.compile(r"module '(?:polars|pl)' has no attribute '(\w+)'")
#: `'Expr' object has no attribute 'div'` — usually a namespace or a spelling away.
_NO_EXPR_ATTRIBUTE = re.compile(
    r"'(Expr|Series|DataFrame|LazyFrame)' object has no attribute '(\w+)'"
)
#: `GroupBy.mean() takes 1 positional argument but 2 were given` — pandas shorthand.
_GROUPBY_ARGS = re.compile(r"GroupBy\.(\w+)\(\) takes \d+ positional argument")
#: `cannot create expression literal for value of type DataFrame` — a frame
#: handed to a context that wanted a column.
_FRAME_LITERAL = re.compile(r"expression literal for value of type (?:DataFrame|LazyFrame)")
#: `division with 'String' datatypes is not allowed` — a missing cast.
_STRING_ARITHMETIC = re.compile(r"with '(?:String|Utf8)' datatypes is not allowed")
#: `NameError: name 'sd_sp_atk' is not defined` — an agg alias used as a variable.
_UNDEFINED_NAME = re.compile(r"NameError: name '(\w+)' is not defined")
#: Where Polars keeps string, date and list operations.
_EXPR_NAMESPACES: Final = ("str", "dt", "list", "arr", "struct", "cat", "bin", "name")
#: The object each error names, so a suggestion is probed against the right one.
_PROBES: Final[dict[str, Any]] = {
    "DataFrame": pl.DataFrame(),
    "LazyFrame": pl.DataFrame().lazy(),
    "Series": pl.Series([1]),
}
#: pandas spells these as methods; Polars uses the operator or the dunder name.
_ARITHMETIC: Final[dict[str, str]] = {
    "div": "truediv",
    "divide": "truediv",
    "multiply": "mul",
    "subtract": "sub",
    "plus": "add",
}
#: Names carried over from pandas that Polars simply calls something else.
_PANDAS_HABITS: Final[dict[str, str]] = {
    "sort_values": "sort",
    "drop_duplicates": "unique",
    "isnull": "is_null",
    "isna": "is_null",
    "notnull": "is_not_null",
    "notna": "is_not_null",
    "fillna": "fill_null",
    "dropna": "drop_nulls",
    "astype": "cast",
    "merge": "join",
    "apply": "map_elements",
    "nlargest": "top_k",
    "nsmallest": "bottom_k",
    "query": "filter",
    "assign": "with_columns",
}
_OPERATORS: Final[dict[str, str]] = {
    "div": "/",
    "divide": "/",
    "multiply": "*",
    "subtract": "-",
    "plus": "+",
}
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
- A column marked `"numeric_text": true` holds numbers stored as text. Cast it \
before any arithmetic: `pl.col("x").cast(pl.Float64, strict=False)`.
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
                prompt = self._retry(question, "", "the reply contained no code", card)
                history.append(Attempt(code="", ok=False, error="no code in the reply"))
                continue

            if approve and on_code is not None and on_code(code) is False:
                raise ProviderError("the proposed code was not approved")

            verdict = check(code)
            if not verdict.allowed:
                history.append(Attempt(code=code, ok=False, refused=list(verdict.reasons)))
                prompt = self._retry(question, code, "; ".join(verdict.reasons), card)
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
            prompt = self._retry(question, code, error, card)

        if not history:
            raise ProviderError("no attempt was made")
        raise ProviderError(
            f"could not answer after {len(history)} attempt(s). "
            f"Last failure: {_summarise(history[-1])}"
        )

    def _retry(self, question: str, code: str, error: str, card: DatasetCard | None = None) -> str:
        """Re-retrieve against the failure as well as the question.

        A traceback names the thing the model got wrong — ``no attribute
        'groupby'`` — which is a far sharper query than the question was.
        """
        return self.reference_for(question, failure=_exception_lines(error)) + _retry_prompt(
            question, code, error, _correction(error, card)
        )


def _ask_prompt(question: str) -> str:
    return f"Question: {question}\n\nWrite a Polars snippet that answers it, assigning to `result`."


def extract_code(payload: dict[str, Any]) -> str:
    """Pull the snippet out, tolerating how models actually reply.

    Two habits, both cheap to forgive: a code fence it was told not to use, and
    a JSON string escaped twice, so the newlines arrive as a literal backslash
    and an ``n``. The second is fatal and silent — Python reads it as a line
    continuation and every retry reproduces it — so it is worth repairing here.
    """
    code = payload.get("code")
    if not isinstance(code, str):
        return ""
    fenced = _FENCE.search(code)
    return _unescape_if_broken((fenced.group(1) if fenced else code).strip())


def _unescape_if_broken(code: str) -> str:
    """Undo double-escaping, but only when it is what breaks the snippet.

    Two readings of a literal ``\\n``, and which one is right depends on what the
    model meant. Between statements it stands for a newline; part-way through a
    chained expression it stands for a continuation, and a real newline there is
    just as broken — ``.select(...)`` on its own line is not valid Python without
    surrounding parentheses. So both are tried and whichever parses wins.

    Verifying before and after is what makes this safe: working code is never
    touched, so a snippet that genuinely splits on ``"\\n"`` keeps its escape.
    """
    if not code or _parses(code):
        return code
    for whitespace in ("\n", " "):
        repaired = code.replace("\\n", whitespace).replace("\\t", "\t")
        if repaired != code and _parses(repaired):
            return repaired
    return code


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


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


def _correction(error: str, card: DatasetCard | None) -> str:
    """What to use instead, when the failure implies a specific answer.

    A traceback says what broke, never what would have worked, so a model can
    spend every remaining attempt rediscovering the same wrong name. Two failures
    do imply their own fix, and both are cheap to state.
    """
    return (
        _existing_columns(error, card)
        or _expression_method(error)
        or _expression_attribute(error)
        or _bare_name(error)
        or _text_arithmetic(error)
        or _frame_as_literal(error)
        or _groupby_shorthand(error)
    )


def _existing_columns(error: str, card: DatasetCard | None) -> str:
    """The real column list, but only when the model just invented one.

    A missing-column error is the one failure where repeating the schema is
    worth its tokens: the model has stopped reading the card and started
    guessing, and nothing else in the retry contradicts the guess.
    """
    if card is None or not _COLUMN_MISSING.search(error):
        return ""
    names = sorted(card.column_names())
    if not names:
        return ""
    shown = ", ".join(repr(name) for name in names[:_COLUMNS_SHOWN])
    if len(names) > _COLUMNS_SHOWN:
        shown += f", and {len(names) - _COLUMNS_SHOWN} more"
    return f"the columns that exist are: {shown}"


def _expression_method(error: str) -> str:
    """``pl.sqrt(x)`` is numpy's shape; in Polars the maths lives on the column.

    Checked against the installed polars rather than a list kept here, so the
    hint cannot drift from the library and is never offered for a name that
    genuinely has no expression form.
    """
    match = _NO_MODULE_ATTRIBUTE.search(error)
    if match is None:
        return ""
    name = match.group(1)
    if hasattr(pl, name) or not hasattr(pl.Expr, name):
        return ""
    return (
        f"there is no `pl.{name}()`, but every expression has `.{name}()` — "
        f'write `pl.col("x").{name}()` rather than `pl.{name}(pl.col("x"))`.'
    )


def _expression_attribute(error: str) -> str:
    """``.div()`` and ``.to_uppercase()`` are real, just not where pandas put them.

    Polars keeps string, date and list operations behind namespaces, and the
    arithmetic aliases under their dunder names. The traceback says only that the
    attribute is missing, which leaves the model guessing at a name it has
    already guessed wrong once.
    """
    match = _NO_EXPR_ATTRIBUTE.search(error)
    if match is None:
        return ""
    owner, name = match.group(1), match.group(2)
    probe = _PROBES.get(owner) if owner in _PROBES else pl.col("x")
    if hasattr(probe, name):
        return ""

    # Polars usually spells it the same way with an underscore, so try that
    # against the real object before consulting any table.
    for split in range(1, len(name)):
        spaced = f"{name[:split]}_{name[split:]}"
        if hasattr(probe, spaced):
            return f"Polars spells it `{spaced}` — write `.{spaced}()`, not `.{name}()`."

    for space in _EXPR_NAMESPACES:
        if hasattr(probe, space) and hasattr(getattr(probe, space), name):
            return (
                f"`.{name}()` lives on the `.{space}` namespace in Polars — "
                f'write `pl.col("x").{space}.{name}()`.'
            )

    alias = _ARITHMETIC.get(name)
    if alias and hasattr(probe, alias):
        return (
            f"Polars has no `.{name}()`; the operator `{_OPERATORS[name]}` works "
            f"directly on expressions, or use `.{alias}()`."
        )

    # Everything else carried over from pandas, checked against the installed
    # polars before it is offered so a suggestion cannot name a missing method.
    equivalent = _PANDAS_HABITS.get(name)
    if equivalent and (hasattr(probe, equivalent) or hasattr(pl.col("x"), equivalent)):
        return f"`.{name}()` is pandas; the Polars equivalent is `.{equivalent}()`."
    return ""


def _bare_name(error: str) -> str:
    """A name that is not defined is nearly always a column the model just made.

    Only `df` and `pl` exist in the snippet's namespace, so an undefined name in
    a Polars chain is an alias from an earlier `.agg()` being used as though it
    were a Python variable — by far the most frequent way these snippets fail.
    """
    match = _UNDEFINED_NAME.search(error)
    if match is None:
        return ""
    name = match.group(1)
    return (
        f"`{name}` is not a Python variable. Only `df` and `pl` exist. If it is a "
        f'column an earlier step created, refer to it as `pl.col("{name}")`; '
        "aliases from `.agg()` do not become names you can use directly."
    )


def _text_arithmetic(error: str) -> str:
    """Numbers that arrived as text, which no dtype in the card gives away."""
    if not _STRING_ARITHMETIC.search(error):
        return ""
    return (
        "that column holds numbers stored as text, so it has to be converted "
        'before any arithmetic: `pl.col("x").cast(pl.Float64, strict=False)`. '
        'The card marks such columns `"numeric_text": true`.'
    )


def _groupby_shorthand(error: str) -> str:
    """``group_by("a").mean("b")`` is pandas; Polars aggregates explicitly."""
    match = _GROUPBY_ARGS.search(error)
    if match is None:
        return ""
    name = match.group(1)
    return (
        f"`GroupBy.{name}()` takes no column in Polars — it applies to every "
        f"column at once. Name the column in `agg` instead: "
        f'`.group_by("k").agg(pl.col("x").{name}())`.'
    )


def _frame_as_literal(error: str) -> str:
    """A whole frame handed to something that wanted one column."""
    if not _FRAME_LITERAL.search(error):
        return ""
    return (
        "a DataFrame was passed where Polars expected an expression. Inside "
        "`select`, `filter`, `with_columns` and `agg`, refer to columns as "
        '`pl.col("x")` rather than slicing the frame first.'
    )


def _retry_prompt(question: str, code: str, error: str, correction: str = "") -> str:
    schema = f"\n--- what to use instead ---\n{correction}\n" if correction else ""
    return (
        f"Question: {question}\n\n"
        f"Your previous snippet failed.\n\n"
        f"--- code ---\n{code}\n\n"
        f"--- failure ---\n{error}\n"
        f"{schema}\n"
        "Fix it. Reply with the corrected snippet, still assigning to `result`. "
        "Remember this is Polars: group_by, filter(pl.col(...)), select, with_columns."
    )
