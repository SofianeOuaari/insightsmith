"""A run, rendered four ways.

One model, four surfaces. Markdown is the source of truth for prose, HTML adds
the charts and the styling, PDF is that HTML printed, and the notebook is the
run made executable again. They are separate functions rather than one with a
format flag because the notebook is not a rendering of a document at all — it
is the same findings rebuilt as code someone can run.

Three things go in every one of them, because §11.4 asks a result to be
auditable: the code that produced each answer, the hash of the card the model
saw, and the caveats the critic raised. A report that shows the number and
drops the caveat would undo the milestone before it.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import polars as pl

from insightsmith.critique import Critique
from insightsmith.errors import MissingDependencyError
from insightsmith.profiling import Profile

__all__ = [
    "MAX_TABLE_ROWS",
    "Finding",
    "Report",
    "render_html",
    "render_markdown",
    "render_notebook",
    "render_pdf",
]

#: A report is read, not queried. Past this many rows a table stops informing
#: and starts hiding the prose around it; the notebook re-runs the code for
#: anyone who wants all of it.
MAX_TABLE_ROWS: Final = 25
_TEMPLATE_DIR: Final = Path(__file__).parent / "templates"
_TEMPLATE: Final = "report.html.j2"


@dataclass(slots=True)
class Finding:
    """One question, its answer, and everything that qualifies it."""

    question: str
    code: str
    narrative: str = ""
    kind: str = "none"
    value: Any = None
    frame: pl.DataFrame | None = None
    critique: Critique | None = None
    #: How many times the coder had to write it. More than one is not a defect,
    #: but it is something a reader is entitled to see.
    attempts: int = 1
    #: Rendered charts, by path. PNG for print, HTML for the interactive one.
    png: Path | None = None
    #: The same figure in the opposite colour mode, so the page can follow the
    #: reader's theme rather than the one the run happened to be drawn in.
    png_flipped: Path | None = None
    chart_html: Path | None = None
    #: Which mode ``png`` was drawn in, which decides which of the two is which.
    dark: bool = False

    @property
    def caveats(self) -> list[str]:
        return [] if self.critique is None else [c.message for c in self.critique.caveats]

    @property
    def confidence(self) -> float | None:
        return None if self.critique is None else self.critique.confidence

    @property
    def verdict(self) -> str:
        return "" if self.critique is None else self.critique.verdict.value

    def table(self, limit: int = MAX_TABLE_ROWS) -> tuple[list[str], list[list[str]], int]:
        """Columns, stringified rows and how many were dropped off the end."""
        if self.frame is None:
            return [], [], 0
        shown = self.frame.head(limit)
        rows = [[_cell(value) for value in row] for row in shown.iter_rows()]
        return list(self.frame.columns), rows, max(0, self.frame.height - shown.height)

    @property
    def png_light(self) -> Path | None:
        """The light-surface figure, whichever pass produced it."""
        return self.png_flipped if self.dark else self.png

    @property
    def png_dark(self) -> Path | None:
        return self.png if self.dark else self.png_flipped

    @property
    def numeric(self) -> list[bool]:
        """Which columns hold numbers, so the template can align them right.

        Read off the dtype rather than guessed from the rendered string: an ID
        column of digits is text that happens to look numeric, and right-aligning
        it invites a reader to compare magnitudes that mean nothing.
        """
        if self.frame is None:
            return []
        return [self.frame.schema[name].is_numeric() for name in self.frame.columns]

    @property
    def tile(self) -> bool:
        """True when the answer is one number, which is a stat tile, not a table."""
        return self.frame is None and self.value is not None

    @property
    def answer(self) -> str:
        """The headline, for a reader who will not read the table."""
        if self.frame is not None:
            return f"{self.frame.height} rows x {self.frame.width} columns"
        return "no result" if self.value is None else _cell(self.value)


@dataclass(slots=True)
class Report:
    """Everything a forge run produced, ready to render."""

    source: Path
    profile: Profile
    findings: list[Finding] = field(default_factory=list)
    #: Ideas that were proposed but not answered, so the reader can see what was
    #: left on the table rather than assuming the list was exhaustive.
    unanswered: list[str] = field(default_factory=list)
    card_hash: str = ""
    engine: str = "polars"
    models: dict[str, str] = field(default_factory=dict)
    #: The mode the figures were drawn in, which the page opens in so that the
    #: first paint always matches. A reader may still switch either way.
    dark: bool = False
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def title(self) -> str:
        return f"{self.source.name} — analysis"

    @property
    def stamp(self) -> str:
        return self.created.strftime("%Y-%m-%d %H:%M UTC")

    @property
    def issues(self) -> list[str]:
        return [
            f"{issue.column}: {issue.message}" if issue.column else issue.message
            for issue in self.profile.issues
        ]

    @property
    def quality_count(self) -> int:
        return len(self.profile.issues)

    @property
    def shape(self) -> str:
        rows = f"~{self.profile.n_rows:,}" if self.profile.estimated else f"{self.profile.n_rows:,}"
        return f"{rows} rows x {self.profile.n_columns} columns"

    @property
    def sampled(self) -> str:
        """The honesty line, or empty when the profile saw the whole file."""
        if not self.profile.estimated:
            return ""
        return (
            f"Profiled on {self.profile.sampled_rows:,} sampled rows, so every figure "
            "below is an estimate of the whole file rather than a measurement of it."
        )


def _cell(value: Any) -> str:
    """One value, short enough to sit in a table cell."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".") if value == value else "NaN"
    if isinstance(value, int):
        return f"{value:,}"
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def render_markdown(report: Report) -> str:
    """The report as Markdown, which is also what the HTML prose is built from."""
    out: list[str] = [f"# {report.title}", "", f"*{report.stamp}*", ""]
    out += ["## The data", "", f"- Source: `{report.source}`", f"- Shape: {report.shape}"]
    out += [f"- Engine: {report.engine}"]
    if report.card_hash:
        out.append(f"- Card hash: `{report.card_hash}`")
    for role, model in sorted(report.models.items()):
        out.append(f"- {role.capitalize()} model: `{model}`")
    out.append("")
    if report.sampled:
        out += [f"> {report.sampled}", ""]
    if report.issues:
        out += ["### Quality notes", ""] + [f"- {line}" for line in report.issues] + [""]

    for index, finding in enumerate(report.findings, 1):
        out += [f"## {index}. {finding.question}", ""]
        if finding.narrative:
            out += [finding.narrative, ""]
        columns, rows, dropped = finding.table()
        if columns:
            out += ["| " + " | ".join(columns) + " |"]
            out += ["| " + " | ".join("---" for _ in columns) + " |"]
            out += ["| " + " | ".join(row) + " |" for row in rows]
            if dropped:
                out.append(f"\n*{dropped:,} further row(s) not shown.*")
            out.append("")
        elif finding.value is not None:
            out += [f"**{finding.answer}**", ""]
        figure = finding.png_light or finding.png
        if figure is not None:
            out += [f"![{finding.question}]({figure.name})", ""]
        if finding.caveats:
            out += ["**Caveats**", ""] + [f"- {c}" for c in finding.caveats] + [""]
        if finding.confidence is not None:
            out += [
                f"Confidence {finding.confidence:.2f} · verdict {finding.verdict} "
                f"· {finding.attempts} attempt(s)",
                "",
            ]
        out += [
            "<details><summary>Code</summary>",
            "",
            "```python",
            finding.code,
            "```",
            "",
            "</details>",
            "",
        ]

    if report.unanswered:
        out += ["## Not answered", ""]
        out += ["These were proposed and left unrun:", ""]
        out += [f"- {q}" for q in report.unanswered] + [""]
    out += _honesty_lines()
    return "\n".join(out).rstrip() + "\n"


def _honesty_lines() -> list[str]:
    """§12, in the deliverable rather than only in the README.

    A report is the artefact that leaves the machine and gets forwarded, so it
    is the one place the limits have to travel with the numbers.
    """
    return [
        "---",
        "",
        "## How to read this",
        "",
        "Every number here was produced by code a language model wrote, shown above each "
        "answer. Models write wrong code confidently, so the code is the claim and the "
        "number is only its output. The critic flags statistical problems it can measure "
        "and misses the ones it cannot. Confidence is an index computed from the caveats, "
        "not a probability that the answer is right.",
        "",
    ]


def render_html(report: Report) -> str:
    """The same report, styled, with the charts inline."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    sections = [
        {
            "finding": finding,
            "columns": finding.table()[0],
            "rows": finding.table()[1],
            "dropped": finding.table()[2],
        }
        for finding in report.findings
    ]
    return env.get_template(_TEMPLATE).render(report=report, sections=sections)


def render_pdf(page: str, *, base_url: Path | None = None) -> bytes:
    """``page`` printed. ``base_url`` is where relative image paths resolve from."""
    try:
        from weasyprint import HTML
    except ImportError as exc:  # pragma: no cover - exercised by a stub in tests
        raise MissingDependencyError("weasyprint", "pdf", purpose="writing a PDF") from exc
    root = str(base_url) if base_url is not None else None
    rendered: bytes = HTML(string=page, base_url=root).write_pdf()
    return rendered


def render_notebook(report: Report) -> str:
    """The run, rebuilt as a notebook that executes.

    Not a transcript of one. The snippets ran against a frame the sandbox had
    already read from Parquet, so the setup cell has to rebuild that same frame
    from the original file, and say plainly when the run was on a sample and
    re-running on the whole file will not reproduce the numbers.
    """
    cells: list[dict[str, Any]] = [
        _markdown_cell(
            f"# {report.title}\n\n"
            f"Generated by insightsmith on {report.stamp}. Every cell below is the code "
            f"that produced the answer above it.\n\n"
            f"- Source: `{report.source}`\n"
            f"- Shape: {report.shape}\n"
            f"- Card hash: `{report.card_hash or 'n/a'}`\n"
            + (f"\n> {report.sampled}\n" if report.sampled else "")
        ),
        _code_cell(_setup_source(report)),
    ]
    for index, finding in enumerate(report.findings, 1):
        heading = f"## {index}. {finding.question}"
        body = f"\n\n{finding.narrative}" if finding.narrative else ""
        if finding.caveats:
            body += "\n\n**Caveats**\n\n" + "\n".join(f"- {c}" for c in finding.caveats)
        if finding.confidence is not None:
            body += (
                f"\n\nConfidence {finding.confidence:.2f} · verdict {finding.verdict}"
                f" · {finding.attempts} attempt(s)"
            )
        cells.append(_markdown_cell(heading + body))
        cells.append(_code_cell(finding.code + "\nresult"))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1) + "\n"


def _setup_source(report: Report) -> str:
    """The cell that rebuilds ``df`` as the sandbox had it."""
    from insightsmith.engine import Engine, spec_for

    try:
        spec = spec_for(Engine(report.engine))
    except ValueError:  # pragma: no cover - engine strings come from the enum
        spec = spec_for(Engine.POLARS)
    note = (
        "# The run was on a sample; reading the whole file here will not reproduce it.\n"
        if report.profile.estimated
        else ""
    )
    return (
        f"{spec.preamble}\n"
        "from insightsmith import load, sniff\n\n"
        f"{note}"
        f"source = {str(report.source)!r}\n"
        f"df = load(sniff(source)).collect()"
        + ("" if spec.engine is Engine.POLARS else ".to_pandas()")
        + "\ndf.head()"
    )


def _markdown_cell(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list[str]:
    """nbformat wants a list of lines, each keeping its newline but the last."""
    split = text.splitlines()
    return [line + "\n" for line in split[:-1]] + split[-1:] if split else []


def escape(text: str) -> str:
    """Exposed for the template's few unescaped spots."""
    return html.escape(text)
