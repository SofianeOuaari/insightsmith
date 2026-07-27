"""The ``ismith`` command line."""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from polars.exceptions import PolarsError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from insightsmith import __version__
from insightsmith.errors import InsightsmithError
from insightsmith.io.sniff import CONFIDENCE_THRESHOLD, Compression, SourceSpec, sniff
from insightsmith.profiling import ColumnProfile, Profile, profile
from insightsmith.profiling.quality import Severity

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Agentic data consultant. 0.1.0 ships the no-LLM foundation: detect, load, profile.",
)
console = Console()
errors = Console(stderr=True)


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[bool, typer.Option("--version", help="Show the version and exit.")] = False,
) -> None:
    if version:
        console.print(f"insightsmith {__version__}")
        raise typer.Exit


@app.command()
def look(
    path: Annotated[Path, typer.Argument(help="Data file to inspect.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the profile as JSON instead of tables.")
    ] = False,
) -> None:
    """Detect a file's real format, then profile it."""
    spec = None
    try:
        spec = sniff(path)
        result = profile(spec)
    except (InsightsmithError, OSError) as exc:
        _fail(exc)
    except PolarsError as exc:
        # The format was detected but the content does not parse. Surface what we
        # assumed, since a wrong guess is the likeliest cause.
        hint = ""
        if spec is not None:
            hint = f" (read as {spec.format.value}, {spec.confidence:.0%} confidence)"
        _fail(f"could not parse {path}{hint}: {_first_line(exc)}")

    if as_json:
        console.print_json(json.dumps(_as_dict(result), default=str))
        return
    _render(result)


def _fail(message: object) -> NoReturn:
    """Report and exit 1.

    ``escape`` matters: without it rich reads the ``[excel]`` in "pip install
    insightsmith[excel]" as a style tag and silently drops it, leaving the user
    with an install command that omits the extra they need.
    """
    errors.print(f"[bold red]error[/]: {escape(str(message))}")
    raise typer.Exit(1)


def _first_line(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0]


def _render(result: Profile) -> None:
    console.print(Panel(_source_lines(result.source), title="source", expand=False))
    console.print(result.summary())
    console.print(_columns_table(result))
    if result.candidate_keys:
        console.print(f"candidate keys: {', '.join(result.candidate_keys)}")
    if result.issues:
        console.print(_issues_table(result))


def _source_lines(spec: SourceSpec) -> str:
    lines = [
        f"[bold]format[/]    {spec.format.value}",
        f"[bold]encoding[/]  {spec.encoding}",
    ]
    if spec.compression is not Compression.NONE:
        lines.append(f"[bold]compressed[/] {spec.compression.value}")
    if spec.dialect is not None:
        d = spec.dialect
        lines.append(
            f"[bold]dialect[/]   delimiter={d.delimiter!r} decimal={d.decimal!r} "
            f"header={d.has_header}"
        )
    style = "green" if spec.confidence >= CONFIDENCE_THRESHOLD else "yellow"
    lines.append(f"[bold]confidence[/] [{style}]{spec.confidence:.0%}[/]")
    # Below the threshold the assumptions are the point, so never hide them.
    lines.extend(f"[yellow]assumed[/]   {w}" for w in spec.warnings)
    return "\n".join(lines)


def _columns_table(result: Profile) -> Table:
    table = Table(title="columns", header_style="bold")
    for column, justify in (
        ("column", "left"),
        ("dtype", "left"),
        ("semantic", "left"),
        ("nulls", "right"),
        ("unique", "right"),
        ("detail", "left"),
    ):
        table.add_column(column, justify=justify)  # type: ignore[arg-type]

    for col in result.columns:
        nulls = f"{col.null_rate:.0%}" if col.null_count else "-"
        table.add_row(
            col.name,
            col.schema.dtype,
            col.schema.semantic.value,
            nulls,
            f"{col.n_unique:,}",
            _detail(col),
        )
    return table


def _detail(col: ColumnProfile) -> str:
    if col.numeric is not None:
        n = col.numeric
        detail = f"min {n.minimum:g} · med {n.median:g} · max {n.maximum:g}"
        if col.iqr_outliers or col.modified_z_outliers:
            detail += f" · outliers {col.iqr_outliers} iqr / {col.modified_z_outliers} mad"
        return detail
    if col.temporal is not None:
        return f"{col.temporal.earliest} → {col.temporal.latest}"
    if col.categorical is not None and col.categorical.top:
        return " · ".join(f"{value} ({count})" for value, count in col.categorical.top[:3])
    if col.text is not None:
        t = col.text
        return f"length {t.min_length}-{t.max_length} (mean {t.mean_length:.1f})"
    return ""


def _issues_table(result: Profile) -> Table:
    table = Table(title="quality notes", header_style="bold")
    table.add_column("severity")
    table.add_column("column")
    table.add_column("note")
    for issue in result.issues:
        colour = "yellow" if issue.severity is Severity.WARNING else "dim"
        table.add_row(f"[{colour}]{issue.severity.value}[/]", issue.column or "-", issue.message)
    return table


def _as_dict(value: Any) -> Any:
    """Dataclasses to plain JSON, with enums and paths flattened to strings."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _as_dict(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_as_dict(v) for v in value]
    if isinstance(value, dict):
        return {k: _as_dict(v) for k, v in value.items()}
    return value
