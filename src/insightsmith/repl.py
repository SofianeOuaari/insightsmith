"""The conversational surface: one file, many questions.

`ask` pays for startup on every question and forgets the last one. This keeps
the profile, the card and the frame in memory and hands the coder what the
session already established, so "now break that down by region" means something.

Rendering lives here rather than being shared with `cli.ask`. The two want
different things: a one-shot answer is printed and scrolled away, where a turn
in a conversation is read next to the turns around it and earns the frame. That
also keeps the import arrow pointing one way, from the CLI into the REPL.

Beautiful, at this milestone, means rich. §9 puts textual, streaming tokens and
a live plot preview at 1.5.0, and reaching for it here would spend that
milestone early and add a dependency for a surface that is still finding its
shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import polars as pl
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from insightsmith.agents.coder import Answer, CoderAgent
from insightsmith.agents.critic import CriticAgent
from insightsmith.agents.ideation import IdeationAgent
from insightsmith.agents.narrator import NarratorAgent, readable
from insightsmith.critique import Critique, Verdict
from insightsmith.engine import Engine
from insightsmith.errors import InsightsmithError
from insightsmith.llm.router import Router
from insightsmith.session import SessionState, Turn

__all__ = ["COMMANDS", "run"]

#: Rows of a result shown inline. A conversation scrolls, so a long table buries
#: the question above it; `/save` writes the whole thing out.
PREVIEW_ROWS: Final = 12
#: Where prompt_toolkit keeps the line history between sessions.
HISTORY_FILE: Final = Path.home() / ".insightsmith" / "history"

_VERDICT_STYLE: Final[dict[str, str]] = {
    Verdict.SOUND.value: "green",
    Verdict.QUALIFIED.value: "yellow",
    Verdict.UNSOUND.value: "red",
}


@dataclass(frozen=True, slots=True)
class Command:
    """One slash command: what it is called, and what it says it does."""

    name: str
    help: str


COMMANDS: Final[tuple[Command, ...]] = (
    Command("/help", "show this list"),
    Command("/columns", "the columns, with types and quality notes"),
    Command("/ideas", "ask the model what is worth analysing"),
    Command("/card", "exactly what the model is shown"),
    Command("/code", "toggle printing the code behind each answer"),
    Command("/chart", "toggle drawing a chart for each answer"),
    Command("/critique", "toggle the statistical critic"),
    Command("/engine", "switch dataframe API: /engine pandas"),
    Command("/save", "forge this session into a report: /save out/"),
    Command("/undo", "forget the last turn"),
    Command("/clear", "forget the whole conversation, keep the file"),
    Command("/exit", "leave (or Ctrl-D)"),
)


def run(
    state: SessionState,
    *,
    console: Console,
    router: Router | None = None,
    read: Callable[[str], str] | None = None,
) -> None:
    """The loop. ``read`` is injected so tests drive it without a terminal."""
    router = router or Router()
    ask_line = read or _reader(state)
    console.print(_banner(state))

    while True:
        try:
            line = ask_line("").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            return
        if not line:
            continue
        if line.startswith("/"):
            if _command(line, state, console=console, router=router) is False:
                return
            continue
        _answer(line, state, console=console, router=router)


def _reader(state: SessionState) -> Callable[[str], str]:
    """A prompt with history, suggestions, and the file's own column names.

    Completing on columns is the one piece of domain knowledge the input line
    can offer: a question fails most often on a column spelled from memory.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import FileHistory

    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        history: Any = FileHistory(str(HISTORY_FILE))
    except OSError:  # pragma: no cover - unwritable home
        history = None

    words = [command.name for command in COMMANDS] + list(state.frame.columns)
    session: Any = PromptSession(
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        completer=WordCompleter(words, ignore_case=True, sentence=True),
        complete_while_typing=False,
    )
    # Escaped rather than literal: a bare chevron trips ruff's ambiguous-character rule.
    prompt = ANSI("\x1b[38;5;39m\u276f\x1b[0m ")
    return lambda _: str(session.prompt(prompt))


def _banner(state: SessionState) -> Panel:
    """What you are talking to, in four lines."""
    profile = state.profile
    rows = f"~{profile.n_rows:,}" if profile.estimated else f"{profile.n_rows:,}"
    head = Text.assemble(
        (state.source.name, "bold"),
        ("  ·  ", "dim"),
        (f"{rows} rows x {profile.n_columns} columns", "default"),
        ("  ·  ", "dim"),
        (state.engine.value, "cyan"),
    )
    notes = len(profile.issues)
    hint = Text.assemble(
        (f"{notes} quality note{'' if notes == 1 else 's'}", "yellow" if notes else "dim"),
        ("  ·  ", "dim"),
        ("/help", "cyan"),
        (" for commands, ", "dim"),
        ("Ctrl-D", "cyan"),
        (" to leave", "dim"),
    )
    return Panel(
        Group(head, Text(""), hint),
        box=ROUNDED,
        border_style="cyan",
        title="[bold]insightsmith[/]",
        title_align="left",
        padding=(1, 2),
    )


def _answer(question: str, state: SessionState, *, console: Console, router: Router) -> None:
    """One turn: ask, run, critique, narrate, remember."""
    with console.status("[dim]thinking...[/]", spinner="dots"):
        try:
            answer = CoderAgent(router=router, engine=state.engine).answer(
                state.card,
                state.frame,
                question,
                critic=CriticAgent(router=router) if state.critique else None,
                profile=state.profile,
                context=state.context(),
            )
        except InsightsmithError as exc:
            console.print(Panel(str(exc), border_style="red", box=ROUNDED, title="failed"))
            return
        story = NarratorAgent(router=router).narrate(
            question, frame=answer.frame, value=answer.value, critique=answer.critique
        )

    index = len(state.turns) + 1
    console.print(_turn_panel(index, question, answer, story, state))
    state.record(
        Turn(
            question=question,
            code=answer.code,
            kind=answer.kind,
            columns=[] if answer.frame is None else list(answer.frame.columns),
            rows=0 if answer.frame is None else answer.frame.height,
            value=answer.value,
            verdict="" if answer.critique is None else answer.critique.verdict.value,
            confidence=None if answer.critique is None else answer.critique.confidence,
            narrative=story,
        )
    )


def _turn_panel(
    index: int, question: str, answer: Answer, story: str, state: SessionState
) -> Panel:
    parts: list[Any] = []
    if story:
        parts.append(Text(story))
    if answer.frame is not None:
        parts += [Text(""), _result_table(answer.frame)]
    elif answer.value is not None:
        parts += [Text(""), Text(str(readable(answer.value)), style="bold cyan")]
    if answer.critique is not None:
        parts += [Text(""), _verdict(answer.critique, len(answer.attempts))]
        for caveat in answer.critique.caveats:
            parts.append(Text(f"  · {caveat.message}", style="yellow"))
    if state.show_code and answer.code:
        parts += [
            Text(""),
            Syntax(answer.code, "python", theme="ansi_dark", background_color="default"),
        ]
    return Panel(
        Group(*parts),
        box=ROUNDED,
        border_style="dim",
        title=f"[cyan]{index}[/] [bold]{question}[/]",
        title_align="left",
        padding=(1, 2),
    )


def _result_table(frame: pl.DataFrame) -> Table:
    """The result, capped. Numeric columns right-aligned, as in the report."""
    table = Table(box=ROUNDED, border_style="dim", header_style="bold", expand=False)
    for name in frame.columns:
        numeric = frame.schema[name].is_numeric()
        table.add_column(name, justify="right" if numeric else "left")
    for row in frame.head(PREVIEW_ROWS).iter_rows():
        table.add_row(*(_cell(value) for value in row))
    if frame.height > PREVIEW_ROWS:
        table.caption = f"{frame.height - PREVIEW_ROWS:,} more row(s)"
    return table


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)[:48]


def _verdict(critique: Critique, attempts: int) -> Text:
    style = _VERDICT_STYLE.get(critique.verdict.value, "dim")
    return Text.assemble(
        ("● ", style),
        (critique.verdict.value, f"bold {style}"),
        ("   confidence ", "dim"),
        (f"{critique.confidence:.2f}", "default"),
        ("   ", "dim"),
        (f"{attempts} attempt{'' if attempts == 1 else 's'}", "dim"),
    )


def _command(line: str, state: SessionState, *, console: Console, router: Router) -> bool:
    """Run a slash command. Returns False to end the session."""
    name, _, argument = line.partition(" ")
    argument = argument.strip()
    name = name.lower()

    if name in ("/exit", "/quit", "/q"):
        console.print("[dim]bye[/]")
        return False
    if name in ("/help", "/?"):
        console.print(_help())
    elif name == "/columns":
        console.print(_columns(state))
    elif name == "/card":
        console.print(
            Panel(
                state.card.to_json(indent=2),
                box=ROUNDED,
                border_style="dim",
                title="[bold]what the model is shown[/]",
                title_align="left",
            )
        )
    elif name == "/ideas":
        _ideas(state, console=console, router=router)
    elif name in ("/code", "/chart", "/critique"):
        _toggle(name, state, console=console)
    elif name == "/engine":
        _engine(argument, state, console=console)
    elif name == "/save":
        _save(argument or "insightsmith-out", state, console=console, router=router)
    elif name == "/undo":
        dropped = state.undo()
        console.print(
            f"[dim]forgot:[/] {dropped.question}" if dropped else "[dim]nothing to forget[/]"
        )
    elif name == "/clear":
        console.print(f"[dim]forgot {state.forget()} turn(s); the file stays loaded[/]")
    else:
        console.print(f"[yellow]no such command:[/] {name}   [dim]try /help[/]")
    return True


def _help() -> Panel:
    table = Table.grid(padding=(0, 3))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="dim")
    for command in COMMANDS:
        table.add_row(command.name, command.help)
    body = Group(
        Text("Ask a question in plain words, or use a command.", style="dim"),
        Text(""),
        table,
    )
    return Panel(
        body, box=ROUNDED, border_style="dim", title="[bold]commands[/]", title_align="left"
    )


def _columns(state: SessionState) -> Table:
    table = Table(box=ROUNDED, border_style="dim", header_style="bold", title_justify="left")
    table.add_column("column")
    table.add_column("type", style="cyan")
    table.add_column("nulls", justify="right")
    table.add_column("unique", justify="right")
    flagged = {issue.column for issue in state.profile.issues if issue.column}
    for column in state.profile.columns:
        name = column.schema.name
        table.add_row(
            Text(name, style="yellow" if name in flagged else "default"),
            str(column.schema.dtype),
            "-" if not column.null_count else f"{column.null_rate:.0%}",
            f"{column.n_unique:,}",
        )
    table.caption = "[yellow]yellow[/] carries a quality note"
    return table


def _ideas(state: SessionState, *, console: Console, router: Router) -> None:
    with console.status("[dim]reading the card...[/]", spinner="dots"):
        try:
            proposed = IdeationAgent(router=router).propose(state.card)
        except InsightsmithError as exc:
            console.print(f"[yellow]no ideas:[/] {exc}")
            return
    table = Table(box=ROUNDED, border_style="dim", header_style="bold")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("question")
    table.add_column("columns", style="dim")
    for number, idea in enumerate(proposed, 1):
        table.add_row(str(number), idea.question, ", ".join(idea.columns[:4]))
    console.print(table)
    console.print("[dim]ask any of them in your own words[/]")


def _toggle(name: str, state: SessionState, *, console: Console) -> None:
    field = {"/code": "show_code", "/chart": "charts", "/critique": "critique"}[name]
    value = not getattr(state, field)
    setattr(state, field, value)
    console.print(f"[dim]{name[1:]}:[/] {'on' if value else 'off'}")


def _engine(argument: str, state: SessionState, *, console: Console) -> None:
    if not argument:
        console.print(f"[dim]engine:[/] {state.engine.value}   [dim]{_engine_names()}[/]")
        return
    try:
        state.engine = Engine(argument.lower())
    except ValueError:
        console.print(f"[yellow]no such engine:[/] {argument}   [dim]{_engine_names()}[/]")
        return
    console.print(f"[dim]engine:[/] {state.engine.value}")


def _engine_names() -> str:
    return " | ".join(engine.value for engine in Engine)


def _save(target: str, state: SessionState, *, console: Console, router: Router) -> None:
    """Forge the conversation so far into a report.

    The session already holds everything a report needs except the frames, which
    is why a turn keeps its result shape and not its rows: re-running the code
    here would double every answer's cost to reproduce what was on screen. So
    `/save` writes the Markdown and the notebook, and the notebook is the thing
    that reproduces the numbers.
    """
    from insightsmith.report import Finding, Report, render_markdown, render_notebook

    if not state.turns:
        console.print("[dim]nothing to save yet[/]")
        return
    out = Path(target)
    out.mkdir(parents=True, exist_ok=True)
    report = Report(
        source=state.source,
        profile=state.profile,
        findings=[
            Finding(
                question=turn.question,
                code=turn.code,
                narrative=turn.narrative,
                kind=turn.kind,
                value=turn.value,
            )
            for turn in state.turns
        ],
        card_hash=getattr(state.card, "hash", ""),
        engine=state.engine.value,
        models=router.resolved,
    )
    written = []
    for name, payload in (
        ("session.md", render_markdown(report)),
        ("session.ipynb", render_notebook(report)),
    ):
        (out / name).write_text(payload, encoding="utf-8")
        written.append(str(out / name))
    console.print(f"[green]saved[/] {len(state.turns)} turn(s)")
    for path in written:
        console.print(f"  [dim]{path}[/]")
