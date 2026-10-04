"""The conversational surface, driven without a terminal.

`run` takes its reader as an argument for exactly this reason: prompt_toolkit
needs a tty, and the loop's behaviour is worth testing without one.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from rich.console import Console

from insightsmith import repl
from insightsmith.engine import Engine
from insightsmith.io.sniff import sniff
from insightsmith.llm.router import Router
from insightsmith.profiling import profile_with_sample
from insightsmith.profiling.card import build_card
from insightsmith.session import SessionState, Turn


@pytest.fixture
def state(tmp_path: Path) -> SessionState:
    source = tmp_path / "sales.csv"
    source.write_text("region,amount\nN,10\nS,20\nE,30\n", encoding="utf-8")
    profile, sample = profile_with_sample(sniff(source))
    return SessionState(
        source=source,
        card=build_card(profile, sample),
        profile=profile,
        frame=sample,
        engine=Engine.POLARS,
    )


def drive(state: SessionState, *lines: str) -> str:
    """Run the loop over a script and return what it printed."""
    console = Console(width=100, force_terminal=False, no_color=True)
    with console.capture() as captured:
        source = iter(lines)
        repl.run(state, console=console, router=Router(), read=lambda _: next(source))
    return captured.get()


def test_the_banner_says_what_you_are_talking_to(state: SessionState) -> None:
    out = drive(state, "/exit")

    assert "sales.csv" in out
    assert "3 rows x 2 columns" in out
    assert "polars" in out


def test_ctrl_d_leaves(state: SessionState) -> None:
    """StopIteration from the reader stands in for EOF."""

    def refuse(_: str) -> str:
        raise EOFError

    console = Console(force_terminal=False, no_color=True)
    with console.capture() as captured:
        repl.run(state, console=console, router=Router(), read=refuse)

    assert "bye" in captured.get()


def test_blank_lines_are_not_questions(state: SessionState) -> None:
    """Pressing enter on an empty prompt must not spend an LLM call."""
    drive(state, "", "   ", "/exit")

    assert state.turns == []


def test_help_lists_every_command(state: SessionState) -> None:
    out = drive(state, "/help", "/exit")

    for command in repl.COMMANDS:
        assert command.name in out


def test_an_unknown_command_says_so_rather_than_asking_it(state: SessionState) -> None:
    """A mistyped slash command must not become a question to the model."""
    out = drive(state, "/nope", "/exit")

    assert "no such command" in out
    assert state.turns == []


def test_toggles_change_the_session_not_a_local(state: SessionState) -> None:
    assert state.show_code is True

    drive(state, "/code", "/chart", "/critique", "/exit")

    assert state.show_code is False
    assert state.charts is True
    assert state.critique is False


def test_engine_switches_and_refuses_a_name_it_does_not_have(state: SessionState) -> None:
    out = drive(state, "/engine pandas", "/engine duckdb", "/exit")

    assert state.engine is Engine.PANDAS, "a valid switch takes effect"
    assert "no such engine" in out


def test_undo_and_clear_report_what_they_did(state: SessionState) -> None:
    state.record(Turn(question="first", code="result = df"))
    state.record(Turn(question="second", code="result = df"))

    out = drive(state, "/undo", "/clear", "/exit")

    assert "second" in out
    assert state.turns == []


def test_save_writes_nothing_when_there_is_nothing(state: SessionState, tmp_path: Path) -> None:
    out = drive(state, f"/save {tmp_path / 'empty'}", "/exit")

    assert "nothing to save" in out
    assert not (tmp_path / "empty").exists()


def test_save_forges_the_conversation(state: SessionState, tmp_path: Path) -> None:
    """The session already holds what a report needs, so /save costs no LLM call."""
    state.record(
        Turn(
            question="revenue by region?",
            code='result = df.group_by("region").agg(pl.col("amount").sum())',
            kind="frame",
            columns=["region", "amount"],
            rows=3,
            narrative="North leads.",
        )
    )

    out = drive(state, f"/save {tmp_path / 'session'}", "/exit")

    assert "saved" in out
    written = (tmp_path / "session" / "session.md").read_text()
    assert "revenue by region?" in written
    assert "group_by" in written, "the code is the claim, in every surface"
    assert (tmp_path / "session" / "session.ipynb").exists()


def test_the_result_table_right_aligns_numbers(state: SessionState) -> None:
    """Same rule as the report: an id of digits is text that looks numeric."""
    frame = pl.DataFrame({"region": ["N"], "total": [1.5], "id": ["007"]})

    table = repl._result_table(frame)

    assert [column.justify for column in table.columns] == ["left", "right", "left"]


def test_a_long_result_is_capped_and_says_so(state: SessionState) -> None:
    frame = pl.DataFrame({"n": list(range(repl.PREVIEW_ROWS + 5))})

    table = repl._result_table(frame)

    assert table.row_count == repl.PREVIEW_ROWS
    assert "5 more row" in str(table.caption)
