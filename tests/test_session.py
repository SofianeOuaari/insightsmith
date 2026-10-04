"""What a conversation remembers, and what it is careful not to.

The transcript goes to the model on every turn, so anything leaked into it
leaks repeatedly. These tests are mostly about that: a turn records the shape
of a result, never its rows.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from insightsmith.engine import Engine
from insightsmith.io.sniff import sniff
from insightsmith.profiling import profile_with_sample
from insightsmith.profiling.card import build_card
from insightsmith.session import MAX_TRANSCRIPT_BYTES, MAX_TURNS, SessionState, Turn


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


def turn(question: str = "q", code: str = "result = df") -> Turn:
    return Turn(question=question, code=code, kind="frame", columns=["region"], rows=3)


def test_a_turn_carries_the_shape_of_a_result_never_its_rows(state: SessionState) -> None:
    """The hard rule, applied to the thing that is sent on every later turn."""
    frame = pl.DataFrame({"region": ["N", "S"], "secret": ["alice@example.com", "bob@x.io"]})
    recorded = Turn(
        question="who?",
        code="result = df",
        kind="frame",
        columns=list(frame.columns),
        rows=frame.height,
    )
    state.record(recorded)

    context = state.context()

    assert "alice@example.com" not in context
    assert "bob@x.io" not in context
    assert "region" in context, "column names are what a follow-up refers to"
    assert "2 row(s)" in context


def test_a_scalar_survives_because_it_is_already_an_aggregate(state: SessionState) -> None:
    """A mean is not a row. Carrying it is what makes "is that high?" answerable."""
    state.record(Turn(question="how many?", code="result = df.height", kind="value", value=4248))

    assert "4248" in state.context()


def test_an_empty_session_sends_no_context(state: SessionState) -> None:
    """Turn one must cost exactly what `ask` costs, or the REPL is a tax."""
    assert state.context() == ""
    assert state.transcript() == ""


def test_the_transcript_keeps_the_recent_turns_and_drops_the_oldest(
    state: SessionState,
) -> None:
    """A follow-up refers to the last turn; turn one is noise competing for tokens."""
    for index in range(MAX_TURNS + 4):
        state.record(turn(question=f"question number {index}"))

    body = state.transcript()

    assert f"question number {MAX_TURNS + 3}" in body, "the newest must survive"
    assert "question number 0" not in body, "the oldest must not"


def test_the_transcript_stays_inside_its_budget(state: SessionState) -> None:
    """A long session must not push the card out of a small context window."""
    for index in range(MAX_TURNS):
        state.record(turn(question=f"q{index}", code="result = df\n" * 80))

    assert len(state.transcript().encode("utf-8")) <= MAX_TRANSCRIPT_BYTES


def test_one_oversized_turn_is_kept_whole_rather_than_cut(state: SessionState) -> None:
    """Half a snippet teaches a wrong API, so the budget yields to the first turn."""
    state.record(turn(code="result = df\n" * 500))

    assert "result = df" in state.transcript()


def test_the_context_tells_the_model_it_cannot_reuse_variables(state: SessionState) -> None:
    """Each snippet runs in a fresh sandbox, so `result` from turn one is gone."""
    state.record(turn())

    assert "cannot reuse variables" in state.context()


def test_undo_and_clear(state: SessionState) -> None:
    state.record(turn(question="first"))
    state.record(turn(question="second"))

    assert state.undo() is not None and [t.question for t in state.turns] == ["first"]
    assert state.forget() == 1
    assert state.turns == []
    assert state.undo() is None, "undo on an empty session is not an error"
