"""Choosing the dataframe API the generated code is written against.

The point of the choice is empirical: sweeping real questions, the largest class
of coder failure was a model reaching for pandas on a Polars frame. These tests
hold the line that matters, which is that advice meant for one engine never
reaches the other, where it would be wrong rather than merely useless.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from insightsmith.agents.coder import _correction, system_prompt_for
from insightsmith.config import load_config
from insightsmith.engine import Engine, spec_for
from insightsmith.errors import ConfigError
from insightsmith.knowledge import retrieve, sections


def test_every_engine_has_a_spec_and_a_guide_that_parses() -> None:
    for engine in Engine:
        spec = spec_for(engine)
        found = sections(spec.guide)
        assert len(found) > 30, f"{engine.value} guide barely parsed"
        assert all(section.body.strip() for section in found)


def test_an_unknown_engine_says_what_the_choices_are() -> None:
    with pytest.raises(ValueError, match="fireducks, pandas, polars"):
        spec_for("modin")


def test_each_engine_is_told_to_write_its_own_api() -> None:
    polars = system_prompt_for(Engine.POLARS)
    fireducks = system_prompt_for(Engine.FIREDUCKS)

    assert "group_by" in polars and "not `groupby()`" not in fireducks
    assert "df.groupby(" in fireducks
    assert "import polars as pl" in polars
    assert "import fireducks.pandas as pd" in fireducks
    # The braces in the JSON contract must survive templating.
    for prompt in (polars, fireducks):
        assert '{"code": "...", "explanation": "..."}' in prompt


def test_polars_advice_never_reaches_fireducks() -> None:
    """`groupby` is a mistake in Polars and correct in FireDucks.

    Handing the Polars correction to a FireDucks snippet would not merely waste
    tokens, it would talk the model out of code that already works.
    """
    error = "AttributeError: 'DataFrame' object has no attribute 'groupby'"

    assert "group_by" in _correction(error, None, Engine.POLARS)
    assert _correction(error, None, Engine.FIREDUCKS) == ""


def test_the_column_list_is_the_one_correction_that_travels(tmp_path: Path) -> None:
    """Which columns exist is a fact about the data, not about the API."""
    from insightsmith.io.sniff import sniff
    from insightsmith.profiling import profile_with_sample
    from insightsmith.profiling.card import build_card

    path = tmp_path / "sales.csv"
    path.write_text("region,revenue\nnorth,120\nsouth,80\n", encoding="utf-8")
    result, sample = profile_with_sample(sniff(path))
    card = build_card(result, sample)
    error = 'ColumnNotFoundError: "phone" not found'

    for engine in Engine:
        assert "the columns that exist are" in _correction(error, card, engine)


def test_each_engine_retrieves_from_its_own_guide() -> None:
    question = "average revenue by region"
    polars = retrieve(question, limit=3, guide=spec_for(Engine.POLARS).guide)
    fireducks = retrieve(question, limit=3, guide=spec_for(Engine.FIREDUCKS).guide)

    assert polars and fireducks
    assert {s.heading for s in polars} != {s.heading for s in fireducks}


def test_the_engine_can_be_set_in_config(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('engine = "fireducks"\n', encoding="utf-8")

    assert load_config(path, environ={}).engine is Engine.FIREDUCKS


def test_config_defaults_to_polars(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[roles]\ncoder = "ollama/x"\n', encoding="utf-8")

    assert load_config(path, environ={}).engine is Engine.POLARS


def test_a_misspelled_engine_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('engine = "panads"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="engine must be one of"):
        load_config(path, environ={})


def test_every_failure_the_benchmark_hit_now_has_an_answer() -> None:
    """Four engine failures in an 80-run benchmark; three got no help at all."""
    cases = [
        (
            Engine.POLARS,
            "AttributeError: 'DataFrame' object has no attribute 'sort_by'",
            "expression method",
        ),
        (Engine.POLARS, "TypeError: 'Expr' object is not subscriptable", "cannot be indexed"),
        (
            Engine.FIREDUCKS,
            "ValueError: Cannot subset columns with a tuple with more than one element.",
            "inner list",
        ),
    ]
    for engine, error, expected in cases:
        assert expected in _correction(error, None, engine), (engine, error)


def test_an_expression_method_is_only_suggested_when_it_exists() -> None:
    """Probed against the installed polars, never a table kept here."""
    assert (
        _correction(
            "AttributeError: 'DataFrame' object has no attribute 'wibble_by'", None, Engine.POLARS
        )
        == ""
    )
    # `sort_by` on an Expr is the model using it correctly, so nothing to say.
    assert (
        _correction("AttributeError: 'Expr' object has no attribute 'sort_by'", None, Engine.POLARS)
        == ""
    )


def test_the_tuple_hint_stays_out_of_polars() -> None:
    """Polars has no `df["a", "b"]` idiom, so the advice would only confuse."""
    error = "ValueError: Cannot subset columns with a tuple with more than one element."
    assert _correction(error, None, Engine.POLARS) == ""


def test_polars_written_under_fireducks_is_named_as_such() -> None:
    """The mirror of the habit this feature exists for, seen in benchmarking."""
    for missing in ("select", "with_columns"):
        correction = _correction(
            f"AttributeError: 'DataFrame' object has no attribute '{missing}'",
            None,
            Engine.FIREDUCKS,
        )
        assert "is Polars" in correction and "pandas" in correction


def test_a_name_neither_library_has_gets_nothing() -> None:
    assert (
        _correction(
            "AttributeError: 'DataFrame' object has no attribute 'wibble'", None, Engine.FIREDUCKS
        )
        == ""
    )


def test_groupby_stays_silent_under_fireducks_despite_being_a_polars_name() -> None:
    """`groupby` is correct here, so neither correction may claim it.

    It exists on neither `pl.DataFrame` nor `pl.Expr`, which is what keeps the
    reached-for-Polars hint away from working FireDucks code.
    """
    assert (
        _correction(
            "AttributeError: 'DataFrame' object has no attribute 'groupby'", None, Engine.FIREDUCKS
        )
        == ""
    )


def test_pandas_is_offered_as_its_own_engine() -> None:
    spec = spec_for(Engine.PANDAS)

    assert spec.alias == "pd"
    assert spec.preamble == "import pandas as pd"
    assert "fireducks" not in system_prompt_for(Engine.PANDAS).lower()
    assert "df.groupby(" in system_prompt_for(Engine.PANDAS)


def test_fireducks_only_chapters_never_reach_a_pandas_snippet() -> None:
    """The guide is FireDucks', and its operational chapters are pandas verbatim.

    What must not travel is everything true of FireDucks and not of pandas: lazy
    execution, the compatibility deviations, fallback tuning, its own API
    extensions. Those would teach a pandas snippet behaviour pandas lacks.
    """
    spec = spec_for(Engine.PANDAS)
    for chapter in ("1", "3", "4", "9", "10"):
        assert chapter in spec.excludes

    hits = retrieve(
        "lazy evaluation and forcing execution",
        limit=8,
        exclude=spec.excludes,
        guide=spec.guide,
    )
    assert all(h.number.split(".")[0] not in spec.excludes for h in hits)


def test_pandas_and_fireducks_share_an_api_so_share_their_advice() -> None:
    """Both are pandas at the surface, so Polars corrections stay away from both."""
    error = "AttributeError: 'DataFrame' object has no attribute 'groupby'"

    assert _correction(error, None, Engine.PANDAS) == ""
    assert _correction(error, None, Engine.FIREDUCKS) == ""
    assert "group_by" in _correction(error, None, Engine.POLARS)


def test_pandas_says_the_same_thing_about_text_columns_as_polars() -> None:
    """pandas raises where polars returns null; it is the same missing cast.

    Four of seven pandas failures on the synthetic corpus were this one error
    on the same file, and none of them got any advice.
    """
    error = "TypeError: agg function failed [how->mean,dtype->object]"
    for engine in Engine:
        correction = _correction(error, None, engine)
        assert "stored as text" in correction, engine
        assert "pd.to_numeric" in correction and "cast(pl.Float64" in correction
