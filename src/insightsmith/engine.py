"""Which dataframe API the generated code is written against.

Everything up to the dataset card stays Polars: sniffing, loading and profiling
are lazy for a reason, and that reason does not change with the engine. What
this selects is narrower and more useful, the API the *model* writes against.

The case for offering FireDucks is empirical. Sweeping real questions across
several datasets, the largest single class of failure was a model reaching for
pandas on a Polars frame: ``groupby``, ``sort_values``, ``fillna``. Those are
not slips a better prompt fixes, they are the weight of everything the model has
read. FireDucks is pandas at the surface, so under it those same habits are
simply correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

__all__ = ["Engine", "EngineSpec", "spec_for"]


class Engine(str, Enum):
    """The dataframe API a snippet is written against."""

    POLARS = "polars"
    PANDAS = "pandas"
    FIREDUCKS = "fireducks"


@dataclass(frozen=True, slots=True)
class EngineSpec:
    """Everything that differs between one engine and another, in one place."""

    engine: Engine
    #: How the engine writes its own name.
    label: str
    #: Package data under ``knowledge/``.
    guide: str
    #: Guide sections the coder must never see. It is handed a frame already in
    #: memory and forbidden to read files or plot, so reading, charting,
    #: installing and the worked case study can only mislead it.
    excludes: tuple[str, ...]
    #: The import root the static gate has to allow.
    import_root: str
    #: The name a snippet refers to the library by.
    alias: str
    #: How the sandbox hands the frame over.
    preamble: str
    #: The API-specific half of the coder's system prompt.
    rules: str


_POLARS_RULES: Final = """\
This is Polars, not pandas. The two APIs differ and pandas methods do not exist:
- `df.group_by("col").agg(pl.col("x").sum())`  not `groupby()` / `.sum()`
- `df.filter(pl.col("x") > 1)`                 not boolean indexing
- `df.select(pl.col("a"), pl.col("b"))`        not `df[["a", "b"]]`
- `df.sort("x", descending=True)`              not `sort_values(ascending=False)`
- `df.with_columns((pl.col("a") / pl.col("b")).alias("ratio"))`\
"""

_FIREDUCKS_RULES: Final = """\
This is FireDucks, which is the pandas API. Write ordinary pandas:
- `df.groupby("col")["x"].sum().reset_index()`
- `df[df["x"] > 1]`
- `df[["a", "b"]]`
- `df.sort_values("x", ascending=False)`
- `df["ratio"] = df["a"] / df["b"]`

Two differences from pandas that matter here:
- Evaluation is lazy and happens when a result is used. Do not rely on timing.
- `isinstance(df, pandas.DataFrame)` is False. Never type-check the frame.\
"""

_PANDAS_RULES: Final = """\
This is pandas. Write ordinary pandas:
- `df.groupby("col")["x"].sum().reset_index()`
- `df[df["x"] > 1]`
- `df[["a", "b"]]`
- `df.sort_values("x", ascending=False)`
- `df["ratio"] = df["a"] / df["b"]`\
"""

_SPECS: Final[dict[Engine, EngineSpec]] = {
    Engine.POLARS: EngineSpec(
        engine=Engine.POLARS,
        label="Polars",
        guide="polars_guide.md",
        excludes=("2", "3", "15", "16"),
        import_root="polars",
        alias="pl",
        preamble="import polars as pl",
        rules=_POLARS_RULES,
    ),
    Engine.PANDAS: EngineSpec(
        engine=Engine.PANDAS,
        label="pandas",
        # The bundled guide is FireDucks', and FireDucks *is* the pandas API, so
        # its operational chapters are pandas verbatim. What is excluded here is
        # everything that is true of FireDucks and not of pandas: the lazy
        # execution model, the compatibility deviations, the fallback tuning and
        # its own API extensions. Handing those to a pandas snippet would teach
        # it behaviour the library does not have.
        guide="fireducks_guide.md",
        excludes=("1", "2", "3", "4", "9", "10", "13", "14", "17"),
        import_root="pandas",
        alias="pd",
        preamble="import pandas as pd",
        rules=_PANDAS_RULES,
    ),
    Engine.FIREDUCKS: EngineSpec(
        engine=Engine.FIREDUCKS,
        label="FireDucks",
        guide="fireducks_guide.md",
        excludes=("2", "13", "14", "17"),
        import_root="fireducks",
        alias="pd",
        preamble="import fireducks.pandas as pd",
        rules=_FIREDUCKS_RULES,
    ),
}


def spec_for(engine: Engine | str) -> EngineSpec:
    """The spec for an engine, by value or by name."""
    try:
        return _SPECS[Engine(engine)]
    except ValueError as exc:
        known = ", ".join(sorted(member.value for member in Engine))
        raise ValueError(f"unknown engine {engine!r}; choose one of {known}") from exc
