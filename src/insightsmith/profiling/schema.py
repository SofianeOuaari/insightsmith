"""Dtypes, semantic types and candidate keys."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

import polars as pl
from polars.exceptions import PolarsError

__all__ = [
    "ColumnSchema",
    "SemanticType",
    "candidate_keys",
    "detect_temporal",
    "infer_schema",
]

# A string column with few distinct values is a category; with many, free text.
_CATEGORICAL_MAX_UNIQUE: Final = 50
_CATEGORICAL_MAX_RATIO: Final = 0.2
_IDENTIFIER_NAMES: Final = re.compile(r"(^|_)(id|uuid|guid|key|code|sku|isbn)($|_)", re.I)


class SemanticType(str, Enum):
    """What a column *means*, as opposed to how it is stored."""

    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IDENTIFIER = "identifier"
    EMPTY = "empty"


#: Tried in order. Day-first and month-first variants sit next to each other so
#: a tie between them can be detected and reported rather than quietly resolved.
TEMPORAL_FORMATS: Final = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m/%d/%y %H:%M:%S",
    "%d/%m/%y %H:%M:%S",
    "%m/%d/%y",
    "%d/%m/%y",
    "%d.%m.%Y",
    "%d-%m-%Y",
)
#: Share of non-null values that must parse before a column counts as temporal.
TEMPORAL_MATCH_SHARE: Final = 0.95
#: A format yielding years outside this range parsed something that isn't a year.
MIN_PLAUSIBLE_YEAR: Final = 1000
MAX_PLAUSIBLE_YEAR: Final = 3000
_MIN_TEMPORAL_VALUES: Final = 3


@dataclass(slots=True)
class ColumnSchema:
    name: str
    dtype: str
    semantic: SemanticType
    nullable: bool
    #: strptime pattern, when a string column was recognised as dates.
    temporal_format: str | None = None
    #: True when day-first and month-first fit equally well.
    temporal_ambiguous: bool = False


def detect_temporal(series: pl.Series) -> tuple[str, bool] | None:
    """Find the datetime format of a string column, if it has one.

    Returns ``(format, ambiguous)``. ``ambiguous`` is set when a day-first and a
    month-first pattern parse the column equally well — ``04/01/10`` is either
    the 4th of January or the 1st of April, and no amount of staring at the
    column settles it. The caller is expected to say so.
    """
    values = series.drop_nulls()
    if values.len() < _MIN_TEMPORAL_VALUES:
        return None

    scores: dict[str, int] = {}
    for fmt in TEMPORAL_FORMATS:
        try:
            parsed = values.str.to_datetime(fmt, strict=False)
        except PolarsError:
            # polars rejects some patterns outright rather than returning nulls.
            continue
        matched = values.len() - parsed.null_count()
        if matched / values.len() < TEMPORAL_MATCH_SHARE:
            continue
        # %Y will happily consume the "10" of a two-digit year and hand back the
        # year 10 AD. Any format implying an implausible year is the wrong one.
        years = parsed.dt.year().drop_nulls()
        earliest, latest = years.min(), years.max()
        if not isinstance(earliest, int) or not isinstance(latest, int):
            continue
        if earliest < MIN_PLAUSIBLE_YEAR or latest > MAX_PLAUSIBLE_YEAR:
            continue
        scores[fmt] = matched

    if not scores:
        return None
    best = max(scores, key=lambda fmt: (scores[fmt], -TEMPORAL_FORMATS.index(fmt)))
    # Derive the rival by swapping the day and month directives rather than
    # listing families: that way the "%H:%M:%S" variants are covered too, which
    # a hand-written list of date-only patterns silently missed.
    rival = None
    if "%m/%d" in best:
        rival = best.replace("%m/%d", "%d/%m")
    elif "%d/%m" in best:
        rival = best.replace("%d/%m", "%m/%d")
    ambiguous = rival is not None and scores.get(rival, -1) == scores[best]
    return best, ambiguous


def infer_schema(frame: pl.DataFrame) -> list[ColumnSchema]:
    """Storage dtype plus a semantic guess, per column."""
    height = frame.height
    out: list[ColumnSchema] = []
    for name in frame.columns:
        series = frame[name]
        non_null = series.drop_nulls()
        schema = ColumnSchema(
            name=name,
            dtype=str(series.dtype),
            semantic=_semantic(name, series, non_null.n_unique(), height),
            nullable=series.null_count() > 0,
        )
        # A string column holding dates is temporal, whatever polars called it.
        if series.dtype == pl.String and schema.semantic is not SemanticType.EMPTY:
            found = detect_temporal(series)
            if found is not None:
                schema.temporal_format, schema.temporal_ambiguous = found
                schema.semantic = SemanticType.TEMPORAL
        out.append(schema)
    return out


def _semantic(name: str, series: pl.Series, n_unique: int, height: int) -> SemanticType:
    if height == 0 or n_unique == 0:
        return SemanticType.EMPTY
    dtype = series.dtype
    if dtype == pl.Boolean:
        return SemanticType.BOOLEAN
    if dtype.is_temporal():
        return SemanticType.TEMPORAL
    if dtype.is_numeric():
        # An integer column that is unique on every row and named like a key is
        # an identifier, not a quantity to average.
        if n_unique == height and _IDENTIFIER_NAMES.search(name):
            return SemanticType.IDENTIFIER
        return SemanticType.NUMERIC
    if dtype == pl.String:
        if n_unique == height and (height > 1 or _IDENTIFIER_NAMES.search(name)):
            return SemanticType.IDENTIFIER
        if n_unique <= _CATEGORICAL_MAX_UNIQUE or n_unique / height <= _CATEGORICAL_MAX_RATIO:
            return SemanticType.CATEGORICAL
        return SemanticType.TEXT
    return SemanticType.CATEGORICAL


def candidate_keys(frame: pl.DataFrame) -> list[str]:
    """Single columns that could serve as a primary key: unique and never null.

    Floats are excluded even when they happen to be unique — a measurement that
    is distinct on every sampled row is a coincidence, not a key.
    """
    if frame.height == 0:
        return []
    return [
        name
        for name in frame.columns
        if not frame[name].dtype.is_float()
        and frame[name].null_count() == 0
        and frame[name].n_unique() == frame.height
    ]
