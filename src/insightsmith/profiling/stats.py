"""Per-column statistics, by kind of column."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import polars as pl

__all__ = [
    "CategoricalStats",
    "NumericStats",
    "TemporalStats",
    "TextStats",
    "categorical_stats",
    "numeric_stats",
    "temporal_stats",
    "text_stats",
]

TOP_VALUES: Final = 5


@dataclass(slots=True)
class NumericStats:
    minimum: float
    maximum: float
    mean: float
    median: float
    std: float | None
    q1: float
    q3: float
    zeros: int
    negatives: int
    skew: float | None


@dataclass(slots=True)
class CategoricalStats:
    n_unique: int
    top: list[tuple[str, int]] = field(default_factory=list)
    imbalance_ratio: float | None = None


@dataclass(slots=True)
class TemporalStats:
    earliest: str
    latest: str
    n_unique: int


@dataclass(slots=True)
class TextStats:
    n_unique: int
    min_length: int
    max_length: int
    mean_length: float
    empty_strings: int


def numeric_stats(series: pl.Series) -> NumericStats | None:
    values = series.drop_nulls()
    if values.is_empty():
        return None
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return NumericStats(
        minimum=float(values.min()),  # type: ignore[arg-type]
        maximum=float(values.max()),  # type: ignore[arg-type]
        mean=float(values.mean()),  # type: ignore[arg-type]
        median=float(values.median()),  # type: ignore[arg-type]
        # std is undefined for a single observation.
        std=None if values.len() < 2 else _maybe_float(values.std()),
        q1=float(q1) if q1 is not None else float(values.min()),  # type: ignore[arg-type]
        q3=float(q3) if q3 is not None else float(values.max()),  # type: ignore[arg-type]
        zeros=int((values == 0).sum()),
        negatives=int((values < 0).sum()),
        skew=None if values.len() < 3 else _maybe_float(values.skew()),
    )


def categorical_stats(series: pl.Series) -> CategoricalStats:
    values = series.drop_nulls()
    if values.is_empty():
        return CategoricalStats(n_unique=0)
    counts = values.value_counts(sort=True)
    name, count = counts.columns[0], counts.columns[1]
    top = [(str(row[name]), int(row[count])) for row in counts.head(TOP_VALUES).to_dicts()]
    frequencies = counts[count]
    ratio = (
        float(frequencies.max()) / float(frequencies.min())  # type: ignore[arg-type]
        if frequencies.len() > 1 and float(frequencies.min()) > 0  # type: ignore[arg-type]
        else None
    )
    return CategoricalStats(n_unique=values.n_unique(), top=top, imbalance_ratio=ratio)


def temporal_stats(series: pl.Series) -> TemporalStats | None:
    values = series.drop_nulls()
    if values.is_empty():
        return None
    return TemporalStats(
        earliest=str(values.min()), latest=str(values.max()), n_unique=values.n_unique()
    )


def text_stats(series: pl.Series) -> TextStats | None:
    values = series.drop_nulls()
    if values.is_empty():
        return None
    lengths = values.str.len_chars()
    return TextStats(
        n_unique=values.n_unique(),
        min_length=int(lengths.min()),  # type: ignore[arg-type]
        max_length=int(lengths.max()),  # type: ignore[arg-type]
        mean_length=float(lengths.mean()),  # type: ignore[arg-type]
        empty_strings=int((values.str.strip_chars() == "").sum()),
    )


def _maybe_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]
