"""Dtypes, semantic types and candidate keys."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

import polars as pl

__all__ = ["ColumnSchema", "SemanticType", "candidate_keys", "infer_schema"]

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


@dataclass(slots=True)
class ColumnSchema:
    name: str
    dtype: str
    semantic: SemanticType
    nullable: bool


def infer_schema(frame: pl.DataFrame) -> list[ColumnSchema]:
    """Storage dtype plus a semantic guess, per column."""
    height = frame.height
    out: list[ColumnSchema] = []
    for name in frame.columns:
        series = frame[name]
        non_null = series.drop_nulls()
        out.append(
            ColumnSchema(
                name=name,
                dtype=str(series.dtype),
                semantic=_semantic(name, series, non_null.n_unique(), height),
                nullable=series.null_count() > 0,
            )
        )
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
