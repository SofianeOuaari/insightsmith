"""Profiling: schema, statistics and quality, composed into a :class:`Profile`.

Large sources are profiled on a sample, and every statistic derived from one is
marked ``estimated`` so a caller can never mistake an approximation for a census.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import polars as pl

from insightsmith.io.loaders import load
from insightsmith.io.sniff import SourceSpec
from insightsmith.profiling.quality import Issue, column_issues, outlier_counts, table_issues
from insightsmith.profiling.schema import (
    ColumnSchema,
    SemanticType,
    candidate_keys,
    infer_schema,
)
from insightsmith.profiling.stats import (
    CategoricalStats,
    NumericStats,
    TemporalStats,
    TextStats,
    categorical_stats,
    numeric_stats,
    temporal_stats,
    text_stats,
)

__all__ = [
    "ColumnProfile",
    "Profile",
    "profile",
    "profile_with_sample",
]

#: Sources larger than this are profiled on a sample (design doc §3).
SAMPLE_THRESHOLD_BYTES: Final = 2 * 1024**3
#: Rows retained when sampling.
SAMPLE_ROWS: Final = 200_000


@dataclass(slots=True)
class ColumnProfile:
    schema: ColumnSchema
    null_count: int
    null_rate: float
    n_unique: int
    estimated: bool
    numeric: NumericStats | None = None
    categorical: CategoricalStats | None = None
    temporal: TemporalStats | None = None
    text: TextStats | None = None
    iqr_outliers: int = 0
    modified_z_outliers: int = 0

    @property
    def name(self) -> str:
        return self.schema.name


@dataclass(slots=True)
class Profile:
    """A source, measured. ``estimated`` is true when built from a sample."""

    source: SourceSpec
    n_rows: int
    n_columns: int
    estimated: bool
    sampled_rows: int
    columns: list[ColumnProfile] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    candidate_keys: list[str] = field(default_factory=list)

    def summary(self) -> str:
        rows = f"~{self.n_rows:,}" if self.estimated else f"{self.n_rows:,}"
        head = f"{self.source.path.name}: {rows} rows x {self.n_columns} columns"
        if self.estimated:
            head += f" (profiled on {self.sampled_rows:,} sampled rows)"
        if self.issues:
            head += f" — {len(self.issues)} quality note(s)"
        return head


def profile_with_sample(
    spec: SourceSpec,
    *,
    threshold_bytes: int = SAMPLE_THRESHOLD_BYTES,
    sample_rows: int = SAMPLE_ROWS,
) -> tuple[Profile, pl.DataFrame]:
    """Profile, and also hand back the sample it was computed from.

    The dataset card needs the rows to draw examples and correlations from, and
    re-reading the source to get them would be both slower and inconsistent.
    """
    return _profile(spec, threshold_bytes=threshold_bytes, sample_rows=sample_rows)


def profile(
    spec: SourceSpec,
    *,
    threshold_bytes: int = SAMPLE_THRESHOLD_BYTES,
    sample_rows: int = SAMPLE_ROWS,
) -> Profile:
    """Profile the source described by ``spec``."""
    return _profile(spec, threshold_bytes=threshold_bytes, sample_rows=sample_rows)[0]


def _profile(
    spec: SourceSpec,
    *,
    threshold_bytes: int = SAMPLE_THRESHOLD_BYTES,
    sample_rows: int = SAMPLE_ROWS,
) -> tuple[Profile, pl.DataFrame]:
    """Profile the source described by ``spec``.

    Sources whose file size exceeds ``threshold_bytes`` are sampled down to
    roughly ``sample_rows`` rows, taken at a fixed stride across the whole file
    rather than from the head, so the sample is not an artefact of row order.
    """
    frame = load(spec)
    size = spec.path.stat().st_size

    if size > threshold_bytes:
        n_rows = int(frame.select(pl.len()).collect().item())
        stride = max(1, math.ceil(n_rows / sample_rows)) if n_rows else 1
        sample = frame.gather_every(stride).head(sample_rows).collect()
        estimated = True
    else:
        sample = frame.collect()
        n_rows = sample.height
        estimated = False

    schemas = infer_schema(sample)
    issues = [*table_issues(sample), *column_issues(sample, schemas)]
    columns = [_column_profile(sample, s, estimated) for s in schemas]

    result = Profile(
        source=spec,
        n_rows=n_rows,
        n_columns=sample.width,
        estimated=estimated,
        sampled_rows=sample.height,
        columns=columns,
        issues=issues,
        candidate_keys=candidate_keys(sample),
    )
    return result, sample


def _column_profile(frame: pl.DataFrame, schema: ColumnSchema, estimated: bool) -> ColumnProfile:
    series = frame[schema.name]
    height = frame.height
    nulls = series.null_count()
    out = ColumnProfile(
        schema=schema,
        null_count=nulls,
        null_rate=nulls / height if height else 0.0,
        n_unique=series.drop_nulls().n_unique(),
        estimated=estimated,
    )

    if schema.semantic is SemanticType.NUMERIC:
        out.numeric = numeric_stats(series)
        out.iqr_outliers, out.modified_z_outliers = outlier_counts(series)
    elif schema.semantic is SemanticType.TEMPORAL:
        # Dates recognised inside a string column need parsing before they can
        # be summarised; genuine date dtypes are already usable.
        if schema.temporal_format is not None:
            series = series.str.to_datetime(schema.temporal_format, strict=False)
        out.temporal = temporal_stats(series)
    elif schema.semantic is SemanticType.TEXT:
        out.text = text_stats(series)
    elif schema.semantic in {SemanticType.CATEGORICAL, SemanticType.BOOLEAN}:
        out.categorical = categorical_stats(series.cast(pl.String))
    return out
