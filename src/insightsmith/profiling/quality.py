"""Data-quality checks: the things worth saying out loud before analysing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import polars as pl

from insightsmith.profiling.schema import ColumnSchema, SemanticType

__all__ = [
    "Issue",
    "Severity",
    "column_issues",
    "duplicate_counts",
    "outlier_counts",
    "table_issues",
]

NULL_RATE_WARN: Final = 0.20
NULL_RATE_HIGH: Final = 0.60
NEAR_CONSTANT_SHARE: Final = 0.95
HIGH_CARDINALITY_RATIO: Final = 0.90
IMBALANCE_WARN: Final = 10.0
IQR_MULTIPLIER: Final = 1.5
# 0.6745 scales MAD to be comparable with a standard deviation.
MAD_SCALE: Final = 0.6745
# Iglewicz & Hoaglin's constant for the mean-absolute-deviation fallback.
MEAN_AD_SCALE: Final = 1.253314
MODIFIED_Z_CUTOFF: Final = 3.5


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"


@dataclass(slots=True)
class Issue:
    kind: str
    message: str
    severity: Severity = Severity.WARNING
    column: str | None = None


def table_issues(frame: pl.DataFrame) -> list[Issue]:
    """Whole-table problems: no rows, no columns, duplicate rows."""
    issues: list[Issue] = []
    if frame.width == 0:
        issues.append(Issue("no_columns", "source has no columns"))
        return issues
    if frame.height == 0:
        issues.append(Issue("no_rows", "source has no rows"))
        return issues

    exact, near = duplicate_counts(frame)
    if exact:
        share = exact / frame.height
        issues.append(
            Issue("duplicate_rows", f"{exact} exactly duplicated rows ({share:.1%} of the sample)")
        )
    if near > exact:
        issues.append(
            Issue(
                "near_duplicate_rows",
                f"{near - exact} further rows duplicate once text is trimmed and case-folded",
                severity=Severity.INFO,
            )
        )
    return issues


def duplicate_counts(frame: pl.DataFrame) -> tuple[int, int]:
    """``(exact, near)`` duplicate row counts.

    "Near" normalises string columns — trimmed and case-folded — so ``" Ada"`` and
    ``"ada"`` collide. It is a superset of the exact count.
    """
    if frame.height == 0:
        return 0, 0
    exact = frame.height - frame.n_unique()
    normalised = frame.with_columns(
        pl.col(name).str.strip_chars().str.to_lowercase()
        for name, dtype in frame.schema.items()
        if dtype == pl.String
    )
    return exact, frame.height - normalised.n_unique()


def column_issues(frame: pl.DataFrame, schemas: list[ColumnSchema]) -> list[Issue]:
    """Per-column problems: nulls, constants, runaway cardinality, imbalance."""
    issues: list[Issue] = []
    height = frame.height
    if height == 0:
        return issues

    for schema in schemas:
        series = frame[schema.name]
        issues.extend(_null_issues(schema.name, series, height))
        issues.extend(_shape_issues(schema, series, height))
        if schema.temporal_ambiguous and schema.temporal_format is not None:
            issues.append(
                Issue(
                    "ambiguous_date_format",
                    f"day-first and month-first both fit; read as {schema.temporal_format}. "
                    f"Check before trusting anything grouped by this column",
                    column=schema.name,
                )
            )
    return issues


def _null_issues(name: str, series: pl.Series, height: int) -> list[Issue]:
    nulls = series.null_count()
    if nulls == height:
        return [Issue("all_null", "every value is null", column=name)]
    rate = nulls / height
    if rate >= NULL_RATE_HIGH:
        return [Issue("high_null_rate", f"{rate:.1%} null", column=name)]
    if rate >= NULL_RATE_WARN:
        return [Issue("null_rate", f"{rate:.1%} null", column=name, severity=Severity.INFO)]
    return []


def _shape_issues(schema: ColumnSchema, series: pl.Series, height: int) -> list[Issue]:
    issues: list[Issue] = []
    values = series.drop_nulls()
    if values.is_empty():
        return issues
    n_unique = values.n_unique()

    if n_unique == 1:
        return [Issue("constant", f"single value {values[0]!r} throughout", column=schema.name)]

    counts = values.value_counts(sort=True)
    share = int(counts[counts.columns[1]][0]) / values.len()
    if share >= NEAR_CONSTANT_SHARE:
        issues.append(
            Issue(
                "near_constant",
                f"one value covers {share:.1%} of non-null rows",
                column=schema.name,
            )
        )

    if (
        schema.semantic in {SemanticType.CATEGORICAL, SemanticType.TEXT}
        and n_unique / height >= HIGH_CARDINALITY_RATIO
    ):
        issues.append(
            Issue(
                "high_cardinality",
                f"{n_unique} distinct values across {height} rows — free text or an id",
                column=schema.name,
                severity=Severity.INFO,
            )
        )

    if schema.semantic in {SemanticType.CATEGORICAL, SemanticType.BOOLEAN} and n_unique > 1:
        frequencies = counts[counts.columns[1]]
        smallest = int(frequencies.min())  # type: ignore[arg-type]
        if smallest > 0:
            ratio = int(frequencies.max()) / smallest  # type: ignore[arg-type]
            if ratio >= IMBALANCE_WARN:
                issues.append(
                    Issue(
                        "class_imbalance",
                        f"most common class is {ratio:.0f}x the rarest",
                        column=schema.name,
                        severity=Severity.INFO,
                    )
                )
    return issues


def outlier_counts(series: pl.Series) -> tuple[int, int]:
    """``(iqr, modified_z)`` outlier counts for a numeric column.

    Two methods because they disagree usefully: the IQR fence is distribution-free
    but sensitive to a wide middle, while the MAD-based modified z-score resists
    the outliers it is looking for.
    """
    values = series.drop_nulls()
    if values.len() < 4:
        return 0, 0

    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr_count = 0
    if q1 is not None and q3 is not None:
        iqr = q3 - q1
        if iqr > 0:
            low, high = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
            iqr_count = int(((values < low) | (values > high)).sum())

    median = _scalar(values.median())
    z_count = 0
    if median is not None:
        deviations = (values - median).abs()
        mad = _scalar(deviations.median()) or 0.0
        scores: pl.Series | None = None
        if mad > 0:
            scores = deviations * MAD_SCALE / mad
        else:
            # MAD collapses to zero whenever over half the values are identical —
            # a near-constant column with one wild value, where an outlier is
            # most obvious. Iglewicz & Hoaglin's documented fallback is the mean
            # absolute deviation, scaled by 1.253314.
            mean_ad = _scalar(deviations.mean()) or 0.0
            if mean_ad > 0:
                scores = deviations / (MEAN_AD_SCALE * mean_ad)
        if scores is not None:
            z_count = int((scores > MODIFIED_Z_CUTOFF).sum())
    return iqr_count, z_count


def _scalar(value: object) -> float | None:
    """Polars aggregations are typed as any scalar; only numbers concern us here."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
