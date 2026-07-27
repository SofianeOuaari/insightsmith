"""Profiling tests: schema inference, statistics, quality checks, sampling."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from insightsmith.io.sniff import sniff
from insightsmith.profiling import profile
from insightsmith.profiling.quality import Severity, duplicate_counts, outlier_counts
from insightsmith.profiling.schema import SemanticType, candidate_keys, infer_schema
from insightsmith.profiling.stats import categorical_stats, numeric_stats, text_stats


def _profile(path: Path, **kwargs: object):
    return profile(sniff(path), **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def test_semantic_types() -> None:
    frame = pl.DataFrame(
        {
            "customer_id": ["a1", "a2", "a3", "a4"],
            "region": ["n", "s", "n", "s"],
            "revenue": [1.5, 2.5, 3.5, 4.5],
            "active": [True, False, True, False],
            "signed": pl.Series(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
            ).str.to_date(),
            "comment": ["a much longer free text value", "another one", "third", "fourth entry"],
        }
    )
    got = {s.name: s.semantic for s in infer_schema(frame)}
    assert got["customer_id"] is SemanticType.IDENTIFIER
    assert got["region"] is SemanticType.CATEGORICAL
    assert got["revenue"] is SemanticType.NUMERIC
    assert got["active"] is SemanticType.BOOLEAN
    assert got["signed"] is SemanticType.TEMPORAL


def test_candidate_keys_exclude_floats() -> None:
    frame = pl.DataFrame({"id": [1, 2, 3], "measure": [0.1, 0.2, 0.3], "grp": ["a", "a", "b"]})
    assert candidate_keys(frame) == ["id"]


def test_candidate_keys_reject_nullable_columns() -> None:
    frame = pl.DataFrame({"id": [1, 2, None]})
    assert candidate_keys(frame) == []


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def test_numeric_stats() -> None:
    stats = numeric_stats(pl.Series([1.0, 2.0, 3.0, 4.0, 0.0, -1.0]))
    assert stats is not None
    assert (stats.minimum, stats.maximum) == (-1.0, 4.0)
    assert stats.median == 1.5
    assert stats.zeros == 1
    assert stats.negatives == 1


def test_numeric_stats_of_a_single_value_has_no_std() -> None:
    stats = numeric_stats(pl.Series([7.0]))
    assert stats is not None
    assert stats.std is None
    assert stats.skew is None


def test_numeric_stats_of_all_nulls_is_none() -> None:
    assert numeric_stats(pl.Series([None, None], dtype=pl.Float64)) is None


def test_categorical_stats_ranks_and_measures_imbalance() -> None:
    stats = categorical_stats(pl.Series(["a"] * 20 + ["b"]))
    assert stats.n_unique == 2
    assert stats.top[0] == ("a", 20)
    assert stats.imbalance_ratio == 20.0


def test_text_stats() -> None:
    stats = text_stats(pl.Series(["ab", "abcd", "  "]))
    assert stats is not None
    assert (stats.min_length, stats.max_length) == (2, 4)
    assert stats.empty_strings == 1


# --------------------------------------------------------------------------- #
# quality
# --------------------------------------------------------------------------- #


def test_duplicate_counts_distinguishes_exact_from_near() -> None:
    frame = pl.DataFrame({"name": ["ada", "ada", " ADA ", "bob"]})
    exact, near = duplicate_counts(frame)
    assert exact == 1  # the second "ada"
    assert near == 2  # plus " ADA " once trimmed and folded


def test_outlier_counts_by_both_methods() -> None:
    """On a spread distribution with one extreme value, both methods agree."""
    values = pl.Series([*[float(i) for i in range(1, 21)], 9999.0])
    iqr, mad = outlier_counts(values)
    assert iqr >= 1
    assert mad >= 1


def test_near_constant_column_defeats_iqr_but_not_the_mad_fallback() -> None:
    """Twenty 10s and one 9999: Q1 == Q3, so the IQR fence is degenerate.

    MAD is zero here too, which is exactly when the mean-absolute-deviation
    fallback earns its place — the outlier is obvious and must not be missed.
    """
    iqr, mad = outlier_counts(pl.Series([*([10.0] * 20), 9999.0]))
    assert iqr == 0
    assert mad == 1


def test_outlier_counts_need_enough_data() -> None:
    assert outlier_counts(pl.Series([1.0, 2.0])) == (0, 0)


def test_messy_file_raises_the_expected_notes(samples: dict[str, Path]) -> None:
    result = _profile(samples["messy"])
    kinds = {issue.kind for issue in result.issues}
    assert "duplicate_rows" in kinds
    assert "near_duplicate_rows" in kinds
    assert "constant" in kinds  # the "team" column is "alpha" throughout
    assert any(i.kind == "constant" and i.column == "team" for i in result.issues)


def test_outliers_are_attributed_to_the_column(samples: dict[str, Path]) -> None:
    result = _profile(samples["messy"])
    score = next(c for c in result.columns if c.name == "score")
    assert score.modified_z_outliers >= 1


def test_null_rate_is_reported(samples: dict[str, Path]) -> None:
    result = _profile(samples["messy"])
    note = next(c for c in result.columns if c.name == "note")
    assert note.null_count > 0
    assert 0 < note.null_rate < 1


def test_severities_are_set(samples: dict[str, Path]) -> None:
    result = _profile(samples["messy"])
    assert {i.severity for i in result.issues} <= {Severity.INFO, Severity.WARNING}


# --------------------------------------------------------------------------- #
# whole-profile behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sample", ["csv", "parquet", "feather", "jsonl", "xlsx_real", "german_csv"]
)
def test_profile_every_loadable_format(samples: dict[str, Path], sample: str) -> None:
    result = _profile(samples[sample])
    assert result.n_rows > 0
    assert result.n_columns > 0
    assert len(result.columns) == result.n_columns
    assert not result.estimated
    assert result.summary()


def test_empty_table_is_reported_not_crashed(tmp_path: Path) -> None:
    path = tmp_path / "headers_only.csv"
    path.write_text("a,b,c\n", encoding="utf-8")
    result = _profile(path)
    assert result.n_rows == 0
    assert {i.kind for i in result.issues} == {"no_rows"}


def test_large_sources_are_sampled_and_marked_estimated(tmp_path: Path) -> None:
    """Every statistic from a sample must be flagged, never passed off as exact."""
    path = tmp_path / "big.csv"
    rows = "\n".join(f"{i},{i % 7}" for i in range(5_000))
    path.write_text(f"n,grp\n{rows}\n", encoding="utf-8")

    # threshold_bytes=0 forces the sampling path without writing a 2 GB file.
    result = _profile(path, threshold_bytes=0, sample_rows=100)
    assert result.estimated
    assert result.sampled_rows <= 100
    assert result.n_rows == 5_000  # the true count, counted lazily
    assert all(c.estimated for c in result.columns)


def test_sampling_spans_the_whole_file(tmp_path: Path) -> None:
    """A strided sample, not the head — otherwise ordered data is misprofiled."""
    path = tmp_path / "ordered.csv"
    rows = "\n".join(str(i) for i in range(1_000))
    path.write_text(f"n\n{rows}\n", encoding="utf-8")

    result = _profile(path, threshold_bytes=0, sample_rows=50)
    numeric = result.columns[0].numeric
    assert numeric is not None
    # Head-only sampling would cap the maximum near 50.
    assert numeric.maximum > 900
