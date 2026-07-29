"""Loader tests: one interface, lazy where the format allows it."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from insightsmith.errors import UnsupportedFormatError
from insightsmith.io.loaders import load
from insightsmith.io.sniff import Format, sniff


@pytest.mark.parametrize(
    "sample",
    ["csv", "tsv", "pipe_csv", "parquet", "feather", "jsonl", "ndjson", "xlsx_real", "json_array"],
)
def test_load_returns_a_lazyframe_that_collects(samples: dict[str, Path], sample: str) -> None:
    frame = load(sniff(samples[sample]))
    assert isinstance(frame, pl.LazyFrame)
    assert frame.collect().height > 0


@pytest.mark.parametrize("sample", ["parquet", "feather", "jsonl", "csv"])
def test_scannable_formats_stay_lazy(samples: dict[str, Path], sample: str) -> None:
    """These must not be materialised on load — that is the point of LazyFrame."""
    frame = load(sniff(samples[sample]))
    # A pushed-down limit is only possible if nothing was collected eagerly.
    assert frame.head(1).collect().height == 1


def test_values_survive_the_round_trip(samples: dict[str, Path]) -> None:
    frame = load(sniff(samples["parquet"])).collect()
    assert frame.columns == ["region", "units", "revenue"]
    assert frame["units"].to_list() == [12, 7, 23, 4, 19]


def test_semicolons_and_decimal_commas_become_real_floats(samples: dict[str, Path]) -> None:
    """cp1252 + ';' + ',' decimal must land as Float64, not strings."""
    frame = load(sniff(samples["german_csv"])).collect()
    assert frame["Umsatz"].dtype == pl.Float64
    assert frame["Umsatz"].to_list() == [120.50, 70.25, 230.00, 40.75]
    # And the transcoding must not mangle the non-ASCII bytes.
    assert frame["Region"].to_list() == ["Nord", "Süd", "Ost", "Köln"]


def test_tsv_separator_is_honoured(samples: dict[str, Path]) -> None:
    frame = load(sniff(samples["tsv"])).collect()
    assert frame.width == 3


def test_comment_lines_are_skipped(samples: dict[str, Path]) -> None:
    frame = load(sniff(samples["csv_commented"])).collect()
    assert frame.columns == ["region", "units"]
    assert frame.height == 3


def test_headerless_csv_keeps_every_row(samples: dict[str, Path]) -> None:
    spec = sniff(samples["csv_no_header"])
    assert spec.dialect is not None
    assert not spec.dialect.has_header
    assert load(spec).collect().height == 4


def test_quoted_delimiters_do_not_split_fields(samples: dict[str, Path]) -> None:
    frame = load(sniff(samples["csv_quoted"])).collect()
    assert frame.width == 3
    assert frame["note"][0] == "a, b, and c"


def test_gzip_csv(samples: dict[str, Path]) -> None:
    assert load(sniff(samples["gz_csv"])).collect().height == 3


def test_gzip_parquet_is_decompressed(samples: dict[str, Path]) -> None:
    spec = sniff(samples["gz_parquet"])
    assert spec.format is Format.PARQUET
    assert load(spec).collect().height == 5


def test_single_member_zip(samples: dict[str, Path]) -> None:
    assert load(sniff(samples["zip_csv"])).collect().height == 3


@pytest.mark.parametrize(
    ("sample", "fmt"),
    [("sqlite", "sqlite"), ("xml", "xml"), ("html", "html"), ("zip_multi", "zip"), ("ods", "ods")],
)
def test_unsupported_formats_say_so(samples: dict[str, Path], sample: str, fmt: str) -> None:
    with pytest.raises(UnsupportedFormatError) as caught:
        load(sniff(samples[sample]))
    assert fmt in str(caught.value)


def test_zstd_refuses_with_a_reason(samples: dict[str, Path]) -> None:
    with pytest.raises(UnsupportedFormatError, match="decompress"):
        load(sniff(samples["zstd_csv"]))


def test_unparseable_dates_do_not_sink_the_whole_file(tmp_path: Path) -> None:
    """polars' try_parse_dates aborts the read on the first date it dislikes.

    A file whose dates it cannot handle must still load, with those values left
    as strings, rather than failing outright.
    """
    path = tmp_path / "sales.csv"
    path.write_text(
        "id,Date,amount\n1,04/01/10 00:00:00,10\n2,07/01/10 00:00:00,20\n3,11/01/11 00:00:00,30\n",
        encoding="utf-8",
    )
    frame = load(sniff(path)).collect()
    assert frame.height == 3
    assert frame["Date"].dtype == pl.String


def test_lying_extension_still_loads_correctly(samples: dict[str, Path]) -> None:
    """A parquet file named .csv loads as parquet, because sniff said so."""
    frame = load(sniff(samples["parquet_as_csv"])).collect()
    assert frame["units"].to_list() == [12, 7, 23, 4, 19]
