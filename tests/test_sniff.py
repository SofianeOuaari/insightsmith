"""Sniffer tests: the magic-byte table, the cascade's vetoes, and the dialect probe."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from insightsmith.io.sniff import (
    CONFIDENCE_THRESHOLD,
    Compression,
    Format,
    SourceSpec,
    sniff,
)

# --------------------------------------------------------------------------- #
# stage 2 — magic bytes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ("parquet", Format.PARQUET),
        ("feather", Format.ARROW),
        ("sqlite", Format.SQLITE),
        ("xls", Format.EXCEL_LEGACY),
        ("xlsx", Format.EXCEL),
        ("ods", Format.ODS),
        ("xml", Format.XML),
        ("xml_no_decl", Format.XML),
        ("html", Format.HTML),
        ("jsonl", Format.JSONL),
        ("json_array", Format.JSON),
        ("json_pretty", Format.JSON),
        ("csv", Format.CSV),
        ("tsv", Format.TSV),
    ],
)
def test_format_detected(samples: dict[str, Path], sample: str, expected: Format) -> None:
    spec = sniff(samples[sample])
    assert spec.format is expected
    assert spec.is_confident, f"{sample} detected at only {spec.confidence}"


def test_binary_magic_beats_a_lying_extension(samples: dict[str, Path]) -> None:
    spec = sniff(samples["parquet_as_csv"])
    assert spec.format is Format.PARQUET
    assert any("extension suggests csv" in w for w in spec.warnings)


def test_text_content_beats_a_lying_extension(samples: dict[str, Path]) -> None:
    spec = sniff(samples["csv_as_parquet"])
    assert spec.format is Format.CSV
    assert any("extension suggests parquet" in w for w in spec.warnings)


def test_ooxml_container_that_is_not_a_spreadsheet(samples: dict[str, Path]) -> None:
    spec = sniff(samples["docx"])
    assert spec.format is Format.ZIP
    assert not spec.is_confident


def test_odf_container_that_is_not_a_spreadsheet(samples: dict[str, Path]) -> None:
    spec = sniff(samples["odt"])
    assert spec.format is Format.ZIP


# --------------------------------------------------------------------------- #
# JSON vs JSONL — the line-2 check
# --------------------------------------------------------------------------- #


def test_jsonl_needs_two_parseable_lines() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        single = Path(tmp) / "one.json"
        single.write_text('{"a":1}\n', encoding="utf-8")
        assert sniff(single).format is Format.JSON

        double = Path(tmp) / "two.json"
        double.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
        assert sniff(double).format is Format.JSONL


def test_truncated_json_is_flagged_not_guessed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        big = Path(tmp) / "big.json"
        rows = ",".join(f'{{"i":{i},"pad":"{"x" * 40}"}}' for i in range(400))
        big.write_text(f"[{rows}]", encoding="utf-8")
        spec = sniff(big)
    assert spec.format is Format.JSON
    assert any("incomplete" in w for w in spec.warnings)


# --------------------------------------------------------------------------- #
# stage 3 — dialect probe
# --------------------------------------------------------------------------- #


def test_german_csv(samples: dict[str, Path]) -> None:
    """Semicolons, decimal commas, cp1252 — the combination that breaks tools."""
    spec = sniff(samples["german_csv"])
    assert spec.format is Format.CSV
    assert spec.dialect is not None
    assert spec.dialect.delimiter == ";"
    assert spec.dialect.decimal == ","
    assert samples["german_csv"].read_bytes().decode(spec.encoding).startswith("Region")


def test_thousands_separator_inverts_the_decimal_mark(samples: dict[str, Path]) -> None:
    spec = sniff(samples["grouped_csv"])
    assert spec.dialect is not None
    assert spec.dialect.delimiter == ";"
    assert spec.dialect.thousands == "."
    assert spec.dialect.decimal == ","


def test_bom_gives_utf_8_sig(samples: dict[str, Path]) -> None:
    assert sniff(samples["csv_bom"]).encoding == "utf-8-sig"


def test_pipe_delimiter(samples: dict[str, Path]) -> None:
    spec = sniff(samples["pipe_csv"])
    assert spec.dialect is not None
    assert spec.dialect.delimiter == "|"


def test_delimiters_inside_quotes_are_not_counted(samples: dict[str, Path]) -> None:
    spec = sniff(samples["csv_quoted"])
    assert spec.dialect is not None
    assert spec.dialect.delimiter == ","


def test_comment_prefix(samples: dict[str, Path]) -> None:
    spec = sniff(samples["csv_commented"])
    assert spec.dialect is not None
    assert spec.dialect.comment_prefix == "#"
    assert spec.dialect.delimiter == ","


@pytest.mark.parametrize(
    "payload",
    [
        "region\nnorth\nsouth\neast\n",
        "This is prose.\nIt has no delimiters.\nJust sentences.\n",
    ],
    ids=["single-column", "prose"],
)
def test_absence_of_a_delimiter_is_reported_as_such(payload: str) -> None:
    """Every row agreeing at zero delimiters is not the same as a varying count."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "one_column.csv"
        path.write_text(payload, encoding="utf-8")
        spec = sniff(path)
    assert not spec.is_confident
    assert any("no delimiter found" in w for w in spec.warnings)
    assert not any("varies" in w for w in spec.warnings)


def test_ragged_rows_do_report_a_varying_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ragged.csv"
        path.write_text("a,b,c\n1,2,3\n4,5\n6,7,8,9\n", encoding="utf-8")
        spec = sniff(path)
    assert spec.dialect is not None
    assert spec.dialect.delimiter == ","
    assert any("varies" in w for w in spec.warnings)


def test_quotes_shield_a_rival_delimiter() -> None:
    """A quoted field full of semicolons must not out-vote the real comma."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shielded.csv"
        path.write_text('a,b\n"x;y;z;w",1\n"p;q;r;s",2\n"m;n;o;p",3\n', encoding="utf-8")
        spec = sniff(path)
    assert spec.dialect is not None
    assert spec.dialect.delimiter == ","
    assert spec.is_confident


def test_utf16_bom_is_decoded() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wide.csv"
        path.write_bytes("region,units\nnord,12\nsud,7\nest,9\n".encode("utf-16"))
        spec = sniff(path)
    assert spec.encoding == "utf-16"
    assert spec.dialect is not None
    assert spec.dialect.delimiter == ","


def test_header_presence(samples: dict[str, Path]) -> None:
    with_header = sniff(samples["csv"])
    without = sniff(samples["csv_no_header"])
    assert with_header.dialect is not None
    assert without.dialect is not None
    assert with_header.dialect.has_header
    assert not without.dialect.has_header


# --------------------------------------------------------------------------- #
# compression
# --------------------------------------------------------------------------- #


def test_gzip_is_unwrapped_and_the_payload_identified(samples: dict[str, Path]) -> None:
    spec = sniff(samples["gz_csv"])
    assert spec.compression is Compression.GZIP
    assert spec.format is Format.CSV
    assert spec.dialect is not None
    assert spec.dialect.delimiter == ","


def test_single_member_zip_is_treated_as_compression(samples: dict[str, Path]) -> None:
    spec = sniff(samples["zip_single"])
    assert spec.compression is Compression.ZIP
    assert spec.format is Format.CSV
    assert any("table.csv" in w for w in spec.warnings)


def test_multi_member_zip_asks_for_a_choice(samples: dict[str, Path]) -> None:
    spec = sniff(samples["zip_multi"])
    assert spec.format is Format.ZIP
    assert any("2 members" in w for w in spec.warnings)


def test_zstd_is_honest_about_not_looking_inside(samples: dict[str, Path]) -> None:
    spec = sniff(samples["zstd_csv"])
    assert spec.compression is Compression.ZSTD
    assert spec.format is Format.CSV  # from the filename, not the content
    assert not spec.is_confident
    assert any("not inspected" in w for w in spec.warnings)


# --------------------------------------------------------------------------- #
# degenerate input
# --------------------------------------------------------------------------- #


def test_empty_file(samples: dict[str, Path]) -> None:
    spec = sniff(samples["empty"])
    assert spec.format is Format.UNKNOWN
    assert spec.confidence == 0.0
    assert spec.warnings == ["file is empty"]


def test_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sniff(tmp_path / "nope.csv")


def test_directory(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        sniff(tmp_path)


def test_spec_carries_its_path(samples: dict[str, Path]) -> None:
    spec = sniff(samples["csv"])
    assert isinstance(spec, SourceSpec)
    assert spec.path == samples["csv"]


# --------------------------------------------------------------------------- #
# property tests
# --------------------------------------------------------------------------- #

_SAFE_TEXT = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzüé"), min_size=1, max_size=8
)


@dataclass(slots=True)
class _Case:
    payload: bytes
    suffix: str
    delimiter: str
    decimal: str


@st.composite
def _delimited(draw: st.DrawFn) -> _Case:
    delimiter = draw(st.sampled_from([",", ";", "\t", "|"]))
    decimal = draw(st.sampled_from([".", ","]))
    assume(not (decimal == "," and delimiter == ","))
    encoding = draw(st.sampled_from(["utf-8", "utf-8-sig", "cp1252", "latin-1"]))
    quoted = draw(st.booleans())
    crlf = draw(st.booleans())
    names = draw(st.lists(_SAFE_TEXT, min_size=4, max_size=10, unique=True))

    def cell(value: str) -> str:
        return f'"{value}"' if quoted else value

    header = delimiter.join(["region", "units", "revenue"])
    rows = [
        delimiter.join([cell(name), str(10 + i), f"1{decimal}5{i}"]) for i, name in enumerate(names)
    ]
    text = ("\r\n" if crlf else "\n").join([header, *rows]) + "\n"
    suffix = {",": ".csv", ";": ".csv", "\t": ".tsv", "|": ".txt"}[delimiter]
    return _Case(text.encode(encoding), suffix, delimiter, decimal)


@given(case=_delimited())
def test_dialect_survives_a_round_trip(case: _Case) -> None:
    """Delimiter and decimal mark must come back out however the file was written."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"table{case.suffix}"
        path.write_bytes(case.payload)
        spec = sniff(path)

    assert spec.format is (Format.TSV if case.delimiter == "\t" else Format.CSV)
    assert spec.dialect is not None
    assert spec.dialect.delimiter == case.delimiter
    assert spec.dialect.decimal == case.decimal
    # Whatever encoding was detected must at least decode the bytes we wrote.
    case.payload.decode(spec.encoding)


@given(
    fmt=st.sampled_from(["parquet", "feather", "jsonl"]),
    misleading=st.sampled_from([".csv", ".json", ".dat", ""]),
    rows=st.integers(min_value=1, max_value=25),
)
def test_binary_formats_ignore_the_extension(fmt: str, misleading: str, rows: int) -> None:
    frame = pl.DataFrame({"i": list(range(rows)), "v": [f"r{i}" for i in range(rows)]})
    expected = {"parquet": Format.PARQUET, "feather": Format.ARROW, "jsonl": Format.JSONL}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"data{misleading}"
        if fmt == "parquet":
            frame.write_parquet(path)
        elif fmt == "feather":
            frame.write_ipc(path)
        else:
            frame.write_ndjson(path)
        spec = sniff(path)

    if fmt == "jsonl" and rows == 1:
        # One record is a JSON document; JSONL needs a second line to prove itself.
        assert spec.format is Format.JSON
    else:
        assert spec.format is expected[fmt]
        assert spec.confidence >= CONFIDENCE_THRESHOLD
