"""Loading, behind one entry point: :func:`load`.

Lazy wherever the format allows it — Parquet, Arrow IPC, NDJSON and UTF-8 CSV
all scan without materialising. Everything else is read eagerly and wrapped with
``.lazy()`` so callers see a single type regardless.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from importlib.util import find_spec
from typing import Final

import polars as pl

from insightsmith.errors import MissingDependencyError, UnsupportedFormatError
from insightsmith.io.sniff import Compression, Dialect, Format, SourceSpec

__all__ = ["LOADABLE_FORMATS", "load"]

LOADABLE_FORMATS: Final = frozenset(
    {
        Format.CSV,
        Format.TSV,
        Format.EXCEL,
        Format.EXCEL_LEGACY,
        Format.PARQUET,
        Format.ARROW,
        Format.JSON,
        Format.JSONL,
    }
)

# polars accepts only these for CSV; anything else has to be transcoded first.
_NATIVE_ENCODINGS: Final = frozenset({"utf-8", "utf8", "ascii", "utf-8-sig"})

_MILESTONES: Final[dict[Format, str]] = {
    Format.SQLITE: "SQL sources arrive in 1.2.0",
    Format.DUCKDB: "SQL sources arrive in 1.2.0",
    Format.XML: "XML streaming arrives in 1.1.0",
    Format.HTML: "HTML tables arrive in 1.1.0",
    Format.ORC: "ORC arrives in 1.1.0",
    Format.HDF5: "HDF5 arrives in 1.1.0",
    Format.SPSS: "SPSS/Stata/SAS arrive in 0.9.0",
    Format.STATA: "SPSS/Stata/SAS arrive in 0.9.0",
    Format.SAS: "SPSS/Stata/SAS arrive in 0.9.0",
    Format.ODS: "only OOXML spreadsheets are supported; re-save as .xlsx",
    Format.ZIP: "point at a single member instead of the archive",
    Format.UNKNOWN: "the format could not be determined",
}


def load(spec: SourceSpec) -> pl.LazyFrame:
    """Load the source described by ``spec``.

    Raises:
        UnsupportedFormatError: if the format is detected but not loadable yet.
        MissingDependencyError: if an optional extra is required and absent.
    """
    if spec.format not in LOADABLE_FORMATS:
        raise UnsupportedFormatError(spec.format.value, detail=_MILESTONES.get(spec.format))
    if spec.compression is Compression.ZSTD:
        raise UnsupportedFormatError(
            "zstd-compressed",
            detail="no stdlib decompressor before Python 3.14; decompress it first",
        )

    payload = _payload(spec)

    if spec.format in {Format.CSV, Format.TSV}:
        return _load_delimited(spec, payload)
    if spec.format in {Format.EXCEL, Format.EXCEL_LEGACY}:
        return _load_excel(spec, payload)
    if spec.format is Format.PARQUET:
        source = io.BytesIO(payload) if payload is not None else spec.path
        return pl.scan_parquet(source)
    if spec.format is Format.ARROW:
        source = io.BytesIO(payload) if payload is not None else spec.path
        return pl.scan_ipc(source)
    if spec.format is Format.JSONL:
        if payload is not None:
            return pl.read_ndjson(io.BytesIO(payload)).lazy()
        return pl.scan_ndjson(spec.path)
    # JSON is a single document: there is nothing to stream.
    return pl.read_json(io.BytesIO(payload) if payload is not None else spec.path).lazy()


def _payload(spec: SourceSpec) -> bytes | None:
    """Bytes polars must be handed directly, or ``None`` to let it open the path.

    Returning ``None`` is the fast path — it keeps scanning lazy.
    """
    if spec.compression is Compression.ZIP:
        return _unzip(spec)

    needs_transcode = (
        spec.format in {Format.CSV, Format.TSV} and spec.encoding.lower() not in _NATIVE_ENCODINGS
    )
    if spec.compression is Compression.GZIP:
        # polars decompresses gzip for CSV itself; other readers do not.
        if spec.format in {Format.CSV, Format.TSV} and not needs_transcode:
            return None
        raw = gzip.decompress(spec.path.read_bytes())
        return _transcode(raw, spec.encoding) if needs_transcode else raw

    if needs_transcode:
        return _transcode(spec.path.read_bytes(), spec.encoding)
    return None


def _unzip(spec: SourceSpec) -> bytes:
    with zipfile.ZipFile(spec.path) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        if len(members) != 1:
            raise UnsupportedFormatError("zip", detail=f"expected one member, found {len(members)}")
        raw = zf.read(members[0])
    if spec.format in {Format.CSV, Format.TSV} and spec.encoding.lower() not in _NATIVE_ENCODINGS:
        return _transcode(raw, spec.encoding)
    return raw


def _transcode(raw: bytes, encoding: str) -> bytes:
    """polars reads UTF-8 only, so cp1252/latin-1/utf-16 sources get re-encoded.

    This materialises the source, which is the price of correctness on files
    polars cannot scan directly.
    """
    return raw.decode(encoding, errors="replace").encode("utf-8")


def _load_delimited(spec: SourceSpec, payload: bytes | None) -> pl.LazyFrame:
    dialect = spec.dialect or Dialect()
    options: dict[str, object] = {
        "separator": dialect.delimiter,
        "has_header": dialect.has_header,
        "quote_char": dialect.quotechar,
        "comment_prefix": dialect.comment_prefix,
        "decimal_comma": dialect.decimal == ",",
        "try_parse_dates": True,
        "infer_schema_length": 10_000,
    }
    if payload is not None:
        return pl.read_csv(io.BytesIO(payload), **options).lazy()  # type: ignore[arg-type]
    return pl.scan_csv(spec.path, **options)  # type: ignore[arg-type]


def _load_excel(spec: SourceSpec, payload: bytes | None) -> pl.LazyFrame:
    if find_spec("fastexcel") is None:
        raise MissingDependencyError("fastexcel", "excel", purpose="reading spreadsheets")
    source = io.BytesIO(payload) if payload is not None else spec.path
    # Default options read the first worksheet, so this is always one frame.
    return pl.read_excel(source, engine="calamine").lazy()
