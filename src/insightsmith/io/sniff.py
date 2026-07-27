"""Format detection: extension hint, then magic bytes, then a text dialect probe.

Each stage can veto the one before it — an extension is a hint, magic bytes are
evidence. The result is a :class:`SourceSpec` carrying a confidence score and the
list of assumptions made along the way, never a bare format string.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final

from charset_normalizer import from_bytes

__all__ = [
    "CONFIDENCE_THRESHOLD",
    "Compression",
    "Dialect",
    "Format",
    "SourceSpec",
    "sniff",
]


class Format(str, Enum):
    """A detected source format. Detection is wider than 0.1.0's loader support."""

    CSV = "csv"
    TSV = "tsv"
    EXCEL = "excel"  # xlsx / xlsm — OOXML spreadsheet
    EXCEL_LEGACY = "excel_legacy"  # xls — OLE2 compound document
    ODS = "ods"
    PARQUET = "parquet"
    ARROW = "arrow"  # feather v2 / Arrow IPC
    ORC = "orc"
    JSON = "json"
    JSONL = "jsonl"
    XML = "xml"
    HTML = "html"
    SQLITE = "sqlite"
    DUCKDB = "duckdb"
    SPSS = "spss"
    STATA = "stata"
    SAS = "sas"
    HDF5 = "hdf5"
    ZIP = "zip"
    UNKNOWN = "unknown"


class Compression(str, Enum):
    """Outer compression wrapper, if any."""

    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"
    ZIP = "zip"


@dataclass(slots=True)
class Dialect:
    """Delimited-text parameters. Only meaningful for :attr:`Format.CSV`/``TSV``."""

    delimiter: str = ","
    quotechar: str = '"'
    doublequote: bool = True
    escapechar: str | None = None
    has_header: bool = True
    decimal: str = "."
    thousands: str | None = None
    comment_prefix: str | None = None
    line_terminator: str = "\n"


@dataclass(slots=True)
class SourceSpec:
    """What we concluded about a source, and how sure we are.

    ``confidence`` below :data:`CONFIDENCE_THRESHOLD` means the caller should
    surface ``warnings`` to the user rather than proceeding silently.
    """

    path: Path
    format: Format
    encoding: str = "utf-8"
    dialect: Dialect | None = None
    compression: Compression = Compression.NONE
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= CONFIDENCE_THRESHOLD


CONFIDENCE_THRESHOLD: Final = 0.8
SAMPLE_BYTES: Final = 8192
PROBE_LINES: Final = 50
CANDIDATE_DELIMITERS: Final = (",", ";", "\t", "|", ":")

# Stage 1 — extension is a hint only.
_EXTENSION_HINTS: Final[dict[str, Format]] = {
    ".csv": Format.CSV,
    ".tsv": Format.TSV,
    ".txt": Format.CSV,
    ".xlsx": Format.EXCEL,
    ".xlsm": Format.EXCEL,
    ".xls": Format.EXCEL_LEGACY,
    ".ods": Format.ODS,
    ".parquet": Format.PARQUET,
    ".feather": Format.ARROW,
    ".arrow": Format.ARROW,
    ".orc": Format.ORC,
    ".json": Format.JSON,
    ".jsonl": Format.JSONL,
    ".ndjson": Format.JSONL,
    ".xml": Format.XML,
    ".html": Format.HTML,
    ".htm": Format.HTML,
    ".sqlite": Format.SQLITE,
    ".db": Format.SQLITE,
    ".duckdb": Format.DUCKDB,
    ".sav": Format.SPSS,
    ".dta": Format.STATA,
    ".sas7bdat": Format.SAS,
    ".h5": Format.HDF5,
    ".hdf5": Format.HDF5,
}

_COMPRESSION_SUFFIXES: Final[dict[str, Compression]] = {
    ".gz": Compression.GZIP,
    ".gzip": Compression.GZIP,
    ".zst": Compression.ZSTD,
    ".zstd": Compression.ZSTD,
    ".zip": Compression.ZIP,
}

# Stage 2 — magic bytes, exactly the table in the design doc.
_MAGIC_GZIP: Final = b"\x1f\x8b"
_MAGIC_ZSTD: Final = b"\x28\xb5\x2f\xfd"
_MAGIC_ZIP: Final = b"PK\x03\x04"
_MAGIC_OLE2: Final = b"\xd0\xcf\x11\xe0"
_MAGIC_SQLITE: Final = b"SQLite format 3\x00"
_MAGIC_PARQUET: Final = b"PAR1"
_MAGIC_ARROW: Final = b"ARROW1"
_MAGIC_ORC: Final = b"ORC"

# Confidence budget. Magic bytes are evidence; an extension alone is a guess.
_CONF_MAGIC: Final = 0.99
_CONF_CONTAINER: Final = 0.97
_CONF_STRUCTURED_TEXT: Final = 0.92
_CONF_DIALECT_AGREED: Final = 0.95
_CONF_DIALECT_DISPUTED: Final = 0.72
_CONF_EXTENSION_ONLY: Final = 0.55
_CONF_OPAQUE: Final = 0.45
_CONF_NONE: Final = 0.1

_NUM_COMMA_THOUSANDS: Final = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_NUM_DOT_THOUSANDS: Final = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")
_NUM_COMMA_DECIMAL: Final = re.compile(r"^-?\d+,\d+$")
_NUM_DOT_DECIMAL: Final = re.compile(r"^-?\d+\.\d+$")


def sniff(
    path: str | Path,
    *,
    sample_bytes: int = SAMPLE_BYTES,
    probe_lines: int = PROBE_LINES,
) -> SourceSpec:
    """Detect the format of ``path`` without trusting its extension.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        IsADirectoryError: if ``path`` is a directory.
    """
    p = Path(path)
    if p.is_dir():
        raise IsADirectoryError(p)
    if not p.exists():
        raise FileNotFoundError(p)

    with p.open("rb") as fh:
        head = fh.read(sample_bytes)

    ext_format, ext_compression = _extension_hint(p)

    if not head:
        return SourceSpec(
            path=p,
            format=Format.UNKNOWN,
            compression=ext_compression,
            confidence=0.0,
            warnings=["file is empty"],
        )

    spec = _detect(p, head, sample_bytes=sample_bytes, probe_lines=probe_lines)
    _reconcile_extension(spec, ext_format)
    return spec


def _extension_hint(path: Path) -> tuple[Format | None, Compression]:
    """Stage 1. ``sales.csv.gz`` hints at gzip-wrapped CSV."""
    suffixes = [s.lower() for s in path.suffixes]
    compression = Compression.NONE
    if suffixes and suffixes[-1] in _COMPRESSION_SUFFIXES:
        compression = _COMPRESSION_SUFFIXES[suffixes.pop()]
    fmt = _EXTENSION_HINTS.get(suffixes[-1]) if suffixes else None
    return fmt, compression


def _detect(path: Path, head: bytes, *, sample_bytes: int, probe_lines: int) -> SourceSpec:
    """Stages 2 and 3, unwrapping any compression first."""
    if head.startswith(_MAGIC_GZIP):
        return _detect_gzip(path, sample_bytes=sample_bytes, probe_lines=probe_lines)
    if head.startswith(_MAGIC_ZSTD):
        return _detect_zstd(path)
    if head.startswith(_MAGIC_ZIP):
        return _detect_zip(path, sample_bytes=sample_bytes, probe_lines=probe_lines)

    fmt = _magic_format(head)
    if fmt is not None:
        return SourceSpec(path=path, format=fmt, confidence=_CONF_MAGIC)

    return _probe_text(path, head, probe_lines=probe_lines)


def _magic_format(head: bytes) -> Format | None:
    """Stage 2, for formats identifiable from a fixed signature."""
    if head.startswith(_MAGIC_PARQUET):
        return Format.PARQUET
    if head.startswith(_MAGIC_ARROW):
        return Format.ARROW
    if head.startswith(_MAGIC_ORC):
        return Format.ORC
    if head.startswith(_MAGIC_SQLITE):
        return Format.SQLITE
    if head.startswith(_MAGIC_OLE2):
        return Format.EXCEL_LEGACY
    return None


def _detect_gzip(path: Path, *, sample_bytes: int, probe_lines: int) -> SourceSpec:
    """Unwrap gzip and detect the payload, keeping ``compression=gzip``."""
    try:
        with gzip.open(path, "rb") as fh:
            inner = fh.read(sample_bytes)
    except (OSError, EOFError) as exc:
        return SourceSpec(
            path=path,
            format=Format.UNKNOWN,
            compression=Compression.GZIP,
            confidence=_CONF_NONE,
            warnings=[f"gzip magic present but the stream could not be read: {exc}"],
        )

    spec = _detect(path, inner, sample_bytes=sample_bytes, probe_lines=probe_lines)
    spec.compression = Compression.GZIP
    return spec


def _detect_zstd(path: Path) -> SourceSpec:
    """zstd has no stdlib decompressor before 3.14, so the payload stays opaque."""
    inner_hint, _ = _extension_hint(path)
    return SourceSpec(
        path=path,
        format=inner_hint or Format.UNKNOWN,
        compression=Compression.ZSTD,
        confidence=_CONF_OPAQUE if inner_hint else _CONF_NONE,
        warnings=[
            "zstd payload not inspected (no stdlib decompressor on this Python); "
            "format taken from the filename"
        ],
    )


def _detect_zip(path: Path, *, sample_bytes: int, probe_lines: int) -> SourceSpec:
    """Disambiguate an OOXML/ODF container from a plain archive."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if "[Content_Types].xml" in names:
                return _ooxml_format(path, zf.read("[Content_Types].xml"))
            # §3 says to inspect [Content_Types].xml, but that part is OOXML-only.
            # Real ODF containers declare themselves in a `mimetype` member, so
            # detecting ODS at all requires checking both.
            if "mimetype" in names:
                return _odf_format(path, zf.read("mimetype"))

            members = [n for n in names if not n.endswith("/")]
            if len(members) == 1:
                with zf.open(members[0]) as member:
                    inner = member.read(sample_bytes)
                spec = _detect(path, inner, sample_bytes=sample_bytes, probe_lines=probe_lines)
                spec.compression = Compression.ZIP
                spec.warnings.append(f"format taken from the archived member {members[0]!r}")
                return spec

            return SourceSpec(
                path=path,
                format=Format.ZIP,
                compression=Compression.ZIP,
                confidence=_CONF_CONTAINER,
                warnings=[f"archive holds {len(members)} members; pick one explicitly"],
            )
    except (zipfile.BadZipFile, OSError) as exc:
        return SourceSpec(
            path=path,
            format=Format.UNKNOWN,
            confidence=_CONF_NONE,
            warnings=[f"zip magic present but the archive could not be opened: {exc}"],
        )


def _ooxml_format(path: Path, declared: bytes) -> SourceSpec:
    """Read the content-type manifest that distinguishes xlsx from ods from docx."""
    if b"spreadsheetml" in declared:
        return SourceSpec(path=path, format=Format.EXCEL, confidence=_CONF_CONTAINER)
    if b"opendocument.spreadsheet" in declared:
        return SourceSpec(path=path, format=Format.ODS, confidence=_CONF_CONTAINER)
    return SourceSpec(
        path=path,
        format=Format.ZIP,
        compression=Compression.ZIP,
        confidence=_CONF_OPAQUE,
        warnings=["OOXML container is not a spreadsheet"],
    )


def _odf_format(path: Path, declared: bytes) -> SourceSpec:
    """ODF declares its type in an uncompressed ``mimetype`` member."""
    if b"opendocument.spreadsheet" in declared:
        return SourceSpec(path=path, format=Format.ODS, confidence=_CONF_CONTAINER)
    return SourceSpec(
        path=path,
        format=Format.ZIP,
        compression=Compression.ZIP,
        confidence=_CONF_OPAQUE,
        warnings=[f"ODF container is not a spreadsheet ({declared[:64].decode(errors='replace')})"],
    )


def _probe_text(path: Path, head: bytes, *, probe_lines: int) -> SourceSpec:
    """Stage 3. Decode, then decide between markup, JSON and delimited text."""
    encoding, text, enc_warnings = _decode(head)
    stripped = text.lstrip()

    if not stripped:
        return SourceSpec(
            path=path,
            format=Format.UNKNOWN,
            encoding=encoding,
            confidence=_CONF_NONE,
            warnings=[*enc_warnings, "no printable content in the sample"],
        )

    if stripped.startswith("<"):
        lowered = stripped[:512].lower()
        is_html = "<html" in lowered or "<!doctype html" in lowered
        return SourceSpec(
            path=path,
            format=Format.HTML if is_html else Format.XML,
            encoding=encoding,
            confidence=_CONF_STRUCTURED_TEXT,
            warnings=enc_warnings,
        )

    if stripped[0] in "{[":
        fmt, confidence, json_warnings = _json_or_jsonl(text)
        return SourceSpec(
            path=path,
            format=fmt,
            encoding=encoding,
            confidence=confidence,
            warnings=[*enc_warnings, *json_warnings],
        )

    return _probe_delimited(
        path, text, encoding=encoding, warnings=enc_warnings, probe_lines=probe_lines
    )


def _decode(head: bytes) -> tuple[str, str, list[str]]:
    """Explicit BOMs win; otherwise ask charset-normalizer."""
    for bom, encoding in (
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe\x00\x00", "utf-32"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    ):
        if head.startswith(bom):
            return encoding, head.decode(encoding, errors="replace"), []

    match = from_bytes(head).best()
    if match is None:
        return (
            "utf-8",
            head.decode("utf-8", errors="replace"),
            ["encoding could not be determined; assumed utf-8"],
        )

    encoding = str(match.encoding).replace("_", "-")
    warnings: list[str] = []
    if encoding not in {"utf-8", "ascii"}:
        warnings.append(f"encoding detected as {encoding}")
    return encoding, str(match), warnings


def _json_or_jsonl(text: str) -> tuple[Format, float, list[str]]:
    """The line-2 test: two consecutive standalone JSON values means JSONL."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and _parses(lines[0]) and _parses(lines[1]):
        return Format.JSONL, _CONF_STRUCTURED_TEXT, []
    if _parses(text):
        return Format.JSON, _CONF_STRUCTURED_TEXT, []
    # An 8 KB window through a large document truncates mid-value, which is not
    # evidence against JSON — only against JSONL.
    return Format.JSON, _CONF_OPAQUE, ["JSON structure incomplete within the sample"]


def _parses(candidate: str) -> bool:
    try:
        json.loads(candidate)
    except (ValueError, RecursionError):
        return False
    return True


def _probe_delimited(
    path: Path,
    text: str,
    *,
    encoding: str,
    warnings: list[str],
    probe_lines: int,
) -> SourceSpec:
    """Pick a delimiter by column-count stability, cross-checked against csv.Sniffer."""
    notes = list(warnings)
    comment_prefix = _comment_prefix(text)
    lines = _content_lines(text, probe_lines=probe_lines, comment_prefix=comment_prefix)
    if not lines:
        return SourceSpec(
            path=path,
            format=Format.UNKNOWN,
            encoding=encoding,
            confidence=_CONF_NONE,
            warnings=[*notes, "no parseable lines in the sample"],
        )

    found = _lowest_variance_delimiter(lines)
    if found is None:
        # No candidate appears even once per line. A one-column export and a
        # paragraph of prose are indistinguishable here, so say so rather than
        # claiming the column count "varies" — every row agrees at zero.
        return SourceSpec(
            path=path,
            format=Format.CSV,
            encoding=encoding,
            dialect=Dialect(
                has_header=_has_header(text, ","),
                comment_prefix=comment_prefix,
                line_terminator="\r\n" if "\r\n" in text else "\n",
            ),
            confidence=_CONF_EXTENSION_ONLY,
            warnings=[*notes, "no delimiter found in the sample; assuming a single column"],
        )

    delimiter, stable = found
    sniffed = _csv_sniffer_delimiter(text)

    if sniffed is not None and sniffed != delimiter:
        notes.append(
            f"csv.Sniffer chose {sniffed!r} but column counts are more stable "
            f"with {delimiter!r}; using {delimiter!r}"
        )
        confidence = _CONF_DIALECT_DISPUTED
    elif not stable:
        notes.append(f"column count varies across rows with {delimiter!r}")
        confidence = _CONF_DIALECT_DISPUTED
    else:
        confidence = _CONF_DIALECT_AGREED

    fields = _split_fields(lines, delimiter)
    decimal, thousands = _numeric_conventions(fields, delimiter)
    dialect = Dialect(
        delimiter=delimiter,
        quotechar=_quotechar(lines),
        has_header=_has_header(text, delimiter),
        decimal=decimal,
        thousands=thousands,
        comment_prefix=comment_prefix,
        line_terminator="\r\n" if "\r\n" in text else "\n",
    )
    return SourceSpec(
        path=path,
        format=Format.TSV if delimiter == "\t" else Format.CSV,
        encoding=encoding,
        dialect=dialect,
        confidence=confidence,
        warnings=notes,
    )


def _comment_prefix(text: str) -> str | None:
    for line in text.splitlines():
        if not line.strip():
            continue
        return "#" if line.lstrip().startswith("#") else None
    return None


def _content_lines(text: str, *, probe_lines: int, comment_prefix: str | None) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        if len(out) >= probe_lines:
            break
        if not line.strip():
            continue
        if comment_prefix and line.lstrip().startswith(comment_prefix):
            continue
        out.append(line)
    # A trailing line may be cut mid-field by the sample window.
    return out[:-1] if len(out) > 2 else out


def _lowest_variance_delimiter(lines: list[str]) -> tuple[str, bool] | None:
    """The real delimiter yields the same field count on every row.

    Returns the winner and whether its column count was perfectly stable, or
    ``None`` when no candidate occurs at least once per line.
    """
    best: str | None = None
    best_key = (float("inf"), 0.0)
    for candidate in CANDIDATE_DELIMITERS:
        counts = [_count_outside_quotes(line, candidate) for line in lines]
        mean = sum(counts) / len(counts)
        if mean < 1:
            continue
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        key = (variance, -mean)
        if key < best_key:
            best_key, best = key, candidate
    if best is None:
        return None
    return best, best_key[0] == 0.0


def _count_outside_quotes(line: str, delimiter: str) -> int:
    count = 0
    in_quotes = False
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == delimiter and not in_quotes:
            count += 1
    return count


def _csv_sniffer_delimiter(text: str) -> str | None:
    try:
        return csv.Sniffer().sniff(text, delimiters="".join(CANDIDATE_DELIMITERS)).delimiter
    except csv.Error:
        return None


def _has_header(text: str, delimiter: str) -> bool:
    try:
        return csv.Sniffer().has_header(text)
    except csv.Error:
        # Fall back to "row 1 is all non-numeric, row 2 is not".
        rows = [line.split(delimiter) for line in text.splitlines()[:2] if line.strip()]
        if len(rows) < 2:
            return True
        first_numeric = any(_looks_numeric(f) for f in rows[0])
        second_numeric = any(_looks_numeric(f) for f in rows[1])
        return not first_numeric and second_numeric


def _looks_numeric(field_value: str) -> bool:
    candidate = field_value.strip().strip('"')
    return bool(
        _NUM_DOT_DECIMAL.match(candidate)
        or _NUM_COMMA_DECIMAL.match(candidate)
        or _NUM_COMMA_THOUSANDS.match(candidate)
        or _NUM_DOT_THOUSANDS.match(candidate)
        or candidate.lstrip("-").isdigit()
    )


def _quotechar(lines: list[str]) -> str:
    doubles = sum(line.count('"') for line in lines)
    singles = sum(line.count("'") for line in lines)
    return "'" if singles > doubles and singles >= 2 else '"'


def _split_fields(lines: list[str], delimiter: str) -> list[str]:
    """Data fields only — the header row would skew the numeric vote."""
    out: list[str] = []
    for line in lines[1:]:
        out.extend(part.strip().strip('"') for part in line.split(delimiter))
    return out


def _numeric_conventions(fields: list[str], delimiter: str) -> tuple[str, str | None]:
    """European ``1.234,56`` versus Anglo ``1,234.56``."""
    comma_grouped = sum(bool(_NUM_COMMA_THOUSANDS.match(f)) for f in fields)
    dot_grouped = sum(bool(_NUM_DOT_THOUSANDS.match(f)) for f in fields)
    if comma_grouped or dot_grouped:
        if comma_grouped >= dot_grouped:
            return ".", ","
        return ",", "."

    comma_decimal = sum(bool(_NUM_COMMA_DECIMAL.match(f)) for f in fields)
    dot_decimal = sum(bool(_NUM_DOT_DECIMAL.match(f)) for f in fields)
    if delimiter != "," and comma_decimal > dot_decimal:
        return ",", None
    return ".", None


def _reconcile_extension(spec: SourceSpec, ext_format: Format | None) -> None:
    """Record — but do not obey — an extension that contradicts the evidence."""
    if ext_format is None or spec.format is Format.UNKNOWN:
        return
    if ext_format is spec.format:
        return
    # csv/tsv differ only by the delimiter we just measured, so this is no conflict.
    if {ext_format, spec.format} <= {Format.CSV, Format.TSV}:
        return
    if spec.confidence <= _CONF_EXTENSION_ONLY:
        spec.format = ext_format
        spec.warnings.append(
            f"content was inconclusive; fell back to the {ext_format.value} extension"
        )
        return
    spec.warnings.append(
        f"extension suggests {ext_format.value} but the content is {spec.format.value}"
    )
