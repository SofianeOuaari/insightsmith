"""Sample sources, generated rather than committed as binaries.

The only fixtures that fake a format instead of writing a real one are the
signature-only cases (`xls`, `zstd_csv`) — the sniffer reads a header, so a header
is all they need. A loader test would need genuine files.
"""

from __future__ import annotations

import gzip
import os
import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path

import polars as pl
import pytest
from hypothesis import settings as hypothesis_settings

# The sniffer is the one module where "looks right" and "is right" diverge, so the
# example budget is tunable: HYPOTHESIS_PROFILE=stress before a release.
hypothesis_settings.register_profile("dev", max_examples=200, deadline=None)
hypothesis_settings.register_profile("stress", max_examples=3000, deadline=None)
hypothesis_settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

_OOXML_SHEET = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
_OOXML_DOC = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"

TABLE = pl.DataFrame(
    {
        "region": ["north", "south", "east", "west", "north"],
        "units": [12, 7, 23, 4, 19],
        "revenue": [120.5, 70.25, 230.0, 40.75, 190.5],
    }
)


def _write_ooxml(path: Path, content_type: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            f'<Override PartName="/xl/workbook.xml" ContentType="{content_type}"/>'
            "</Types>",
        )


def _write_odf(path: Path, mimetype: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", mimetype)
        zf.writestr("content.xml", "<office:document-content/>")


@pytest.fixture(scope="session")
def samples(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One directory of every source shape the sniffer must recognise."""
    d = tmp_path_factory.mktemp("samples")
    out: dict[str, Path] = {}

    def put(name: str, filename: str) -> Path:
        out[name] = d / filename
        return out[name]

    put("csv", "plain.csv").write_text(
        "region,units,revenue\nnorth,12,120.50\nsouth,7,70.25\neast,23,230.00\n",
        encoding="utf-8",
    )
    put("csv_bom", "bom.csv").write_text("region,units\nnorth,12\nsouth,7\n", encoding="utf-8-sig")
    # The case §11 calls out: semicolons, decimal commas, cp1252.
    put("german_csv", "umsatz.csv").write_bytes(
        "Region;Menge;Umsatz\nNord;12;120,50\nSüd;7;70,25\nOst;23;230,00\nKöln;4;40,75\n".encode(
            "cp1252"
        )
    )
    put("grouped_csv", "grouped.csv").write_text(
        "city;population;area\nBerlin;3.664.088;891,68\nHamburg;1.852.478;755,22\n"
        "München;1.488.202;310,70\n",
        encoding="utf-8",
    )
    put("tsv", "plain.tsv").write_text(
        "region\tunits\trevenue\nnorth\t12\t120.50\nsouth\t7\t70.25\neast\t23\t230.0\n",
        encoding="utf-8",
    )
    put("pipe_csv", "pipes.csv").write_text(
        "region|units|revenue\nnorth|12|120.50\nsouth|7|70.25\neast|23|230.0\n",
        encoding="utf-8",
    )
    put("csv_commented", "commented.csv").write_text(
        "# exported 2026-01-01\n# source: warehouse\nregion,units\nnorth,12\nsouth,7\neast,23\n",
        encoding="utf-8",
    )
    put("csv_quoted", "quoted.csv").write_text(
        'region,note,units\n"north","a, b, and c",12\n"south","x, y",7\n"east","p, q",23\n',
        encoding="utf-8",
    )
    put("csv_no_header", "headerless.csv").write_text(
        "north,12,120.50\nsouth,7,70.25\neast,23,230.00\nwest,4,40.75\n",
        encoding="utf-8",
    )

    TABLE.write_parquet(put("parquet", "table.parquet"))
    TABLE.write_ipc(put("feather", "table.feather"))
    put("jsonl", "rows.jsonl").write_text(
        '{"region":"north","units":12}\n{"region":"south","units":7}\n', encoding="utf-8"
    )
    put("json_array", "rows.json").write_text(
        '[{"region":"north","units":12},{"region":"south","units":7}]', encoding="utf-8"
    )
    put("json_pretty", "config.json").write_text(
        '{\n  "region": "north",\n  "units": 12\n}\n', encoding="utf-8"
    )

    _write_ooxml(put("xlsx", "book.xlsx"), _OOXML_SHEET)
    _write_ooxml(put("docx", "letter.docx"), _OOXML_DOC)
    _write_odf(put("ods", "book.ods"), "application/vnd.oasis.opendocument.spreadsheet")
    _write_odf(put("odt", "letter.odt"), "application/vnd.oasis.opendocument.text")

    # OLE2 signature only — enough for detection, not a readable workbook.
    put("xls", "legacy.xls").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512)
    # zstd frame magic only; there is no stdlib compressor to make a real one.
    put("zstd_csv", "table.csv.zst").write_bytes(b"\x28\xb5\x2f\xfd" + b"\x00" * 64)

    # `with sqlite3.connect(...)` manages the transaction, NOT the connection —
    # it never closes. On 3.13 the leaked handle surfaces as an unraisable
    # exception during GC, which filterwarnings=error turns into a failure in
    # whichever unrelated test happened to be running. closing() shuts it.
    with closing(sqlite3.connect(put("sqlite", "store.db"))) as con, con:
        con.execute("CREATE TABLE sales (region TEXT, units INT)")
        con.execute("INSERT INTO sales VALUES ('north', 12)")

    csv_bytes = b"region,units\nnorth,12\nsouth,7\neast,23\n"
    with gzip.open(put("gz_csv", "table.csv.gz"), "wb") as fh:
        fh.write(csv_bytes)
    with zipfile.ZipFile(put("zip_single", "one.zip"), "w") as zf:
        zf.writestr("table.csv", csv_bytes.decode())
    with zipfile.ZipFile(put("zip_multi", "many.zip"), "w") as zf:
        zf.writestr("a.csv", csv_bytes.decode())
        zf.writestr("b.csv", csv_bytes.decode())

    put("xml", "rows.xml").write_text(
        '<?xml version="1.0"?><rows><row><region>north</region></row></rows>',
        encoding="utf-8",
    )
    put("xml_no_decl", "bare.xml").write_text(
        "<rows><row><region>north</region></row></rows>", encoding="utf-8"
    )
    put("html", "table.html").write_text(
        "<!doctype html><html><body><table><tr><td>north</td></tr></table></body></html>",
        encoding="utf-8",
    )
    put("empty", "empty.csv").write_bytes(b"")

    # Extension lies; content does not.
    TABLE.write_parquet(put("parquet_as_csv", "actually_parquet.csv"))
    put("csv_as_parquet", "actually_csv.parquet").write_text(
        "region,units\nnorth,12\nsouth,7\n", encoding="utf-8"
    )

    # A genuine workbook, for the loaders. xlsxwriter is a dev dependency; the
    # published package reads with fastexcel and never needs a writer.
    TABLE.write_excel(put("xlsx_real", "real.xlsx"))
    TABLE.write_ndjson(put("ndjson", "table.ndjson"))
    with zipfile.ZipFile(put("zip_csv", "zipped.zip"), "w") as zf:
        zf.writestr("inner.csv", "region,units\nnorth,12\nsouth,7\neast,23\n")
    with gzip.open(put("gz_parquet", "table.parquet.gz"), "wb") as fh:
        fh.write(out["parquet"].read_bytes())

    # Deliberately grubby: nulls, a duplicate row, a near-duplicate differing
    # only by case and spacing, a constant column, and one wild outlier.
    put("messy", "messy.csv").write_text(
        "name,team,score,note\n"
        "ada,alpha,10,x\n"
        "ada,alpha,10,x\n"
        " ADA ,alpha,10,x\n"
        "bob,alpha,12,\n"
        "cyd,alpha,11,\n"
        "dee,alpha,9999,\n"
        "eve,alpha,13,\n"
        "fay,alpha,,\n",
        encoding="utf-8",
    )
    return out
