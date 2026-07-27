<div align="center">

# insightsmith

**Forging insight from raw data.**

An agentic data consultant that runs on your own machine.

</div>

---

> ### 0.1.0 is the foundation, not the whole thing
>
> This release ships the parts that need no LLM: **format detection, loading and
> profiling**, behind `ismith look`. It is useful on its own — point it at a file
> and it tells you what the file really is and what's wrong with the data.
>
> The agentic half — proposing analyses, writing and sandbox-executing code,
> critiquing the statistics, writing a report — arrives across 0.4.0–0.8.0.
> Nothing in this release talks to a model or the network.

---

## What it does today

```bash
pip install insightsmith
ismith look data/sales.csv
```

```
╭──────────────── source ─────────────────╮
│ format    csv                           │
│ encoding  cp1252                        │
│ dialect   delimiter=';' decimal=','     │
│ confidence 95%                          │
│ assumed   read as cp1252; charset-      │
│           normalizer suggested cp775    │
╰─────────────────────────────────────────╯
umsatz.csv: 5 rows x 3 columns
        columns
┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ column ┃ dtype   ┃ semantic    ┃ nulls ┃ unique ┃ detail                ┃
┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ region │ String  │ categorical │     - │      4 │ Nord (2) · Süd (1)    │
│ menge  │ Int64   │ numeric     │     - │      5 │ min 4 · med 12 · max… │
│ umsatz │ Float64 │ numeric     │     - │      5 │ min 40.75 · med 120.5 │
└────────┴─────────┴─────────────┴───────┴────────┴───────────────────────┘
candidate keys: menge
```

Add `--json` for the same profile as machine-readable output.

**Format detection doesn't trust the extension.** A three-stage cascade — extension
hint, then magic bytes, then a text-dialect probe — where each stage can veto the
one before it. A Parquet file named `.csv` is loaded as Parquet, and you're told
the extension lied. Every result carries a confidence score and the list of
assumptions behind it; below 80% those assumptions are printed rather than hidden.

It is built for the files that actually turn up: semicolon-delimited cp1252 CSVs
with decimal commas and thousands separators, BOMs, comment preambles, quoted
fields containing the delimiter, gzip and single-member zip wrappers.

**Profiling** reports per-column dtype and semantic type, null rates, cardinality,
numeric summaries with outlier counts by two different methods, temporal ranges,
text lengths, category frequencies, candidate keys, and quality notes: duplicate
and near-duplicate rows, constant and near-constant columns, runaway cardinality,
class imbalance.

### As a library

```python
from insightsmith import load, profile, sniff

spec = sniff("data/sales.csv")
print(spec.format, spec.encoding, spec.confidence, spec.warnings)

frame = load(spec)  # a Polars LazyFrame — nothing read yet
result = profile(spec)
print(result.summary())
for issue in result.issues:
    print(issue.severity.value, issue.column, issue.message)
```

Polars `LazyFrame` is the internal representation throughout, so Parquet, Arrow,
NDJSON and UTF-8 CSV are scanned rather than loaded.

**Formats loadable in 0.1.0:** csv, tsv, xlsx/xlsm (with `[excel]`), xls, parquet,
feather/arrow, json, jsonl/ndjson. Detected-but-not-yet-loadable formats — sqlite,
duckdb, xml, html, ods, orc, hdf5, spss, stata, sas — say so, and name the release
that will handle them.

## Install

```bash
pip install insightsmith            # csv, tsv, parquet, arrow, json, jsonl
pip install insightsmith[excel]     # + xlsx / xls
pip install insightsmith[pandas]    # + a .to_pandas() escape hatch
```

The base install is four dependencies — polars, typer, rich, charset-normalizer.
No torch, no pandas, no agent framework. Extras stay optional on purpose.

## What it won't do

Worth stating plainly, in advance:

- **Large files are profiled on a sample.** Above a size threshold the profile is
  built from a strided sample of the rows, and every affected statistic is marked
  `estimated`. Row counts remain exact; distributions are approximate.
- **Encoding detection is a guess on small files.** Single-byte codepages are
  genuinely ambiguous in a few hundred bytes. Sparse non-ASCII text is read as
  cp1252 and the substitution is reported, but it can still be wrong.
- **Statistics have edges.** The IQR fence is degenerate when the middle 50% of a
  column is a single value, so outliers are counted by a MAD-based modified
  z-score as well, and both numbers are shown. They disagree usefully.
- **"Near-duplicate" means one specific thing** — identical once string columns
  are trimmed and case-folded. It is not fuzzy matching.
- **Thousands separators are detected but not stripped.** polars has no option for
  them, so such columns may load as strings.
- **No XML row-unit discovery, and zstd payloads aren't inspected** (no stdlib
  decompressor before Python 3.14).

And for the releases still to come: LLMs write wrong code confidently, the sandbox
will be defense-in-depth rather than a security boundary, and a statistical critic
reduces but does not remove statistical nonsense.

## License

Apache-2.0. See [LICENSE](LICENSE).
