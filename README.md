<div align="center">

# insightsmith

**Forging insight from raw data.**

An agentic data consultant that runs on your own machine.

</div>

---

> ### The foundation, not yet the whole thing
>
> Shipped so far: **format detection, loading and profiling** (`ismith look`),
> **hardware probing with model-fit recommendation** (`ismith doctor`), the
> **provider layer** routing roles to local or cloud models (`ismith models`), and
> the **dataset card plus ideation** behind `ismith look --ideas`.
>
> Still to come across 0.5.0–0.8.0: sandboxed code execution, visualisation, the
> statistical critic, and reports. Nothing sends your data anywhere unless you
> configure a remote provider yourself.

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

### Which local model fits your machine

```bash
ismith doctor
```

Probes CPU, RAM, disk and GPU, then sizes each catalogued model against what it
finds — **per role**, because routing and planning are different jobs and your
machine may afford one but not the other.

The arithmetic is the point. Weights come from the quantisation's bytes-per-param;
the KV cache from `2 × layers × n_kv_heads × head_dim × context × 2 bytes`. Using
`n_kv_heads` rather than the attention-head count is what makes it right: on a
4 GB laptop GPU at 8k context, `qwen3:8b` (8 KV heads) needs 1.21 GB of cache
against 4.92 GB of weights, while `deepseek-coder:6.7b` — same size, but 32 KV
heads and no grouped-query attention — needs **4.29 GB of cache against 4.20 GB
of weights**. A rule of thumb based on parameter count alone cannot tell you that.

Models that don't fit outright get a partial-offload layer count rather than a
shrug, and anything larger than your RAM is excluded with a reason.

### Asking a model what's worth analysing

```bash
ismith look data/sales.csv --ideas    # ranked analyses
ismith look data/sales.csv --card     # exactly what the model will be shown
```

**The dataframe is never pasted into a prompt.** Everything an agent sees arrives
through a *dataset card*: a compact JSON summary of schema, per-column statistics,
quality flags, a correlation shortlist, and a few stratified example rows with
obvious PII masked. Three things follow, and they are the reason for the design:

- **Token cost is flat regardless of file size.** A 589 KB, 4248-row, 20-column
  file produces a 4.8 KB card — and so would a 40 GB one. That is what makes an
  8B local model workable.
- **No raw records leave the machine.** Sensitive columns are redacted by name,
  recognisable values by pattern.
- **The card hashes**, so the same data yields the same plan and results cache.

Ideas come back ranked, each naming the columns it needs — and **any idea
referencing a column the card doesn't contain is discarded before you see it**.
That one check removes most hallucination for the price of a set-membership test.
Because the quality notes are on the card too, a model will hedge appropriately:
given a date column flagged as ambiguous, it proposes parsing it "with caution".

### Wiring up a model

```bash
ismith models
```

Shows what each role resolves to, whether it stays on your machine, and **how it
will be asked for structured output**. Configure it in `~/.insightsmith/config.toml`:

```toml
[roles]
planner = "ollama/qwen3:8b"
coder   = "ollama/qwen2.5-coder:7b"
cheap   = "ollama/qwen3:4b"

[budget]
max_usd_per_session = 0.50
local_only = true
```

One class covers every backend that speaks the OpenAI wire format — OpenAI,
OpenRouter, DeepInfra, Together, Groq, Fireworks, Mistral, Gemini's compat
endpoint, and local vLLM / llama.cpp / LM Studio — because they differ only by
base URL and key. Ollama is written natively instead, since `/api/show`,
`/api/ps` and `keep_alive` are the whole reason to run locally.

**Capabilities are read, not assumed.** Ollama reports whether a model supports
tool-calling, so the router picks its approach up front rather than failing
mid-run: tool-calling where available, JSON mode next, and otherwise prompted
JSON with a bounded retry that feeds the parse failure back. Small models wrap
JSON in prose and code fences no matter how firmly told not to, so that path is
the common case, not an edge case.

**`local_only = true` is a hard failure, not a warning.** Point a role at a
remote provider with it set and loading the config raises, naming the offending
role. A privacy switch that only warned would not be a privacy switch.

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

**Formats loadable today:** csv, tsv, xlsx/xlsm (with `[excel]`), xls, parquet,
feather/arrow, json, jsonl/ndjson. Detected-but-not-yet-loadable formats — sqlite,
duckdb, xml, html, ods, orc, hdf5, spss, stata, sas — say so, and name the release
that will handle them.

## Install

```bash
pip install insightsmith            # csv, tsv, parquet, arrow, json, jsonl
pip install insightsmith[excel]     # + xlsx / xls
pip install insightsmith[pandas]    # + a .to_pandas() escape hatch
```

The base install is six dependencies — polars, typer, rich, charset-normalizer,
psutil and httpx (plus a TOML backport on Python 3.10 only). No torch, no pandas,
no agent framework. Extras stay optional on purpose.

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
- **PII masking is best-effort, not a guarantee.** It catches values that look
  like contact details or identifiers and blanks columns whose names say they
  hold personal data. It cannot recognise a person's name in free text, an
  address split across columns, or an identifier in a format it hasn't seen. If
  data must not leave the machine, set `local_only` — masking is defence in
  depth, not a substitute for keeping it local. `--card` shows exactly what would
  be sent, and `include_examples=False` omits sample values entirely.
- **Dates are inferred, and `04/01/10` is genuinely ambiguous.** Date columns load
  as strings and the format is inferred afterwards, so a file polars cannot parse
  still loads. Where day-first and month-first fit equally well, the chosen format
  is reported as a quality note rather than picked silently — check it before
  trusting anything grouped by that column.
- **Thousands separators are detected but not stripped.** polars has no option for
  them, so such columns may load as strings.
- **No XML row-unit discovery, and zstd payloads aren't inspected** (no stdlib
  decompressor before Python 3.14).
- **Throughput figures are estimates, not benchmarks.** `ismith doctor` derives
  tok/s from published peak memory bandwidth, which no real decode loop reaches.
  Devices missing from that table report `unknown` rather than a plausible
  substitute, and the catalog only contains models whose layer and KV-head counts
  were read from a running Ollama — never guessed.

And for the releases still to come: LLMs write wrong code confidently, the sandbox
will be defense-in-depth rather than a security boundary, and a statistical critic
reduces but does not remove statistical nonsense.

## License

Apache-2.0. See [LICENSE](LICENSE).
