<div align="center">

<img src="https://raw.githubusercontent.com/SofianeOuaari/insightsmith/main/assets/logo.png" alt="insightsmith" width="160">

# insightsmith

**An agentic data consultant that runs on your own machine.**

[![CI](https://github.com/SofianeOuaari/insightsmith/actions/workflows/ci.yml/badge.svg)](https://github.com/SofianeOuaari/insightsmith/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/insightsmith.svg)](https://pypi.org/project/insightsmith/)
[![Python](https://img.shields.io/pypi/pyversions/insightsmith.svg)](https://pypi.org/project/insightsmith/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Point it at a data file. It detects the real format, profiles the data, proposes
analyses, writes and sandbox-executes the code, critiques the statistics, and
forges a report.

</div>

<img src="https://raw.githubusercontent.com/SofianeOuaari/insightsmith/main/assets/report_generation_example_1.png" alt="An insightsmith report: 14 findings from a 510-row loan file" width="100%">

## Features

- **Local-first.** A local model on your hardware is the default, not a degraded
  fallback. Set `local_only = true` and a remote model becomes a hard error.
- **Your rows never reach the model.** Every agent sees a *dataset card*: a 2 to
  5 KB JSON summary with PII masked. Token cost is flat whether the file is 600 KB
  or 40 GB.
- **A statistical critic.** Every answer is checked against a fixed list of
  computable problems (tiny groups, skewed means, silent nulls, outlier-driven
  correlations) and carries the caveats that fired.
- **Auditable by construction.** Every result ships with the code that produced
  it, the hash of the card the model saw, and a notebook that re-runs it.
- **Hardware-aware.** `ismith doctor` reads your GPU and RAM and tells you which
  models actually fit, with the KV-cache maths done properly.
- **Format sniffing that works on messy files.** Semicolon CSVs, decimal commas,
  cp1252, ambiguous dates. It says what it assumed and how sure it is.

## Install

```bash
pip install insightsmith              # csv, tsv, parquet, arrow, json, jsonl
pip install insightsmith[excel]       # + xlsx / xls
pip install insightsmith[viz]         # + charts (matplotlib, plotly)
pip install insightsmith[pandas]      # + a .to_pandas() escape hatch
pip install insightsmith[stats]       # + scipy / statsmodels / scikit-learn
pip install insightsmith[fireducks]   # + the pandas-compatible engine
pip install insightsmith[pdf]         # + printing a report to PDF
```

Base install is seven dependencies: polars, typer, rich, charset-normalizer,
psutil, httpx and jinja2. No torch, no pandas, no agent framework.

You also need [Ollama](https://ollama.com) with a model pulled, for everything
except `ismith look`.

```bash
ismith init      # writes the config, checks Ollama, offers to pull a model that fits
```

## Usage

| Command | What it does |
|---|---|
| `ismith look` | Detect the format, profile the data. No LLM. |
| `ismith ask` | Answer one question by writing and running code. |
| `ismith forge` | Full autonomous pass, written up as a report. |
| `ismith doctor` | Probe the machine, recommend models that fit. |
| `ismith models` | Show which model each role resolves to. |
| `ismith init` | Set up config and a local model. |

### Profile a file

```bash
ismith look data/sales.csv
ismith look data/sales.csv --json     # machine-readable
ismith look data/sales.csv --card     # exactly what a model would be shown
```

Prints the detected format with a confidence score and what was assumed, then a
column table (dtype, semantic type, nulls, cardinality, distribution) and any
quality notes. Works with no model installed.

### Ask a question

```bash
ismith ask data/sales.csv "which region grew fastest last quarter?"
ismith ask data/sales.csv "correlation between discount and churn" --chart
ismith ask data/sales.csv "average order value by segment" --engine pandas
```

The code is generated, checked by an AST gate, run in a sandboxed subprocess, and
printed alongside the answer. If it crashes, the traceback is fed back and it
tries again, up to three times.

<img src="https://raw.githubusercontent.com/SofianeOuaari/insightsmith/main/assets/chart-example.png" alt="A bar chart produced by ismith ask --chart" width="620">

Useful flags: `--approve` (see the code before it runs), `--no-code`, `--critique/--no-critique`,
`--engine polars|pandas|fireducks`, `--chart`, `--dark`, `--json`.

### See what is worth analysing

```bash
ismith look data/sales.csv --ideas
```

<img src="https://raw.githubusercontent.com/SofianeOuaari/insightsmith/main/assets/ideas-example.png" alt="Eight ranked analysis ideas, each naming the columns it needs" width="720">

Ideas come back ranked, each naming the columns it needs. **Any idea referencing a
column the card does not contain is discarded before you see it**, which removes
most hallucination for the price of a set-membership test.

### Forge a report

```bash
ismith forge data/sales.csv -o out/                        # propose analyses, answer them, write it up
ismith forge data/sales.csv "which region grew?" -o out/   # or ask your own
ismith forge data/sales.csv -o out/ -n 15 --pdf
```

Four files land in the output directory:

| File | What it is |
|---|---|
| `report.html` | The report, with figures inline, a contents rail, light and dark |
| `report.md` | The same content as Markdown, for a repo or a wiki |
| `report.ipynb` | The run rebuilt as a notebook that executes |
| `report.pdf` | With `--pdf`, and the `pdf` extra installed |

Every finding carries the code that produced it, the caveats the critic raised, a
confidence index and how many attempts it took. A question that fails is listed
under "Not answered" rather than dropped.

<img src="https://raw.githubusercontent.com/SofianeOuaari/insightsmith/main/assets/report_generation_example_2.png" alt="One finding: the narrative, the result table and the chart behind it" width="680">

The notebook is not a transcript. Its first cell rebuilds the dataframe from the
original file so every later cell runs, and it says so when the run was on a
sample and re-running will not reproduce the numbers.

### Pick a model for your machine

```bash
ismith doctor        # GPU, RAM, and the models that fit
ismith models        # which model answers as coder, critic, narrator
```

## Configuration

`~/.insightsmith/config.toml`, written on first run and commented.

```toml
local_only = true          # a remote model becomes a hard error
engine = "polars"          # polars | pandas | fireducks

[roles]
ideation = "ollama/qwen3:8b"
coder    = "ollama/qwen3:8b"
critic   = "ollama/qwen3:8b"
narrator = "ollama/qwen3:8b"
```

## As a library

```python
from insightsmith import sniff, load, profile

spec = sniff("data/sales.csv")
print(spec.format, spec.encoding, spec.confidence, spec.warnings)

frame = load(spec)  # a Polars LazyFrame; nothing read yet
result = profile(spec)
print(result.summary())
```

## How it works

```text
sniff -> load -> profile -> card -+- ideation -> ranked ideas
                                  |
                                  +- question -> coder -> sandbox -> critic
                                                   ^        |
                                                   +- retry -+   (max 3, traceback fed back)
                                                            |
                                                            +-> viz -> narrator -> report
```

Polars `LazyFrame` is the internal representation, so Parquet, Arrow, NDJSON and
UTF-8 CSV are scanned rather than loaded. `--engine` picks the API the *generated
snippet* is written against; Parquet is the handover, so a result is a Polars
frame whichever engine ran.

The coder is given worked examples from a bundled recipe book matched to the
shape of your question. Measured on qwen3:8b across 15 questions, that took
answered-on-first-attempt from 13/15 to 15/15 and halved the median time.

**Formats loadable today:** csv, tsv, xlsx/xlsm (`[excel]`), xls, parquet,
feather/arrow, json, jsonl/ndjson. Formats that are detected but not yet loadable
say so rather than failing obscurely.

## Limits

Worth stating plainly, in advance.

- **LLMs write wrong code confidently.** `ismith ask` prints the code for exactly
  that reason. The retry loop fixes code that *crashes*; it cannot tell that a
  snippet ran cleanly and answered the wrong question.
- **The critic reduces statistical nonsense; it does not remove it.** It checks a
  fixed list of computable things, so it cannot catch a confounder, survivorship
  bias, or a question that was the wrong question. A clean verdict means nothing
  on the list fired, not that the analysis is sound. Confidence is an index
  computed from the caveats, not a probability that the answer is right.
- **The sandbox is defence in depth, not a security boundary.** See
  [SECURITY.md](SECURITY.md). Resource limits are POSIX-only; on Windows the AST
  gate and the timeout are all there is.
- **Large files are profiled on a sample.** Above a size threshold every affected
  statistic is marked `estimated`. Row counts stay exact, distributions are
  approximate, and an answer computed on such a file carries a caveat.
- **PII masking is best-effort.** It catches values that look like contact details
  or identifiers. It cannot recognise a name in free text or an identifier in a
  format it has not seen. If data must not leave the machine, set `local_only`.
- **`--chart` often has nothing to draw.** Roughly two answers in five are a
  single number, and the chart is skipped with the reason given rather than a
  meaningless figure produced.
- **Dates are inferred, and `04/01/10` is genuinely ambiguous.** Where day-first
  and month-first both fit, the chosen format is reported as a quality note
  rather than picked silently.
- **Throughput figures are estimates, not benchmarks.** `ismith doctor` derives
  tok/s from published peak memory bandwidth, which no real decode loop reaches.
  Devices missing from that table report `unknown`.
- **FireDucks does not run everywhere.** Linux x86_64 and macOS arm64 only, which
  is why Polars is the default.

## Contributing

```bash
uv sync --all-extras
uv run pytest
uv run ruff check --fix && uv run ruff format
uv run mypy src
```

Conventional Commits. One feature, one branch, one squashed PR.

## License

Apache-2.0. See [LICENSE](LICENSE).
