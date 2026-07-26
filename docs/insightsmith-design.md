# insightsmith — agentic data consultant

> Forging insight from raw data.
> `pip install insightsmith` · `ismith look sales.parquet`

PyPI name checked: **`insightsmith` is free**. Other free names if you change your mind: `augury`, `datawright`, `scryer`, `oracular`, `augurio`, `tabularis`.
Taken (don't bother): `augur`, `datasmith`, `sibyl`, `pythia`, `kestrel`, `dossier`, `sleuth`, `ferret`, `haruspex`, `vizier`, `numina`.

Distribution name `insightsmith`, import name `insightsmith`, and **ship two console scripts**: `insightsmith` and the short alias `ismith`. Eleven characters is too many to type forty times a day, and if you don't provide the alias users will invent their own.

---

## 1. Product shape

Three entry points, same core:

```bash
ismith doctor                        # hardware probe + model recommendations
ismith look data/sales.xlsx          # sniff, profile, print analysis ideas
ismith ask data/sales.parquet "correlation between discount and churn, plot it"
ismith forge data/*.csv -o out/      # full autonomous pass → HTML/PDF report
ismith chat data/sales.csv           # interactive REPL session
```

`forge` instead of `report` is the one place the smith metaphor earns its keep — it's the command that hammers everything into a finished deliverable. Keep `report` as a hidden alias so nobody has to guess.

```python
from insightsmith import Consultant

c = Consultant("data/sales.parquet", model="qwen3:14b")
print(c.profile.summary())
print(c.ideas())                     # ranked analysis recommendations
r = c.ask("what drives revenue variance by region?")
r.figures[0].save("fig.png"); print(r.code, r.narrative)
```

---

## 2. Repository layout

```
insightsmith/
├── pyproject.toml            # hatchling, PEP 621, extras
├── README.md  CHANGELOG.md  LICENSE (Apache-2.0)
├── .pre-commit-config.yaml   # ruff, ruff-format, mypy
├── .github/workflows/        # ci.yml, release.yml, docs.yml
├── docs/                     # mkdocs-material
├── tests/
│   ├── fixtures/             # tiny csv/xlsx/parquet/xml/jsonl samples
│   ├── cassettes/            # recorded LLM responses (vcr.py)
│   └── ...
└── src/insightsmith/
    ├── __init__.py           # public API only: Consultant, Profile, Result
    ├── cli.py                # typer + rich
    ├── config.py             # pydantic-settings ← env, ~/.insightsmith/config.toml
    ├── errors.py
    ├── hardware/
    │   ├── probe.py          # cpu, ram, os, arch  (psutil, platform)
    │   ├── accel.py          # nvidia-smi / rocm-smi / system_profiler
    │   ├── catalog.yaml      # shipped model catalog
    │   └── recommend.py      # fit scoring + tok/s estimate
    ├── io/
    │   ├── sniff.py          # extension → magic bytes → content probe
    │   ├── loaders.py        # per-format readers, lazy where possible
    │   └── registry.py       # entry-point-based format plugins
    ├── profiling/
    │   ├── schema.py         # dtypes, semantic types, candidate keys
    │   ├── stats.py          # numeric/categorical/temporal/text
    │   ├── quality.py        # nulls, dupes, outliers, leakage, imbalance
    │   └── card.py           # → compact JSON "dataset card" for the LLM
    ├── llm/
    │   ├── base.py           # Provider protocol + Capabilities
    │   ├── openai_compat.py  # one class, many backends
    │   ├── ollama.py         # native: /api/ps, /api/pull, /api/show
    │   ├── gemini.py
    │   ├── registry.py
    │   └── router.py         # role → model (planner/coder/vision/cheap)
    ├── execution/
    │   ├── sandbox.py        # isolated subprocess runner
    │   ├── tools.py          # tool/function schemas
    │   └── artifacts.py      # figures, tables, files
    ├── agents/
    │   ├── base.py
    │   ├── ideation.py       # "you should look at X, then Y"
    │   ├── analyst.py        # question → plan
    │   ├── coder.py          # plan → pandas/polars code
    │   ├── critic.py         # validate code + result sanity
    │   ├── viz.py
    │   └── graph.py          # orchestration state machine
    ├── viz/
    │   ├── theme.py
    │   └── render.py         # matplotlib static, plotly interactive
    ├── report/
    │   └── builder.py        # jinja2 → md/html → pdf
    └── memory/
        └── store.py          # sqlite: sessions, findings, code, cost
```

`src/` layout is not optional — it prevents the classic "tests pass locally because they import the source tree, fail after install" bug.

---

## 3. Format detection (`io/sniff.py`)

Don't trust the extension. Three-stage cascade, each stage can veto:

1. **Extension hint** — `.csv .tsv .txt .xlsx .xls .xlsm .parquet .feather .arrow .orc .json .jsonl .ndjson .xml .html .sqlite .db .duckdb .sav .dta .sas7bdat .h5 .zip .gz .zst`
2. **Magic bytes** — read first 8 KB:
   - `PAR1` → Parquet · `ARROW1` → Arrow IPC · `PK\x03\x04` → zip container (xlsx/ods/or actual zip — inspect `[Content_Types].xml`)
   - `\xd0\xcf\x11\xe0` → legacy OLE2 (xls/doc) · `SQLite format 3\x00` → SQLite
   - `<?xml` or `<` → XML/HTML · `{` / `[` → JSON vs JSONL (does line 2 also parse?)
3. **Text dialect probe** — `csv.Sniffer` on a sample plus your own tie-break: count candidate delimiters per line, pick the one with the lowest variance across the first 50 lines. Detect encoding with `charset-normalizer`, decimal comma vs point, thousands separator, header presence, quoting, and comment prefixes.

Return a `SourceSpec(format, encoding, dialect, compression, confidence, warnings)` rather than a bare string. `confidence < 0.8` → tell the user what you assumed. For XML, run a streaming pass with `lxml.iterparse` to find the repeating element that becomes the row unit — that's the part every tool gets wrong.

Loaders return **Polars LazyFrame** by default with a pandas escape hatch (`.to_pandas()`). Polars gives you lazy scanning of Parquet/CSV, so profiling a 40 GB file doesn't OOM. Above a size threshold, profile on a reservoir sample and label every statistic as estimated.

---

## 4. Hardware probing and model recommendation

**Never `os.system`.** It gives no stdout capture, no timeout, and shell-injects. Always:

```python
subprocess.run(cmd_list, capture_output=True, text=True, timeout=5, check=False)
```

with `cmd` as a list, `shell=False`, wrapped so a missing binary is a soft failure. And prefer a library over a shell-out where one exists:

| Signal | Source |
|---|---|
| CPU model, cores, freq | `platform`, `psutil.cpu_count(logical=False)`, `/proc/cpuinfo`, `sysctl -n machdep.cpu.brand_string` |
| RAM total / available | `psutil.virtual_memory()` |
| NVIDIA GPU | `nvidia-smi --query-gpu=name,memory.total,memory.used,compute_cap --format=csv,noheader,nounits` |
| AMD GPU | `rocm-smi --showmeminfo vram --json` |
| Apple Silicon | `system_profiler SPHardwareDataType -json` + `sysctl hw.memsize`; unified memory, usable ≈ 0.70 × total |
| Intel/Arc, iGPU | `lspci`, `clinfo` (best-effort, low confidence) |
| Disk free | `shutil.disk_usage` (model weights need room) |
| Ollama present | `shutil.which("ollama")`, then `GET /api/tags` |

### Fit math — the actual differentiator

Weights, for a GGUF quant:

```
bytes_per_param ≈ {fp16: 2.00, Q8_0: 1.06, Q6_K: 0.82, Q5_K_M: 0.70, Q4_K_M: 0.60, Q3_K_M: 0.48}
W_gb = params_B × bytes_per_param
```

KV cache, which everyone forgets and which dominates at long context:

```
KV_gb = 2 × n_layers × n_kv_heads × head_dim × ctx_len × kv_bytes / 1e9
```

`n_kv_heads` (not `n_heads`) is the point — GQA models like Qwen3 and Llama 3 cut this 4–8×. Pull these from `/api/show` → `model_info` so you're not guessing.

```
need_gb = (W_gb + KV_gb) × 1.10          # runtime + activation overhead
```

Decision rule:

- `need_gb ≤ 0.85 × VRAM` → full GPU offload, recommend it
- `need_gb ≤ 0.85 × unified_mem` (Apple) → recommend, note thermals
- else → compute partial offload `n_gpu_layers = floor(VRAM_free × 0.85 / (W_gb / n_layers))` and warn
- `need_gb > RAM` → exclude entirely

Throughput estimate — decoding is memory-bandwidth-bound, so:

```
tok/s ≈ (mem_bandwidth_GB/s × efficiency) / W_gb     # efficiency ≈ 0.6–0.8
```

Ship a bandwidth lookup: DDR4-3200 dual channel ≈ 51 GB/s, DDR5-5600 ≈ 90, M-series 100–800, RTX 4090 ≈ 1008, A100 ≈ 1555, H100 ≈ 3350. Now `ismith doctor` can say **"qwen3:14b Q4_K_M at 8k ctx: 9.8 GB, fits your 16 GB, expect ~28 tok/s"** — a real number, not a vibe. That single feature will get you stars.

### Catalog

`catalog.yaml` ships with the package, refreshable via `ismith doctor --refresh`. Each entry: ollama tag, params, layers, kv heads, head dim, context, license, roles (`coder`/`reasoner`/`vision`/`embed`), quality tier. Recommend **per role**, not one winner: a small fast model for routing and summarizing, a coder model for the code agent, a bigger reasoner for planning.

Rough tiers to seed it: ≤8 GB → `qwen3:4b`, `llama3.2:3b`, `gemma3:4b`; 8–16 GB → `qwen3:8b`, `qwen2.5-coder:7b`, `mistral-nemo`; 16–24 GB → `qwen3:14b`, `qwen2.5-coder:14b`; 24–48 GB → `qwen3:32b`, `qwen2.5-coder:32b`, `devstral`; 48 GB+ → `llama3.3:70b-q4`, `gpt-oss:120b`.

---

## 5. Provider layer — don't write six SDKs

Almost everything speaks the OpenAI Chat Completions wire format. So:

- **One** `OpenAICompatProvider(base_url, api_key, headers)` covers OpenAI, OpenRouter, DeepInfra, Together, Groq, Fireworks, Mistral, local vLLM/llama.cpp/LM Studio, and Gemini's compat endpoint. Config is just a table of base URLs.
- **`OllamaProvider`** written natively, because local needs things the OpenAI shape can't express: `/api/tags`, `/api/pull` with progress, `/api/show` for the metadata your fit math needs, `/api/ps` for what's resident, `keep_alive`, and `num_gpu`/`num_ctx` options.
- **`GeminiProvider`** native only if you want thinking budgets and long-context caching; otherwise use its compat endpoint.
- Offer `pip install insightsmith[litellm]` as an escape valve for the long tail. Don't make it a hard dependency.

```python
class Provider(Protocol):
    name: str
    def capabilities(self, model: str) -> Capabilities: ...
    def chat(self, messages, *, tools=None, **kw) -> Completion: ...
    def stream(self, messages, **kw) -> Iterator[Chunk]: ...
```

`Capabilities`: `context_window`, `tool_calling`, `json_mode`, `vision`, `cost_in/out_per_mtok`, `max_output`. The router reads capabilities, so an agent that needs tool-calling never gets handed a model that lacks it — instead it degrades to a prompted-JSON path with a retry-on-parse-failure loop. That degradation path is essential for small local models.

Model strings are `provider/model`: `ollama/qwen3:14b`, `openrouter/anthropic/claude-sonnet-4.5`, `deepinfra/meta-llama/Llama-3.3-70B-Instruct`.

Config:

```toml
# ~/.insightsmith/config.toml
[roles]
planner = "ollama/qwen3:14b"
coder   = "ollama/qwen2.5-coder:14b"
cheap   = "ollama/qwen3:4b"
vision  = "gemini/gemini-2.5-flash"

[budget]
max_usd_per_session = 0.50
local_only = false
```

---

## 6. The grounding trick

Never paste the dataframe into the prompt. Build a **dataset card**: a 2–5 KB JSON object with schema, per-column stats, semantic type guesses, quality flags, correlation shortlist, and *k* stratified example rows with obvious PII masked. Every agent sees the card, not the data. Consequences:

- Token cost stays flat regardless of file size, so 4B local models actually work
- No raw records leave the machine; `local_only = true` hard-fails if any remote provider is configured — worth advertising loudly, since "don't upload my data to an API" is the reason most people won't touch existing tools
- Reproducible: same card → same plan, cacheable by card hash

Ideation prompt is then constrained: return ranked `{question, rationale, method, columns, expected_artifact, effort}` objects, and reject any that reference a column not in the card. Cheap validation, kills most hallucination.

---

## 7. Code execution sandbox

The coder agent writes Python; you have to run it. Layers, cheapest first:

1. **Static gate** — parse with `ast`, walk it, reject `import os/sys/socket/subprocess/shutil/requests`, `open()` outside the allowlist, `eval`, `exec`, `__import__`, dunder attribute access. Allowlist `polars, pandas, numpy, scipy, sklearn, statsmodels, matplotlib, plotly, seaborn`.
2. **Process isolation** — `subprocess` with `sys.executable -I -S`, `cwd` a fresh tempdir, `env` scrubbed, `timeout=60`, stdin closed. Never `exec()` in-process; one `while True` and your CLI is gone.
3. **Resource caps** — a `preexec_fn` calling `resource.setrlimit` for `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_NPROC`, `RLIMIT_FSIZE` (POSIX only; document the Windows gap).
4. **Contract** — the snippet gets `df` preloaded and must assign `result` and/or `fig`. Results come back over a pickle/Arrow file in the tempdir, not stdout parsing.
5. **Optional hardening** — `ismith --sandbox=docker` mounting the data read-only with `--network=none`. Ship it as an extra, off by default.
6. **Human in the loop** — `--approve` mode prints the code and waits. Default on for `forge` runs that write files.

Failures are fuel: feed traceback + the offending code back to the coder agent, max 3 retries, then surface honestly.

---

## 8. Agent graph

```
sniff → load → profile → card
                          ├─ ideation ──→ ranked ideas
                          └─ on question:
                             planner → coder → sandbox → critic
                                          ↑         │
                                          └──retry───┘
                                                    ↓
                                          viz → narrator → artifact
```

The **critic** is what separates this from a toy. It checks: did the code answer the question that was asked; is the sample size adequate; is a Pearson correlation being reported on non-linear or heavily-outliered data; is a "significant" p-value the product of 40 untracked comparisons; is a group with n=3 being described as a trend. Have it emit `{verdict, caveats[], confidence}` and print the caveats in the report. Statistical honesty as a headline feature is genuinely uncommon in this space.

Use LangGraph if you want the checkpointing and streaming for free — you already know it. But a hand-rolled state machine over a `SessionState` dataclass is ~200 lines, has no dependency churn, and you keep control. My recommendation: hand-rolled core, LangGraph as an optional adapter.

---

## 9. Version roadmap

Pre-1.0 is where you're allowed to break things. Use it.

| Version | Theme | Ships |
|---|---|---|
| **0.0.1** | name squat | empty package, README only. Do this today. |
| **0.1.0** | foundation | `sniff`, loaders (csv/xlsx/parquet/json), `profiling`, `ismith look`, no LLM at all. Genuinely useful standalone. |
| **0.2.0** | hardware | `probe`, `accel`, catalog, fit math, tok/s estimate, `ismith doctor`. |
| **0.3.0** | providers | `Provider` protocol, Ollama native, OpenAI-compat multi-backend, router, config file, `ismith models`. |
| **0.4.0** | ideation | dataset card, PII masking, `ismith look --ideas`, ranked recommendations with column validation. |
| **0.5.0** | execution | AST gate, subprocess sandbox, rlimits, coder agent, `ismith ask` returning numbers. |
| **0.6.0** | visuals | viz agent, theme, static + interactive, artifact store. |
| **0.7.0** | critique | critic agent, caveats, retry loop, confidence scoring. |
| **0.8.0** | reporting | jinja2 → HTML/PDF, `ismith forge`, notebook export. |
| **0.9.0** | polish | memory/sqlite sessions, cost tracking, budget caps, `ismith chat` REPL, xml/sqlite/stata/sas loaders, mkdocs site, 80% coverage. |
| **1.0.0** | stability | public API frozen, semver commitment, deprecation policy, `py.typed`, tested 3.10–3.13 × linux/mac/win. |
| 1.1.0 | more formats: XML streaming, HDF5, Avro, ORC, glob/multi-file, `--sample` strategies |
| 1.2.0 | SQL: duckdb, sqlite, postgres, `ismith ask "postgres://..." "..."` — agent writes SQL not pandas |
| 1.3.0 | statistics agent: assumption checks, effect sizes, multiple-comparison correction, power |
| 1.4.0 | caching (card hash → plan → result), prompt cache, resumable sessions |
| 1.5.0 | TUI (textual), streaming tokens, live plot preview |
| 1.6.0 | embeddings + semantic search over columns/docs; text-column analysis |
| **2.0.0** | breaking: async core, plugin architecture via entry points (`insightsmith.loaders`, `insightsmith.agents`, `insightsmith.providers`), multi-dataset joins with FK inference, `ismith serve` FastAPI + OpenAPI, session sharing |
| 2.1–2.5 | dbt/warehouse connectors, scheduled monitoring, data-drift detection, Slack/Teams bot, RBAC |
| **3.0.0** | modelling agent: feature engineering, baseline AutoML with leakage detection, model cards, and an eval harness that benchmarks *itself* — pinned question/answer pairs on public datasets, scored per model, so users pick a local model on evidence |

Two things to decide early because they're painful later: `src/` layout (do it) and whether the public API is sync or async (make the core sync with an async transport underneath; go async-first at 2.0).

---

## 10. GitHub and git strategy

### Setup

```bash
mkdir insightsmith && cd insightsmith && git init -b main
uv init --lib --package insightsmith      # or hatch new
gh repo create insightsmith --public --source=. \
   --description "Agentic data consultant for local and cloud LLMs"
```

Branch protection on `main`: require PR, require CI green, require linear history, no force-push. Yes, even solo — it stops the 1 a.m. direct-push-to-main that breaks the release.

`pyproject.toml` essentials:

```toml
[project]
name = "insightsmith"
requires-python = ">=3.10"
dependencies = ["polars", "pyarrow", "typer", "rich", "pydantic", "pydantic-settings", "psutil", "httpx", "charset-normalizer", "jinja2"]

[project.optional-dependencies]
pandas   = ["pandas", "openpyxl", "xlrd"]
viz      = ["matplotlib", "plotly", "kaleido"]
stats    = ["scipy", "statsmodels", "scikit-learn"]
gemini   = ["google-genai"]
litellm  = ["litellm"]
sql      = ["duckdb", "sqlalchemy", "psycopg[binary]"]
all      = ["insightsmith[pandas,viz,stats,gemini,sql]"]

[project.scripts]
insightsmith = "insightsmith.cli:app"
ismith       = "insightsmith.cli:app"

[project.entry-points."insightsmith.loaders"]
# third parties register formats here
```

Keep the base install light — no torch, no pandas by default. `httpx` is all you need to talk to every provider.

### Commit and history discipline

Conventional Commits, enforced by a `commit-msg` pre-commit hook:

```
feat(hardware): estimate decode throughput from memory bandwidth
fix(sniff): treat BOM-prefixed CSV as utf-8-sig
perf(profiling): reservoir-sample files over 2 GB
refactor(llm)!: rename Provider.complete to Provider.chat

BREAKING CHANGE: ...
```

The commit *type* drives the changelog and the version bump. Use `git-cliff` to generate `CHANGELOG.md` from history — write commits well once and never hand-edit a changelog again.

### Making the history tell the story

One roadmap row = one GitHub milestone. One feature = one issue = one short-lived branch = one squash-merged PR closing that issue.

```bash
git switch -c feat/hardware-probe
# ... commit freely, messy is fine, it gets squashed
gh pr create --fill --milestone 0.2.0
gh pr merge --squash --delete-branch
```

Result: `main` is linear, each commit is one shippable feature referencing its PR and issue, and `git log --oneline` between tags reads as a release note. Then:

```bash
git switch main && git pull
# bump version, regenerate changelog
git commit -am "chore(release): v0.2.0"
git tag -a v0.2.0 -m "v0.2.0 — hardware probing and model recommendation"
git push --follow-tags
```

Tag push triggers release. Never tag a commit that isn't on `main`.

**One caution about backfilling:** don't rewrite dates or fabricate a history you didn't live. Recruiters and collaborators read `git log --stat`, and a repo where 40 "incremental" commits share one timestamp reads worse than an honest three-week history. If you're starting from an existing prototype, land it as an honest `feat: initial prototype` and let the real work accumulate from there.

### CI

`ci.yml` — on push and PR: ruff check, ruff format --check, mypy, pytest across 3.10/3.11/3.12/3.13 × ubuntu/macos/windows, coverage to Codecov. Mock all network; record real provider responses once with `vcr.py` cassettes.

`release.yml` — on `v*` tag: build sdist+wheel, publish to PyPI via **Trusted Publishing** (OIDC, no API token in secrets — this is the modern way and takes five minutes to configure), then `gh release create` with `git-cliff` notes attached.

`docs.yml` — mkdocs-material to GitHub Pages on merge to main.

Also add: `dependabot.yml`, issue templates, `CONTRIBUTING.md`, a `SECURITY.md` explaining the sandbox threat model explicitly (you're shipping an LLM code executor — say so plainly and describe the mitigations; it builds trust rather than eroding it).

### Testing the hard parts

- **Sniffer** — property-based with `hypothesis`: generate a table, serialize to every format with random dialects/encodings, assert round-trip detection. This is where your bugs will live.
- **Hardware** — never touch real hardware in CI. Fixture files of captured `nvidia-smi` / `system_profiler` / `/proc/cpuinfo` output across a dozen machines, parsed offline. Ask people to contribute theirs; it's a great low-friction first issue.
- **Fit math** — golden tests: known model + known ctx → expected GB, ±5%.
- **Agents** — cassettes for determinism, plus a small nightly live-eval job on a pinned question set.

---

## 11. Differentiators worth prioritizing

Ordered by how much they'd distinguish this from the existing crowd:

1. **Hardware-aware local model recommendation with real throughput numbers.** Nobody does this well, and it's the first thing every local-LLM user wants to know.
2. **Local-only privacy guarantee** — a hard-failing switch, plus PII masking in the card. This maps directly onto your research area and is a defensible reason for institutional users to adopt it.
3. **Statistical critic** producing explicit caveats. Turns "the LLM printed a number" into something a real analyst can sign off on.
4. **Auditability** — every result carries the code that produced it, the card hash, the model, and the token cost. Export a runnable notebook.
5. **Format sniffing that actually works** on messy European CSVs (semicolons, decimal commas, cp1252) and awkward nested XML.

## 12. What to be honest about in the README

State the failure modes up front: LLMs write wrong code confidently; the sandbox is defense-in-depth, not a security boundary; profiling large files is sampled; the critic reduces but does not remove statistical nonsense. A README that names its limits gets taken more seriously than one that doesn't — and it pre-empts the first angry issue.
