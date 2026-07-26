# Claude Code kickoff pack for `insightsmith`

Don't paste one giant prompt asking for the whole roadmap. Agentic coding degrades badly past a few hundred lines of unreviewed output, and 0.1.0→3.0 in one shot gets you a plausible-looking skeleton with a broken sniffer and no tests. Three-part approach:

1. Save the design doc into the repo, so Claude Code reads it instead of you re-describing it
2. Commit a `CLAUDE.md` so every session starts with the same rules
3. Run one prompt per milestone

---

## Step 0 — before you open Claude Code

```bash
mkdir insightsmith && cd insightsmith
git init -b main
mkdir -p docs
cp ~/Downloads/insightsmith-design.md docs/DESIGN.md
git add -A && git commit -m "docs: add design specification"
claude
```

---

## Step 1 — `CLAUDE.md` (paste this prompt first)

> Read `docs/DESIGN.md` in full. Then write a `CLAUDE.md` at the repo root that captures the non-negotiable engineering rules for this project so future sessions don't have to re-read the whole spec. It must include, concisely:
>
> - What the project is, in three sentences
> - `src/` layout, Python ≥3.10, hatchling, distribution name `insightsmith`, console scripts `insightsmith` and `ismith`
> - Polars LazyFrame is the internal dataframe; pandas only via explicit `.to_pandas()` and only under the `[pandas]` extra
> - Base install stays light: no torch, no pandas, no langchain in `[project.dependencies]`
> - **Never** `os.system`, never `shell=True`, never bare `exec()`/`eval()`. External binaries only via `subprocess.run(list, capture_output=True, text=True, timeout=N, check=False)` with a missing-binary soft failure
> - Never send raw data rows to an LLM — everything goes through the dataset card in `profiling/card.py`
> - Type hints on all public functions, `py.typed` shipped, mypy clean
> - Every module gets tests in the same PR; network is always mocked in tests
> - Conventional Commits, one feature per branch, squash merge, tags `vX.Y.Z`
> - The commands to run: `uv run pytest`, `uv run ruff check --fix`, `uv run ruff format`, `uv run mypy src`
>
> Keep it under 100 lines. It is context, not documentation. Then commit it as `docs: add CLAUDE.md engineering rules`.

Review what it writes. This file is the highest-leverage thing in the repo — every later session inherits it, so an error here compounds.

---

## Step 2 — bootstrap prompt (v0.1.0)

> Read `docs/DESIGN.md` and `CLAUDE.md`.
>
> Implement **v0.1.0 only**: the no-LLM foundation. Scope is exactly sections 2 and 3 of the design doc, nothing else. Do not create the `llm/`, `agents/`, `execution/`, or `report/` packages yet — not even stubs.
>
> Deliver:
>
> 1. `pyproject.toml` — hatchling, PEP 621, `src/` layout, the dependency and extras table from section 10 but with only the deps 0.1.0 actually needs, both console scripts, `py.typed`
> 2. `.pre-commit-config.yaml` (ruff, ruff-format, mypy), `.gitignore`, `LICENSE` (Apache-2.0), a real `README.md` that includes the limitations section from design doc §12
> 3. `src/insightsmith/io/sniff.py` — the three-stage cascade from §3 returning a `SourceSpec` dataclass with a `confidence` float and `warnings` list. Implement the magic-byte table exactly as specified, including the zip-container disambiguation and the JSON-vs-JSONL line-2 check.
> 4. `src/insightsmith/io/loaders.py` — csv/tsv, xlsx/xls, parquet, feather, json/jsonl. Return `pl.LazyFrame` where the format supports lazy scanning, eager otherwise, behind one `load(spec) -> LazyFrame` interface.
> 5. `src/insightsmith/profiling/` — `schema.py`, `stats.py`, `quality.py`. Numeric, categorical, temporal and text column profiles; quality checks for null rate, exact and near duplicates, constant and near-constant columns, high-cardinality categoricals, outliers via IQR and modified z-score, class imbalance. Files above a configurable size threshold profile on a reservoir sample and every resulting statistic carries an `estimated: bool`.
> 6. `src/insightsmith/cli.py` — typer + rich. Just `ismith look PATH` for now, with a readable table output and `--json`.
> 7. `tests/` — pytest. Include a `hypothesis` property test for the sniffer: generate a small table, round-trip it through every supported format with randomised delimiters, encodings (utf-8, utf-8-sig, cp1252, latin-1), decimal separators and quoting, and assert the sniffer recovers the format and dialect. Generate the fixture files in a `conftest.py` fixture rather than committing binaries where you can.
> 8. `.github/workflows/ci.yml` — ruff, mypy, pytest on 3.10–3.13 × ubuntu/macos/windows
>
> Work in this order and stop for my review after each: (a) pyproject + tooling + CI, (b) sniff.py + its tests, (c) loaders.py + tests, (d) profiling + tests, (e) cli.py + README.
>
> Ask me before adding any dependency not already listed in the design doc. If part of the spec is ambiguous, ask rather than guessing.

The staged "stop for review" is the important line. Without it you get 3,000 lines to review at once and you won't.

---

## Step 3 — per-milestone prompt template

Reuse this for 0.2.0 onward, one milestone per session, fresh context each time:

> Read `CLAUDE.md` and section {N} of `docs/DESIGN.md`.
>
> Current version is {previous}. Implement **{target} only** — {one-line theme}. Do not touch anything outside the packages that milestone names, and do not add stubs for later milestones.
>
> Before writing code: give me a numbered implementation plan with the files you'll create or modify and the tests you'll add. Wait for my approval.
>
> Then implement it in reviewable steps, running `uv run pytest && uv run ruff check && uv run mypy src` after each. Conventional Commits, one commit per logical unit, on branch `feat/{slug}`.
>
> When it's green: bump the version, regenerate `CHANGELOG.md` with git-cliff, and give me the `gh pr create` command — don't merge it yourself.

Filled in for the next two:

**0.2.0** — theme: *hardware probing and model recommendation*. Add: "Implement the fit math from §4 exactly as written, including the KV-cache formula using `n_kv_heads`. Bandwidth figures live in a data table, not inline constants. Hardware parsers must take raw text as input so they're testable without hardware — commit captured `nvidia-smi`, `rocm-smi`, `system_profiler` and `/proc/cpuinfo` output as fixtures, and add golden tests asserting known model + known context → expected GB within 5%."

**0.3.0** — theme: *provider layer*. Add: "One `OpenAICompatProvider` for all OpenAI-wire-format backends driven by a base-URL table; `OllamaProvider` native. Implement the capability-based fallback: when a model lacks tool-calling, degrade to prompted JSON with a bounded retry-on-parse-failure loop. All provider tests use recorded cassettes — no live calls in CI."

---

## Two things to say out loud in the sandbox milestone (0.5.0)

Claude Code will be building an LLM code executor, which is exactly the kind of thing safety training makes it cautious about. Frame it accurately and it'll help properly:

> This is a defense-in-depth sandbox for running LLM-generated analysis code on the user's own machine, with their own data, at their explicit request. Implement all six layers from §7 and be adversarial about the AST gate — assume the model being sandboxed may produce dangerous code by accident. Then write `SECURITY.md` documenting honestly what this does and does not protect against, including that it is not a security boundary against a deliberately malicious prompt, and that `RLIMIT` caps are POSIX-only.

---

## What to keep an eye on in review

- **Invented dependencies.** It will reach for `pandas-profiling`/`ydata-profiling`, `python-magic`, `langchain` unprompted. Each one you accept makes the base install heavier and the profiler less yours.
- **Over-stubbed future work.** Empty `agents/` modules with `pass` bodies in 0.1.0 rot immediately. The prompts above forbid it; enforce it.
- **The sniffer.** This is the one place where "looks right" and "is right" diverge most. Actually run it against a real semicolon-delimited, decimal-comma, cp1252 German CSV before you believe the tests.
- **Fabricated benchmark numbers.** If it writes bandwidth or tok/s figures into the README, check them. Don't ship numbers you haven't measured on at least one machine.
