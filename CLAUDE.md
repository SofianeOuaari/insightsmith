# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`insightsmith` is an agentic data consultant: point it at a data file and it detects the real format, profiles
the data, then uses an LLM to propose analyses, write and sandbox-execute the code, critique the statistics, and
forge a report. It is local-first — a local model on the user's own hardware is the default path, not a degraded
fallback, so raw data never has to leave the machine. The differentiators are hardware-aware model
recommendation, a hard privacy guarantee, and a statistical critic that emits caveats.

Full spec: [docs/insightsmith-design.md](docs/insightsmith-design.md) (sections are cited below).
Build plan: [docs/insightsmith-claude-code-prompts.md](docs/insightsmith-claude-code-prompts.md).

## Working rules

- **One milestone at a time** (§9). Implement only the target version's scope; do **not** create packages or
  stub modules for later milestones — empty `agents/` files with `pass` bodies rot immediately.
- **Current state: docs only** — no `pyproject.toml`, no `src/`. 0.0.1 is a name reservation.
- Ask before adding a dependency the design doc doesn't name; don't reach for `ydata-profiling`,
  `python-magic`, or `langchain`. If the spec is ambiguous, ask rather than guessing.

## Commands

```bash
uv run pytest                  # single test: uv run pytest tests/test_sniff.py::test_name
uv run ruff check --fix
uv run ruff format
uv run mypy src
```

## Packaging

- **`src/` layout** — non-negotiable; it prevents "tests pass locally, break after install".
- Python ≥3.10, hatchling, PEP 621. Distribution and import name `insightsmith`.
- Two console scripts, both → `insightsmith.cli:app`: `insightsmith` and the alias `ismith`.
- Ship `py.typed`.
- **Base install stays light.** No torch, no pandas, no langchain in `[project.dependencies]`. `httpx` is
  enough to talk to every provider. Heavy things live in extras: `pandas`, `viz`, `stats`, `gemini`,
  `litellm`, `sql`, `all`.

## Hard rules

- **Polars `LazyFrame` is the internal dataframe.** pandas only via an explicit `.to_pandas()` escape hatch,
  and only under the `[pandas]` extra. Lazy scanning is why profiling a 40 GB file doesn't OOM.
- **Never send raw data rows to an LLM.** Everything an agent sees goes through the dataset card built in
  `profiling/card.py` — a 2–5 KB JSON summary: schema, per-column stats, quality flags, and *k* stratified
  example rows with PII masked. This keeps token cost flat regardless of file size, makes runs cacheable by
  card hash, and is what makes `local_only = true` meaningful (it must hard-fail, not warn).
- **Never `os.system`, never `shell=True`, never bare `exec()`/`eval()`.** External binaries only via
  `subprocess.run(cmd_list, capture_output=True, text=True, timeout=N, check=False)`, wrapped so a missing
  binary is a soft failure. LLM-generated code runs in the `execution/sandbox.py` subprocess, never in-process.
- **Type hints on all public functions; mypy clean.**
- **Every module gets tests in the same PR.** Network is always mocked — recorded cassettes, never live calls.
- Hardware parsers take **raw text** as input so they are testable without hardware. Never touch real hardware
  in CI; use committed fixture captures of `nvidia-smi` / `rocm-smi` / `system_profiler` / `/proc/cpuinfo`.
- Don't ship benchmark numbers (bandwidth, tok/s) that haven't been measured on a real machine.

## Orientation

Two surfaces over one core — the `ismith` CLI (typer + rich) and the `Consultant` API; `__init__.py` exports only `Consultant`, `Profile`, `Result`.

```text
sniff → load → profile → card
                          ├─ ideation ──→ ranked ideas
                          └─ on question:
                             planner → coder → sandbox → critic
                                          ↑         │
                                          └──retry───┘   (max 3, traceback fed back)
                                                    ↓
                                          viz → narrator → artifact
```

- **`io/sniff.py`** (§3) returns a `SourceSpec(format, encoding, dialect, compression, confidence, warnings)`,
  never a bare string. Three-stage cascade — extension, magic bytes, text-dialect probe — each can veto;
  `confidence < 0.8` means saying what was assumed. This is where "looks right" and "is right" diverge most:
  property-test it with hypothesis across formats, delimiters, encodings, and decimal commas.
- **`hardware/recommend.py`** (§4) — the KV-cache formula uses `n_kv_heads`, not `n_heads`; GQA cuts it 4–8×,
  and the real values come from Ollama's `/api/show`. Bandwidth figures live in a data table, not inline
  constants. Recommend **per role** (planner / coder / cheap / vision), never one winner.
- **`llm/`** (§5) — one `OpenAICompatProvider` covers every OpenAI-wire-format backend via a base-URL table;
  `OllamaProvider` is native because local needs `/api/show`, `/api/pull`, `/api/ps`, `keep_alive`, `num_ctx`.
  `router.py` maps role → model by reading `Capabilities`; a model lacking tool-calling must degrade to
  prompted JSON with bounded retry-on-parse-failure. Model strings are `provider/model`.
- **`execution/sandbox.py`** (§7) — six layers: AST import allowlist, isolated `sys.executable -I -S` subprocess
  in a fresh tempdir with scrubbed env, POSIX `setrlimit` caps, a contract where the snippet gets `df` and
  assigns `result`/`fig` returned over a file (never stdout parsing), optional docker, `--approve`. It is
  defense-in-depth, **not** a security boundary — `SECURITY.md` must say so plainly.
- **`agents/critic.py`** (§8) is what separates this from a toy: it emits `{verdict, caveats[], confidence}` and
  the caveats get printed. Orchestration is a hand-rolled state machine over a `SessionState` dataclass;
  LangGraph, if ever, is an optional adapter — never a core dependency.

## Git

Conventional Commits (`feat(hardware): …`, `fix(sniff): …`, `refactor(llm)!: …`) — the type drives the changelog
and the version bump. One feature = one issue = one short-lived branch = one squash-merged PR; `main` stays
linear. `CHANGELOG.md` is git-cliff-generated, never hand-edited. Tags are `vX.Y.Z`, only ever on `main`. Don't
merge PRs or push tags unless asked.
