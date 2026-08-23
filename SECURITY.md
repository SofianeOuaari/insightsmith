# Security

insightsmith runs code written by a language model against your data, on your
machine, at your request. That is the product. This document says what the
sandbox around it does and — more importantly — what it does not do.

## Reporting a vulnerability

Open a [security advisory](https://github.com/SofianeOuaari/insightsmith/security/advisories/new)
rather than a public issue. Include the version, your platform, and a snippet
that reproduces the behaviour.

## The threat model, stated plainly

The sandbox is built on the assumption that **a model may produce dangerous code
by accident**: a stray `import os`, a path that walks out of the working
directory, a loop that never ends, an allocation that takes the machine down.
That is the realistic failure, and the layers below address it.

**It is not a security boundary against a deliberately malicious prompt.** If
someone can choose the text sent to your model, treat anything it writes as
hostile, and do not rely on this sandbox to contain it. Python is not a language
in which untrusted code can be contained by inspection; a sufficiently determined
bypass of an AST filter always exists. Use OS-level isolation — a container, a
VM, a separate unprivileged user — if that is your situation.

## What is actually enforced

| Layer | What it does | Where it fails |
|---|---|---|
| **AST gate** | Allowlists imports and refuses `eval`, `exec`, `compile`, `__import__`, `open`, `getattr`, `globals`, and any `_`-prefixed attribute | Static analysis only. It cannot reason about values, and a novel route through an allowed library is not covered |
| **Separate process** | `sys.executable -I`, fresh temp directory, scrubbed environment (no credentials, no proxies, no `PYTHON*`), stdin closed | Same user, same filesystem permissions as you |
| **Wall-clock timeout** | Kills after 60s by default | — |
| **`RLIMIT_CPU`** | Terminates a looping snippet | **POSIX only** |
| **`RLIMIT_AS`** | Caps address space at 4 GB | **Linux only** — see below |
| **`RLIMIT_FSIZE`** | Caps written file size at 256 MB | **POSIX only** |
| **Data staging** | The snippet gets a Parquet copy in a scratch directory, never a path into your tree | It can still reach any file your user can, if it finds a way to open one |
| **`--approve`** | Prints the code and waits for you | Off by default for `ismith ask` |

### The memory cap is Linux-only

`RLIMIT_AS` caps *virtual* address space, not memory actually touched, and Rust
allocators reserve virtual space far in excess of what they use. On Linux a
polars import peaks around 400 MB and a 4 GB cap leaves real work alone. On
macOS the reservation is large enough that the same cap kills the import before
any snippet runs, so it is not applied there. `SandboxResult.memory_capped`
reports whether it was, and a snippet that allocates without bound on macOS will
be stopped by the CPU limit or the timeout rather than by a memory ceiling.

### On Windows, the resource limits do not exist

`resource.setrlimit` is POSIX-only. On Windows the AST gate and the wall-clock
timeout are the only limits that apply: a snippet that allocates without bound
can exhaust memory, and one that writes without bound can fill the disk.
`SandboxResult.limits_enforced` reports which situation you are in.

### The process runs as you

There is no privilege separation. The child has your user's permissions and can
reach anything you can. The gate is what stands between a generated snippet and
your home directory, and the gate is a filter, not a jail.

## Deliberate deviations from the design document

**`-I` without `-S`.** §7 specifies both. `-S` skips `site`, which removes
`site-packages` and makes `import polars` fail, so the snippet could not do its
job. `-I` alone still discards `PYTHONPATH`, other `PYTHON*` variables and user
site-packages, which is the injection route that matters. The cost is that `.pth`
files already installed in the environment still execute.

**Results are JSON or Parquet, never pickle.** §7 suggests returning results in a
pickle file. Unpickling is arbitrary code execution *in the parent process*,
which would hand back everything the sandbox had just taken away. Scalars come
back as JSON, frames as Parquet.

**`RLIMIT_NPROC` is not set by default.** §7 lists it. It is per-UID rather than
per-process — it counts every process your user already has — so any workable cap
is either far beyond anything a snippet could reach or immediately fatal. It also
caps threads, and polars spawns them at import, so a low value kills the process
before the snippet runs. Process spawning is prevented by the gate refusing `os`,
`subprocess` and `multiprocessing` instead. `Limits(processes=…)` sets it if you
want it.

## What you can do

- Run `ismith ask --approve` to see every snippet before it executes.
- Set `local_only = true` in `~/.insightsmith/config.toml` so no data leaves the
  machine.
- Work on a copy of sensitive data, in a directory containing nothing else.
- On untrusted input, run the whole tool inside a container.
