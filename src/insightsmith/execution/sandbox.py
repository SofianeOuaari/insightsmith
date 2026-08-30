"""Running generated code in a separate, constrained process.

Layers two, three and four of design doc §7. The snippet never runs in this
process: one ``while True`` in-process and the CLI is gone, and one bad
attribute access could reach the caller's own objects.

Two deliberate departures from §7, both load-bearing:

* **``-I`` without ``-S``.** ``-S`` skips ``site``, which removes
  ``site-packages`` and makes ``import polars`` fail — the snippet could not do
  its job. ``-I`` alone still discards ``PYTHONPATH``, ``PYTHON*`` variables and
  user site-packages, which is the injection route that matters.
* **Results come back as JSON or Parquet, never pickle.** §7 suggests a pickle
  file, but unpickling is arbitrary code execution *in the parent*, which would
  hand back everything the sandbox just took away.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import polars as pl

from insightsmith.execution.gate import Verdict, check

__all__ = ["DEFAULT_LIMITS", "Limits", "SandboxResult", "run"]

#: Whether a virtual-address-space cap can be applied without breaking imports.
_MEMORY_CAP_SUPPORTED: Final = sys.platform.startswith("linux")

_RUNNER: Final = """\
import json, sys, pathlib
import polars as pl

_dir = pathlib.Path(sys.argv[1])
df = pl.read_parquet(_dir / "input.parquet")
result = None
fig = None

try:
    _src = (_dir / "snippet.py").read_text(encoding="utf-8")
    exec(compile(_src, "snippet.py", "exec"), globals())
    # A LazyFrame is a query plan, not an answer. Collecting it here rather than
    # rejecting it keeps the guide's lazy-first advice usable, and doing it
    # inside the try means a plan that fails to execute reports the real error.
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
except BaseException:
    import traceback
    (_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    raise SystemExit(1)

out = {}
if isinstance(result, (pl.DataFrame, pl.Series)):
    frame = result.to_frame() if isinstance(result, pl.Series) else result
    frame.write_parquet(_dir / "result.parquet")
    out["kind"] = "frame"
    out["rows"] = frame.height
    out["columns"] = frame.columns
elif result is not None:
    try:
        json.dumps(result)
        out["kind"] = "value"
        out["value"] = result
    except TypeError:
        out["kind"] = "repr"
        out["value"] = repr(result)[:2000]
else:
    out["kind"] = "none"

(_dir / "result.json").write_text(json.dumps(out), encoding="utf-8")
"""


@dataclass(slots=True, frozen=True)
class Limits:
    """Resource ceilings for the child process.

    Enforced through ``resource.setrlimit`` on POSIX. **Windows has no
    equivalent and these are silently unavailable there** — the timeout and the
    AST gate are the only limits that apply. SECURITY.md says so plainly.
    """

    timeout_seconds: float = 60.0
    cpu_seconds: int = 60
    address_space_bytes: int = 4 * 1024**3
    file_size_bytes: int = 256 * 1024**2
    #: RLIMIT_NPROC, or None to leave it alone — the default.
    #:
    #: §7 lists it, but it is per-UID rather than per-process: it counts every
    #: process the user already has, so any workable cap is either far above
    #: anything a snippet could reach or instantly fatal. Worse, it caps threads
    #: too, and polars spawns them at import, so a low value panics before the
    #: snippet runs. Forking is prevented by the gate refusing os, subprocess
    #: and multiprocessing instead.
    processes: int | None = None


DEFAULT_LIMITS: Final = Limits()
#: Signal raised when RLIMIT_CPU is breached; absent on Windows.
_SIGXCPU: Final = getattr(signal, "SIGXCPU", 24)
#: Seconds between the soft and hard CPU limits, so SIGXCPU lands first.
_CPU_GRACE: Final = 5


@dataclass(slots=True)
class SandboxResult:
    """What came back. ``ok`` is false for a refusal, a crash or a timeout."""

    ok: bool
    kind: str = "none"
    value: Any = None
    frame: pl.DataFrame | None = None
    stdout: str = ""
    stderr: str = ""
    traceback: str = ""
    refused: list[str] = field(default_factory=list)
    timed_out: bool = False
    #: True when any rlimit was applied (POSIX only).
    limits_enforced: bool = os.name == "posix"
    #: True when the address-space cap was applied. Linux only — see Limits.
    memory_capped: bool = _MEMORY_CAP_SUPPORTED

    @property
    def summary(self) -> str:
        if self.refused:
            return "refused by the static gate"
        if self.timed_out:
            return "exceeded its time limit"
        if not self.ok:
            return "raised an exception"
        if self.kind == "frame" and self.frame is not None:
            return f"{self.frame.height} rows x {self.frame.width} columns"
        return f"{self.value!r}"


def run(
    source: str,
    frame: pl.DataFrame,
    *,
    limits: Limits = DEFAULT_LIMITS,
    gate: Verdict | None = None,
) -> SandboxResult:
    """Screen ``source``, then run it against ``frame`` in a child process.

    The snippet receives ``df`` and is expected to assign ``result``. It sees a
    Parquet copy of ``frame`` in a scratch directory, never the original file,
    so it cannot reach anything the caller did not hand it.
    """
    verdict = gate if gate is not None else check(source)
    if not verdict.allowed:
        return SandboxResult(ok=False, refused=list(verdict.reasons))

    # ignore_cleanup_errors: on Windows a killed child can still hold a handle to
    # input.parquet when the directory is torn down, and a PermissionError there
    # would mask the real result.
    with tempfile.TemporaryDirectory(
        prefix="insightsmith-", ignore_cleanup_errors=True
    ) as workspace:
        work = Path(workspace)
        (work / "snippet.py").write_text(source, encoding="utf-8")
        (work / "runner.py").write_text(_RUNNER, encoding="utf-8")
        frame.write_parquet(work / "input.parquet")

        completed, timed_out = _spawn(work, limits)
        if timed_out:
            return SandboxResult(ok=False, timed_out=True, stderr="killed after the time limit")

        error = work / "error.txt"
        if completed.returncode != 0:
            # A resource limit kills by signal, leaving no Python traceback. Say
            # which limit was hit rather than implying the code raised.
            killed = _killed_by(completed.returncode)
            return SandboxResult(
                ok=False,
                stdout=completed.stdout,
                stderr=completed.stderr,
                traceback=(
                    error.read_text(encoding="utf-8")
                    if error.is_file()
                    else (killed or completed.stderr or "the snippet exited non-zero")
                ),
                timed_out=completed.returncode == -_SIGXCPU,
            )
        return _collect(work, completed)


def _killed_by(returncode: int) -> str:
    """Describe a signal death, which is what a breached rlimit looks like."""
    if returncode >= 0:
        return ""
    try:
        name = signal.Signals(-returncode).name
    except ValueError:  # pragma: no cover - unknown signal number
        return f"killed by signal {-returncode}"
    explanation = {
        "SIGXCPU": "exceeded the CPU time limit — it is probably looping",
        "SIGXFSZ": "tried to write a file larger than the limit allows",
        "SIGKILL": "was killed outright — a resource limit or the OS out-of-memory killer",
        "SIGSEGV": "crashed",
    }.get(name, "was killed")
    return f"the snippet {explanation} ({name})"


def _spawn(work: Path, limits: Limits) -> tuple[subprocess.CompletedProcess[str], bool]:
    command = [sys.executable, "-I", str(work / "runner.py"), str(work)]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, see module docstring
            command,
            capture_output=True,
            text=True,
            timeout=limits.timeout_seconds,
            check=False,
            cwd=work,
            env=_scrubbed_env(),
            stdin=subprocess.DEVNULL,
            preexec_fn=_apply_limits(limits) if os.name == "posix" else None,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(command, 1, "", ""), True
    return completed, False


def _scrubbed_env() -> dict[str, str]:
    """A minimal environment: no credentials, no proxies, no PYTHON* settings."""
    keep = ["PATH", "LANG", "LC_ALL", "TMPDIR"]
    if os.name == "nt":
        # Windows needs more than POSIX to start a process at all, and its temp
        # directory lives in TEMP/TMP rather than TMPDIR. Stripping these left
        # the child unable to probe its own CPU.
        keep += [
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "PATHEXT",
            "COMSPEC",
            "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE",
        ]
    env = {name: os.environ[name] for name in keep if name in os.environ}

    # Bound the thread pool explicitly. RLIMIT_NPROC cannot do it (see Limits),
    # and an unbounded pool on a many-core box is its own denial of service.
    env["POLARS_MAX_THREADS"] = "4"

    # The parent already imported polars on this machine, so the CPU check has
    # passed. Re-running it here is not a second safety net: with a minimal
    # environment the CPUID probe can fail, and polars reads an empty flag set as
    # "every flag missing" rather than "could not detect", raising
    # `unknown feature flag: 'sse3'` before the snippet runs.
    env["POLARS_SKIP_CPU_CHECK"] = "1"
    return env


def _apply_limits(limits: Limits) -> Callable[[], None]:
    """Build the preexec callable that caps the child's resources."""

    def apply() -> None:  # pragma: no cover - runs in the forked child
        import resource

        # Soft below hard on purpose. With both equal, SIGXCPU and SIGKILL
        # arrive together and the kill looks unexplained; the gap lets SIGXCPU
        # terminate it cleanly so the cause is legible.
        resource.setrlimit(
            resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + _CPU_GRACE)
        )
        # RLIMIT_AS caps *virtual* address space, and Rust allocators reserve it
        # far in excess of what they touch. On Linux a polars import peaks around
        # 400 MB and the cap is safe; on Darwin the reservation is large enough
        # that a virtual cap kills the import before any snippet runs. Applied
        # only where it has been verified to leave legitimate work alone.
        if _MEMORY_CAP_SUPPORTED and limits.address_space_bytes:
            resource.setrlimit(
                resource.RLIMIT_AS, (limits.address_space_bytes, limits.address_space_bytes)
            )
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_size_bytes, limits.file_size_bytes))
        if limits.processes is not None:
            resource.setrlimit(resource.RLIMIT_NPROC, (limits.processes, limits.processes))

    return apply


def _collect(work: Path, completed: subprocess.CompletedProcess[str]) -> SandboxResult:
    manifest = work / "result.json"
    if not manifest.is_file():
        return SandboxResult(
            ok=False,
            stdout=completed.stdout,
            stderr=completed.stderr,
            traceback="the snippet produced no result",
        )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    kind = str(payload.get("kind", "none"))
    frame = None
    if kind == "frame":
        frame = pl.read_parquet(work / "result.parquet")
    return SandboxResult(
        ok=True,
        kind=kind,
        value=payload.get("value"),
        frame=frame,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
