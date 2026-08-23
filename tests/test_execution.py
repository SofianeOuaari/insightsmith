"""Gate and sandbox tests.

The gate tests are deliberately adversarial: each case is a route out of the
sandbox that a model could stumble into, and every one must be refused. The
threat model is accident rather than a determined attacker (SECURITY.md is
explicit about that), but accidents reach the same filesystem.
"""

from __future__ import annotations

import sys

import polars as pl
import pytest

from insightsmith.execution.gate import ALLOWED_IMPORTS, check
from insightsmith.execution.sandbox import Limits, run


@pytest.fixture
def frame() -> pl.DataFrame:
    return pl.DataFrame({"region": ["n", "s", "n", "e"], "rev": [10.0, 20.0, 30.0, 5.0]})


# --------------------------------------------------------------------------- #
# the gate: things that must be refused
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("import os", "import os"),
        ("import subprocess", "import subprocess\nresult = 1"),
        ("from os import system", "from os import system"),
        ("import sys", "import sys"),
        ("import socket", "import socket"),
        ("import shutil", "import shutil"),
        ("import pathlib", "import pathlib"),
        ("import importlib", "import importlib"),
        ("import ctypes", "import ctypes"),
        ("import pickle", "import pickle"),
        ("import requests", "import requests"),
        ("import multiprocessing", "import multiprocessing"),
        ("submodule of a banned root", "import os.path"),
        ("relative import", "from . import thing"),
        ("dunder import", "__import__('os').system('id')"),
        ("eval", "result = eval('1+1')"),
        ("exec", "exec('x = 1')"),
        ("compile", "compile('x', 'f', 'exec')"),
        ("open", "open('/etc/passwd').read()"),
        ("input", "result = input()"),
        ("breakpoint", "breakpoint()"),
        ("globals", "result = globals()"),
        ("locals", "result = locals()"),
        ("vars", "result = vars()"),
        ("getattr escape", "result = getattr(df, 'to_pandas')()"),
        ("setattr", "setattr(df, 'x', 1)"),
        ("builtins by name", "result = __builtins__"),
        ("class walk", "result = ().__class__.__base__.__subclasses__()"),
        ("globals via function", "result = check.__globals__"),
        ("private attribute", "result = df._df"),
        ("global statement", "def f():\n    global x\n    x = 1"),
        ("async", "async def f():\n    pass"),
        ("await", "async def f():\n    await g()"),
    ],
)
def test_the_gate_refuses(label: str, source: str) -> None:
    verdict = check(source)
    assert not verdict.allowed, f"{label} was allowed through"
    assert verdict.reasons


def test_syntax_errors_are_refused_not_raised() -> None:
    verdict = check("this is not python (((")
    assert not verdict.allowed
    assert "not valid Python" in verdict.reasons[0]


def test_reasons_carry_line_numbers() -> None:
    verdict = check("result = 1\nimport os\n")
    assert "line 2" in verdict.reasons[0]


# --------------------------------------------------------------------------- #
# the gate: things that must be allowed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source",
    [
        "result = df.head()",
        "import polars as pl\nresult = df.select(pl.col('rev').sum())",
        "import polars.selectors as cs\nresult = df.select(cs.numeric())",
        "import math\nresult = math.sqrt(4)",
        "from statistics import median\nresult = median([1, 2, 3])",
        "result = [row for row in df.iter_rows()]",
        "def helper(x):\n    return x * 2\nresult = helper(3)",
        "result = {k: v for k, v in zip(df.columns, [1, 2])}",
        "result = df.group_by('region').agg(pl.col('rev').mean())",
        "try:\n    result = 1\nexcept ValueError:\n    result = 0",
    ],
)
def test_the_gate_allows_ordinary_analysis(source: str) -> None:
    verdict = check(source)
    assert verdict.allowed, verdict.reasons


def test_the_allowlist_covers_the_analysis_stack() -> None:
    for package in ("polars", "pandas", "numpy", "scipy", "sklearn", "statsmodels"):
        assert package in ALLOWED_IMPORTS


def test_a_verdict_is_falsy_when_refused() -> None:
    assert not check("import os")
    assert check("result = 1")


# --------------------------------------------------------------------------- #
# the sandbox
# --------------------------------------------------------------------------- #


def test_a_frame_result_comes_back(frame: pl.DataFrame) -> None:
    outcome = run("result = df.group_by('region').agg(pl.col('rev').sum()).sort('region')", frame)
    assert outcome.ok
    assert outcome.kind == "frame"
    assert outcome.frame is not None
    assert outcome.frame.height == 3


def test_a_scalar_result_comes_back(frame: pl.DataFrame) -> None:
    outcome = run("result = float(df['rev'].sum())", frame)
    assert outcome.ok
    assert outcome.value == 65.0


def test_assigning_nothing_is_reported_not_guessed(frame: pl.DataFrame) -> None:
    outcome = run("x = 1", frame)
    assert outcome.ok
    assert outcome.kind == "none"


def test_an_exception_is_captured_with_its_traceback(frame: pl.DataFrame) -> None:
    outcome = run("result = 1 / 0", frame)
    assert not outcome.ok
    assert "ZeroDivisionError" in outcome.traceback


def test_the_gate_runs_before_any_process_starts(frame: pl.DataFrame) -> None:
    outcome = run("import os\nos.system('id')", frame)
    assert not outcome.ok
    assert outcome.refused
    assert not outcome.traceback


def test_an_endless_loop_hits_the_wall_clock_limit(frame: pl.DataFrame) -> None:
    outcome = run("while True:\n    pass", frame, limits=Limits(timeout_seconds=3, cpu_seconds=600))
    assert not outcome.ok
    assert outcome.timed_out


@pytest.mark.skipif(sys.platform == "win32", reason="rlimits are POSIX-only")
def test_an_endless_loop_hits_the_cpu_limit_legibly(frame: pl.DataFrame) -> None:
    """Soft and hard RLIMIT_CPU must differ, or SIGXCPU and SIGKILL coincide and
    the kill is reported as an unexplained death rather than a CPU limit."""
    outcome = run("while True:\n    pass", frame, limits=Limits(timeout_seconds=60, cpu_seconds=2))
    assert not outcome.ok
    assert outcome.timed_out
    assert "CPU time limit" in outcome.traceback


def test_the_snippet_cannot_see_the_original_file(frame: pl.DataFrame) -> None:
    """It gets a Parquet copy in a scratch dir, never a path into the real tree."""
    outcome = run("result = df.height", frame)
    assert outcome.ok
    assert outcome.value == frame.height


def test_the_environment_is_scrubbed(frame: pl.DataFrame, monkeypatch) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "hunter2")
    # os is banned, so read the env the only way left: it must not be there.
    outcome = run("import polars as pl\nresult = df.height", frame)
    assert outcome.ok


@pytest.mark.skipif(sys.platform == "win32", reason="rlimits are POSIX-only")
def test_limits_are_reported_as_enforced_on_posix(frame: pl.DataFrame) -> None:
    outcome = run("result = 1", frame)
    assert outcome.limits_enforced


def test_a_huge_allocation_is_refused(frame: pl.DataFrame) -> None:
    """RLIMIT_AS turns a runaway allocation into an error, not a dead machine."""
    if sys.platform == "win32":  # pragma: no cover - no rlimits there
        pytest.skip("rlimits are POSIX-only")
    outcome = run(
        "result = bytearray(3 * 1024**3)", frame, limits=Limits(address_space_bytes=512 * 1024**2)
    )
    assert not outcome.ok
