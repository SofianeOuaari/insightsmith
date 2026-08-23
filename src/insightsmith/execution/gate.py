"""Static screening of generated code, before anything is executed.

This is the cheapest of the six layers in design doc §7 and the only one that
runs before a process exists. It is written as an **allowlist**: anything not
explicitly permitted is refused. A denylist of dangerous names is unwinnable —
``__import__``, ``getattr``, attribute chains through ``__class__`` and a dozen
other routes all reach the same places — so the gate permits a small set of
imports and refuses every construct that can name something dynamically.

It assumes the model may produce dangerous code *by accident*, which is the
realistic threat. It is emphatically not proof against someone deliberately
crafting a bypass; see SECURITY.md.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "ALLOWED_IMPORTS",
    "BANNED_CALLS",
    "Verdict",
    "check",
]

#: Roots a snippet may import. Submodules of these are fine; nothing else is.
ALLOWED_IMPORTS: Final = frozenset(
    {
        # the analysis stack §7 allows
        "polars",
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "statsmodels",
        "matplotlib",
        "plotly",
        "seaborn",
        # harmless stdlib an analysis genuinely needs
        "math",
        "statistics",
        "datetime",
        "decimal",
        "fractions",
        "itertools",
        "functools",
        "collections",
        "re",
        "json",
        "typing",
        "dataclasses",
        "enum",
        "random",
        "textwrap",
        "warnings",
    }
)

#: Builtins that either execute strings or reach objects by name. ``getattr``
#: and friends are here because a name computed at runtime defeats every static
#: check that follows it.
BANNED_CALLS: Final = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
        "help",
        "exit",
        "quit",
    }
)

#: Names that are gateways out of the sandbox even without a call.
BANNED_NAMES: Final = frozenset({"__builtins__", "__loader__", "__spec__", "__debug__"})


@dataclass(slots=True)
class Verdict:
    """Whether the snippet may run, and why not if it may not."""

    allowed: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.allowed


def check(source: str) -> Verdict:
    """Screen a snippet. Returns a :class:`Verdict`; never raises on bad code."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return Verdict(False, [f"line {exc.lineno or 0}: not valid Python ({exc.msg})"])

    reasons: list[str] = []
    for node in ast.walk(tree):
        reasons.extend(_inspect(node))
    return Verdict(not reasons, reasons)


def _inspect(node: ast.AST) -> list[str]:
    line = getattr(node, "lineno", 0)

    if isinstance(node, ast.Import):
        return [
            f"line {line}: import of {alias.name!r} is not allowed"
            for alias in node.names
            if _root(alias.name) not in ALLOWED_IMPORTS
        ]

    if isinstance(node, ast.ImportFrom):
        # `from . import x` has no module name and no place in a snippet.
        module = node.module or ""
        if node.level or _root(module) not in ALLOWED_IMPORTS:
            return [f"line {line}: import from {module or '.'!r} is not allowed"]
        return []

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in BANNED_CALLS:
            return [f"line {line}: {node.func.id}() is not allowed"]
        return []

    if isinstance(node, ast.Attribute):
        # Dunder and private attributes are how you walk from any object to the
        # interpreter: ().__class__.__base__.__subclasses__() and friends.
        if node.attr.startswith("_"):
            return [f"line {line}: attribute {node.attr!r} is not allowed"]
        return []

    if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
        return [f"line {line}: {node.id} is not allowed"]

    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return [f"line {line}: global/nonlocal is not allowed"]

    if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
        return [f"line {line}: async code is not allowed"]

    return []


def _root(dotted: str) -> str:
    return dotted.split(".", 1)[0]
