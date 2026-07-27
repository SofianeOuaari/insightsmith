"""Exceptions raised by insightsmith.

Every one carries an actionable message: what was refused, and what to do next.
"""

from __future__ import annotations

__all__ = [
    "InsightsmithError",
    "MissingDependencyError",
    "UnsupportedFormatError",
]


class InsightsmithError(Exception):
    """Base class, so callers can catch everything this package raises."""


class UnsupportedFormatError(InsightsmithError):
    """A format was detected correctly but cannot be loaded yet."""

    def __init__(self, fmt: str, *, detail: str | None = None) -> None:
        message = f"cannot load {fmt} sources yet"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)
        self.format = fmt


class MissingDependencyError(InsightsmithError):
    """An optional extra is needed for this source."""

    def __init__(self, package: str, extra: str, *, purpose: str) -> None:
        super().__init__(
            f"{purpose} needs {package!r}: install it with `pip install insightsmith[{extra}]`"
        )
        self.package = package
        self.extra = extra
