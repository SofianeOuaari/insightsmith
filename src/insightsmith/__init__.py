"""insightsmith — an agentic data consultant that runs on your own hardware.

0.1.0 ships the no-LLM foundation: detect a source's real format, load it as a
Polars ``LazyFrame``, and profile it. ``Consultant`` and the agent graph arrive
with the later milestones.
"""

from importlib.metadata import PackageNotFoundError, version

from insightsmith.errors import (
    InsightsmithError,
    MissingDependencyError,
    UnsupportedFormatError,
)
from insightsmith.io.loaders import load
from insightsmith.io.sniff import Compression, Dialect, Format, SourceSpec, sniff
from insightsmith.profiling import ColumnProfile, Profile, profile

try:
    __version__ = version("insightsmith")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "ColumnProfile",
    "Compression",
    "Dialect",
    "Format",
    "InsightsmithError",
    "MissingDependencyError",
    "Profile",
    "SourceSpec",
    "UnsupportedFormatError",
    "__version__",
    "load",
    "profile",
    "sniff",
]
