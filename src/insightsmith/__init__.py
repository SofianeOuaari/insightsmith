"""insightsmith — an agentic data consultant that runs on your own hardware."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("insightsmith")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
