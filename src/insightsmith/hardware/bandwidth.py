"""Memory bandwidth lookup.

Decoding is memory-bandwidth-bound, so throughput follows from this table rather
than from any benchmark run here.

**Every figure is theoretical peak from the vendor's published specification, not
a measurement.** Report derived throughput as an estimate, never as a benchmark.
Unknown hardware returns ``None``, and callers must say "unknown" rather than
substitute a plausible-looking number.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["BANDWIDTH_GB_S", "SYSTEM_MEMORY_GB_S", "lookup_device", "lookup_system_memory"]

#: Device name pattern -> theoretical peak bandwidth in GB/s (vendor specs).
#: Patterns are matched case-insensitively against the reported device name.
BANDWIDTH_GB_S: Final[dict[str, float]] = {
    r"rtx\s*4090": 1008.0,
    r"rtx\s*4080": 717.0,
    r"rtx\s*3090": 936.0,
    r"a100": 1555.0,
    r"h100": 3350.0,
    r"\bm1\b": 68.25,
    r"m1\s*pro": 200.0,
    r"m1\s*max": 400.0,
    r"m1\s*ultra": 800.0,
    r"\bm2\b": 100.0,
    r"m2\s*pro": 200.0,
    r"m2\s*max": 400.0,
    r"m2\s*ultra": 800.0,
    r"\bm3\b": 100.0,
    r"m3\s*pro": 150.0,
    r"m3\s*max": 400.0,
}

#: System RAM bandwidth by memory generation, dual channel (GB/s).
SYSTEM_MEMORY_GB_S: Final[dict[str, float]] = {
    "ddr4-3200": 51.2,
    "ddr5-5600": 89.6,
}

#: Used when the memory generation is unknown — a conservative DDR4 figure.
DEFAULT_SYSTEM_GB_S: Final = 51.2

#: Fraction of peak bandwidth a real decode loop achieves.
EFFICIENCY: Final = 0.70


def lookup_device(name: str) -> float | None:
    """Theoretical peak bandwidth for a device, or ``None`` if not in the table.

    ``None`` is a normal answer. It means throughput cannot be estimated for this
    device, and the caller must say so rather than guess.
    """
    lowered = name.lower()
    best: tuple[int, float] | None = None
    for pattern, value in BANDWIDTH_GB_S.items():
        found = re.search(pattern, lowered)
        if found is None:
            continue
        # Prefer the most specific match: "m2 max" must beat "m2".
        span = found.end() - found.start()
        if best is None or span > best[0]:
            best = (span, value)
    return None if best is None else best[1]


def lookup_system_memory(generation: str | None = None) -> float:
    """Bandwidth for CPU-side inference, defaulting conservatively."""
    if generation is None:
        return DEFAULT_SYSTEM_GB_S
    return SYSTEM_MEMORY_GB_S.get(generation.lower(), DEFAULT_SYSTEM_GB_S)
