"""Accelerator detection: NVIDIA, AMD, Apple silicon, and a best-effort fallback.

As with :mod:`probe`, parsers take raw text. ``tests/fixtures/hardware`` holds
captured output so none of this needs a GPU to test.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

from insightsmith.hardware.probe import SystemInfo, run_command

__all__ = [
    "Accelerator",
    "Vendor",
    "detect_accelerators",
    "parse_apple_hardware",
    "parse_lspci",
    "parse_nvidia_smi",
    "parse_ollama_list",
    "parse_rocm_smi",
]

_MIB_PER_GB: Final = 1000.0  # nvidia-smi reports MiB; GB here is decimal, as in §4
_BYTES_PER_GB: Final = 1e9
#: Apple's unified memory is shared with the OS; this is the share a model may use.
APPLE_USABLE_SHARE: Final = 0.70


class Vendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    INTEL = "intel"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Accelerator:
    vendor: Vendor
    name: str
    memory_total_gb: float | None = None
    memory_free_gb: float | None = None
    compute_capability: str | None = None
    #: Apple silicon shares one pool with the CPU, which changes the fit rule.
    unified: bool = False
    #: Below 1.0 when the figures came from a weak source such as lspci.
    confidence: float = 1.0


def parse_nvidia_smi(text: str) -> list[Accelerator]:
    """Parse the CSV form of ``nvidia-smi --query-gpu=...``.

    Expects ``name,memory.total,memory.used,compute_cap`` with
    ``--format=csv,noheader,nounits``.
    """
    out: list[Accelerator] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        total = _to_float(parts[1])
        used = _to_float(parts[2])
        out.append(
            Accelerator(
                vendor=Vendor.NVIDIA,
                name=parts[0],
                memory_total_gb=None if total is None else total / _MIB_PER_GB,
                memory_free_gb=(
                    None if total is None or used is None else (total - used) / _MIB_PER_GB
                ),
                compute_capability=parts[3] if len(parts) > 3 and parts[3] else None,
            )
        )
    return out


def parse_rocm_smi(text: str) -> list[Accelerator]:
    """Parse ``rocm-smi --showmeminfo vram --json``."""
    try:
        payload = json.loads(text)
    except ValueError:
        return []
    if not isinstance(payload, dict):
        return []

    out: list[Accelerator] = []
    for card, values in sorted(payload.items()):
        if not isinstance(values, dict):
            continue
        total = _first_numeric(values, "vram total memory")
        used = _first_numeric(values, "vram total used memory")
        out.append(
            Accelerator(
                vendor=Vendor.AMD,
                name=str(values.get("Card Series") or values.get("Card series") or card),
                memory_total_gb=None if total is None else total / _BYTES_PER_GB,
                memory_free_gb=(
                    None if total is None or used is None else (total - used) / _BYTES_PER_GB
                ),
            )
        )
    return out


def parse_apple_hardware(text: str) -> Accelerator | None:
    """Parse ``system_profiler SPHardwareDataType -json`` into a unified-memory device."""
    try:
        item = json.loads(text)["SPHardwareDataType"][0]
    except (ValueError, KeyError, IndexError):
        return None

    chip = str(item.get("chip_type") or item.get("cpu_type") or "Apple silicon")
    memory = item.get("physical_memory")
    total = None
    if isinstance(memory, str):
        found = re.match(r"([\d.]+)\s*(GB|TB)", memory.strip(), re.I)
        if found:
            total = float(found.group(1)) * (1000 if found.group(2).upper() == "TB" else 1)
    elif isinstance(memory, (int, float)):
        total = float(memory)

    return Accelerator(
        vendor=Vendor.APPLE,
        name=chip,
        memory_total_gb=total,
        # Usable, not total: the OS and everything else live in the same pool.
        memory_free_gb=None if total is None else total * APPLE_USABLE_SHARE,
        unified=True,
    )


def parse_lspci(text: str) -> list[Accelerator]:
    """Last resort. Yields names with no memory figures, hence low confidence."""
    out: list[Accelerator] = []
    for line in text.strip().splitlines():
        if not re.search(r"VGA compatible controller|3D controller|Display controller", line):
            continue
        description = line.split(":", 2)[-1].strip()
        lowered = description.lower()
        if "nvidia" in lowered:
            vendor = Vendor.NVIDIA
        elif "amd" in lowered or "advanced micro devices" in lowered or "radeon" in lowered:
            vendor = Vendor.AMD
        elif "intel" in lowered:
            vendor = Vendor.INTEL
        else:
            vendor = Vendor.UNKNOWN
        out.append(Accelerator(vendor=vendor, name=description, confidence=0.3))
    return out


def parse_ollama_list(text: str) -> list[str]:
    """Parse ``ollama list`` into model tags."""
    tags: list[str] = []
    for index, line in enumerate(text.strip().splitlines()):
        if index == 0 and line.upper().startswith("NAME"):
            continue
        first = line.split()
        if first and ":" in first[0]:
            tags.append(first[0])
    return tags


def detect_accelerators(system: SystemInfo) -> list[Accelerator]:
    """Probe the running machine, strongest evidence first."""
    nvidia = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    found: list[Accelerator] = list(parse_nvidia_smi(nvidia)) if nvidia else []

    amd = run_command(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if amd:
        found.extend(parse_rocm_smi(amd))

    if system.is_apple_silicon:
        profiled = run_command(["system_profiler", "SPHardwareDataType", "-json"])
        if profiled:
            apple = parse_apple_hardware(profiled)
            if apple is not None:
                found.append(apple)

    if not found:
        listed = run_command(["lspci"])
        if listed:
            found.extend(parse_lspci(listed))
    return found


def detect_installed_models() -> list[str]:
    """Model tags Ollama already has locally, via the CLI rather than the API."""
    listed = run_command(["ollama", "list"], timeout=10.0)
    return parse_ollama_list(listed) if listed else []


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _first_numeric(values: dict[str, object], needle: str) -> float | None:
    for key, value in values.items():
        if needle in key.lower():
            try:
                return float(str(value))
            except ValueError:
                return None
    return None
