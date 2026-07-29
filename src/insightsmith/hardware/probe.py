"""CPU, memory, OS and disk.

Every parser takes raw text, so the whole module is testable against captured
output and CI never has to touch real hardware.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import psutil

__all__ = [
    "CpuInfo",
    "MemoryInfo",
    "SystemInfo",
    "parse_proc_cpuinfo",
    "parse_sysctl_brand",
    "parse_system_profiler_cpu",
    "probe_system",
    "run_command",
]

_BYTES_PER_GB: Final = 1e9
_COMMAND_TIMEOUT: Final = 5.0


@dataclass(slots=True)
class CpuInfo:
    model: str
    physical_cores: int | None = None
    logical_cores: int | None = None
    max_mhz: float | None = None


@dataclass(slots=True)
class MemoryInfo:
    total_gb: float
    available_gb: float


@dataclass(slots=True)
class SystemInfo:
    os_name: str
    os_release: str
    arch: str
    cpu: CpuInfo
    memory: MemoryInfo
    disk_free_gb: float
    warnings: list[str] = field(default_factory=list)

    @property
    def is_apple_silicon(self) -> bool:
        return self.os_name == "Darwin" and self.arch in {"arm64", "aarch64"}


def run_command(cmd: Sequence[str], *, timeout: float = _COMMAND_TIMEOUT) -> str | None:
    """Run an external binary and return stdout, or ``None`` on any failure.

    A missing binary is a soft failure, not an exception: probing a machine that
    lacks ``nvidia-smi`` is the normal case, not an error.
    """
    if not cmd or shutil.which(cmd[0]) is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, never a shell string
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def parse_proc_cpuinfo(text: str) -> CpuInfo:
    """Parse Linux ``/proc/cpuinfo``.

    Physical cores are counted as distinct ``(physical id, core id)`` pairs,
    which is what separates 8 real cores from 16 hyperthreads.
    """
    model = ""
    logical = 0
    frequencies: list[float] = []
    pairs: set[tuple[str, str]] = set()
    physical_id = core_id = None

    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "processor":
            logical += 1
            physical_id = core_id = None
        elif key == "model name" and not model:
            model = value
        elif key == "cpu MHz":
            with_suppressed = _to_float(value)
            if with_suppressed is not None:
                frequencies.append(with_suppressed)
        elif key == "physical id":
            physical_id = value
        elif key == "core id":
            core_id = value
        if physical_id is not None and core_id is not None:
            pairs.add((physical_id, core_id))

    return CpuInfo(
        model=model or platform.processor() or "unknown",
        physical_cores=len(pairs) or None,
        logical_cores=logical or None,
        max_mhz=max(frequencies) if frequencies else None,
    )


def parse_sysctl_brand(text: str) -> str:
    """Parse ``sysctl -n machdep.cpu.brand_string`` (macOS)."""
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def parse_system_profiler_cpu(text: str) -> CpuInfo | None:
    """Parse ``system_profiler SPHardwareDataType -json`` (macOS)."""
    try:
        payload = json.loads(text)
        item = payload["SPHardwareDataType"][0]
    except (ValueError, KeyError, IndexError):
        return None

    cores = item.get("number_processors")
    physical = logical = None
    if isinstance(cores, int):
        physical = logical = cores
    elif isinstance(cores, str):
        # Apple silicon reports "proc 10:4:6" — total:performance:efficiency.
        found = re.findall(r"\d+", cores)
        if found:
            physical = logical = int(found[0])

    return CpuInfo(
        model=str(item.get("chip_type") or item.get("cpu_type") or "unknown"),
        physical_cores=physical,
        logical_cores=logical,
    )


def probe_cpu() -> CpuInfo:
    """Best available CPU description for the running machine."""
    proc_cpuinfo = Path("/proc/cpuinfo")
    if proc_cpuinfo.is_file():
        try:
            return parse_proc_cpuinfo(proc_cpuinfo.read_text(errors="replace"))
        except OSError:
            pass

    if platform.system() == "Darwin":
        profiled = run_command(["system_profiler", "SPHardwareDataType", "-json"])
        if profiled:
            parsed = parse_system_profiler_cpu(profiled)
            if parsed is not None:
                return parsed
        brand = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return CpuInfo(
                model=parse_sysctl_brand(brand),
                physical_cores=psutil.cpu_count(logical=False),
                logical_cores=psutil.cpu_count(logical=True),
            )

    return CpuInfo(
        model=platform.processor() or platform.machine() or "unknown",
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
    )


def probe_system() -> SystemInfo:
    """Probe the machine this process is running on."""
    memory = psutil.virtual_memory()
    warnings: list[str] = []
    try:
        free = shutil.disk_usage(Path.cwd()).free / _BYTES_PER_GB
    except OSError:  # pragma: no cover - unreadable cwd
        free = 0.0
        warnings.append("could not read free disk space")

    return SystemInfo(
        os_name=platform.system(),
        os_release=platform.release(),
        arch=platform.machine(),
        cpu=probe_cpu(),
        memory=MemoryInfo(
            total_gb=memory.total / _BYTES_PER_GB,
            available_gb=memory.available / _BYTES_PER_GB,
        ),
        disk_free_gb=free,
        warnings=warnings,
    )


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None
