"""Hardware probing and model-fit recommendation."""

from insightsmith.hardware.accel import (
    Accelerator,
    Vendor,
    detect_accelerators,
    detect_installed_models,
)
from insightsmith.hardware.probe import CpuInfo, MemoryInfo, SystemInfo, probe_system
from insightsmith.hardware.recommend import (
    Catalog,
    Fit,
    ModelSpec,
    Placement,
    Recommendation,
    fit_model,
    kv_cache_gb,
    load_catalog,
    recommend,
    tokens_per_second,
    weights_gb,
)

__all__ = [
    "Accelerator",
    "Catalog",
    "CpuInfo",
    "Fit",
    "MemoryInfo",
    "ModelSpec",
    "Placement",
    "Recommendation",
    "SystemInfo",
    "Vendor",
    "detect_accelerators",
    "detect_installed_models",
    "fit_model",
    "kv_cache_gb",
    "load_catalog",
    "probe_system",
    "recommend",
    "tokens_per_second",
    "weights_gb",
]
