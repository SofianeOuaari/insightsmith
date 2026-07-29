"""Fit math and per-role model recommendation, following design doc §4.

The KV-cache term uses ``n_kv_heads``, not ``n_heads``. Grouped-query attention
cuts that four- to eight-fold on modern models, and using the wrong one
overstates the cache by the same factor — which is why a "7B model needs 4 GB"
rule of thumb falls apart at long context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from importlib.resources import files
from pathlib import Path
from typing import Final

from insightsmith.hardware.accel import Accelerator, Vendor
from insightsmith.hardware.bandwidth import EFFICIENCY, lookup_device, lookup_system_memory
from insightsmith.hardware.probe import SystemInfo

__all__ = [
    "Catalog",
    "Fit",
    "ModelSpec",
    "Placement",
    "Recommendation",
    "kv_cache_gb",
    "load_catalog",
    "recommend",
    "tokens_per_second",
    "weights_gb",
]

#: Runtime and activation overhead on top of weights plus cache (§4).
OVERHEAD: Final = 1.10
#: Headroom factor: never plan to fill a device completely.
USABLE_SHARE: Final = 0.85
#: Bytes per KV element. fp16 cache.
KV_BYTES: Final = 2
DEFAULT_CONTEXT: Final = 8192


class Placement(str, Enum):
    GPU = "gpu"
    UNIFIED = "unified"
    PARTIAL = "partial"
    CPU = "cpu"
    EXCLUDED = "excluded"


@dataclass(slots=True)
class ModelSpec:
    tag: str
    params_b: float
    n_layers: int
    n_kv_heads: int
    head_dim: int
    context_length: int
    default_quant: str
    roles: list[str] = field(default_factory=list)
    architecture: str = ""
    license: str = ""
    tier: str = ""
    source: str = ""
    note: str = ""


@dataclass(slots=True)
class Catalog:
    bytes_per_param: dict[str, float]
    models: list[ModelSpec]

    def for_role(self, role: str) -> list[ModelSpec]:
        return [m for m in self.models if role in m.roles]


@dataclass(slots=True)
class Fit:
    """What a model costs, and where it can run."""

    weights_gb: float
    kv_cache_gb: float
    total_gb: float
    context: int
    placement: Placement
    device: str
    n_gpu_layers: int | None = None
    tokens_per_second: float | None = None
    reason: str = ""


@dataclass(slots=True)
class Recommendation:
    role: str
    model: ModelSpec
    fit: Fit
    installed: bool = False


def weights_gb(params_b: float, quant: str, bytes_per_param: dict[str, float]) -> float:
    """``W_gb = params_B x bytes_per_param``."""
    per_param = bytes_per_param.get(quant)
    if per_param is None:
        per_param = bytes_per_param.get("Q4_K_M", 0.6)
    return params_b * per_param


def kv_cache_gb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context: int,
    kv_bytes: int = KV_BYTES,
) -> float:
    """``KV_gb = 2 x n_layers x n_kv_heads x head_dim x ctx x kv_bytes / 1e9``.

    The leading 2 is for the key and the value tensors.
    """
    return 2 * n_layers * n_kv_heads * head_dim * context * kv_bytes / 1e9


def tokens_per_second(bandwidth_gb_s: float | None, weights: float) -> float | None:
    """``tok/s ~ bandwidth x efficiency / W_gb``.

    ``None`` when the device's bandwidth is unknown — an unknown number is
    reported as unknown, never replaced by a guess.
    """
    if bandwidth_gb_s is None or weights <= 0:
        return None
    return bandwidth_gb_s * EFFICIENCY / weights


def load_catalog(path: Path | None = None) -> Catalog:
    """Load the shipped catalog, or one supplied by the caller."""
    if path is None:
        raw = (files("insightsmith.hardware") / "catalog.json").read_text(encoding="utf-8")
    else:
        raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    models = [
        ModelSpec(
            tag=entry["tag"],
            params_b=float(entry["params_b"]),
            n_layers=int(entry["n_layers"]),
            n_kv_heads=int(entry["n_kv_heads"]),
            head_dim=int(entry["head_dim"]),
            context_length=int(entry["context_length"]),
            default_quant=str(entry.get("default_quant", "Q4_K_M")),
            roles=list(entry.get("roles", [])),
            architecture=str(entry.get("architecture", "")),
            license=str(entry.get("license", "")),
            tier=str(entry.get("tier", "")),
            source=str(entry.get("source", "")),
            note=str(entry.get("note", "")),
        )
        for entry in payload.get("models", [])
    ]
    return Catalog(bytes_per_param=dict(payload.get("bytes_per_param", {})), models=models)


def fit_model(
    model: ModelSpec,
    catalog: Catalog,
    system: SystemInfo,
    accelerators: list[Accelerator],
    *,
    context: int = DEFAULT_CONTEXT,
    quant: str | None = None,
) -> Fit:
    """Decide where ``model`` can run at ``context`` tokens, and how fast."""
    context = min(context, model.context_length)
    weights = weights_gb(model.params_b, quant or model.default_quant, catalog.bytes_per_param)
    cache = kv_cache_gb(model.n_layers, model.n_kv_heads, model.head_dim, context)
    need = (weights + cache) * OVERHEAD

    device = _best_device(accelerators)

    if device is not None and device.memory_total_gb:
        budget = device.memory_total_gb * USABLE_SHARE
        if need <= budget:
            placement = Placement.UNIFIED if device.unified else Placement.GPU
            reason = f"{need:.1f} GB needed, {device.memory_total_gb:.1f} GB available" + (
                " — watch thermals on sustained runs" if device.unified else ""
            )
            return _finish(
                weights, cache, need, context, placement, device.name, None, device.name, reason
            )

        # Partial offload: how many whole layers fit in the device budget?
        per_layer = weights / model.n_layers if model.n_layers else weights
        free = (device.memory_free_gb or device.memory_total_gb) * USABLE_SHARE
        layers = int(free / per_layer) if per_layer > 0 else 0
        layers = max(0, min(layers, model.n_layers))
        if layers and need <= system.memory.total_gb:
            return _finish(
                weights,
                cache,
                need,
                context,
                Placement.PARTIAL,
                device.name,
                layers,
                None,
                f"{need:.1f} GB needed but only {budget:.1f} GB usable on the device; "
                f"offload {layers}/{model.n_layers} layers and expect the rest to be slow",
            )

    if need > system.memory.total_gb:
        return _finish(
            weights,
            cache,
            need,
            context,
            Placement.EXCLUDED,
            device.name if device else "cpu",
            None,
            None,
            f"{need:.1f} GB needed exceeds {system.memory.total_gb:.1f} GB of system memory",
        )

    return _finish(
        weights,
        cache,
        need,
        context,
        Placement.CPU,
        "cpu",
        None,
        None,
        f"{need:.1f} GB fits in system memory; no usable accelerator found",
    )


def recommend(
    system: SystemInfo,
    accelerators: list[Accelerator],
    catalog: Catalog,
    *,
    roles: list[str] | None = None,
    context: int = DEFAULT_CONTEXT,
    installed: list[str] | None = None,
) -> list[Recommendation]:
    """Best model per role — never one overall winner (§4).

    A small fast model for routing and a bigger one for planning are different
    jobs, and the machine may afford one but not the other.
    """
    wanted = roles or ["planner", "coder", "cheap", "reasoner", "embed"]
    have = set(installed or [])
    out: list[Recommendation] = []

    for role in wanted:
        candidates = catalog.for_role(role)
        if not candidates:
            continue
        scored = [
            (model, fit_model(model, catalog, system, accelerators, context=context))
            for model in candidates
        ]
        runnable = [(m, f) for m, f in scored if f.placement is not Placement.EXCLUDED]
        if not runnable:
            continue
        # Prefer the best placement, then the largest model that still fits.
        order = {
            Placement.GPU: 0,
            Placement.UNIFIED: 0,
            Placement.PARTIAL: 1,
            Placement.CPU: 2,
            Placement.EXCLUDED: 3,
        }
        model, fit = min(runnable, key=lambda pair: (order[pair[1].placement], -pair[0].params_b))
        out.append(Recommendation(role=role, model=model, fit=fit, installed=model.tag in have))
    return out


def _best_device(accelerators: list[Accelerator]) -> Accelerator | None:
    usable = [a for a in accelerators if a.memory_total_gb and a.vendor is not Vendor.UNKNOWN]
    if not usable:
        return None
    return max(usable, key=lambda a: a.memory_total_gb or 0.0)


def _finish(
    weights: float,
    cache: float,
    need: float,
    context: int,
    placement: Placement,
    device: str,
    layers: int | None,
    bandwidth_device: str | None,
    reason: str,
) -> Fit:
    bandwidth: float | None
    if placement is Placement.CPU:
        bandwidth = lookup_system_memory()
    elif bandwidth_device is not None:
        bandwidth = lookup_device(bandwidth_device)
    else:
        bandwidth = None
    return Fit(
        weights_gb=weights,
        kv_cache_gb=cache,
        total_gb=need,
        context=context,
        placement=placement,
        device=device,
        n_gpu_layers=layers,
        tokens_per_second=tokens_per_second(bandwidth, weights),
        reason=reason,
    )
