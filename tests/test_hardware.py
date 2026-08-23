"""Hardware parsing and fit math.

Parsers are exercised against files in ``tests/fixtures/hardware``. The
``nvidia-smi``, ``/proc/cpuinfo``, ``lspci`` and ``ollama list`` fixtures were
captured from a real machine; the two marked SYNTHETIC were hand-written from
documented output shapes because no AMD or Apple hardware was available. Nothing
here touches real hardware.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from insightsmith.hardware.accel import (
    Accelerator,
    Vendor,
    detect_accelerators,
    detect_installed_models,
    parse_apple_hardware,
    parse_lspci,
    parse_nvidia_smi,
    parse_ollama_list,
    parse_rocm_smi,
)
from insightsmith.hardware.bandwidth import lookup_device, lookup_system_memory
from insightsmith.hardware.probe import (
    CpuInfo,
    MemoryInfo,
    SystemInfo,
    parse_proc_cpuinfo,
    parse_sysctl_brand,
    parse_system_profiler_cpu,
    run_command,
)
from insightsmith.hardware.recommend import (
    OVERHEAD,
    Placement,
    fit_model,
    kv_cache_gb,
    load_catalog,
    recommend,
    tokens_per_second,
    weights_gb,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hardware"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _system(total_gb: float = 32.0) -> SystemInfo:
    return SystemInfo(
        os_name="Linux",
        os_release="6.8.0",
        arch="x86_64",
        cpu=CpuInfo(model="test", physical_cores=8, logical_cores=16),
        memory=MemoryInfo(total_gb=total_gb, available_gb=total_gb / 2),
        disk_free_gb=500.0,
    )


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #


def test_proc_cpuinfo_separates_cores_from_threads() -> None:
    cpu = parse_proc_cpuinfo(_fixture("proc-cpuinfo-i7-11800h.txt"))
    assert "i7-11800H" in cpu.model
    assert cpu.logical_cores == 16
    assert cpu.physical_cores == 8  # not 16: hyperthreading
    assert cpu.max_mhz is not None


def test_proc_cpuinfo_survives_garbage() -> None:
    cpu = parse_proc_cpuinfo("not a cpuinfo file at all\n\n")
    assert cpu.logical_cores is None
    assert cpu.physical_cores is None


def test_sysctl_brand() -> None:
    assert parse_sysctl_brand("Apple M2 Max\n") == "Apple M2 Max"
    assert parse_sysctl_brand("") == ""


def test_system_profiler_cpu() -> None:
    cpu = parse_system_profiler_cpu(_fixture("system-profiler-SYNTHETIC.json"))
    assert cpu is not None
    assert cpu.model == "Apple M2 Max"
    assert cpu.physical_cores == 12  # from "proc 12:8:4"


def test_system_profiler_cpu_rejects_junk() -> None:
    assert parse_system_profiler_cpu("{not json") is None
    assert parse_system_profiler_cpu("{}") is None


def test_run_command_missing_binary_is_a_soft_failure() -> None:
    assert run_command(["definitely-not-a-real-binary-xyz"]) is None
    assert run_command([]) is None


# --------------------------------------------------------------------------- #
# accelerators
# --------------------------------------------------------------------------- #


def test_nvidia_smi() -> None:
    gpus = parse_nvidia_smi(_fixture("nvidia-smi-rtx-a2000.csv"))
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.vendor is Vendor.NVIDIA
    assert "A2000" in gpu.name
    assert gpu.memory_total_gb == pytest.approx(4.096)
    assert gpu.memory_free_gb is not None and gpu.memory_free_gb < gpu.memory_total_gb
    assert gpu.compute_capability == "8.6"


def test_nvidia_smi_handles_multiple_and_blank_lines() -> None:
    gpus = parse_nvidia_smi("GPU A, 24576, 100, 8.9\n\nGPU B, 24576, 200, 8.9\n")
    assert [g.name for g in gpus] == ["GPU A", "GPU B"]


def test_nvidia_smi_of_junk_yields_nothing() -> None:
    assert parse_nvidia_smi("command not found") == []


def test_rocm_smi() -> None:
    gpus = parse_rocm_smi(_fixture("rocm-smi-SYNTHETIC.json"))
    assert len(gpus) == 2
    assert gpus[0].vendor is Vendor.AMD
    assert gpus[0].memory_total_gb == pytest.approx(25.75, rel=0.01)


def test_rocm_smi_of_junk_yields_nothing() -> None:
    assert parse_rocm_smi("not json") == []
    assert parse_rocm_smi("[]") == []


def test_apple_hardware_reserves_memory_for_the_os() -> None:
    device = parse_apple_hardware(_fixture("system-profiler-SYNTHETIC.json"))
    assert device is not None
    assert device.unified
    assert device.memory_total_gb == pytest.approx(64.0)
    # Unified memory is shared, so usable is well below total.
    assert device.memory_free_gb == pytest.approx(64.0 * 0.70)


def test_lspci_is_low_confidence() -> None:
    gpus = parse_lspci(_fixture("lspci-vga-i7-11800h.txt"))
    assert {g.vendor for g in gpus} == {Vendor.INTEL, Vendor.NVIDIA}
    assert all(g.confidence < 1.0 for g in gpus)
    assert all(g.memory_total_gb is None for g in gpus)


def test_ollama_list() -> None:
    tags = parse_ollama_list(_fixture("ollama-list.txt"))
    assert "qwen3:8b" in tags
    assert all(":" in tag for tag in tags)


def test_ollama_list_when_empty() -> None:
    assert parse_ollama_list("NAME    ID    SIZE    MODIFIED\n") == []


def test_detection_prefers_nvidia_over_the_lspci_fallback(monkeypatch) -> None:
    """lspci is only consulted when nothing better answered."""
    calls: list[str] = []

    def fake_run(cmd, **_):
        calls.append(cmd[0])
        if cmd[0] == "nvidia-smi":
            return _fixture("nvidia-smi-rtx-a2000.csv")
        return None

    monkeypatch.setattr("insightsmith.hardware.accel.run_command", fake_run)
    found = detect_accelerators(_system())
    assert [g.vendor for g in found] == [Vendor.NVIDIA]
    assert "lspci" not in calls


def test_detection_falls_back_to_lspci_when_nothing_else_answers(monkeypatch) -> None:
    def fake_run(cmd, **_):
        return _fixture("lspci-vga-i7-11800h.txt") if cmd[0] == "lspci" else None

    monkeypatch.setattr("insightsmith.hardware.accel.run_command", fake_run)
    found = detect_accelerators(_system())
    assert found
    assert all(g.confidence < 1.0 for g in found)


def test_detection_on_a_machine_with_no_tools_returns_nothing(monkeypatch) -> None:
    monkeypatch.setattr("insightsmith.hardware.accel.run_command", lambda *_, **__: None)
    assert detect_accelerators(_system()) == []


def test_installed_models_when_ollama_is_absent(monkeypatch) -> None:
    monkeypatch.setattr("insightsmith.hardware.accel.run_command", lambda *_, **__: None)
    assert detect_installed_models() == []


# --------------------------------------------------------------------------- #
# bandwidth
# --------------------------------------------------------------------------- #


def test_bandwidth_prefers_the_most_specific_match() -> None:
    assert lookup_device("Apple M2 Max") == 400.0
    assert lookup_device("Apple M2") == 100.0


def test_bandwidth_of_unknown_device_is_none() -> None:
    """Unknown must stay unknown; a plausible substitute would be a fabrication."""
    assert lookup_device("NVIDIA RTX A2000 Laptop GPU") is None


def test_system_memory_bandwidth_defaults_conservatively() -> None:
    assert lookup_system_memory("ddr5-5600") == 89.6
    assert lookup_system_memory(None) == lookup_system_memory("nonsense")


# --------------------------------------------------------------------------- #
# fit math — golden values from §4
# --------------------------------------------------------------------------- #


def test_weights_follow_the_bytes_per_param_table() -> None:
    table = {"Q4_K_M": 0.6, "fp16": 2.0}
    assert weights_gb(8.0, "Q4_K_M", table) == pytest.approx(4.8)
    assert weights_gb(8.0, "fp16", table) == pytest.approx(16.0)


def test_unknown_quant_falls_back_rather_than_crashing() -> None:
    assert weights_gb(8.0, "Q2_K_XS", {"Q4_K_M": 0.6}) == pytest.approx(4.8)


def test_kv_cache_formula() -> None:
    """2 x layers x kv_heads x head_dim x ctx x bytes / 1e9, per §4."""
    # qwen3:8b at 8k: 2*36*8*128*8192*2/1e9
    assert kv_cache_gb(36, 8, 128, 8192) == pytest.approx(1.2079, rel=0.001)


def test_kv_cache_uses_kv_heads_not_attention_heads() -> None:
    """The whole point of GQA: 32 heads but 8 KV heads is a 4x smaller cache."""
    gqa = kv_cache_gb(32, 8, 128, 8192)
    mha = kv_cache_gb(32, 32, 128, 8192)
    assert mha == pytest.approx(gqa * 4)


def test_kv_cache_grows_linearly_with_context() -> None:
    assert kv_cache_gb(36, 8, 128, 16384) == pytest.approx(kv_cache_gb(36, 8, 128, 8192) * 2)


def test_golden_total_for_qwen3_8b_at_8k() -> None:
    """Known model + known context -> expected GB, within 5% (kickoff pack)."""
    catalog = load_catalog()
    model = next(m for m in catalog.models if m.tag == "qwen3:8b")
    weights = weights_gb(model.params_b, model.default_quant, catalog.bytes_per_param)
    cache = kv_cache_gb(model.n_layers, model.n_kv_heads, model.head_dim, 8192)
    total = (weights + cache) * OVERHEAD
    assert weights == pytest.approx(4.92, rel=0.05)
    assert cache == pytest.approx(1.21, rel=0.05)
    assert total == pytest.approx(6.74, rel=0.05)


def test_golden_total_for_a_model_without_gqa() -> None:
    """deepseek-coder:6.7b has 32 KV heads: cache roughly equals its weights."""
    catalog = load_catalog()
    model = next(m for m in catalog.models if m.tag == "deepseek-coder:6.7b")
    cache = kv_cache_gb(model.n_layers, model.n_kv_heads, model.head_dim, 8192)
    assert cache == pytest.approx(4.295, rel=0.05)


def test_tokens_per_second() -> None:
    assert tokens_per_second(1008.0, 4.8) == pytest.approx(1008.0 * 0.70 / 4.8)


def test_tokens_per_second_is_none_without_bandwidth() -> None:
    assert tokens_per_second(None, 4.8) is None


# --------------------------------------------------------------------------- #
# placement decisions
# --------------------------------------------------------------------------- #


def _gpu(total: float, free: float | None = None) -> Accelerator:
    return Accelerator(
        vendor=Vendor.NVIDIA,
        name="Test GPU",
        memory_total_gb=total,
        memory_free_gb=total if free is None else free,
    )


def test_a_model_that_fits_goes_fully_on_the_gpu() -> None:
    catalog = load_catalog()
    model = next(m for m in catalog.models if m.tag == "qwen3:8b")
    fit = fit_model(model, catalog, _system(), [_gpu(24.0)], context=8192)
    assert fit.placement is Placement.GPU
    assert fit.n_gpu_layers is None


def test_a_model_that_does_not_fit_gets_partial_offload() -> None:
    catalog = load_catalog()
    model = next(m for m in catalog.models if m.tag == "qwen3:8b")
    fit = fit_model(model, catalog, _system(), [_gpu(4.096)], context=8192)
    assert fit.placement is Placement.PARTIAL
    assert fit.n_gpu_layers is not None
    assert 0 < fit.n_gpu_layers < model.n_layers


def test_a_model_larger_than_system_memory_is_excluded() -> None:
    catalog = load_catalog()
    model = next(m for m in catalog.models if m.tag == "qwen3:8b")
    fit = fit_model(model, catalog, _system(total_gb=2.0), [], context=8192)
    assert fit.placement is Placement.EXCLUDED
    assert "exceeds" in fit.reason


def test_no_accelerator_falls_back_to_cpu_with_an_estimate() -> None:
    catalog = load_catalog()
    model = next(m for m in catalog.models if m.tag == "qwen3:8b")
    fit = fit_model(model, catalog, _system(), [], context=8192)
    assert fit.placement is Placement.CPU
    assert fit.tokens_per_second is not None  # system RAM bandwidth is known


def test_unified_memory_is_its_own_placement() -> None:
    catalog = load_catalog()
    model = next(m for m in catalog.models if m.tag == "qwen3:8b")
    apple = Accelerator(
        vendor=Vendor.APPLE, name="Apple M2 Max", memory_total_gb=64.0, unified=True
    )
    fit = fit_model(model, catalog, _system(64.0), [apple], context=8192)
    assert fit.placement is Placement.UNIFIED
    assert "thermals" in fit.reason


def test_context_is_capped_at_the_model_maximum() -> None:
    catalog = load_catalog()
    model = next(m for m in catalog.models if m.tag == "deepseek-coder:6.7b")
    fit = fit_model(model, catalog, _system(), [_gpu(80.0)], context=999_999)
    assert fit.context == model.context_length


# --------------------------------------------------------------------------- #
# recommendation
# --------------------------------------------------------------------------- #


def test_recommends_per_role_not_one_winner() -> None:
    catalog = load_catalog()
    picks = recommend(_system(), [_gpu(24.0)], catalog, context=8192)
    roles = [p.role for p in picks]
    assert "coder" in roles
    assert len(roles) == len(set(roles))  # one per role


def test_installed_models_are_marked() -> None:
    catalog = load_catalog()
    picks = recommend(_system(), [_gpu(24.0)], catalog, context=8192, installed=["qwen3:8b"])
    assert any(p.installed for p in picks)


def test_a_tiny_machine_gets_fewer_recommendations() -> None:
    catalog = load_catalog()
    roomy = recommend(_system(64.0), [_gpu(24.0)], catalog, context=8192)
    cramped = recommend(_system(1.0), [], catalog, context=8192)
    assert len(cramped) < len(roomy)


def test_catalog_entries_are_self_consistent() -> None:
    """Guards against an entry added by guesswork rather than from /api/show."""
    catalog = load_catalog()
    assert catalog.models
    for model in catalog.models:
        assert model.n_kv_heads > 0
        assert model.n_layers > 0
        assert model.head_dim > 0
        assert model.context_length > 0
        assert model.source == "verified", f"{model.tag} is not marked verified"
        # An entry with no roles is never recommended, so it must say why —
        # otherwise a role gets dropped in a refactor and nobody notices.
        assert model.roles or model.note, (
            f"{model.tag} has no role and no note explaining the omission"
        )


def test_a_model_excluded_from_a_role_explains_itself() -> None:
    """deepseek-coder:6.7b is catalogued for fit maths but not recommended.

    It measured 0/3 on a Polars group-by where qwen3:8b scored 3/3, writing
    pandas `groupby()` and ignoring corrective retries. Recommending a "coder"
    model that cannot code is worse than recommending nothing.
    """
    catalog = load_catalog()
    coder = next(m for m in catalog.models if m.tag == "deepseek-coder:6.7b")
    assert coder.roles == []
    assert "0/3" in coder.note


def test_some_model_still_covers_the_coder_role() -> None:
    assert load_catalog().for_role("coder"), "no model can be recommended for coding"
