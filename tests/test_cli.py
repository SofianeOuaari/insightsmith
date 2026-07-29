"""CLI tests, driven through typer's runner so exit codes are real."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from insightsmith import __version__
from insightsmith.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "look" in result.stdout


def test_look_renders_tables(samples: dict[str, Path]) -> None:
    result = runner.invoke(app, ["look", str(samples["csv"])])
    assert result.exit_code == 0
    assert "csv" in result.stdout
    assert "columns" in result.stdout
    assert "region" in result.stdout


def test_look_surfaces_assumptions_for_the_german_csv(samples: dict[str, Path]) -> None:
    result = runner.invoke(app, ["look", str(samples["german_csv"])])
    assert result.exit_code == 0
    assert "cp1252" in result.stdout


def test_look_reports_quality_notes(samples: dict[str, Path]) -> None:
    result = runner.invoke(app, ["look", str(samples["messy"])])
    assert result.exit_code == 0
    assert "quality notes" in result.stdout


def test_look_json_is_valid_and_complete(samples: dict[str, Path]) -> None:
    result = runner.invoke(app, ["look", str(samples["csv"]), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["n_rows"] == 3
    assert payload["n_columns"] == 3
    assert payload["estimated"] is False
    assert payload["source"]["format"] == "csv"
    assert [c["schema"]["name"] for c in payload["columns"]] == ["region", "units", "revenue"]


def test_look_json_has_no_unserialisable_leftovers(samples: dict[str, Path]) -> None:
    """Enums, Paths and dataclasses must all flatten — no reprs in the output."""
    result = runner.invoke(app, ["look", str(samples["messy"]), "--json"])
    assert "object at 0x" not in result.stdout
    assert "SemanticType." not in result.stdout


def test_unsupported_format_exits_nonzero_with_a_reason(samples: dict[str, Path]) -> None:
    result = runner.invoke(app, ["look", str(samples["sqlite"])])
    assert result.exit_code == 1
    assert "sqlite" in result.output


def test_missing_file_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["look", str(tmp_path / "nope.csv")])
    assert result.exit_code != 0


def test_doctor_runs_on_whatever_machine_it_finds(monkeypatch) -> None:
    """No hardware is touched: the collectors are stubbed, the maths is real."""
    from insightsmith.hardware.probe import CpuInfo, MemoryInfo, SystemInfo

    system = SystemInfo(
        os_name="Linux",
        os_release="6.8.0",
        arch="x86_64",
        cpu=CpuInfo(model="Test CPU", physical_cores=8, logical_cores=16),
        memory=MemoryInfo(total_gb=32.0, available_gb=16.0),
        disk_free_gb=500.0,
    )
    monkeypatch.setattr("insightsmith.cli.probe_system", lambda: system)
    monkeypatch.setattr("insightsmith.cli.detect_accelerators", lambda _: [])
    monkeypatch.setattr("insightsmith.cli.detect_installed_models", list)

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Test CPU" in result.stdout
    assert "none detected" in result.stdout
    assert "kv cache" in result.stdout


def test_doctor_json(monkeypatch) -> None:
    from insightsmith.hardware.probe import CpuInfo, MemoryInfo, SystemInfo

    system = SystemInfo(
        os_name="Linux",
        os_release="6.8.0",
        arch="x86_64",
        cpu=CpuInfo(model="Test CPU", physical_cores=8, logical_cores=16),
        memory=MemoryInfo(total_gb=32.0, available_gb=16.0),
        disk_free_gb=500.0,
    )
    monkeypatch.setattr("insightsmith.cli.probe_system", lambda: system)
    monkeypatch.setattr("insightsmith.cli.detect_accelerators", lambda _: [])
    monkeypatch.setattr("insightsmith.cli.detect_installed_models", list)

    result = runner.invoke(app, ["doctor", "--json", "--context", "4096"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["context"] == 4096
    assert payload["system"]["cpu"]["model"] == "Test CPU"
    assert payload["recommendations"]
    fit = payload["recommendations"][0]["fit"]
    assert fit["weights_gb"] > 0
    assert fit["kv_cache_gb"] > 0


def test_doctor_says_unknown_rather_than_inventing_a_throughput(monkeypatch) -> None:
    """An unlisted device must print "unknown", never a plausible-looking number."""
    from insightsmith.hardware.accel import Accelerator, Vendor
    from insightsmith.hardware.probe import CpuInfo, MemoryInfo, SystemInfo

    system = SystemInfo(
        os_name="Linux",
        os_release="6.8.0",
        arch="x86_64",
        cpu=CpuInfo(model="Test CPU", physical_cores=8, logical_cores=16),
        memory=MemoryInfo(total_gb=32.0, available_gb=16.0),
        disk_free_gb=500.0,
    )
    unlisted = Accelerator(
        vendor=Vendor.NVIDIA, name="NVIDIA RTX A2000 Laptop GPU", memory_total_gb=24.0
    )
    monkeypatch.setattr("insightsmith.cli.probe_system", lambda: system)
    monkeypatch.setattr("insightsmith.cli.detect_accelerators", lambda _: [unlisted])
    monkeypatch.setattr("insightsmith.cli.detect_installed_models", list)

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "unknown" in result.stdout
    # rich wraps the disclaimer, so compare on collapsed whitespace.
    assert "not measured on this machine" in " ".join(result.stdout.split())


def test_unparseable_content_fails_cleanly_not_with_a_traceback(tmp_path: Path) -> None:
    """Text that sniffs as CSV but will not parse must not dump a polars stack."""
    path = tmp_path / "script.sh"
    path.write_text(
        'case "$msg" in\n"a quoted thing"*)\n  echo one\n  ;;\nesac\n', encoding="utf-8"
    )
    result = runner.invoke(app, ["look", str(path)])
    assert result.exit_code == 1
    assert "could not parse" in result.output
    assert "Traceback" not in result.output
    assert "confidence" in result.output


def test_extra_is_named_in_the_missing_dependency_message(monkeypatch, tmp_path: Path) -> None:
    """rich must not eat the `[excel]` in the suggested pip command."""
    from insightsmith.errors import MissingDependencyError

    def boom(*_: object, **__: object) -> None:
        raise MissingDependencyError("fastexcel", "excel", purpose="reading spreadsheets")

    monkeypatch.setattr("insightsmith.cli.profile", boom)
    path = tmp_path / "book.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    result = runner.invoke(app, ["look", str(path)])
    assert result.exit_code == 1
    assert "insightsmith[excel]" in result.output
