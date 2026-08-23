"""CLI tests, driven through typer's runner so exit codes are real."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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


@pytest.fixture
def stub_ollama(monkeypatch):
    """Replace every Ollama construction with a MockTransport-backed one.

    load_config merges defaults, so roles the test config never mentions still
    resolve to ollama/... — stubbing only the configured role would leave those
    opening real sockets.
    """
    import httpx

    from insightsmith.llm.ollama import OllamaProvider

    clients: list[httpx.Client] = []

    def install(show: dict) -> None:
        client = httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=show))
        )
        clients.append(client)
        monkeypatch.setattr(
            "insightsmith.llm.registry.OllamaProvider", lambda **_: OllamaProvider(client=client)
        )

    yield install
    for client in clients:
        client.close()


def test_models_lists_roles(tmp_path: Path, monkeypatch, stub_ollama) -> None:
    stub_ollama(
        {"capabilities": ["completion", "tools"], "model_info": {"q.context_length": 40960}}
    )
    config = tmp_path / "config.toml"
    config.write_text('[roles]\nplanner = "ollama/qwen3:8b"\n', encoding="utf-8")
    monkeypatch.setenv("INSIGHTSMITH_CONFIG", str(config))

    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "planner" in result.stdout
    assert "local" in result.stdout
    assert "tool calling" in result.stdout


def test_models_json(tmp_path: Path, monkeypatch, stub_ollama) -> None:
    stub_ollama({"capabilities": ["completion"], "model_info": {"q.context_length": 8192}})
    config = tmp_path / "config.toml"
    config.write_text('[roles]\nplanner = "ollama/llama3.2:3b"\n', encoding="utf-8")
    monkeypatch.setenv("INSIGHTSMITH_CONFIG", str(config))

    result = runner.invoke(app, ["models", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    planner = next(r for r in payload["roles"] if r["role"] == "planner")
    # No tool-calling, so it must have chosen a degraded strategy rather than fail.
    assert planner["tool_calling"] is False
    assert planner["strategy"] in {"json_mode", "prompted_json"}


def test_models_reports_local_only_violation_and_exits_nonzero(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[roles]\nplanner = "openai/gpt-4o-mini"\n\n[budget]\nlocal_only = true\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("INSIGHTSMITH_CONFIG", str(config))

    result = runner.invoke(app, ["models"])
    assert result.exit_code == 1
    assert "local_only" in result.output


def test_models_marks_an_unreachable_provider_without_crashing(
    tmp_path: Path, monkeypatch, stub_ollama
) -> None:
    """A missing API key must be reported per role, not abort the whole command."""
    stub_ollama({"capabilities": ["completion"], "model_info": {"q.context_length": 8192}})
    config = tmp_path / "config.toml"
    config.write_text('[roles]\nplanner = "openai/gpt-4o-mini"\n', encoding="utf-8")
    monkeypatch.setenv("INSIGHTSMITH_CONFIG", str(config))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "unreachable" in result.stdout


def test_look_card_shows_what_a_model_would_see(samples: dict[str, Path]) -> None:
    result = runner.invoke(app, ["look", str(samples["csv"]), "--card"])
    assert result.exit_code == 0
    assert "dataset card" in result.stdout
    assert "hash" in result.stdout


def test_look_ideas(tmp_path: Path, monkeypatch, stub_ollama) -> None:
    import httpx

    from insightsmith.llm.ollama import OllamaProvider

    body = '{"ideas": [{"question": "Which region earns most?", "rationale": "r", '
    body += '"method": "group by region", "columns": ["region", "revenue"], '
    body += '"expected_artifact": "chart", "effort": "low"}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        if "show" in str(request.url):
            return httpx.Response(
                200,
                json={"capabilities": ["completion"], "model_info": {"q.context_length": 8192}},
            )
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": body}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        "insightsmith.llm.registry.OllamaProvider", lambda **_: OllamaProvider(client=client)
    )
    config = tmp_path / "config.toml"
    config.write_text('[roles]\nplanner = "ollama/test"\n', encoding="utf-8")
    monkeypatch.setenv("INSIGHTSMITH_CONFIG", str(config))

    data = tmp_path / "sales.csv"
    data.write_text("region,revenue\nnorth,120\nsouth,80\neast,95\nwest,60\n", encoding="utf-8")
    result = runner.invoke(app, ["look", str(data), "--ideas"])
    client.close()

    assert result.exit_code == 0
    assert "analysis ideas" in result.stdout
    assert "Which region earns most?" in result.stdout


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

    monkeypatch.setattr("insightsmith.cli.profile_with_sample", boom)
    path = tmp_path / "book.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    result = runner.invoke(app, ["look", str(path)])
    assert result.exit_code == 1
    assert "insightsmith[excel]" in result.output


def _stub_ideas_provider(monkeypatch, count: int = 12):
    """A model that always offers `count` valid ideas, so the cap is what limits."""
    import httpx

    from insightsmith.llm.ollama import OllamaProvider

    ideas = [
        {
            "question": f"Question {i}?",
            "rationale": "r",
            "method": "m",
            "columns": ["region"],
            "expected_artifact": "table",
            "effort": "low",
        }
        for i in range(count)
    ]
    body = json.dumps({"ideas": ideas})

    def handler(request: httpx.Request) -> httpx.Response:
        if "show" in str(request.url):
            return httpx.Response(
                200,
                json={"capabilities": ["completion"], "model_info": {"q.context_length": 8192}},
            )
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": body}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        "insightsmith.llm.registry.OllamaProvider", lambda **_: OllamaProvider(client=client)
    )
    return client


def _ideas_fixture(tmp_path: Path, monkeypatch) -> Path:
    config = tmp_path / "config.toml"
    config.write_text('[roles]\nplanner = "ollama/test"\n', encoding="utf-8")
    monkeypatch.setenv("INSIGHTSMITH_CONFIG", str(config))
    data = tmp_path / "sales.csv"
    data.write_text("region,revenue\nnorth,120\nsouth,80\neast,95\n", encoding="utf-8")
    return data


def test_max_ideas_caps_how_many_come_back(tmp_path: Path, monkeypatch) -> None:
    client = _stub_ideas_provider(monkeypatch)
    data = _ideas_fixture(tmp_path, monkeypatch)
    result = runner.invoke(app, ["look", str(data), "--ideas", "--max-ideas", "3", "--json"])
    client.close()
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)["ideas"]) == 3


def test_max_ideas_has_a_short_form(tmp_path: Path, monkeypatch) -> None:
    client = _stub_ideas_provider(monkeypatch)
    data = _ideas_fixture(tmp_path, monkeypatch)
    result = runner.invoke(app, ["look", str(data), "-n", "2", "--json"])
    client.close()
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)["ideas"]) == 2


def test_a_count_alone_implies_ideas(tmp_path: Path, monkeypatch) -> None:
    """Otherwise --max-ideas would be accepted and silently ignored."""
    client = _stub_ideas_provider(monkeypatch)
    data = _ideas_fixture(tmp_path, monkeypatch)
    result = runner.invoke(app, ["look", str(data), "--max-ideas", "1", "--json"])
    client.close()
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)["ideas"]) == 1


def test_the_default_is_eight(tmp_path: Path, monkeypatch) -> None:
    client = _stub_ideas_provider(monkeypatch)
    data = _ideas_fixture(tmp_path, monkeypatch)
    result = runner.invoke(app, ["look", str(data), "--ideas", "--json"])
    client.close()
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)["ideas"]) == 8


def test_a_nonsensical_count_is_rejected(tmp_path: Path, monkeypatch) -> None:
    data = _ideas_fixture(tmp_path, monkeypatch)
    result = runner.invoke(app, ["look", str(data), "-n", "0"])
    assert result.exit_code != 0


def test_no_ideas_flag_means_no_model_is_contacted(samples: dict[str, Path]) -> None:
    """Plain `look` must stay offline; the autouse guard fails any real request."""
    result = runner.invoke(app, ["look", str(samples["csv"])])
    assert result.exit_code == 0
    assert "analysis ideas" not in result.stdout


def _stub_coder(monkeypatch, code: str):
    import httpx

    from insightsmith.llm.ollama import OllamaProvider

    body = json.dumps({"code": code, "explanation": "does the thing"})

    def handler(request: httpx.Request) -> httpx.Response:
        if "show" in str(request.url):
            return httpx.Response(
                200,
                json={"capabilities": ["completion"], "model_info": {"q.context_length": 8192}},
            )
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": body}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        "insightsmith.llm.registry.OllamaProvider", lambda **_: OllamaProvider(client=client)
    )
    return client


def _ask_fixture(tmp_path: Path, monkeypatch) -> Path:
    config = tmp_path / "config.toml"
    config.write_text('[roles]\ncoder = "ollama/test"\n', encoding="utf-8")
    monkeypatch.setenv("INSIGHTSMITH_CONFIG", str(config))
    data = tmp_path / "sales.csv"
    data.write_text("region,revenue\nnorth,120\nsouth,80\neast,95\n", encoding="utf-8")
    return data


def test_ask_returns_a_number(tmp_path: Path, monkeypatch) -> None:
    client = _stub_coder(monkeypatch, "result = float(df['revenue'].sum())")
    data = _ask_fixture(tmp_path, monkeypatch)
    result = runner.invoke(app, ["ask", str(data), "total revenue?"])
    client.close()
    assert result.exit_code == 0
    assert "295" in result.stdout
    assert "code" in result.stdout


def test_ask_json(tmp_path: Path, monkeypatch) -> None:
    client = _stub_coder(monkeypatch, "result = float(df['revenue'].sum())")
    data = _ask_fixture(tmp_path, monkeypatch)
    result = runner.invoke(app, ["ask", str(data), "total?", "--json"])
    client.close()
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["value"] == 295.0
    assert payload["attempts"] == 1
    assert "df" in payload["code"]


def test_ask_renders_a_table_for_a_frame(tmp_path: Path, monkeypatch) -> None:
    client = _stub_coder(monkeypatch, "result = df.sort('revenue', descending=True).head(2)")
    data = _ask_fixture(tmp_path, monkeypatch)
    result = runner.invoke(app, ["ask", str(data), "top two?"])
    client.close()
    assert result.exit_code == 0
    assert "north" in result.stdout


def test_ask_can_hide_the_code(tmp_path: Path, monkeypatch) -> None:
    client = _stub_coder(monkeypatch, "result = df.height")
    data = _ask_fixture(tmp_path, monkeypatch)
    result = runner.invoke(app, ["ask", str(data), "rows?", "--no-code"])
    client.close()
    assert result.exit_code == 0
    assert "3" in result.stdout


def test_ask_fails_cleanly_when_the_model_cannot_do_it(tmp_path: Path, monkeypatch) -> None:
    client = _stub_coder(monkeypatch, "result = df['missing'].sum()")
    data = _ask_fixture(tmp_path, monkeypatch)
    result = runner.invoke(app, ["ask", str(data), "impossible?"])
    client.close()
    assert result.exit_code == 1
    assert "could not answer" in result.output
    # The snippet's traceback is shown on purpose — it is the user's window into
    # what the model got wrong. What must not appear is our own stack.
    assert "snippet.py" in result.output
    assert "insightsmith/cli.py" not in result.output
    assert "insightsmith/agents" not in result.output


def test_models_names_the_config_path_when_there_is_none(
    tmp_path: Path, monkeypatch, stub_ollama
) -> None:
    """Nothing creates the file, so the CLI has to say where it goes."""
    from insightsmith.config import DEFAULT_CONFIG_PATH

    stub_ollama({"capabilities": ["completion"], "model_info": {"q.context_length": 8192}})
    monkeypatch.setenv("INSIGHTSMITH_CONFIG", str(tmp_path / "absent.toml"))

    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    flat = " ".join(result.stdout.split())
    assert "none found" in flat
    assert str(DEFAULT_CONFIG_PATH.name) in flat
