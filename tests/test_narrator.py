"""Saying what the numbers mean, and not saying more than they support."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import polars as pl
import pytest

from insightsmith.agents.narrator import NarratorAgent, readable
from insightsmith.config import load_config
from insightsmith.critique import Caveat, Critique, Severity, Verdict
from insightsmith.llm.ollama import OllamaProvider
from insightsmith.llm.router import Router

_OPEN: list[httpx.Client] = []


@pytest.fixture(autouse=True)
def _close_clients():
    yield
    while _OPEN:
        _OPEN.pop().close()


def _narrator(tmp_path: Path, reply: str) -> tuple[NarratorAgent, list[str]]:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "show" in str(request.url):
            return httpx.Response(
                200,
                json={"capabilities": ["completion"], "model_info": {"q.context_length": 8192}},
            )
        prompts.append(request.content.decode())
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": reply}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    _OPEN.append(client)
    config = tmp_path / "config.toml"
    config.write_text('[roles]\ncheap = "ollama/test"\n', encoding="utf-8")
    router = Router(config=load_config(config, environ={}))
    router._providers["ollama"] = OllamaProvider(client=client)
    return NarratorAgent(router=router), prompts


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (6.647887323943662, "6.648"),
        (7.0, "7"),
        (1234567.0, "1,234,567"),
        (42, "42"),
        (None, "None"),
        ("Engineer", "Engineer"),
        (float("nan"), "nan"),
    ],
)
def test_numbers_are_written_the_way_a_person_writes_them(value, expected) -> None:
    """Fifteen digits of precision the data never had is not a number, it is noise."""
    assert readable(value) == expected


def test_the_headline_and_the_detail_are_separate_sentences(tmp_path: Path) -> None:
    """Otherwise they run together: "a gap of 4.4 points The table shows"."""
    agent, _ = _narrator(
        tmp_path, json.dumps({"headline": "Engineers sleep best at 8.4", "detail": "The rest."})
    )
    story = agent.narrate("q", frame=pl.DataFrame({"k": ["a"], "v": [1.0]}))

    assert story == "Engineers sleep best at 8.4. The rest."


def test_an_existing_full_stop_is_not_doubled(tmp_path: Path) -> None:
    agent, _ = _narrator(tmp_path, json.dumps({"headline": "Engineers win.", "detail": "Yes."}))
    assert agent.narrate("q", frame=pl.DataFrame({"k": ["a"], "v": [1.0]})) == "Engineers win. Yes."


def test_the_caveats_travel_with_the_numbers(tmp_path: Path) -> None:
    """A narration that ignores a known caveat is worse than no narration."""
    agent, prompts = _narrator(tmp_path, json.dumps({"headline": "h", "detail": "d"}))
    critique = Critique(
        verdict=Verdict.QUALIFIED,
        caveats=[Caveat("mean-on-skewed", Severity.WARNING, "revenue is heavily skewed")],
    )

    agent.narrate("q", frame=pl.DataFrame({"k": ["a"], "v": [1.0]}), critique=critique)

    assert "heavily skewed" in prompts[0]


def test_personal_data_in_a_result_is_masked_before_it_is_narrated(tmp_path: Path) -> None:
    """A snippet may assign `df.head(20)`, which is raw records under a new name."""
    agent, prompts = _narrator(tmp_path, json.dumps({"headline": "h", "detail": "d"}))
    frame = pl.DataFrame({"email": ["ada@example.com", "alan@example.com"], "spend": [120.0, 80.0]})

    agent.narrate("who spent most?", frame=frame)

    assert "ada@example.com" not in prompts[0]
    assert "alan@example.com" not in prompts[0]


def test_a_long_result_is_capped(tmp_path: Path) -> None:
    agent, prompts = _narrator(tmp_path, json.dumps({"headline": "h", "detail": "d"}))
    frame = pl.DataFrame({"k": [f"r{i}" for i in range(200)], "v": [float(i) for i in range(200)]})

    agent.narrate("q", frame=frame)

    assert "and 175 more rows" in prompts[0]
    assert "r199" not in prompts[0]


def test_nothing_to_narrate_asks_nothing(tmp_path: Path) -> None:
    agent, prompts = _narrator(tmp_path, json.dumps({"headline": "h"}))

    assert agent.narrate("q") == ""
    assert agent.narrate("q", frame=pl.DataFrame({"k": []})) == ""
    assert prompts == []


def test_a_narration_that_fails_costs_the_reader_nothing(tmp_path: Path) -> None:
    """It is a courtesy on top of an answer they already have."""
    agent, _ = _narrator(tmp_path, "not json at all")

    assert agent.narrate("q", frame=pl.DataFrame({"k": ["a"], "v": [1.0]})) == ""
