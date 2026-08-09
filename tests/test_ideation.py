"""Ideation: the column check that kills most hallucination."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from insightsmith.agents.ideation import IdeationAgent, validate_ideas
from insightsmith.config import load_config
from insightsmith.errors import ProviderError
from insightsmith.io.sniff import sniff
from insightsmith.llm.ollama import OllamaProvider
from insightsmith.llm.router import Router
from insightsmith.profiling import profile_with_sample
from insightsmith.profiling.card import build_card

_OPEN: list[httpx.Client] = []


@pytest.fixture(autouse=True)
def _close_clients():
    yield
    while _OPEN:
        _OPEN.pop().close()


@pytest.fixture
def card(samples: dict[str, Path]):
    result, sample = profile_with_sample(sniff(samples["csv"]))
    return build_card(result, sample)


def _agent(tmp_path: Path, reply: str) -> IdeationAgent:
    """An agent wired to a model that always answers `reply`."""
    show = {"capabilities": ["completion"], "model_info": {"q.context_length": 8192}}

    def handler(request: httpx.Request) -> httpx.Response:
        if "show" in str(request.url):
            return httpx.Response(200, json=show)
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": reply}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    _OPEN.append(client)
    config_path = tmp_path / "config.toml"
    config_path.write_text('[roles]\nplanner = "ollama/test"\n', encoding="utf-8")
    router = Router(config=load_config(config_path, environ={}))
    router._providers["ollama"] = OllamaProvider(client=client)
    return IdeationAgent(router=router)


def _idea(**overrides: Any) -> dict[str, Any]:
    base = {
        "question": "Which region earns most?",
        "rationale": "revenue varies by region",
        "method": "group by region, sum revenue",
        "columns": ["region", "revenue"],
        "expected_artifact": "chart",
        "effort": "low",
    }
    return {**base, **overrides}


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def test_valid_ideas_are_kept_and_ranked(card) -> None:
    ideas = validate_ideas({"ideas": [_idea(), _idea(question="Trend over units?")]}, card)
    assert [idea.rank for idea in ideas] == [1, 2]
    assert ideas[0].expected_artifact == "chart"


def test_an_invented_column_disqualifies_the_idea(card) -> None:
    """The check §6 calls cheap validation that kills most hallucination."""
    payload = {
        "ideas": [
            _idea(columns=["customer_lifetime_value"]),
            _idea(question="real one", columns=["region"]),
        ]
    }
    ideas = validate_ideas(payload, card)
    assert len(ideas) == 1
    assert ideas[0].question == "real one"


def test_a_partially_invented_column_list_is_also_rejected(card) -> None:
    ideas = validate_ideas({"ideas": [_idea(columns=["region", "not_a_column"])]}, card)
    assert ideas == []


def test_ideas_without_columns_are_rejected(card) -> None:
    assert validate_ideas({"ideas": [_idea(columns=[])]}, card) == []


def test_ideas_without_a_question_are_rejected(card) -> None:
    assert validate_ideas({"ideas": [_idea(question="  ")]}, card) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"ideas": [_idea()]},
        [_idea()],
        {"ideas": '[{"question": "Which region earns most?", "columns": ["region"]}]'},
    ],
    ids=["wrapped", "bare-list", "json-in-a-string"],
)
def test_the_envelope_models_actually_return_is_tolerated(payload, card) -> None:
    """Rejecting a good answer over its wrapper is its own kind of failure."""
    assert validate_ideas(payload, card)


def test_unknown_enum_values_fall_back_rather_than_dropping_the_idea(card) -> None:
    ideas = validate_ideas(
        {"ideas": [_idea(expected_artifact="interpretive dance", effort="enormous")]}, card
    )
    assert ideas[0].expected_artifact == "table"
    assert ideas[0].effort == "medium"


def test_the_limit_is_honoured(card) -> None:
    payload = {"ideas": [_idea(question=f"q{i}") for i in range(20)]}
    assert len(validate_ideas(payload, card, limit=3)) == 3


def test_junk_yields_nothing(card) -> None:
    assert validate_ideas({"ideas": ["a string", 42, None]}, card) == []


# --------------------------------------------------------------------------- #
# the agent
# --------------------------------------------------------------------------- #


def test_agent_returns_validated_ideas(tmp_path: Path, card) -> None:
    reply = '{"ideas": [{"question": "Which region earns most?", "rationale": "r", '
    reply += '"method": "group by", "columns": ["region", "revenue"], '
    reply += '"expected_artifact": "chart", "effort": "low"}]}'
    ideas = _agent(tmp_path, reply).propose(card)
    assert len(ideas) == 1
    assert ideas[0].columns == ["region", "revenue"]


def test_agent_survives_a_fenced_reply(tmp_path: Path, card) -> None:
    reply = (
        'Sure!\n```json\n{"ideas": [{"question": "Q", "columns": ["region"], '
        '"method": "m", "rationale": "r", "expected_artifact": "table", "effort": "low"}]}\n```'
    )
    assert _agent(tmp_path, reply).propose(card)


def test_agent_fails_honestly_when_every_idea_is_hallucinated(tmp_path: Path, card) -> None:
    reply = '{"ideas": [{"question": "Q", "columns": ["invented"], "method": "m", '
    reply += '"rationale": "r", "expected_artifact": "table", "effort": "low"}]}'
    with pytest.raises(ProviderError, match="not in the dataset"):
        _agent(tmp_path, reply).propose(card)


def test_the_agent_sends_the_card_and_never_the_rows(tmp_path: Path, samples) -> None:
    """The guarantee that makes local_only meaningful: no raw records on the wire."""
    sent: list[str] = []
    show = {"capabilities": ["completion"], "model_info": {"q.context_length": 8192}}

    def handler(request: httpx.Request) -> httpx.Response:
        if "show" in str(request.url):
            return httpx.Response(200, json=show)
        sent.append(request.content.decode())
        body = '{"ideas": [{"question": "Q", "columns": ["region"], "method": "m", '
        body += '"rationale": "r", "expected_artifact": "table", "effort": "low"}]}'
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": body}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    _OPEN.append(client)

    people = tmp_path / "people.csv"
    people.write_text(
        "customer_name,email,region,spend\n"
        "Ada Lovelace,ada@example.com,north,120\n"
        "Alan Turing,alan@example.com,south,80\n",
        encoding="utf-8",
    )
    result, sample = profile_with_sample(sniff(people))
    built = build_card(result, sample)

    config_path = tmp_path / "config.toml"
    config_path.write_text('[roles]\nplanner = "ollama/test"\n', encoding="utf-8")
    router = Router(config=load_config(config_path, environ={}))
    router._providers["ollama"] = OllamaProvider(client=client)
    IdeationAgent(router=router).propose(built)

    payload = "\n".join(sent)
    assert payload, "nothing was sent"
    for leaked in ("Ada Lovelace", "ada@example.com", "Alan Turing"):
        assert leaked not in payload, f"{leaked} was sent to the model"
