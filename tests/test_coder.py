"""Coder agent: write, run, and retry on the traceback."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from insightsmith.agents.coder import CoderAgent, extract_code
from insightsmith.config import load_config
from insightsmith.errors import ProviderError
from insightsmith.execution.sandbox import Limits
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
def data(tmp_path: Path):
    path = tmp_path / "sales.csv"
    path.write_text("region,revenue\nnorth,120\nsouth,80\neast,95\nwest,60\n", encoding="utf-8")
    result, sample = profile_with_sample(sniff(path))
    return build_card(result, sample), sample


def _agent(tmp_path: Path, *replies: str):
    """A model that returns each reply in turn, recording the prompts it saw."""
    queue = list(replies)
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "show" in str(request.url):
            return httpx.Response(
                200,
                json={"capabilities": ["completion"], "model_info": {"q.context_length": 8192}},
            )
        prompts.append(request.content.decode())
        text = queue.pop(0) if queue else replies[-1]
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": text}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    _OPEN.append(client)
    config = tmp_path / "config.toml"
    config.write_text('[roles]\ncoder = "ollama/test"\n', encoding="utf-8")
    router = Router(config=load_config(config, environ={}))
    router._providers["ollama"] = OllamaProvider(client=client)
    return CoderAgent(router=router, limits=Limits(timeout_seconds=30)), prompts


def _code(source: str, explanation: str = "") -> str:
    return json.dumps({"code": source, "explanation": explanation})


def test_extract_code_strips_a_fence() -> None:
    assert extract_code({"code": "```python\nresult = 1\n```"}) == "result = 1"
    assert extract_code({"code": "result = 1"}) == "result = 1"
    assert extract_code({"code": 42}) == ""


def test_a_working_snippet_returns_its_value(tmp_path: Path, data) -> None:
    card, frame = data
    agent, prompts = _agent(tmp_path, _code("result = float(df['revenue'].sum())", "sums it"))
    answer = agent.answer(card, frame, "total revenue?")
    assert answer.value == 355.0
    assert answer.explanation == "sums it"
    assert len(prompts) == 1


def test_a_frame_answer_comes_back_as_a_frame(tmp_path: Path, data) -> None:
    card, frame = data
    agent, _ = _agent(tmp_path, _code("result = df.sort('revenue', descending=True).head(2)"))
    answer = agent.answer(card, frame, "top two?")
    assert answer.frame is not None
    assert answer.frame.height == 2


def test_a_traceback_is_fed_back_and_the_retry_succeeds(tmp_path: Path, data) -> None:
    """§7: failures are fuel. The second prompt must carry the actual error."""
    card, frame = data
    agent, prompts = _agent(
        tmp_path,
        _code("result = df['nope'].sum()"),
        _code("result = float(df['revenue'].sum())"),
    )
    answer = agent.answer(card, frame, "total?")
    assert answer.value == 355.0
    assert len(prompts) == 2
    assert "nope" in prompts[1], "the retry must show the code that failed"
    assert "--- failure ---" in prompts[1]


def test_refused_code_is_fed_back_without_ever_running(tmp_path: Path, data) -> None:
    card, frame = data
    agent, prompts = _agent(
        tmp_path,
        _code("import os\nresult = os.listdir('/')"),
        _code("result = df.height"),
    )
    answer = agent.answer(card, frame, "how many rows?")
    assert answer.value == 4
    assert "not allowed" in prompts[1]
    assert not answer.attempts[0].ok
    assert answer.attempts[0].refused


def test_it_gives_up_honestly_after_the_attempt_budget(tmp_path: Path, data) -> None:
    card, frame = data
    agent, prompts = _agent(tmp_path, _code("result = df['missing'].sum()"))
    with pytest.raises(ProviderError, match="could not answer after 2 attempt"):
        agent.answer(card, frame, "total?", attempts=2)
    assert len(prompts) == 2


def test_a_reply_with_no_code_is_retried(tmp_path: Path, data) -> None:
    card, frame = data
    agent, _ = _agent(tmp_path, json.dumps({"explanation": "hmm"}), _code("result = df.height"))
    assert agent.answer(card, frame, "rows?").value == 4


def test_approval_can_refuse_and_nothing_runs(tmp_path: Path, data) -> None:
    card, frame = data
    agent, _ = _agent(tmp_path, _code("result = df.height"))
    with pytest.raises(ProviderError, match="not approved"):
        agent.answer(card, frame, "rows?", approve=True, on_code=lambda _: False)


def test_approval_can_accept(tmp_path: Path, data) -> None:
    card, frame = data
    agent, _ = _agent(tmp_path, _code("result = df.height"))
    seen: list[str] = []
    answer = agent.answer(
        card, frame, "rows?", approve=True, on_code=lambda c: seen.append(c) or True
    )
    assert answer.value == 4
    assert "df.height" in seen[0]


def test_the_model_is_shown_the_card_and_not_the_rows(tmp_path: Path) -> None:
    """The privacy guarantee has to hold for the coder too, not only ideation."""
    path = tmp_path / "people.csv"
    path.write_text(
        "customer_name,email,spend\nAda Lovelace,ada@example.com,120\n"
        "Alan Turing,alan@example.com,80\n",
        encoding="utf-8",
    )
    result, sample = profile_with_sample(sniff(path))
    card = build_card(result, sample)

    agent, prompts = _agent(tmp_path, _code("result = float(df['spend'].sum())"))
    agent.answer(card, sample, "total spend?")

    body = "\n".join(prompts)
    for leaked in ("Ada Lovelace", "ada@example.com", "Alan Turing"):
        assert leaked not in body, f"{leaked} was sent to the model"


def test_the_sandbox_sees_the_real_data_even_though_the_model_does_not(
    tmp_path: Path, data
) -> None:
    """The card is masked; the frame the snippet runs against is not."""
    card, frame = data
    agent, _ = _agent(tmp_path, _code("result = df['region'].to_list()"))
    answer = agent.answer(card, frame, "regions?")
    assert answer.value == ["north", "south", "east", "west"]


def test_the_prompt_carries_polars_reference_for_the_question(tmp_path: Path, data) -> None:
    """The guide is the cheap half of §7: prevent the wrong API, don't retry it."""
    card, frame = data
    agent, prompts = _agent(tmp_path, _code("result = float(df['revenue'].sum())"))
    agent.answer(card, frame, "what is the total revenue by region?")

    assert "Polars reference" in prompts[0]
    assert "8.1 Basic group_by / agg" in prompts[0]
    assert "`df` is already in memory" in prompts[0]


def test_the_reference_can_be_switched_off(tmp_path: Path, data) -> None:
    card, frame = data
    agent, prompts = _agent(tmp_path, _code("result = float(df['revenue'].sum())"))
    agent.guide = False
    agent.answer(card, frame, "what is the total revenue by region?")

    assert "Polars reference" not in prompts[0]
    assert "Question: what is the total revenue by region?" in prompts[0]


def test_the_retry_retrieves_against_the_failure_too(tmp_path: Path, data) -> None:
    """A traceback names the mistake more sharply than the question ever did."""
    card, frame = data
    agent, prompts = _agent(
        tmp_path,
        _code("result = df.groupby('region').revenue.sum()"),
        _code("result = df.group_by('region').agg(pl.col('revenue').sum())"),
    )
    answer = agent.answer(card, frame, "revenue per region?")

    assert answer.frame is not None
    assert "17 Common Pitfalls and Anti-Patterns" in prompts[1], "the failure was not scored"
    assert "17 Common Pitfalls and Anti-Patterns" not in prompts[0]


def test_the_reference_never_points_the_coder_at_a_file(tmp_path: Path, data) -> None:
    """Reading, charting and installing are excluded: the coder may do none of them."""
    card, frame = data
    agent, prompts = _agent(tmp_path, _code("result = df.height"))
    agent.answer(card, frame, "read the csv and plot revenue over time")

    for banned in ("## 3.2 CSV specifics", "## 15.2 Matplotlib", "## 2 Installation"):
        assert banned not in prompts[0], banned


def test_a_snippet_that_assigns_nothing_is_a_failure_not_an_answer(tmp_path: Path, data) -> None:
    """`result` inside a function is invisible to the runner — and a None answer."""
    card, frame = data
    agent, prompts = _agent(
        tmp_path,
        _code("def total():\n    result = df['revenue'].sum()\n    return result"),
        _code("result = float(df['revenue'].sum())"),
    )
    answer = agent.answer(card, frame, "total revenue?")

    assert answer.value == 355.0
    assert "never assigned `result` at the top level" in prompts[1]
    assert not answer.attempts[0].ok
