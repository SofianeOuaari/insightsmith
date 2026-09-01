"""Coder agent: write, run, and retry on the traceback."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from insightsmith.agents.coder import Attempt, CoderAgent, _summarise, _tail, extract_code
from insightsmith.config import load_config
from insightsmith.critique import Verdict
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


def test_the_final_failure_is_one_readable_line_not_a_traceback(tmp_path: Path, data) -> None:
    """A polars stack buries the one line that says what went wrong under twenty."""
    card, frame = data
    agent, _ = _agent(tmp_path, _code("result = pl.DataFrame({'a': [{'b': object()}]})"))

    with pytest.raises(ProviderError) as caught:
        agent.answer(card, frame, "anything?", attempts=1)

    message = str(caught.value)
    assert "\n" not in message
    assert "Traceback (most recent call last)" not in message
    assert "site-packages" not in message
    assert "TypeError" in message, "the exception class must survive whatever follows it"
    assert "snippet.py line" in message, "the reader still needs to know where"


def test_a_refusal_reports_the_reason_rather_than_a_traceback(tmp_path: Path, data) -> None:
    card, frame = data
    agent, _ = _agent(tmp_path, _code("import os\nresult = os.getcwd()"))

    with pytest.raises(ProviderError, match="not allowed"):
        agent.answer(card, frame, "anything?", attempts=1)


def test_the_full_traceback_is_still_kept_for_inspection(tmp_path: Path, data) -> None:
    """One line at the terminal; the whole thing on the attempt."""
    card, frame = data
    agent, _ = _agent(tmp_path, _code("result = df['nope'].sum()"), _code("result = df.height"))
    answer = agent.answer(card, frame, "rows?")

    assert answer.value == 4
    assert "Traceback (most recent call last)" in answer.attempts[0].error


def test_the_exception_survives_context_a_library_appends_after_it() -> None:
    """polars 1.44 adds a context stack and a hint below the exception line.

    A fixed tail window cuts the class out and leaves the reader with only the
    hint, so the summary anchors on the exception instead of counting backwards.
    """
    error = (
        "Traceback (most recent call last):\n"
        '  File "snippet.py", line 1, in <module>\n'
        "TypeError: nested objects are not allowed\n"
        "\n"
        "This error occurred with the following context stack:\n"
        "\t[1] while constructing Series 'a'\n"
        "\n"
        "Hint: Try setting `strict=False` to allow passing data with mixed types."
    )
    summary = _summarise(Attempt(code="x", ok=False, error=error))

    assert summary.startswith("snippet.py line 1: TypeError: nested objects are not allowed")
    assert "while constructing" not in summary, "indented context is still stack, not message"


def test_a_long_traceback_is_cut_on_a_line_boundary() -> None:
    """Cutting mid-line leaves half a frame at column zero, reading as the error."""
    frames = "\n".join(
        f'  File "/very/long/path/to/site-packages/polars/module_{i}.py", line {i}, in f'
        for i in range(60)
    )
    error = f"Traceback (most recent call last):\n{frames}\nValueError: the real problem"

    tail = _tail(error, 1500)
    assert len(tail) <= 1500
    assert all(line.startswith((" ", "V")) for line in tail.splitlines()), tail
    assert _summarise(Attempt(code="x", ok=False, error=tail)) == "ValueError: the real problem"


def test_a_single_line_longer_than_the_budget_still_comes_back() -> None:
    assert _tail("x" * 4000, 100) == "x" * 100


def _critic_router(tmp_path: Path, *judgements: bool):
    """A critic whose model answers each judgement in turn."""
    from insightsmith.agents.critic import CriticAgent

    verdicts = list(judgements)

    def handler(request: httpx.Request) -> httpx.Response:
        if "show" in str(request.url):
            return httpx.Response(
                200,
                json={"capabilities": ["completion"], "model_info": {"q.context_length": 8192}},
            )
        answered = verdicts.pop(0) if verdicts else judgements[-1]
        body = json.dumps(
            {"answers_the_question": answered, "reason": "" if answered else "it totals, not rates"}
        )
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": body}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    _OPEN.append(client)
    config = tmp_path / "critic.toml"
    config.write_text('[roles]\ncritic = "ollama/test"\n', encoding="utf-8")
    router = Router(config=load_config(config, environ={}))
    router._providers["ollama"] = OllamaProvider(client=client)
    return CriticAgent(router=router)


def test_a_snippet_that_answers_the_wrong_question_is_sent_back(tmp_path: Path, data) -> None:
    """§8's critic → retry arrow: the snippet ran, it just answered something else."""
    card, frame = data
    profile, _ = profile_with_sample(sniff(tmp_path / "sales.csv"))
    agent, prompts = _agent(
        tmp_path,
        _code("result = float(df['revenue'].sum())"),
        _code("result = df.group_by('region').agg(pl.col('revenue').sum())"),
    )
    critic = _critic_router(tmp_path, False, True)

    answer = agent.answer(card, frame, "revenue per region?", critic=critic, profile=profile)

    assert answer.frame is not None, "the retry's snippet should be the one kept"
    assert len(prompts) == 2
    assert "does not answer the question" in prompts[1]
    assert "it totals, not rates" in prompts[1]
    assert answer.critique is not None and answer.critique.verdict is not Verdict.UNSOUND


def test_statistical_caveats_never_trigger_a_retry(tmp_path: Path, data) -> None:
    """Rewriting the snippet cannot make the data less skewed, so it rides along."""
    card, frame = data
    profile, _ = profile_with_sample(sniff(tmp_path / "sales.csv"))
    profile.estimated = True
    profile.sampled_rows = 2
    agent, prompts = _agent(tmp_path, _code("result = float(df['revenue'].sum())"))
    critic = _critic_router(tmp_path, True)

    answer = agent.answer(card, frame, "total revenue?", critic=critic, profile=profile)

    assert len(prompts) == 1, "a caveat about the data is not the coder's to fix"
    assert answer.value == 355.0
    assert answer.critique is not None
    assert "sampled-source" in {c.code for c in answer.critique.caveats}
    assert answer.critique.verdict is Verdict.QUALIFIED


def test_the_last_attempt_returns_the_answer_marked_unsound(tmp_path: Path, data) -> None:
    """A number with a loud warning beats an exception and no number at all."""
    card, frame = data
    profile, _ = profile_with_sample(sniff(tmp_path / "sales.csv"))
    agent, _ = _agent(tmp_path, _code("result = float(df['revenue'].sum())"))
    critic = _critic_router(tmp_path, False)

    answer = agent.answer(
        card, frame, "revenue growth rate?", attempts=1, critic=critic, profile=profile
    )

    assert answer.value == 355.0
    assert answer.critique is not None
    assert answer.critique.verdict is Verdict.UNSOUND
    assert answer.critique.caveats[0].code == "wrong-question"


def test_without_a_critic_nothing_changes(tmp_path: Path, data) -> None:
    card, frame = data
    agent, _ = _agent(tmp_path, _code("result = float(df['revenue'].sum())"))
    answer = agent.answer(card, frame, "total revenue?")

    assert answer.value == 355.0
    assert answer.critique is None
