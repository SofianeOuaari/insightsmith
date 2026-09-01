"""The critic: what is measured, what is asked, and what the score means.

§8's checks are all arithmetic, and these tests hold them to that. Every case
builds a result that *should* be flagged and one that should not, because a
critic that flags everything is as useless as one that flags nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import polars as pl
import pytest

from insightsmith.agents.critic import CriticAgent
from insightsmith.config import load_config
from insightsmith.critique import (
    MIN_GROUP_ROWS,
    Caveat,
    Severity,
    Verdict,
    confidence_for,
    review,
    verdict_for,
)
from insightsmith.io.sniff import sniff
from insightsmith.llm.ollama import OllamaProvider
from insightsmith.llm.router import Router
from insightsmith.profiling import profile_with_sample

_OPEN: list[httpx.Client] = []


@pytest.fixture(autouse=True)
def _close_clients():
    yield
    while _OPEN:
        _OPEN.pop().close()


def _profile(tmp_path: Path, text: str, name: str = "data.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return profile_with_sample(sniff(path))


@pytest.fixture
def sales(tmp_path: Path):
    rows = "\n".join(f"r{i % 4},{i * 3}.0,{i}.5" for i in range(60))
    return _profile(tmp_path, f"region,revenue,margin\n{rows}\n")


def _codes(caveats: list[Caveat]) -> set[str]:
    return {caveat.code for caveat in caveats}


# --------------------------------------------------------------------------- #
# the measured checks
# --------------------------------------------------------------------------- #


def test_a_clean_answer_is_flagged_for_nothing(sales) -> None:
    """A critic that flags everything is as useless as one that flags nothing."""
    profile, _ = sales
    caveats = review(
        question="total revenue by region",
        code='result = df.group_by("region").agg(pl.col("revenue").sum())',
        profile=profile,
        frame=pl.DataFrame({"region": ["a", "b"], "revenue": [10.0, 20.0]}),
    )
    assert caveats == []


def test_a_sampled_source_says_the_answer_describes_the_sample(tmp_path: Path) -> None:
    """The gap this closes: a stride sample answered as though it were a census."""
    rows = "\n".join(f"r{i},{i}.0" for i in range(400))
    profile, _ = _profile(tmp_path, f"k,v\n{rows}\n")
    profile.estimated = True
    profile.sampled_rows = 200

    caveats = review(question="q", code='pl.col("v").sum()', profile=profile)

    found = next(c for c in caveats if c.code == "sampled-source")
    assert found.severity is Severity.SERIOUS
    assert "200" in found.message and "not the file" in found.message


def test_a_source_of_a_dozen_rows_is_not_a_population(tmp_path: Path) -> None:
    profile, _ = _profile(tmp_path, "k,v\n" + "\n".join(f"a,{i}" for i in range(12)) + "\n")
    caveats = review(question="q", code='pl.col("v").mean()', profile=profile)
    assert "small-source" in _codes(caveats)


def test_groups_of_three_are_called_out(sales) -> None:
    """§8: "is a group with n=3 being described as a trend"."""
    profile, _ = sales
    frame = pl.DataFrame({"region": ["a", "b", "c"], "n": [40, 38, 3], "revenue": [1.0, 2.0, 3.0]})

    caveats = review(question="q", code="x", profile=profile, frame=frame)

    found = next(c for c in caveats if c.code == "tiny-groups")
    assert "1 of 3 groups" in found.message
    assert str(MIN_GROUP_ROWS) in found.message


def test_a_healthy_count_column_is_left_alone(sales) -> None:
    profile, _ = sales
    frame = pl.DataFrame({"region": ["a", "b"], "n": [40, 38]})
    assert "tiny-groups" not in _codes(review(question="q", code="x", profile=profile, frame=frame))


def test_pearson_on_an_outliered_column_is_flagged(tmp_path: Path) -> None:
    """§8's named example, measured off the profile's own outlier counts."""
    body = "\n".join(f"{i},{i}" for i in range(200))
    profile, _ = _profile(tmp_path, f"a,b\n{body}\n900000,3\n900001,4\n900002,5\n")

    caveats = review(
        question="are a and b correlated",
        code='result = df.select(pl.corr("a", "b"))',
        profile=profile,
    )

    found = next(c for c in caveats if c.code == "correlation-outliers")
    assert found.severity is Severity.SERIOUS
    assert "spearman" in found.message


def test_spearman_is_the_answer_so_it_is_not_flagged(tmp_path: Path) -> None:
    body = "\n".join(f"{i},{i}" for i in range(200))
    profile, _ = _profile(tmp_path, f"a,b\n{body}\n900000,3\n900001,4\n900002,5\n")

    caveats = review(
        question="q",
        code='result = df.select(pl.corr("a", "b", method="spearman"))',
        profile=profile,
    )
    assert "correlation-outliers" not in _codes(caveats)


def test_a_mean_on_a_skewed_column_suggests_the_median(tmp_path: Path) -> None:
    body = "\n".join("1" for _ in range(200))
    profile, _ = _profile(tmp_path, f"v\n{body}\n50000\n60000\n70000\n")

    caveats = review(question="q", code='result = df["v"].mean()', profile=profile)

    found = next(c for c in caveats if c.code == "mean-on-skewed")
    assert "median" in found.message


def test_nulls_the_code_never_mentions_are_surfaced(tmp_path: Path) -> None:
    # Real column names on purpose: csv.Sniffer compares field lengths, so a
    # header of "k,v" over values of "a,1" gives it nothing to detect.
    rows = "\n".join(f"r{i % 3},{'' if i % 4 == 3 else i}" for i in range(60))
    profile, _ = _profile(tmp_path, f"region,revenue\n{rows}\n")

    caveats = review(question="q", code='result = df["revenue"].mean()', profile=profile)

    found = next(c for c in caveats if c.code == "unacknowledged-nulls")
    assert "denominator" in found.message


def test_code_that_handles_its_nulls_is_not_lectured(tmp_path: Path) -> None:
    # A blank in the first data row makes header detection ambiguous, so the
    # gaps start further down.
    rows = "\n".join(f"a,{'' if i % 4 == 3 else i}" for i in range(60))
    profile, _ = _profile(tmp_path, f"k,v\n{rows}\n")

    caveats = review(question="q", code='result = df.drop_nulls()["v"].mean()', profile=profile)
    assert "unacknowledged-nulls" not in _codes(caveats)


def test_forty_p_values_in_one_table_is_a_multiple_comparison_problem(sales) -> None:
    """§8: "is a significant p-value the product of 40 untracked comparisons"."""
    profile, _ = sales
    frame = pl.DataFrame(
        {"group": [f"g{i}" for i in range(40)], "p_value": [0.01] * 3 + [0.5] * 37}
    )

    caveats = review(question="q", code="x", profile=profile, frame=frame)

    found = next(c for c in caveats if c.code == "multiple-comparisons")
    assert found.severity is Severity.SERIOUS
    assert "40 p-values" in found.message and "3 fall" in found.message


def test_a_division_by_zero_is_fatal_not_a_number(sales) -> None:
    profile, _ = sales
    frame = pl.DataFrame({"k": ["a"], "ratio": [float("inf")]})

    found = next(
        c
        for c in review(question="q", code="x", profile=profile, frame=frame)
        if c.code == "non-finite"
    )
    assert found.fatal


def test_a_scalar_nan_is_caught_too(sales) -> None:
    profile, _ = sales
    caveats = review(question="q", code="x", profile=profile, value=float("nan"))
    assert "non-finite" in _codes(caveats)


def test_a_trend_needs_more_than_two_points(sales) -> None:
    profile, _ = sales
    frame = pl.DataFrame({"month": [1, 2], "v": [1.0, 2.0]})

    caveats = review(
        question="what is the trend over time?", code="x", profile=profile, frame=frame
    )
    assert "short-trend" in _codes(caveats)


def test_a_column_named_n_does_not_match_every_snippet(tmp_path: Path) -> None:
    """`_named_in` uses the quotes Polars needs, or one-letter names match anything."""
    rows = "\n".join(f"{i},{i}" for i in range(60))
    profile, _ = _profile(tmp_path, f"n,other\n{rows}\n")

    caveats = review(question="q", code="result = df.mean()", profile=profile)
    assert "mean-on-skewed" not in _codes(caveats)


# --------------------------------------------------------------------------- #
# verdict and confidence
# --------------------------------------------------------------------------- #


def test_the_verdict_separates_a_wrong_answer_from_a_shaky_one() -> None:
    shaky = [Caveat("small-source", Severity.WARNING, "…")]
    broken = [Caveat("non-finite", Severity.SERIOUS, "…", fatal=True)]

    assert verdict_for([], None) is Verdict.SOUND
    assert verdict_for(shaky, True) is Verdict.QUALIFIED
    assert verdict_for(broken, True) is Verdict.UNSOUND
    assert verdict_for([], False) is Verdict.UNSOUND


def test_confidence_falls_monotonically_as_caveats_accumulate() -> None:
    note = Caveat("a", Severity.NOTE, "…")
    warning = Caveat("b", Severity.WARNING, "…")
    serious = Caveat("c", Severity.SERIOUS, "…")

    assert confidence_for([], True) == 1.0
    scores = [
        confidence_for([note], True),
        confidence_for([warning], True),
        confidence_for([serious], True),
        confidence_for([note, warning, serious], True),
    ]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_answering_a_different_question_costs_more_than_any_caveat() -> None:
    serious = [Caveat("c", Severity.SERIOUS, "…")]
    assert confidence_for([], False) < confidence_for(serious, True)


def test_confidence_is_reproducible() -> None:
    """It is a derived index, not an opinion — the same inputs must score alike."""
    caveats = [Caveat("a", Severity.WARNING, "…"), Caveat("b", Severity.NOTE, "…")]
    assert confidence_for(caveats, True) == confidence_for(list(caveats), True)


# --------------------------------------------------------------------------- #
# the one judgement that is asked rather than measured
# --------------------------------------------------------------------------- #


def _critic(tmp_path: Path, *replies: str) -> tuple[CriticAgent, list[str]]:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "show" in str(request.url):
            return httpx.Response(
                200,
                json={"capabilities": ["completion"], "model_info": {"q.context_length": 8192}},
            )
        prompts.append(request.content.decode())
        text = replies[min(len(prompts) - 1, len(replies) - 1)] if replies else "{}"
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": text}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    _OPEN.append(client)
    config = tmp_path / "critic.toml"
    config.write_text('[roles]\ncritic = "ollama/test"\n', encoding="utf-8")
    router = Router(config=load_config(config, environ={}))
    router._providers["ollama"] = OllamaProvider(client=client)
    return CriticAgent(router=router), prompts


def _judgement(answered: bool, reason: str = "") -> str:
    return json.dumps({"answers_the_question": answered, "reason": reason})


def test_the_critic_never_sees_a_result_row(tmp_path: Path, sales) -> None:
    """A result can be `df.head(20)`, which is raw data. Only its shape may go."""
    profile, _ = sales
    frame = pl.DataFrame({"customer": ["Ada Lovelace", "Alan Turing"], "spend": [120.0, 80.0]})
    agent, prompts = _critic(tmp_path, _judgement(True))

    agent.review(
        question="who spent most?", code="result = df.head(2)", profile=profile, frame=frame
    )

    body = "\n".join(prompts)
    for leaked in ("Ada Lovelace", "Alan Turing", "120"):
        assert leaked not in body, f"{leaked} reached the critic"
    assert "customer" in body and "2 row(s)" in body


def test_a_result_that_answers_something_else_is_unsound(tmp_path: Path, sales) -> None:
    profile, _ = sales
    agent, _ = _critic(tmp_path, _judgement(False, "it totals revenue; the question asked a rate"))

    critique = agent.review(
        question="what is the revenue growth rate?",
        code='result = df["revenue"].sum()',
        profile=profile,
        value=1234.0,
    )

    assert critique.verdict is Verdict.UNSOUND
    assert critique.caveats[0].code == "wrong-question"
    assert "rate" in critique.caveats[0].message
    assert critique.confidence < 0.3


def test_an_unreachable_critic_does_not_become_approval(tmp_path: Path, sales) -> None:
    """Silence is not a verdict of "fine" — the measured caveats still stand."""
    profile, _ = sales
    agent, _ = _critic(tmp_path, "not json at all")
    profile.estimated = True
    profile.sampled_rows = 10

    critique = agent.review(question="q", code="x", profile=profile)

    assert critique.answered is None
    assert "sampled-source" in {caveat.code for caveat in critique.caveats}
    assert critique.verdict is Verdict.QUALIFIED


def test_the_model_can_be_left_out_entirely(tmp_path: Path, sales) -> None:
    profile, _ = sales
    agent, prompts = _critic(tmp_path, _judgement(True))
    agent.consult_model = False

    critique = agent.review(question="q", code="x", profile=profile)

    assert prompts == [], "no model should have been consulted"
    assert critique.verdict is Verdict.SOUND
    assert critique.confidence == 1.0


def test_caveat_prose_agrees_in_number(tmp_path: Path) -> None:
    """These sentences are the deliverable, so they have to read as English."""
    body = "\n".join(f"{i},{i}" for i in range(200))
    spikes = "\n".join(f"{900000 + i},{900000 + i}" for i in range(3))
    profile, _ = _profile(tmp_path, f"alpha,beta\n{body}\n{spikes}\n")

    found = next(
        c
        for c in review(question="q", code='df.select(pl.corr("alpha", "beta"))', profile=profile)
        if c.code == "correlation-outliers"
    )
    assert "alpha and beta carry outliers" in found.message
