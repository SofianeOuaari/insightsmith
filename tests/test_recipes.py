"""The recipe books: worked answers by question shape, one per dataframe API.

The guides are API reference, which answers "what does group_by do". These
answer "here is the whole snippet for a question of this shape", which is a
different kind of help and the one a model short of composition needs.
"""

from __future__ import annotations

import pytest

from insightsmith.agents.coder import RECIPE_BUDGET, CoderAgent
from insightsmith.engine import Engine, spec_for
from insightsmith.knowledge import retrieve, sections
from insightsmith.llm.router import Router


def test_every_engine_has_a_recipe_book_that_parses() -> None:
    for engine in Engine:
        book = sections(spec_for(engine).recipes)
        assert len(book) >= 10, engine
        assert all(section.body.strip() for section in book)


def test_pandas_and_fireducks_share_one_book() -> None:
    """They share an API, so two files would only be a copy waiting to drift."""
    assert spec_for(Engine.PANDAS).recipes == spec_for(Engine.FIREDUCKS).recipes
    assert spec_for(Engine.POLARS).recipes != spec_for(Engine.PANDAS).recipes


def test_each_book_is_written_in_its_own_dialect() -> None:
    polars = "\n".join(s.body for s in sections(spec_for(Engine.POLARS).recipes))
    pandas = "\n".join(s.body for s in sections(spec_for(Engine.PANDAS).recipes))

    assert "group_by(" in polars and "pl.col(" in polars
    assert "groupby(" in pandas and "pl.col(" not in pandas


def test_the_two_books_cover_the_same_shapes() -> None:
    """One catalogue, two renderings: a shape missing from one is a gap."""
    polars = [s.title for s in sections(spec_for(Engine.POLARS).recipes)]
    pandas = [s.title for s in sections(spec_for(Engine.PANDAS).recipes)]

    assert polars == pandas


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which Sub Grade has the highest ratio of balance to credit limit?", "Ratio"),
        ("total revenue by region", "Total"),
        ("monthly revenue trend", "over time"),
        ("how many nulls are in each column", "Missing"),
        ("is there a correlation between spend and conversions", "Correlation"),
        ("what percentage of revenue comes from each channel", "Share"),
        ("find the outliers in the amount column", "Outliers"),
        # The eight shapes a harvest of 179 ideation questions found uncovered.
        ("what is the churn rate by region", "Rate of a yes or no"),
        ("which cities have the highest average price", "Top N groups"),
        ("average satisfaction by age group", "cut into bands"),
        ("what is the distribution of annual salaries", "Distribution of a numeric"),
        ("how do impressions vary by channel and quarter", "two categories"),
        ("most common waiting time across departments", "Most common"),
        ("how does occupation affect stress levels", "affect another"),
        ("is the difference between variant and control real", "difference between two groups"),
    ],
)
def test_a_question_finds_the_shape_that_answers_it(question: str, expected: str) -> None:
    hits = retrieve(question, limit=1, guide=spec_for(Engine.POLARS).recipes, by_shape=True)

    assert hits and expected.lower() in hits[0].title.lower(), (question, [h.title for h in hits])


def test_a_recipe_is_matched_on_its_shape_not_its_prose() -> None:
    """The bug this guards, found by harvesting real ideation questions.

    BM25 rewards rare terms, and the rarest words in a recipe book are the
    incidental nouns in its examples. The correlation recipe happened to say
    "Reporting Spearman beside Pearson costs nothing", and that stray verb was
    enough to win every question containing the word *cost* — including plain
    group-by-means, the most common question there is.
    """
    book = spec_for(Engine.POLARS).recipes
    question = "What is the average cost for each region?"

    by_prose = retrieve(question, limit=1, guide=book)
    by_shape = retrieve(question, limit=1, guide=book, by_shape=True)

    assert by_prose[0].title == "Correlation between two numeric columns"
    assert by_shape[0].title == "Average of a measure by category"


def test_every_recipe_is_its_own_best_match() -> None:
    """A new recipe must cover new ground, not compete with a neighbour.

    Two exceptions are allowed and named: both are one-word queries between two
    recipes that are genuinely both about categories, where no lexical score can
    separate them. Anything else means a shape was added twice.
    """
    book = spec_for(Engine.POLARS).recipes
    strays = []
    for recipe in sections(book):
        asks = next(line for line in recipe.body.splitlines() if line.startswith("Questions like:"))
        for ask in asks.removeprefix("Questions like:").rstrip(".").split(";"):
            hits = retrieve(ask, limit=1, guide=book, by_shape=True)
            if not hits or hits[0].title != recipe.title:
                strays.append((recipe.title, ask.strip()))

    assert [ask for _, ask in strays] == [
        "how many of each category",
        "sales by region and product type",
    ], strays


def test_the_ratio_recipe_carries_its_correctness_lesson() -> None:
    """The shape that went wrong in the wild: a ratio by category.

    Reviewing a real answer, the arithmetic was right but nothing said whether
    it was the ratio of sums or the mean of per-row ratios, and the winning
    group rested on two rows. A recipe can carry that where reference cannot.
    """
    ratio = next(s for s in sections(spec_for(Engine.POLARS).recipes) if "Ratio" in s.title)

    assert "ratio of the sums" in ratio.body
    assert "mean of the per-row ratios" in ratio.body
    assert "pl.len()" in ratio.body, "it should keep the group size"
    assert "> 0" in ratio.body, "and guard the zero denominator"


def test_the_new_shapes_carry_their_correctness_lessons() -> None:
    """A recipe exists for the trap it teaches, not for the syntax it shows."""
    book = {s.title: s.body for s in sections(spec_for(Engine.POLARS).recipes)}

    rate = book["Rate of a yes or no outcome by category"]
    assert "not a count" in rate, "a rate is a mean of a flag, not a row count"
    assert "pl.len()" in rate and ">= 30" in rate, "and it must carry and guard n"

    top = book["Top N groups by an aggregate"]
    assert "not `sort().head()`" in top, "aggregate first, then sort"

    bands = book["A measure by a numeric column cut into bands"]
    assert "one group per distinct value" in bands
    assert "moving them can move the conclusion" in bands, "bands are a choice"

    spread = book["Distribution of a numeric column"]
    assert "not `value_counts`" in spread
    assert "median beside the mean" in spread, "so skew is visible"

    real = book["Whether a difference between two groups is real"]
    assert "straddling zero" in real, "an interval, not a verdict"
    assert "one crossing the line" in real, "and the multiple-comparisons warning"

    affect = book["Does one column affect another"]
    assert "causal word" in affect, "the narrator is forbidden to explain causes"


def test_recipes_are_on_by_default() -> None:
    """Measured on qwen3:8b: 15/15 against 13/15, 1.00 attempts against 1.67."""
    assert CoderAgent(router=Router()).recipes is True


def test_a_recipe_reaches_the_prompt_when_switched_on() -> None:
    agent = CoderAgent(router=Router(), recipes=True)

    reference = agent.reference_for("total revenue by region")

    assert "worked answer" in reference
    assert "group_by" in reference


def test_the_recipe_stays_inside_its_budget() -> None:
    """One shape, not a shortlist: a menu invites picking the nearest."""
    agent = CoderAgent(router=Router(), recipes=True, guide=False)

    reference = agent.reference_for("total revenue by region")

    assert len(reference.encode("utf-8")) <= RECIPE_BUDGET + 200
