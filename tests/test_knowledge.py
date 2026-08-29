"""The bundled Polars guide, and picking the few sections a question needs."""

from __future__ import annotations

import re

import pytest

from insightsmith.knowledge import CODER_EXCLUDES, GUIDE_FILE, reference, retrieve, sections
from insightsmith.knowledge.guide import guide_text
from insightsmith.knowledge.retrieve import stem, tokenize


def _top_level_title(number: str) -> str:
    """A top-level section's title, whether or not it has a body of its own."""
    for section in sections():
        if section.number == number:
            return section.title
        if section.number.startswith(f"{number}."):
            return section.parent
    raise AssertionError(f"the guide has no section {number}")


# --------------------------------------------------------------------------- guide


def test_the_guide_ships_as_package_data() -> None:
    """Read through importlib, which is what an installed wheel has to satisfy."""
    text = guide_text()
    assert GUIDE_FILE.endswith(".md")
    assert text.startswith("# 1. Core Concepts")
    assert len(text) > 20_000


def test_every_section_has_a_number_a_title_and_a_body() -> None:
    found = sections()
    assert len(found) > 50
    for section in found:
        assert re.fullmatch(r"\d{1,2}(\.\d{1,2})?", section.number), section.number
        assert section.title.strip()
        assert section.body.strip(), f"{section.number} is an empty heading"


def test_section_numbers_are_unique() -> None:
    numbers = [section.number for section in sections()]
    assert len(numbers) == len(set(numbers))


def test_a_subsection_carries_its_parent_for_scoring_only() -> None:
    """8.3 never says "aggregation"; its parent does, and that is how it is found."""
    dynamic = next(s for s in sections() if s.number == "8.3")
    assert dynamic.parent == "Aggregations and group_by"
    assert "Aggregations" in dynamic.searchable
    assert "Aggregations" not in dynamic.render()


def test_render_leads_with_the_heading() -> None:
    section = sections()[0]
    assert section.render().startswith(f"## {section.heading}")
    assert section.size_bytes == len(section.render().encode("utf-8"))


def test_code_fences_are_balanced() -> None:
    """An odd fence would spill a section's code into the prose after it."""
    for section in sections():
        assert section.body.count("```") % 2 == 0, section.number


@pytest.mark.parametrize(
    ("number", "title"),
    [
        ("2", "Installation and Environment"),
        ("3", "Reading and Writing Data (I/O)"),
        ("15", "Charting and Visualization"),
        ("16", "End-to-End Worked Case Study"),
    ],
)
def test_the_coder_exclusions_still_name_what_they_meant_to(number: str, title: str) -> None:
    """Pinned so that rebuilding from a new PDF fails here, not silently in a prompt.

    The coder is handed a DataFrame and forbidden to read files or plot, so these
    four sections can only point it somewhere it is not allowed to go.
    """
    assert number in CODER_EXCLUDES
    assert _top_level_title(number) == title


# --------------------------------------------------------------------------- tokens


def test_an_identifier_is_indexed_every_way_it_gets_written() -> None:
    tokens = tokenize("df.group_by('x')")
    assert {"group_by", "group", "by", "groupby"} <= set(tokens)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("outliers", "outlier"),
        ("correlated", "correlation"),
        ("monthly", "month"),
        ("aggregate", "aggregations"),
        ("queries", "query"),
    ],
)
def test_variants_of_a_word_stem_together(left: str, right: str) -> None:
    assert stem(left) == stem(right)


def test_stemming_leaves_short_and_non_plural_words_alone() -> None:
    for word in ("is", "as", "less", "col", "sum", "std"):
        assert stem(word) == word


# --------------------------------------------------------------------------- retrieval


@pytest.mark.parametrize(
    ("question", "number"),
    [
        ("total revenue by region", "8.1"),
        ("rolling 7 day average of sales", "8.4"),
        ("are price and quantity correlated", "13.2"),
        ("find the outliers in the amount column", "13.3"),
        ("join the two tables on customer id", "10.1"),
        ("how many nulls are in each column", "11"),
        ("convert the date string to a date", "7.2"),
        ("rank customers by lifetime spend", "9"),
    ],
)
def test_a_question_reaches_the_section_that_answers_it(question: str, number: str) -> None:
    hits = retrieve(question, limit=3, exclude=CODER_EXCLUDES)
    assert number in [section.number for section in hits], [s.heading for s in hits]


def test_a_pandas_traceback_reaches_the_pandas_pitfalls() -> None:
    """The retry path: the error names the mistake more sharply than the question did."""
    hits = retrieve(
        "AttributeError: 'DataFrame' object has no attribute 'groupby'",
        limit=3,
        exclude=CODER_EXCLUDES,
    )
    assert "17" in [section.number for section in hits], [s.heading for s in hits]


def test_excluding_a_top_level_section_excludes_its_subsections() -> None:
    hits = retrieve("read a csv file with a semicolon separator", limit=8, exclude=("3",))
    assert not [section for section in hits if section.number.split(".")[0] == "3"]
    assert retrieve("read a csv file with a semicolon separator", limit=8)


def test_a_question_of_pure_filler_retrieves_nothing() -> None:
    """Better a shorter prompt than four sections chosen by the word "the"."""
    assert retrieve("") == ()
    assert retrieve("what is it") == ()


def test_ranking_is_deterministic() -> None:
    once = retrieve("average revenue per customer", limit=5)
    twice = retrieve("average revenue per customer", limit=5)
    assert [s.number for s in once] == [s.number for s in twice]


# --------------------------------------------------------------------------- budget


def test_the_reference_stays_inside_its_budget() -> None:
    for budget in (300, 800, 2_000, 4_000):
        block = reference("total revenue by region", budget=budget)
        assert len(block.encode("utf-8")) <= budget, budget


def test_sections_arrive_whole_rather_than_halved() -> None:
    block = reference("total revenue by region", budget=4_000)
    for section in retrieve("total revenue by region", limit=5):
        if section.heading in block:
            assert section.body in block, f"{section.number} was cut in half"


def test_a_single_oversized_section_is_clipped_with_its_fence_closed() -> None:
    """A tight budget still returns something, and never an open code fence."""
    block = reference("group_by_dynamic time bucketed downsampling", budget=400)
    assert block
    assert len(block.encode("utf-8")) <= 400
    assert block.count("```") % 2 == 0


def test_a_budget_of_nothing_returns_nothing() -> None:
    assert reference("total revenue by region", budget=0) == ""
