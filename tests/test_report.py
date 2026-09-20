"""Forging a run into something you can hand to someone else.

The load-bearing assertions here are not about formatting. They are that the
code, the card hash and the caveats survive every rendering: a report that
shows a number and drops what qualifies it would undo 0.7.0 on the way out.
"""

from __future__ import annotations

import html
import json
import pathlib
from pathlib import Path

import polars as pl
import pytest

from insightsmith.critique import Caveat, Critique, Verdict
from insightsmith.errors import MissingDependencyError
from insightsmith.io.sniff import sniff
from insightsmith.profiling import Profile, profile_with_sample
from insightsmith.profiling.quality import Severity
from insightsmith.report import (
    Finding,
    Report,
    render_html,
    render_markdown,
    render_notebook,
    render_pdf,
)
from insightsmith.report.builder import MAX_TABLE_ROWS

CODE = 'result = df.group_by("region").agg(pl.col("amount").sum())'


def carries(page: str, text: str) -> bool:
    """True when ``text`` reached the page, escaped or not.

    The HTML template autoescapes, which is the point: a model wrote the code
    and the question, so neither is trusted markup. That turns the quotes in a
    snippet into entities — markupsafe's ``&#34;``, not ``html.escape``'s
    ``&quot;`` — so the page is unescaped rather than the needle escaped.
    """
    return text in html.unescape(page)


CAVEAT = "3 group(s) rest on fewer than 30 rows."


@pytest.fixture
def profile(tmp_path: Path) -> Profile:
    source = tmp_path / "sales.csv"
    source.write_text("region,amount\nN,10\nS,20\nE,30\n", encoding="utf-8")
    return profile_with_sample(sniff(source))[0]


@pytest.fixture
def report(profile: Profile) -> Report:
    critique = Critique(
        verdict=Verdict.QUALIFIED,
        caveats=[Caveat(code="tiny-groups", severity=Severity.WARNING, message=CAVEAT)],
        confidence=0.62,
    )
    return Report(
        source=profile.source.path,
        profile=profile,
        findings=[
            Finding(
                question="Which region sells most?",
                code=CODE,
                narrative="North leads on total amount.",
                frame=pl.DataFrame({"region": ["N", "S"], "total": [1200.5, 900.25]}),
                critique=critique,
                attempts=2,
            ),
            Finding(
                question="How many orders?", code="result = df.height", kind="value", value=1043
            ),
        ],
        unanswered=["What drives returns?"],
        card_hash="ab12cd34",
        engine="polars",
        models={"coder": "ollama:qwen3:8b"},
    )


@pytest.mark.parametrize("render", [render_markdown, render_html])
def test_every_rendering_carries_the_code_the_hash_and_the_caveat(render, report: Report) -> None:
    """§11.4: a result is auditable or it is hearsay."""
    page = render(report)

    assert carries(page, CODE), "the code is the claim; the number is only its output"
    assert carries(page, "ab12cd34"), "the hash ties the answer to a state of the file"
    assert carries(page, CAVEAT), "a caveat dropped on the way out undoes the critic"
    assert "ollama:qwen3:8b" in page


@pytest.mark.parametrize("render", [render_markdown, render_html])
def test_every_rendering_states_its_own_limits(render, report: Report) -> None:
    """§12 belongs in the deliverable, which is the thing that gets forwarded."""
    page = render(report)

    assert "wrong code confidently" in page
    assert "not a probability" in page


@pytest.mark.parametrize("render", [render_markdown, render_html])
def test_unanswered_questions_are_named_not_hidden(render, report: Report) -> None:
    """A reader must not read five findings as the whole of what was asked."""
    assert "What drives returns?" in render(report)


def test_a_sampled_profile_says_so(profile: Profile, report: Report) -> None:
    object.__setattr__(profile, "estimated", True)
    object.__setattr__(profile, "sampled_rows", 1000)

    assert "sampled rows" in render_markdown(report)
    assert "estimate" in render_html(report)


def test_a_long_table_is_cut_and_the_cut_is_declared(profile: Profile) -> None:
    """Silently truncating a table is how a report tells a lie by omission."""
    frame = pl.DataFrame({"n": list(range(MAX_TABLE_ROWS + 40))})
    finding = Finding(question="everything", code="result = df", frame=frame)

    _columns, rows, dropped = finding.table()

    assert len(rows) == MAX_TABLE_ROWS
    assert dropped == 40
    page = render_markdown(Report(source=profile.source.path, profile=profile, findings=[finding]))
    assert "40 further row(s) not shown" in page


def test_a_finding_without_a_critique_claims_no_confidence(profile: Profile) -> None:
    """--no-critique must read as "not checked", never as "checked and fine"."""
    finding = Finding(question="q", code="result = 1", value=1)

    assert finding.confidence is None
    assert finding.caveats == []
    assert "confidence" not in render_html(
        Report(source=profile.source.path, profile=profile, findings=[finding])
    )


def test_the_html_escapes_what_the_model_wrote(profile: Profile) -> None:
    """The question and the code reach the page from a model, so neither is trusted."""
    finding = Finding(
        question="<script>alert(1)</script>",
        code="result = '</pre><script>alert(2)</script>'",
    )

    page = render_html(Report(source=profile.source.path, profile=profile, findings=[finding]))

    assert "<script>alert(1)</script>" not in page
    assert "<script>alert(2)</script>" not in page
    assert "&lt;script&gt;" in page


def test_the_notebook_is_valid_nbformat(report: Report) -> None:
    document = json.loads(render_notebook(report))

    assert document["nbformat"] == 4
    assert document["metadata"]["kernelspec"]["name"] == "python3"
    kinds = [cell["cell_type"] for cell in document["cells"]]
    assert kinds[0] == "markdown" and kinds[1] == "code", "prose, then the setup cell"
    for cell in document["cells"]:
        assert isinstance(cell["source"], list), "nbformat wants a list of lines"
        assert all(isinstance(line, str) for line in cell["source"])


def test_the_notebook_runs(tmp_path: Path) -> None:
    """ "Export a runnable notebook" is testable, so it is tested rather than claimed."""
    source = tmp_path / "sales.csv"
    source.write_text("region,amount\nN,10\nS,20\nE,30\n", encoding="utf-8")
    profile = profile_with_sample(sniff(source))[0]
    report = Report(
        source=source,
        profile=profile,
        findings=[Finding(question="How many rows?", code="result = df.height")],
    )

    scope: dict[str, object] = {}
    for cell in json.loads(render_notebook(report))["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "cell", "exec"), scope)  # noqa: S102

    assert scope["result"] == 3, "the setup cell must rebuild df as the sandbox had it"


def test_the_notebook_warns_when_the_run_was_on_a_sample(report: Report) -> None:
    """Re-running a sampled analysis on the whole file gives different numbers."""
    object.__setattr__(report.profile, "estimated", True)

    setup = "".join(json.loads(render_notebook(report))["cells"][1]["source"])

    assert "will not reproduce" in setup


def test_pdf_without_the_extra_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """The HTML is already on disk, so this is a missing surface, not a failed run."""
    import builtins

    real = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object) -> object:
        if name == "weasyprint":
            raise ImportError("no weasyprint")
        return real(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", refuse)

    with pytest.raises(MissingDependencyError, match=r"insightsmith\[pdf\]"):
        render_pdf("<html></html>")


def test_a_figure_is_kept_in_both_modes_so_the_page_can_follow_the_reader(
    profile: Profile,
) -> None:
    """A report is written once and read later, on someone else's dark setting.

    The PNG's colours are baked at forge time; the page's are chosen at read
    time. Without both bakes a light figure lands on a dark page, which is what
    the first screenshot of this template showed.
    """
    light = Finding(
        question="q",
        code="result = df",
        png=Path("fig.png"),
        png_flipped=Path("fig-dark.png"),
        dark=False,
    )
    drawn_dark = Finding(
        question="q",
        code="result = df",
        png=Path("fig.png"),
        png_flipped=Path("fig-light.png"),
        dark=True,
    )

    assert (light.png_light, light.png_dark) == (Path("fig.png"), Path("fig-dark.png"))
    assert (drawn_dark.png_light, drawn_dark.png_dark) == (
        Path("fig-light.png"),
        Path("fig.png"),
    )

    page = render_html(Report(source=profile.source.path, profile=profile, findings=[light]))
    assert "prefers-color-scheme: dark" in page
    assert 'class="fig-dark" src="fig-dark.png"' in page
    assert 'class="fig-light" src="fig.png"' in page


def test_the_figure_cannot_disagree_with_the_page_it_sits_on() -> None:
    """Found in a screenshot of a real report: a light page with a dark chart.

    The first version used <picture> with a <source media="(prefers-color-scheme:
    dark)">, and a matching source beats the img's own src, so a reader whose OS
    is dark but who had asked this page for light kept getting the dark figure.
    Both bakes are now plain images switched by the same cascade as the colour
    tokens, which is the only way the two cannot drift apart.
    """
    page = pathlib.Path("src/insightsmith/report/templates/report.html.j2").read_text(
        encoding="utf-8"
    )

    assert "<picture>" not in page, "a matching <source> overrides img.src"
    assert ':root[data-theme="light"] .fig-dark { display: none; }' in page
    assert ':root[data-theme="dark"] .fig-light { display: none; }' in page
    printing = page.split("@media print")[1]
    assert ".fig-dark { display: none !important; }" in printing
    assert ".fig-light { display: block !important; }" in printing


def test_one_baked_figure_is_rendered_plainly(profile: Profile) -> None:
    """`ask` renders a single mode, so the picture element would be a lie."""
    finding = Finding(question="q", code="result = df", png=Path("fig.png"))

    page = render_html(Report(source=profile.source.path, profile=profile, findings=[finding]))

    assert 'class="fig-light"' not in page, "one bake is not a pair to switch between"
    assert 'src="fig.png"' in page


def test_markdown_takes_the_light_figure(profile: Profile) -> None:
    """A Markdown viewer has no theme to ask, so it gets the one that always reads."""
    finding = Finding(
        question="q",
        code="result = df",
        png=Path("f.png"),
        png_flipped=Path("f-light.png"),
        dark=True,
    )

    assert "f-light.png" in render_markdown(
        Report(source=profile.source.path, profile=profile, findings=[finding])
    )


def test_the_page_opens_in_the_mode_the_figures_were_drawn_in(profile: Profile) -> None:
    """--dark means the figures are dark, so the first paint must be too."""
    findings = [Finding(question="q", code="result = df")]
    base = {"source": profile.source.path, "profile": profile, "findings": findings}

    assert 'data-theme="auto"' in render_html(Report(**base))
    assert 'data-theme="dark"' in render_html(Report(**base, dark=True))


def test_a_status_hue_is_never_the_only_signal(profile: Profile) -> None:
    """Status colours are reserved and ship with a label: #fab219 is 1.79:1 on
    the light surface, so the word beside the dot is what carries the verdict."""
    critique = Critique(verdict=Verdict.QUALIFIED, caveats=[], confidence=0.5)
    finding = Finding(question="q", code="result = df", critique=critique)

    page = render_html(Report(source=profile.source.path, profile=profile, findings=[finding]))

    assert "qualified</span>" in page.replace("\n", "").replace("  ", "")
