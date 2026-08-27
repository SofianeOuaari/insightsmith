"""Theme, renderer, artifact store and chart selection.

The palette's numbers were measured with a CVD and contrast validator; these
tests protect the *rules* those numbers imply, so a future edit cannot quietly
loosen them.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from insightsmith.agents.viz import default_spec, validate_spec
from insightsmith.execution.artifacts import ArtifactStore, slugify
from insightsmith.viz.render import MAX_CATEGORIES, ChartSpec, Form, render_html, render_png
from insightsmith.viz.theme import MAX_SERIES, SCATTER_MAX_SERIES, theme_for


@pytest.fixture
def sales() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Product Type": ["Coffee", "Espresso", "Herbal Tea", "Tea"],
            "Total Sales": [216828.0, 222996.0, 207214.0, 172773.0],
        }
    )


# --------------------------------------------------------------------------- #
# theme
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_both_modes_carry_the_full_validated_palette(mode: str) -> None:
    theme = theme_for(mode)
    assert len(theme.series) == MAX_SERIES
    assert all(c.startswith("#") and len(c) == 7 for c in theme.series)


def test_dark_is_a_selected_palette_not_an_inversion() -> None:
    """Dark steps were chosen for the dark surface; flipping light would fail it."""
    light, dark = theme_for("light"), theme_for("dark")
    assert light.series != dark.series
    assert light.surface != dark.surface


def test_a_ninth_series_is_refused_rather_than_invented() -> None:
    """A generated hue is indistinguishable from an existing one under CVD."""
    theme = theme_for("light")
    assert theme.colour(MAX_SERIES - 1)
    with pytest.raises(IndexError, match="fold the tail"):
        theme.colour(MAX_SERIES)


def test_scatter_caps_lower_than_bars() -> None:
    """All-pairs forms compare every series; a fourth slot fails the floor."""
    assert SCATTER_MAX_SERIES == 3
    assert SCATTER_MAX_SERIES < MAX_SERIES


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("form", [Form.BAR, Form.COLUMN, Form.LINE, Form.AREA, Form.SCATTER])
def test_every_form_renders_a_png(sales: pl.DataFrame, form: Form) -> None:
    png = render_png(ChartSpec(form=form, x="Product Type", y="Total Sales"), sales)
    assert png.startswith(b"\x89PNG")
    assert len(png) > 1000


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_both_modes_render(sales: pl.DataFrame, mode: str) -> None:
    spec = ChartSpec(form=Form.BAR, x="Product Type", y="Total Sales")
    assert render_png(spec, sales, mode=mode).startswith(b"\x89PNG")


def test_the_interactive_version_has_hover(sales: pl.DataFrame) -> None:
    """Hover is the point of the HTML form: exact values without the axis."""
    html = render_html(ChartSpec(form=Form.BAR, x="Product Type", y="Total Sales"), sales)
    assert "hovertemplate" in html
    assert "plotly" in html.lower()


def test_magnitude_bars_are_ranked(sales: pl.DataFrame) -> None:
    """A bar chart's job is ranking; unsorted defeats it."""
    from insightsmith.viz.render import _prepare

    prepared = _prepare(ChartSpec(form=Form.BAR, x="Product Type", y="Total Sales"), sales)
    values = prepared["Total Sales"].to_list()
    assert values == sorted(values, reverse=True)


def test_an_ordered_axis_keeps_its_order() -> None:
    """Sorting a time axis by value would destroy the thing being shown."""
    from insightsmith.viz.render import _prepare

    frame = pl.DataFrame({"month": [1, 2, 3], "sales": [30.0, 10.0, 20.0]})
    prepared = _prepare(ChartSpec(form=Form.COLUMN, x="month", y="sales"), frame)
    assert prepared["month"].to_list() == [1, 2, 3]


def test_a_long_tail_is_folded_not_drawn(sales: pl.DataFrame) -> None:
    from insightsmith.viz.render import _prepare

    frame = pl.DataFrame(
        {"k": [f"cat{i}" for i in range(60)], "v": [float(60 - i) for i in range(60)]}
    )
    prepared = _prepare(ChartSpec(form=Form.BAR, x="k", y="v"), frame)
    assert prepared.height == MAX_CATEGORIES
    assert "other" in prepared["k"].to_list()


def test_too_many_series_fold_rather_than_grow_the_palette() -> None:
    """Folding must leave a slot for "other" itself, or the palette refuses it."""
    from insightsmith.viz.render import _prepare

    frame = pl.DataFrame(
        {
            "x": list(range(24)),
            "y": [float(i) for i in range(24)],
            "g": [f"g{i % 12}" for i in range(24)],
        }
    )
    prepared = _prepare(ChartSpec(form=Form.LINE, x="x", y="y", series="g"), frame)
    assert prepared["g"].n_unique() <= MAX_SERIES
    assert "other" in prepared["g"].to_list()


def test_scatter_folds_at_three(sales: pl.DataFrame) -> None:
    from insightsmith.viz.render import _prepare

    frame = pl.DataFrame({"x": [1.0] * 10, "y": [1.0] * 10, "g": [f"g{i % 5}" for i in range(10)]})
    prepared = _prepare(ChartSpec(form=Form.SCATTER, x="x", y="y", series="g"), frame)
    assert prepared["g"].n_unique() <= SCATTER_MAX_SERIES
    assert "other" in prepared["g"].to_list()


def test_a_chart_naming_a_missing_column_is_refused(sales: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="not in the result"):
        render_png(ChartSpec(form=Form.BAR, x="nope", y="Total Sales"), sales)


# --------------------------------------------------------------------------- #
# choosing a form
# --------------------------------------------------------------------------- #


def test_the_shape_of_the_data_implies_a_form() -> None:
    cases = {
        Form.COLUMN: pl.DataFrame({"k": ["a", "b"], "v": [1.0, 2.0]}),
        Form.LINE: pl.DataFrame({"year": [2020, 2021, 2022], "v": [1.0, 2.0, 3.0]}),
        Form.SCATTER: pl.DataFrame({"spend": [5.0, 1.0, 3.0], "profit": [3.0, 4.0, 2.0]}),
    }
    for expected, frame in cases.items():
        spec = default_spec(frame)
        assert spec is not None and spec.form is expected


def test_long_labels_get_horizontal_bars() -> None:
    frame = pl.DataFrame({"k": ["a very long category name indeed", "b"], "v": [1.0, 2.0]})
    spec = default_spec(frame)
    assert spec is not None and spec.form is Form.BAR


def test_a_result_with_nothing_numeric_gets_no_chart() -> None:
    assert default_spec(pl.DataFrame({"a": ["x"], "b": ["y"]})) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"form": "bar", "x": "missing", "y": "v"},
        {"form": "bar", "x": "k", "y": "k"},  # y must be numeric
        {"form": "pie", "x": "k", "y": "v"},  # not a form we draw
        {"form": "bar"},
        "not a dict",
    ],
)
def test_an_unusable_spec_is_rejected(payload: object) -> None:
    frame = pl.DataFrame({"k": ["a"], "v": [1.0]})
    assert validate_spec(payload, frame) is None


def test_a_series_column_that_duplicates_an_axis_is_dropped() -> None:
    frame = pl.DataFrame({"k": ["a"], "v": [1.0]})
    spec = validate_spec({"form": "bar", "x": "k", "y": "v", "series": "k"}, frame)
    assert spec is not None
    assert spec.series is None


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #


def test_artifacts_record_where_they_came_from(tmp_path: Path) -> None:
    """A figure found months later must be traceable to the data behind it."""
    store = ArtifactStore(tmp_path / "out")
    artifact = store.write_bytes(
        "sales by type.png", b"\x89PNG", question="which type sells most?", card_hash="abc123"
    )
    assert artifact.path.exists()
    entry = json.loads(store.manifest_path.read_text())[0]
    assert entry["question"] == "which type sells most?"
    assert entry["card_hash"] == "abc123"
    assert entry["created"]


def test_a_repeated_name_does_not_overwrite(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "out")
    first = store.write_bytes("chart.png", b"a")
    second = store.write_bytes("chart.png", b"b")
    assert first.path != second.path
    assert first.path.read_bytes() == b"a"
    assert len(store.entries()) == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Total Sales by Product Type!", "total-sales-by-product-type"),
        ("   ", "artifact"),
        ("../../etc/passwd", "etc-passwd"),
    ],
)
def test_slugify_produces_a_safe_name(raw: str, expected: str) -> None:
    assert slugify(raw) == expected


def test_a_corrupt_manifest_does_not_break_the_next_write(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "out")
    store.write_bytes("a.png", b"a")
    store.manifest_path.write_text("{ not json", encoding="utf-8")
    store.write_bytes("b.png", b"b")
    assert len(store.entries()) == 1
