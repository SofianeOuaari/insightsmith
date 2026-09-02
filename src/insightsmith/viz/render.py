"""Drawing a :class:`ChartSpec`, statically and interactively.

The model chooses a *spec* — a form and which columns fill which role — and this
module draws it. It never writes plotting code. That keeps every chart on the
same theme, makes the output reproducible for a given spec, and means no
matplotlib written by a language model runs anywhere.

Two rules the renderer enforces rather than trusts:

* **One series gets one hue.** Colouring every bar differently when there is
  only one series is the most common chart mistake there is: it implies an
  identity distinction that the data does not contain.
* **Scatter caps at three series** (see :mod:`insightsmith.viz.theme`).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

import polars as pl

from insightsmith.errors import MissingDependencyError
from insightsmith.viz.theme import MAX_SERIES, SCATTER_MAX_SERIES, Theme, theme_for

__all__ = ["ChartSpec", "Form", "render_html", "render_png"]

#: Beyond this many categories a bar chart is a wall; the tail folds into "other".
MAX_CATEGORIES: Final = 20
#: Bar thickness for one series, and the width a whole group of them shares.
_SINGLE_BAR: Final = 0.62
_BAR_GROUP: Final = 0.8
_OTHER: Final = "other"


class Form(str, Enum):
    """Chart forms, named for the job each does."""

    BAR = "bar"  # magnitude across categories, horizontal — long labels fit
    COLUMN = "column"  # magnitude across categories, vertical
    LINE = "line"  # trend over an ordered axis
    AREA = "area"  # trend, single series, magnitude matters
    SCATTER = "scatter"  # relationship between two measures


@dataclass(slots=True)
class ChartSpec:
    """What to draw. Validated against the frame before it reaches a renderer."""

    form: Form
    x: str
    y: str
    series: str | None = None
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        """The columns this spec needs, each named once.

        A spec that puts the same column on both axes is nonsense, but it must
        not be a crash: selecting a name twice raises ``DuplicateError`` deep in
        polars, which surfaces to the reader as chart machinery leaking.
        """
        named: list[str] = []
        for name in (self.x, self.y, self.series):
            if name and name not in named:
                named.append(name)
        return named


def render_png(spec: ChartSpec, frame: pl.DataFrame, *, mode: str = "light") -> bytes:
    """Draw ``spec`` as a PNG.

    Raises:
        MissingDependencyError: if the ``viz`` extra is not installed.
    """
    plt = _pyplot()
    theme = theme_for(mode)
    from insightsmith.viz.theme import matplotlib_rc

    data = _prepare(spec, frame)
    with plt.rc_context(matplotlib_rc(theme)):
        figure, axes = plt.subplots(figsize=(8, 4.5), dpi=144)
        _draw(axes, spec, data, theme)
        _label(axes, spec, theme)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(figure)
    return buffer.getvalue()


def render_html(spec: ChartSpec, frame: pl.DataFrame, *, mode: str = "light") -> str:
    """Draw ``spec`` as a self-contained interactive HTML fragment.

    Hover is the point of the interactive version — a reader can read exact
    values off the marks instead of estimating them against the axis.
    """
    go = _graph_objects()
    theme = theme_for(mode)
    from insightsmith.viz.theme import plotly_template

    data = _prepare(spec, frame)
    parts = _split(spec, data)
    form = _effective_form(spec, parts)
    horizontal = form is Form.BAR

    figure = go.Figure()
    for index, (name, part) in enumerate(parts):
        colour = theme.colour(index)
        xs, ys = part[spec.x].to_list(), part[spec.y].to_list()
        if form in {Form.LINE, Form.AREA}:
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    name=name,
                    mode="lines",
                    line={"color": colour, "width": 2},
                    fill="tozeroy" if form is Form.AREA else None,
                    hovertemplate=f"%{{x}}<br>{spec.y}: %{{y}}<extra>{name}</extra>",
                )
            )
        elif form is Form.SCATTER:
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    name=name,
                    mode="markers",
                    marker={"color": colour, "size": 9},
                    hovertemplate=f"{spec.x}: %{{x}}<br>{spec.y}: %{{y}}<extra>{name}</extra>",
                )
            )
        else:
            figure.add_trace(
                go.Bar(
                    x=ys if horizontal else xs,
                    y=xs if horizontal else ys,
                    name=name,
                    orientation="h" if horizontal else "v",
                    marker={"color": colour},
                    hovertemplate=f"%{{{'y' if horizontal else 'x'}}}<br>"
                    f"{spec.y}: %{{{'x' if horizontal else 'y'}}}<extra>{name}</extra>",
                )
            )

    # Merge rather than spread-and-override: the template already carries a
    # `title` key (its font), so passing title= alongside it collides.
    layout: dict[str, Any] = dict(plotly_template(theme)["layout"])
    layout["title"] = {**layout.get("title", {}), "text": spec.title or ""}
    layout["xaxis"] = {
        **layout.get("xaxis", {}),
        "title": {"text": spec.x_label or (spec.y if horizontal else spec.x)},
    }
    layout["yaxis"] = {
        **layout.get("yaxis", {}),
        "title": {"text": spec.y_label or (spec.x if horizontal else spec.y)},
    }
    if horizontal:
        # matplotlib inverts the y axis so the largest bar reads first. Plotly
        # counts categories upward from the origin, which would hand the same
        # spec back as its own mirror image.
        layout["yaxis"] = {**layout["yaxis"], "autorange": "reversed"}
    layout["barmode"] = "group"
    layout["showlegend"] = spec.series is not None
    layout["margin"] = {"l": 60, "r": 24, "t": 56, "b": 48}
    figure.update_layout(**layout)
    return str(figure.to_html(include_plotlyjs="cdn", full_html=False, div_id=None))


# --------------------------------------------------------------------------- #
# shared preparation
# --------------------------------------------------------------------------- #


def _prepare(spec: ChartSpec, frame: pl.DataFrame) -> pl.DataFrame:
    """Validate the spec against the frame and fold any oversized tail."""
    missing = [c for c in spec.columns if c not in frame.columns]
    if missing:
        msg = f"chart references columns not in the result: {', '.join(missing)}"
        raise ValueError(msg)

    data = frame.select(spec.columns).drop_nulls()
    if spec.form in {Form.LINE, Form.AREA}:
        data = _order_axis(data, spec)
    if spec.series is None:
        return _fold_categories(data, spec)

    counts = data[spec.series].value_counts(sort=True)
    names = counts[spec.series].to_list()
    cap = SCATTER_MAX_SERIES if spec.form is Form.SCATTER else MAX_SERIES
    if len(names) > cap:
        # Never solve too-many-series by generating hues: fold the tail. "other"
        # occupies a slot itself, so keep one fewer — otherwise the fold pushes
        # the total to cap + 1 and the palette refuses the last one.
        keep = set(names[: cap - 1])
        data = data.with_columns(
            pl.when(pl.col(spec.series).is_in(list(keep)))
            .then(pl.col(spec.series))
            .otherwise(pl.lit(_OTHER))
            .alias(spec.series)
        )
    return data


def _order_axis(data: pl.DataFrame, spec: ChartSpec) -> pl.DataFrame:
    """Sort an ordered x axis, because a line claims the sequence means something.

    ``group_by`` does not maintain order, and a grouped aggregate is exactly
    where "revenue by month" comes from — drawn in hash order a trend reads as a
    scribble rather than a trend. Only genuinely ordered dtypes are sorted: a
    string axis is left alone, because its order may already be deliberate
    (month names, size buckets) and alphabetising it would be worse.
    """
    dtype = data.schema[spec.x]
    if dtype.is_temporal() or dtype.is_numeric():
        return data.sort(spec.x)
    return data


def _fold_categories(data: pl.DataFrame, spec: ChartSpec) -> pl.DataFrame:
    """Rank magnitude bars, and fold an oversized tail.

    A bar chart's job is ranking, so an unsorted one defeats its own purpose —
    but only when the axis has no order of its own. A temporal or numeric x
    carries meaning in its sequence, and reordering it would destroy that.
    """
    if spec.form not in {Form.BAR, Form.COLUMN}:
        return data
    if data.schema[spec.x] == pl.String:
        data = data.sort(spec.y, descending=True)
    if data.height <= MAX_CATEGORIES:
        return data
    ordered = data.sort(spec.y, descending=True)
    head = ordered.head(MAX_CATEGORIES - 1)
    tail_total = ordered.tail(data.height - MAX_CATEGORIES + 1)[spec.y].sum()
    # "other" is a label, so the axis has to hold labels. A numeric axis being
    # folded is already categorical in everything but dtype — and putting a
    # string into an Int64 column is a TypeError from deep inside polars.
    if head.schema[spec.x] != pl.String:
        head = head.with_columns(pl.col(spec.x).cast(pl.String))
    tail = pl.DataFrame({spec.x: [_OTHER], spec.y: [tail_total]}, schema=head.schema)
    return pl.concat([head, tail])


def _effective_form(spec: ChartSpec, parts: list[tuple[str, pl.DataFrame]]) -> Form:
    """The form actually drawn, which is the spec's unless it cannot be read.

    Filled areas stacked on one surface hide each other, and where they overlap
    they mix a colour that is in no validated slot. An area chart is a
    single-series form; asked for several, draw the lines instead of drawing
    something the reader cannot decode.
    """
    if spec.form is Form.AREA and len(parts) > 1:
        return Form.LINE
    return spec.form


def _split(spec: ChartSpec, data: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """One (name, frame) per series, in a stable order."""
    if spec.series is None:
        return [(spec.y, data)]
    names = data[spec.series].unique(maintain_order=True).to_list()
    return [(str(n), data.filter(pl.col(spec.series) == n)) for n in names]


# --------------------------------------------------------------------------- #
# matplotlib
# --------------------------------------------------------------------------- #


def _draw(axes: Any, spec: ChartSpec, data: pl.DataFrame, theme: Theme) -> None:
    parts = _split(spec, data)
    form = _effective_form(spec, parts)
    single = len(parts) == 1

    if form in {Form.BAR, Form.COLUMN}:
        _draw_bars(axes, spec, form, parts, theme, single=single)
    else:
        for index, (name, part) in enumerate(parts):
            # One series means one hue. A different colour per bar would imply an
            # identity distinction the data does not have.
            colour = theme.series[0] if single else theme.colour(index)
            xs, ys = part[spec.x].to_list(), part[spec.y].to_list()
            if form is Form.LINE:
                axes.plot(xs, ys, color=colour, label=name)
            elif form is Form.AREA:
                axes.fill_between(range(len(xs)), ys, color=colour, alpha=0.85, label=name)
                axes.set_xticks(range(len(xs)))
                axes.set_xticklabels(xs)
            else:
                axes.scatter(xs, ys, color=colour, label=name, s=64, edgecolor=theme.surface)

    if form is Form.BAR:
        axes.invert_yaxis()  # largest at the top, reading order
        axes.grid(axis="x")
        axes.grid(axis="y", visible=False)
    if not single:
        axes.legend(loc="best")


def _draw_bars(
    axes: Any,
    spec: ChartSpec,
    form: Form,
    parts: list[tuple[str, pl.DataFrame]],
    theme: Theme,
    *,
    single: bool,
) -> None:
    """Bars for every series, side by side within each category.

    Drawing each series at the same position does not overlap harmlessly: the
    series drawn first disappears underneath, and what is left reads as a
    *stacked* chart, so the reader takes totals off it that the data never
    contained. Explicit slots make the grouping real.
    """
    categories = _categories(parts, spec.x)
    slot = {name: index for index, name in enumerate(categories)}
    span = (_SINGLE_BAR if single else _BAR_GROUP) / len(parts)
    horizontal = form is Form.BAR

    for index, (name, part) in enumerate(parts):
        colour = theme.series[0] if single else theme.colour(index)
        offset = (index - (len(parts) - 1) / 2) * span
        at = [slot[key] + offset for key in part[spec.x].to_list()]
        values = part[spec.y].to_list()
        size = span if single else span * 0.9
        if horizontal:
            bars = axes.barh(at, values, color=colour, label=name, height=size)
        else:
            bars = axes.bar(at, values, color=colour, label=name, width=size)
        _value_labels(axes, bars, single, theme)

    ticks = list(range(len(categories)))
    labels = [str(name) for name in categories]
    if horizontal:
        axes.set_yticks(ticks)
        axes.set_yticklabels(labels)
    else:
        axes.set_xticks(ticks)
        axes.set_xticklabels(labels)


def _categories(parts: list[tuple[str, pl.DataFrame]], column: str) -> list[Any]:
    """Every category any series uses, in the order they first appear."""
    seen: list[Any] = []
    known: set[Any] = set()
    for _, part in parts:
        for value in part[column].to_list():
            if value not in known:
                known.add(value)
                seen.append(value)
    return seen


def _value_labels(axes: Any, bars: Any, single: bool, theme: Theme) -> None:
    """Put the number on the bar.

    Reading a value off a gridline is guesswork, and three palette slots sit
    below 3:1 on the light surface — the relief rule requires a visible label or
    a table wherever colour alone would carry the meaning. Labels go on only for
    a single series; on grouped bars they collide.
    """
    if not single:
        return
    axes.bar_label(
        bars,
        fmt=_compact,
        padding=4,
        color=theme.ink_secondary,
        fontsize=9,
    )


def _compact(value: float) -> str:
    """1234567 -> 1.2M. Precision the eye cannot use is noise."""
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= limit:
            # removesuffix, not replace: replace would eat a ".0" anywhere.
            return f"{value / limit:.1f}".removesuffix(".0") + suffix
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _label(axes: Any, spec: ChartSpec, theme: Theme) -> None:
    if spec.title:
        axes.set_title(spec.title, loc="left", color=theme.ink)
    horizontal = spec.form is Form.BAR
    axes.set_xlabel(spec.x_label or (spec.y if horizontal else spec.x))
    axes.set_ylabel(spec.y_label or (spec.x if horizontal else spec.y))


# --------------------------------------------------------------------------- #
# optional imports
# --------------------------------------------------------------------------- #


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: never try to open a window
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via the error path
        raise MissingDependencyError("matplotlib", "viz", purpose="drawing charts") from exc
    return plt


def _graph_objects() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("plotly", "viz", purpose="interactive charts") from exc
    return go
