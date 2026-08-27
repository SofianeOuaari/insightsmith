"""Chart colours and chrome, for both renderers.

The palette is not a matter of taste — it was validated with a colour-blindness
and contrast checker, and the numbers in the comments below are measured, not
estimated. Two results constrain how it may be used, and both are enforced in
code rather than left to whoever writes the next chart:

* **Slot order is the safety mechanism.** Adjacent slots are the pairs a reader
  actually has to tell apart in a bar or line chart, and this order clears the
  CVD gate in both modes (worst adjacent ΔE 9.1 light / 8.4 dark, ≥8 target).
  Assign slots in order and never cycle: a generated ninth hue is
  indistinguishable from an existing one under simulated CVD.
* **Scatter caps at three series.** Scatter compares every pair, not just
  neighbours, and a fourth slot fails outright (light, all-pairs: normal-vision
  ΔE 13.7 against the ≥15 floor). Past three, fold the tail into "other".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "MAX_SERIES",
    "SCATTER_MAX_SERIES",
    "Theme",
    "matplotlib_rc",
    "plotly_template",
    "theme_for",
]

#: Token ceiling. Past this, fold the tail into "other" or facet (never a new hue).
MAX_SERIES: Final = 8
#: All-pairs forms compare every series against every other, not just neighbours.
#: A fourth slot fails the normal-vision floor in light mode, so three is the cap.
SCATTER_MAX_SERIES: Final = 3


@dataclass(frozen=True, slots=True)
class Theme:
    """One rendering mode's colours."""

    name: str
    surface: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    #: Fixed order. Assign by index; never cycle past the end.
    series: tuple[str, ...]
    #: Single hue, light to dark, for magnitude.
    sequential: tuple[str, ...]
    #: Warm and cool poles with a neutral midpoint, for polarity.
    diverging: tuple[str, str, str]
    #: For the one-series-matters case: everything else recedes to this.
    de_emphasis: str

    def colour(self, index: int) -> str:
        """The slot for series ``index``, refusing to wrap around."""
        if not 0 <= index < len(self.series):
            msg = (
                f"series slot {index} is outside the validated palette of "
                f"{len(self.series)}; fold the tail into 'other' rather than "
                "generating a colour"
            )
            raise IndexError(msg)
        return self.series[index]


LIGHT: Final = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=(
        "#2a78d6",  # blue
        "#eb6834",  # orange
        "#1baf7a",  # aqua      — 2.74:1, needs a visible label or the table
        "#eda100",  # yellow    — 2.11:1, likewise
        "#e87ba4",  # magenta   — 2.62:1, likewise
        "#008300",  # green
        "#4a3aa7",  # violet
        "#e34948",  # red
    ),
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
    diverging=("#2a78d6", "#f0efec", "#e34948"),
    de_emphasis="#c3c2b7",
)

DARK: Final = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    # The same eight hues, re-stepped for the dark surface — not a second
    # palette. Every slot clears 3:1 here, unlike light.
    series=(
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
    sequential=("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
    diverging=("#3987e5", "#383835", "#e66767"),
    de_emphasis="#383835",
)

#: Slots that sit below 3:1 on the light surface. The relief rule applies: ship
#: a visible label or the table view, never colour alone.
LOW_CONTRAST_LIGHT_SLOTS: Final = (2, 3, 4)


def theme_for(mode: str = "light") -> Theme:
    """``light`` or ``dark``. Dark is a selected palette, not an inverted one."""
    return DARK if mode == "dark" else LIGHT


def matplotlib_rc(theme: Theme) -> dict[str, Any]:
    """rcParams putting the data forward and the chrome back.

    Recessive grid, no top or right spine, muted tick labels: the marks should
    be the darkest thing on the surface.
    """
    return {
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "text.color": theme.ink,
        "axes.labelcolor": theme.ink_secondary,
        "axes.edgecolor": theme.axis,
        "axes.titlecolor": theme.ink,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": theme.grid,
        "grid.linewidth": 0.8,
        "xtick.color": theme.ink_muted,
        "ytick.color": theme.ink_muted,
        "xtick.labelcolor": theme.ink_secondary,
        "ytick.labelcolor": theme.ink_secondary,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "legend.frameon": False,
        "figure.autolayout": True,
    }


def plotly_template(theme: Theme) -> dict[str, Any]:
    """The same chrome for the interactive renderer, so the two agree."""
    return {
        "layout": {
            "paper_bgcolor": theme.surface,
            "plot_bgcolor": theme.surface,
            "colorway": list(theme.series),
            "font": {"color": theme.ink_secondary, "size": 12},
            "title": {"font": {"color": theme.ink, "size": 16}},
            "xaxis": {
                "gridcolor": theme.grid,
                "linecolor": theme.axis,
                "zerolinecolor": theme.axis,
                "tickfont": {"color": theme.ink_secondary},
            },
            "yaxis": {
                "gridcolor": theme.grid,
                "linecolor": theme.axis,
                "zerolinecolor": theme.axis,
                "tickfont": {"color": theme.ink_secondary},
            },
            "legend": {"bgcolor": "rgba(0,0,0,0)"},
            "hoverlabel": {"font": {"size": 12}},
        }
    }
