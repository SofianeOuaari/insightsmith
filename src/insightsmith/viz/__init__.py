"""Charting: a validated spec in, a themed figure out."""

from insightsmith.viz.render import ChartSpec, Form, render_html, render_png
from insightsmith.viz.theme import MAX_SERIES, SCATTER_MAX_SERIES, Theme, theme_for

__all__ = [
    "MAX_SERIES",
    "SCATTER_MAX_SERIES",
    "ChartSpec",
    "Form",
    "Theme",
    "render_html",
    "render_png",
    "theme_for",
]
