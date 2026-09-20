"""Forging a finished deliverable out of a run.

The rest of the package answers one question at a time and prints to a
terminal. This is where a run becomes something to hand to someone else: one
document carrying every question, the code that answered it, the caveats the
critic raised, and the card hash that ties it all to a specific state of a
specific file.
"""

from __future__ import annotations

from insightsmith.report.builder import (
    Finding,
    Report,
    render_html,
    render_markdown,
    render_notebook,
    render_pdf,
)

__all__ = [
    "Finding",
    "Report",
    "render_html",
    "render_markdown",
    "render_notebook",
    "render_pdf",
]
