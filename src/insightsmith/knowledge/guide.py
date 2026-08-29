"""Parsing the bundled Polars guide into retrievable sections.

The guide ships as Markdown, not as the PDF it came from: converting it once at
authoring time (``scripts/build_polars_guide.py``) keeps a PDF parser out of the
base install for a file that only changes when the guide is rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Final

__all__ = ["GUIDE_FILE", "Section", "guide_text", "sections"]

#: Package data, alongside this module.
GUIDE_FILE: Final = "polars_guide.md"
#: ``# 8. Aggregations and group_by`` / ``## 8.3 group_by_dynamic — ...``
_HEADING = re.compile(r"^(#{1,2}) (\d{1,2}(?:\.\d{1,2})?)\.? +(\S.*)$")
_FENCE: Final = "```"


@dataclass(frozen=True, slots=True)
class Section:
    """One numbered chunk of the guide, the unit retrieval ranks and returns."""

    number: str
    title: str
    body: str
    #: Title of the enclosing top-level section, when there is one. It is scored
    #: along with the body, so "aggregation" finds 8.3 even though 8.3's own
    #: title says only "group_by_dynamic".
    parent: str = ""

    @property
    def heading(self) -> str:
        return f"{self.number} {self.title}"

    @property
    def searchable(self) -> str:
        parent = "" if self.parent == self.title else self.parent
        return f"{parent}\n{self.title}\n{self.body}"

    def render(self) -> str:
        """How the section appears in a prompt."""
        return f"## {self.heading}\n{self.body}"

    @property
    def size_bytes(self) -> int:
        return len(self.render().encode("utf-8"))


@lru_cache(maxsize=1)
def guide_text() -> str:
    """The raw Markdown, read once."""
    return (
        resources.files("insightsmith.knowledge").joinpath(GUIDE_FILE).read_text(encoding="utf-8")
    )


@lru_cache(maxsize=1)
def sections() -> tuple[Section, ...]:
    """Every section with a body, in document order.

    Headings that only introduce subsections carry no body of their own and are
    dropped — they would be retrieved as empty context.
    """
    found: list[Section] = []
    parent = ""
    number = title = ""
    body: list[str] = []
    in_fence = False

    def close() -> None:
        text = "\n".join(body).strip()
        if number and text:
            found.append(Section(number=number, title=title, body=text, parent=parent))
        body.clear()

    for line in guide_text().splitlines():
        if line.startswith(_FENCE):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match is None:
            body.append(line)
            continue
        close()
        level, number, title = match.group(1), match.group(2), match.group(3).strip()
        if level == "#":
            parent = title
    close()
    return tuple(found)
