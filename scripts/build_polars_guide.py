"""Turn ``assets/polars_data_consulting_guide.pdf`` into shipped Markdown.

The guide is reference material the coder agent retrieves from, so it has to be
text at runtime. Parsing a PDF on every run would mean a PDF dependency in the
base install for a file that never changes, so the conversion happens here, once,
and the Markdown is committed.

Run it after replacing the PDF::

    python scripts/build_polars_guide.py

Needs ``pdftotext`` (poppler-utils) on PATH.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "assets" / "polars_data_consulting_guide.pdf"
OUT = ROOT / "src" / "insightsmith" / "knowledge" / "polars_guide.md"

#: A heading is "12." or "12.3" followed by a title, flush against the margin.
HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\.?\s+(\S.*)$")
#: Front matter runs to the first body heading; two tables of contents precede it.
BODY_STARTS = "1. Core Concepts"
#: Indentation at or beyond this, in a block that looks like code, means code.
CODE_INDENT = 5
CODE_MARKERS = ("pl.", "df.", "import ", "px.", "plt.", "pip install", "lf.")
#: Three spaces inside a line mean aligned columns, not a wrapped sentence.
COLUMN_GAP = re.compile(r"\S {3,}\S")
#: Only a lowercase word or trailing punctuation continues the line above.
CONTINUATION = re.compile(r"\s*[a-z(.,;:)\-]")
#: Curly punctuation costs tokens and buys nothing in a reference for a model.
SUBSTITUTIONS = str.maketrans(
    dict.fromkeys("\u2018\u2019", "'") | dict.fromkeys("\u201c\u201d", '"')
)


def extract(pdf: Path) -> str:
    """Run pdftotext, or fail with something actionable."""
    try:
        done = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            ["pdftotext", "-layout", str(pdf), "-"],  # noqa: S607 - resolved via PATH
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("pdftotext not found — install poppler-utils")
    if done.returncode != 0:
        sys.exit(f"pdftotext failed: {done.stderr.strip()}")
    return done.stdout


def looks_like_code(block: list[str]) -> bool:
    """Indented, and mentioning something only code mentions."""
    indents = [len(line) - len(line.lstrip()) for line in block if line.strip()]
    if not indents or min(indents) < CODE_INDENT:
        return False
    joined = "\n".join(block)
    return any(marker in joined for marker in CODE_MARKERS) or "=" in joined


def looks_like_table(block: list[str]) -> bool:
    """Two or more lines with an interior run of spaces: aligned columns."""
    return sum(bool(COLUMN_GAP.search(line.strip())) for line in block) >= 2


def unwrap(block: list[str]) -> list[str]:
    """Undo the PDF's hard line breaks, conservatively.

    A line is a continuation only when it opens with something that cannot start
    a sentence or a list item. Guessing wrong therefore leaves a stray break in a
    paragraph, never two thoughts welded together.
    """
    out: list[str] = []
    for raw in block:
        line = raw.rstrip()
        if out and CONTINUATION.match(line):
            out[-1] = f"{out[-1]} {line.strip()}"
        else:
            out.append(line)
    return out


def render_block(block: list[str]) -> list[str]:
    """Fence and dedent code, keep aligned tables verbatim, unwrap the rest."""
    if looks_like_code(block):
        pad = min(len(line) - len(line.lstrip()) for line in block if line.strip())
        body = [line[pad:].rstrip() if line.strip() else "" for line in block]
        return ["```python", *body, "```"]
    if looks_like_table(block):
        return [line.rstrip() for line in block]
    return unwrap(block)


def convert(text: str) -> str:
    """Body only, headings promoted to Markdown, code fenced."""
    lines = text.translate(SUBSTITUTIONS).replace("\f", "").splitlines()
    # Two tables of contents precede the body; the last hit is the real heading.
    start = max(i for i, line in enumerate(lines) if line.strip() == BODY_STARTS)

    out: list[str] = []
    block: list[str] = []

    def flush() -> None:
        if block:
            out.extend(render_block(block))
            block.clear()

    for line in lines[start:]:
        stripped = line.strip()
        match = HEADING.match(line) if line[:1].isdigit() else None
        if match and not looks_like_code([line]):
            flush()
            number, title = match.groups()
            depth, dot = ("##", "") if "." in number else ("#", ".")
            out.extend(["", f"{depth} {number}{dot} {title.strip()}", ""])
            continue
        if not stripped:
            flush()
            if out and out[-1]:
                out.append("")
            continue
        block.append(line)
    flush()

    markdown = "\n".join(out)
    # Consecutive snippets arrive as separate blocks; one fence reads better.
    markdown = markdown.replace("```\n\n```python\n", "")
    return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"


def main() -> None:
    if not PDF.is_file():
        sys.exit(f"missing {PDF.relative_to(ROOT)}")
    markdown = convert(extract(PDF))
    OUT.write_text(markdown, encoding="utf-8")
    headings = markdown.count("\n#")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(markdown):,} bytes, {headings} headings")


if __name__ == "__main__":
    main()
