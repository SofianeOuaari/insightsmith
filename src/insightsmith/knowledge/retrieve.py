"""Ranking guide sections against a question, on a byte budget.

BM25 over the guide's sixty-odd sections, in pure stdlib. A vector index would be a better
retriever and a worse dependency: embeddings would mean a model download, a
second round trip per question, and a cache to invalidate, to choose between a
few dozen documents that a lexical score already separates cleanly. Should the
guide ever grow into a library, this is the seam to replace.

The tokenizer is the part that earns its keep. ``group_by`` yields ``group_by``,
``group``, ``by`` and ``groupby``, so a question phrased in English, an API name,
and a pandas-flavoured ``AttributeError`` all reach the same section. Stemming is
crude on purpose — "series" becomes "sery" — because the corpus and the query run
through the same function, and agreeing matters more here than being right.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from insightsmith.knowledge.guide import Section, sections

__all__ = [
    "CODER_EXCLUDES",
    "DEFAULT_BUDGET",
    "DEFAULT_LIMIT",
    "FOCUS_WEIGHT",
    "reference",
    "retrieve",
    "stem",
    "tokenize",
]

#: Roughly half a dataset card — enough for four or five sections beside it.
DEFAULT_BUDGET: Final = 2_560
DEFAULT_LIMIT: Final = 5
#: How much harder a failure counts than the question it came from. A traceback
#: names the mistake — ``no attribute 'groupby'`` — where the question only names
#: the goal, so on a retry the traceback is the sharper query of the two.
FOCUS_WEIGHT: Final = 3.0
#: Top-level sections the coder must not see. It is handed a DataFrame that is
#: already in memory and forbidden to read files or plot, so reading, charting,
#: installing and the end-to-end case study can only send it somewhere it is not
#: allowed to go. Pinned by number, and by title in ``test_knowledge.py`` so that
#: rebuilding the guide from a new PDF fails loudly rather than quietly
#: excluding the wrong four sections.
CODER_EXCLUDES: Final = ("2", "3", "15", "16")

_K1: Final = 1.5
_B: Final = 0.75
#: The blank line between two rendered sections, and the fence a clip may reopen.
_JOIN_BYTES: Final = 2
_FENCE_BYTES: Final = 4
_WORD = re.compile(r"[a-z][a-z0-9_]*|[0-9]+")
#: (suffix, replacement, shortest word the rule may apply to). Longest first.
_SUFFIXES: Final = (
    ("ation", "at", 7),
    ("ate", "at", 7),
    ("ing", "", 6),
    ("ed", "", 5),
    ("ly", "", 5),
)
#: A trailing "s" that is not a plural.
_KEEP_S: Final = ("ss", "us", "is")


def stem(word: str) -> str:
    """Fold the endings that separate a question from the API it is asking about.

    "outliers" has to reach "outlier", and "correlated" and "correlation" both
    have to reach the section on correlation. One plural rule, then one suffix
    rule; anything more would need a real stemmer and a dependency.
    """
    if len(word) > 4 and word.endswith("ies"):
        word = f"{word[:-3]}y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith(_KEEP_S):
        word = word[:-1]
    for suffix, replacement, minimum in _SUFFIXES:
        if len(word) >= minimum and word.endswith(suffix):
            return word[: -len(suffix)] + replacement
    return word


#: Dropped from a *question*, never from the guide. "How many nulls?" otherwise
#: lands on joins, because `how="left"` and "one-to-many" are what the guide says
#: those words for. The corpus keeps them so that asking about `how=` still works.
# fmt: off
_NOISE_WORDS: Final = (
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "calculate", "can",
    "compute", "did", "do", "does", "each", "find", "for", "from", "get", "give",
    "had", "has", "have", "how", "i", "in", "into", "is", "it", "its", "like",
    "list", "make", "many", "me", "my", "need", "of", "on", "or", "our", "out",
    "please", "show", "so", "some", "tell", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "us", "use", "want", "was",
    "we", "were", "what", "which", "who", "whose", "why", "will", "would", "you",
    "your",
)
# fmt: on
_QUERY_NOISE: Final = frozenset(map(stem, _NOISE_WORDS))


def tokenize(text: str) -> list[str]:
    """Words and identifiers, stemmed, plus the two ways an identifier is written."""
    out: list[str] = []
    for match in _WORD.finditer(text.lower()):
        token = match.group()
        out.append(stem(token))
        if "_" in token:
            out.extend(stem(part) for part in token.split("_") if part)
            out.append(stem(token.replace("_", "")))
    return out


@dataclass(frozen=True, slots=True)
class _Index:
    docs: tuple[Counter[str], ...]
    lengths: tuple[int, ...]
    average_length: float
    idf: dict[str, float]

    def score(self, terms: Mapping[str, float], doc: int) -> float:
        counts = self.docs[doc]
        norm = _K1 * (1 - _B + _B * self.lengths[doc] / self.average_length)
        total = 0.0
        for term, weight in terms.items():
            frequency = counts.get(term, 0)
            if frequency:
                total += weight * self.idf[term] * frequency * (_K1 + 1) / (frequency + norm)
        return total


@lru_cache(maxsize=1)
def _index() -> _Index:
    corpus = [Counter(tokenize(section.searchable)) for section in sections()]
    lengths = [sum(counts.values()) for counts in corpus]
    seen: Counter[str] = Counter()
    for counts in corpus:
        seen.update(counts.keys())
    total = len(corpus)
    idf = {term: math.log(1 + (total - n + 0.5) / (n + 0.5)) for term, n in seen.items()}
    return _Index(
        docs=tuple(corpus),
        lengths=tuple(lengths),
        average_length=(sum(lengths) / total) if total else 1.0,
        idf=idf,
    )


def retrieve(
    query: str,
    *,
    focus: str = "",
    limit: int = DEFAULT_LIMIT,
    exclude: tuple[str, ...] = (),
) -> tuple[Section, ...]:
    """The sections that best match ``query``, best first.

    ``focus`` is extra query text weighted :data:`FOCUS_WEIGHT` times as heavily —
    a traceback, on a retry. ``exclude`` drops whole top-level sections by number,
    so a caller can rule out advice its own rules forbid. Sections that match
    nothing are left out rather than padded in: a weak match is worse than a
    longer prompt.
    """
    index = _index()
    candidates = {
        term: weight
        for term, weight in _weigh(query, focus).items()
        if term in index.idf and term not in _QUERY_NOISE
    }
    if not candidates:
        return ()

    ranked: list[tuple[float, int]] = []
    for position, section in enumerate(sections()):
        if section.number.split(".")[0] in exclude:
            continue
        score = index.score(candidates, position)
        if score > 0:
            ranked.append((score, position))
    # Descending by score; document order breaks ties, so the result is stable.
    ranked.sort(key=lambda pair: (-pair[0], pair[1]))
    return tuple(sections()[position] for _, position in ranked[:limit])


def _weigh(query: str, focus: str) -> dict[str, float]:
    """Term weights for the two halves of a query, the focus counting for more."""
    terms: dict[str, float] = {}
    for text, weight in ((query, 1.0), (focus, FOCUS_WEIGHT)):
        for term in tokenize(text):
            terms[term] = terms.get(term, 0.0) + weight
    return terms


def reference(
    query: str,
    *,
    focus: str = "",
    budget: int = DEFAULT_BUDGET,
    limit: int = DEFAULT_LIMIT,
    exclude: tuple[str, ...] = (),
) -> str:
    """Rendered sections for ``query``, stopping before ``budget`` bytes.

    A section is taken whole or not at all — half a code example teaches the
    wrong API as readily as no example. The one exception is a first section
    that alone exceeds the budget, which is cut at a line boundary so that a
    tight budget still returns something.
    """
    chosen: list[str] = []
    used = 0
    for section in retrieve(query, focus=focus, limit=limit, exclude=exclude):
        rendered = section.render()
        size = len(rendered.encode("utf-8"))
        if used + size <= budget:
            chosen.append(rendered)
            used += size + _JOIN_BYTES
        elif not chosen:
            chosen.append(_clip(rendered, budget))
            break
    return "\n\n".join(chosen)


def _clip(text: str, budget: int) -> str:
    """Cut at the last line that fits, leaving any open code fence closed."""
    kept: list[str] = []
    used = 0
    for line in text.splitlines():
        size = len(line.encode("utf-8")) + 1
        if used + size > budget - _FENCE_BYTES:
            break
        kept.append(line)
        used += size
    if sum(line.startswith("```") for line in kept) % 2:
        kept.append("```")
    return "\n".join(kept)
