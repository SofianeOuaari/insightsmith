"""The dataset card: what an agent is allowed to see.

Never paste a dataframe into a prompt. Everything an agent knows about the data
arrives through this object — schema, per-column statistics, quality flags, a
correlation shortlist, and a handful of stratified example rows with obvious PII
masked. Three consequences follow, and they are the reason the design insists on
it (§6):

* Token cost is flat regardless of file size, so a 4B local model is workable on
  a 40 GB source.
* No raw records leave the machine.
* The card hashes, so the same data yields the same plan and results cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Final

import polars as pl

from insightsmith.profiling import ColumnProfile, Profile
from insightsmith.profiling.pii import is_sensitive_column, mask_value
from insightsmith.profiling.schema import SemanticType

__all__ = ["DEFAULT_EXAMPLES", "MAX_CARD_BYTES", "DatasetCard", "build_card"]

#: The card must stay small enough to sit in a small model's context (§6).
MAX_CARD_BYTES: Final = 5_120
DEFAULT_EXAMPLES: Final = 5
#: Only report correlations strong enough to be worth a question.
CORRELATION_FLOOR: Final = 0.5
MAX_CORRELATIONS: Final = 8
_MAX_TOP_VALUES: Final = 3
_MAX_CELL_CHARS: Final = 40


@dataclass(slots=True)
class DatasetCard:
    """A compact, model-safe description of a source."""

    name: str
    n_rows: int
    n_columns: int
    estimated: bool
    columns: list[dict[str, Any]] = field(default_factory=list)
    quality: list[dict[str, str]] = field(default_factory=list)
    correlations: list[dict[str, Any]] = field(default_factory=list)
    candidate_keys: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)
    masked_kinds: list[str] = field(default_factory=list)
    redacted_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "rows": self.n_rows,
            "columns_count": self.n_columns,
            "estimated": self.estimated,
            "columns": self.columns,
        }
        for key, value in (
            ("quality", self.quality),
            ("correlations", self.correlations),
            ("candidate_keys", self.candidate_keys),
            ("examples", self.examples),
            ("redacted_columns", self.redacted_columns),
            ("masked", self.masked_kinds),
        ):
            if value:
                payload[key] = value
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, default=str)

    @property
    def size_bytes(self) -> int:
        return len(self.to_json().encode("utf-8"))

    @property
    def hash(self) -> str:
        """Stable digest, so identical data gives an identical plan."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]

    def column_names(self) -> set[str]:
        return {str(column["name"]) for column in self.columns}


def build_card(
    result: Profile,
    frame: pl.DataFrame | None = None,
    *,
    examples: int = DEFAULT_EXAMPLES,
    max_bytes: int = MAX_CARD_BYTES,
    include_examples: bool = True,
) -> DatasetCard:
    """Build a card from a profile, and optionally the sample it was built from.

    ``frame`` supplies example rows and correlations. Without it the card still
    describes the data, just without either. Pass ``include_examples=False`` to
    omit sample values entirely, which is the strictest posture short of not
    calling a model at all.
    """
    masked_kinds: set[str] = set()
    redacted = [c.name for c in result.columns if is_sensitive_column(c.name)]

    card = DatasetCard(
        name=result.source.path.name,
        n_rows=result.n_rows,
        n_columns=result.n_columns,
        estimated=result.estimated,
        columns=[_column_entry(c) for c in result.columns],
        quality=[
            {"column": issue.column or "", "issue": issue.kind, "detail": issue.message}
            for issue in result.issues
        ],
        candidate_keys=list(result.candidate_keys),
        redacted_columns=redacted,
    )

    if frame is not None:
        card.correlations = _correlations(frame, result)
        if include_examples and examples > 0:
            card.examples = _examples(frame, result, examples, masked_kinds)

    card.masked_kinds = sorted(masked_kinds)
    _fit_budget(card, max_bytes)
    return card


def _column_entry(column: ColumnProfile) -> dict[str, Any]:
    """One compact line per column. Verbosity here is what blows the budget."""
    entry: dict[str, Any] = {
        "name": column.name,
        "type": column.schema.semantic.value,
        "dtype": column.schema.dtype,
    }
    if column.null_rate:
        entry["null_rate"] = round(column.null_rate, 3)
    entry["unique"] = column.n_unique

    if column.numeric is not None:
        n = column.numeric
        entry["stats"] = {
            "min": _round(n.minimum),
            "median": _round(n.median),
            "max": _round(n.maximum),
        }
        if column.iqr_outliers or column.modified_z_outliers:
            entry["outliers"] = {"iqr": column.iqr_outliers, "mad": column.modified_z_outliers}
    elif column.temporal is not None:
        entry["range"] = [column.temporal.earliest, column.temporal.latest]
    elif column.categorical is not None and column.categorical.top:
        if is_sensitive_column(column.name):
            entry["top"] = "<redacted>"
        else:
            entry["top"] = [
                {"value": _shorten(value), "n": count}
                for value, count in column.categorical.top[:_MAX_TOP_VALUES]
            ]
    elif column.text is not None:
        entry["length"] = [column.text.min_length, column.text.max_length]
    return entry


def _correlations(frame: pl.DataFrame, result: Profile) -> list[dict[str, Any]]:
    """Numeric pairs worth asking about, strongest first."""
    numeric = [
        c.name
        for c in result.columns
        if c.schema.semantic is SemanticType.NUMERIC and c.name in frame.columns
    ]
    if len(numeric) < 2 or frame.height < 3:
        return []

    found: list[dict[str, Any]] = []
    for index, left in enumerate(numeric):
        for right in numeric[index + 1 :]:
            try:
                value = frame.select(pl.corr(left, right)).item()
            except (pl.exceptions.PolarsError, ValueError):
                continue
            if value is None:
                continue
            coefficient = float(value)
            if abs(coefficient) >= CORRELATION_FLOOR:
                found.append({"a": left, "b": right, "r": round(coefficient, 2)})

    found.sort(key=lambda item: -abs(float(item["r"])))
    return found[:MAX_CORRELATIONS]


def _examples(
    frame: pl.DataFrame, result: Profile, count: int, masked_kinds: set[str]
) -> list[dict[str, str]]:
    """A few rows, spread across the data rather than taken from the top.

    The head of a file is often sorted, so the first rows misrepresent it. Where
    a low-cardinality categorical exists, take one row per category; otherwise
    step evenly through the frame.
    """
    if frame.height == 0:
        return []
    indices = _stratified_indices(frame, result, count)
    rows: list[dict[str, str]] = []
    for index in indices:
        row: dict[str, str] = {}
        for name, value in frame.row(index, named=True).items():
            masked, kinds = mask_value(value, column=name)
            masked_kinds.update(kinds)
            row[name] = _shorten(masked)
        rows.append(row)
    return rows


def _stratified_indices(frame: pl.DataFrame, result: Profile, count: int) -> list[int]:
    strata = next(
        (
            c.name
            for c in result.columns
            if c.schema.semantic is SemanticType.CATEGORICAL
            and 1 < c.n_unique <= count
            and c.name in frame.columns
        ),
        None,
    )
    if strata is not None:
        seen: dict[str, int] = {}
        for position, value in enumerate(frame[strata].to_list()):
            key = str(value)
            if key not in seen:
                seen[key] = position
            if len(seen) >= count:
                break
        return sorted(seen.values())

    step = max(1, frame.height // count)
    return list(range(0, frame.height, step))[:count]


def _fit_budget(card: DatasetCard, max_bytes: int) -> None:
    """Trim until the card fits, dropping the least load-bearing parts first.

    A card that outgrows the budget defeats its purpose, so this is enforced
    rather than hoped for. Column entries are the last thing to go, because an
    agent that cannot see every column will invent one.
    """
    if card.size_bytes <= max_bytes:
        return

    while card.examples and card.size_bytes > max_bytes:
        card.examples.pop()
    while len(card.correlations) > 1 and card.size_bytes > max_bytes:
        card.correlations.pop()
    while len(card.quality) > 3 and card.size_bytes > max_bytes:
        card.quality.pop()

    # Per-column detail goes last and in order of dispensability. A wide frame
    # spends its whole budget here, and losing a column entirely is worse than
    # losing its statistics: an agent that cannot see a column will invent one.
    for key in ("outliers", "top", "range", "length", "stats", "null_rate", "unique", "dtype"):
        if card.size_bytes <= max_bytes:
            return
        for entry in card.columns:
            entry.pop(key, None)


def _round(value: float) -> float | int:
    rounded = round(value, 4)
    return int(rounded) if rounded == int(rounded) else rounded


def _shorten(text: str) -> str:
    text = str(text)
    return text if len(text) <= _MAX_CELL_CHARS else text[: _MAX_CELL_CHARS - 1] + "…"
