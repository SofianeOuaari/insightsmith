"""What can be checked about an answer without asking a model.

§8 calls the critic "what separates this from a toy", and the checks it names —
sample size, a Pearson correlation on outliered data, forty untracked
comparisons, a trend read off three rows — have one thing in common: **every one
of them is computable.** Asking an 8B model whether a result is statistically
sound produces confident prose with nothing behind it, which is the failure this
milestone exists to reduce, not to reproduce.

So the division is deliberate. This module measures; it never guesses, never
calls a provider, and returns the same caveats for the same inputs. The one
genuine judgement — did this code answer the question that was asked — is the
only part handed to a model, in :mod:`insightsmith.agents.critic`.

Everything here reads the :class:`~insightsmith.profiling.Profile`, not the
dataset card. The card is trimmed to fit a context window and may have dropped
the very statistic a check needs; the profile is complete, and none of this
leaves the machine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

import polars as pl

from insightsmith.profiling import ColumnProfile, Profile

__all__ = [
    "Caveat",
    "Critique",
    "Severity",
    "Verdict",
    "confidence_for",
    "review",
    "verdict_for",
]

#: Below this many rows, an aggregate is describing individuals.
MIN_ROWS: Final = 30
#: A group this small cannot support a comparison, let alone a trend.
MIN_GROUP_ROWS: Final = 5
#: Fraction of a column that must be outlying before Pearson is the wrong tool.
OUTLIER_RATE: Final = 0.01
#: |skew| past which the mean stops describing a typical value.
SKEW_LIMIT: Final = 2.0
#: Null rate worth mentioning when the code never acknowledges it.
NULL_RATE: Final = 0.05
#: This many p-values in one table is a multiple-comparison problem.
MANY_COMPARISONS: Final = 10
#: Points needed before a line is a trend rather than two dots.
MIN_TREND_POINTS: Final = 3
#: Phrasings that ask for one row per something.
_BREAKDOWN = re.compile(
    r"\b(?:per|by|for each|for every|across|grouped by|broken down by)\s+(?:the\s+)?(.+)",
    re.IGNORECASE,
)
#: How many words after the grouping word may name a column ("market size").
_BREAKDOWN_WORDS: Final = 3


class Severity(str, Enum):
    """How much a caveat should change what the reader does."""

    NOTE = "note"  # worth knowing
    WARNING = "warning"  # the answer can mislead if read plainly
    SERIOUS = "serious"  # the answer does not support what it invites you to conclude


class Verdict(str, Enum):
    """The headline, for a reader who reads nothing else."""

    SOUND = "sound"  # answers the question, nothing flagged
    QUALIFIED = "qualified"  # answers the question; read the caveats
    UNSOUND = "unsound"  # does not answer it, or the number cannot be used


_WEIGHT: Final[dict[Severity, float]] = {
    Severity.NOTE: 0.05,
    Severity.WARNING: 0.15,
    Severity.SERIOUS: 0.30,
}
#: What it costs to have answered a different question than the one asked.
_UNANSWERED_WEIGHT: Final = 0.80
_ORDER: Final[dict[Severity, int]] = {Severity.SERIOUS: 0, Severity.WARNING: 1, Severity.NOTE: 2}


@dataclass(frozen=True, slots=True)
class Caveat:
    """One thing that is true about this answer and easy to miss."""

    code: str
    severity: Severity
    message: str
    #: True when the number itself cannot be used, as opposed to needing care.
    fatal: bool = False


@dataclass(slots=True)
class Critique:
    """A verdict, what it rests on, and how much of the answer survives it."""

    verdict: Verdict
    caveats: list[Caveat] = field(default_factory=list)
    confidence: float = 1.0
    #: The model's judgement, or None when it was not asked.
    answered: bool | None = None
    answered_reason: str = ""

    @property
    def clean(self) -> bool:
        return self.verdict is Verdict.SOUND


def review(
    *,
    question: str,
    code: str,
    profile: Profile,
    frame: pl.DataFrame | None = None,
    value: Any = None,
) -> list[Caveat]:
    """Every caveat that can be measured, most serious first.

    ``frame`` is the result when the answer is a table, ``value`` when it is a
    single number. Neither is sent anywhere — the checks run here.
    """
    found = [
        _sampled_source(profile),
        _small_source(profile),
        _tiny_groups(frame),
        _correlation_on_outliers(code, profile),
        _mean_on_skewed(code, profile),
        _unacknowledged_nulls(code, profile),
        _multiple_comparisons(frame),
        _non_finite(frame, value),
        _short_trend(question, frame),
        _ungrouped_result(question, profile, frame, value),
    ]
    caveats = [caveat for caveat in found if caveat is not None]
    caveats.sort(key=lambda caveat: _ORDER[caveat.severity])
    return caveats


def verdict_for(caveats: list[Caveat], answered: bool | None) -> Verdict:
    """Answering a different question is a different failure from a shaky one."""
    if answered is False or any(caveat.fatal for caveat in caveats):
        return Verdict.UNSOUND
    return Verdict.QUALIFIED if caveats else Verdict.SOUND


def confidence_for(caveats: list[Caveat], answered: bool | None) -> float:
    """A bounded index derived from the caveats found.

    **This is not a probability that the answer is correct**, and nothing here
    measures that. It is a monotone summary: each caveat discounts it by a fixed
    weight for its severity, so more or worse caveats always score lower and the
    same inputs always score the same. Its only job is to sort and to signal;
    read the caveats, which say something specific.
    """
    score = 1.0
    for caveat in caveats:
        score *= 1.0 - _WEIGHT[caveat.severity]
    if answered is False:
        score *= 1.0 - _UNANSWERED_WEIGHT
    return round(max(score, 0.0), 2)


# --------------------------------------------------------------------------- #
# the checks
# --------------------------------------------------------------------------- #


def _sampled_source(profile: Profile) -> Caveat | None:
    """A stride sample answers about the sample, and says so nowhere else."""
    if not profile.estimated:
        return None
    return Caveat(
        code="sampled-source",
        severity=Severity.SERIOUS,
        message=(
            f"the file was too large to read whole, so this was computed on "
            f"{profile.sampled_rows:,} sampled rows out of roughly {profile.n_rows:,}. "
            "It describes the sample, not the file."
        ),
    )


def _small_source(profile: Profile) -> Caveat | None:
    if profile.n_rows >= MIN_ROWS or profile.n_rows <= 0:
        return None
    return Caveat(
        code="small-source",
        severity=Severity.WARNING,
        message=(
            f"the source has {profile.n_rows} rows, so any aggregate here is "
            "describing a handful of individuals rather than a population."
        ),
    )


def _tiny_groups(frame: pl.DataFrame | None) -> Caveat | None:
    """§8's "a group with n=3 being described as a trend", measured."""
    if frame is None or frame.height == 0:
        return None
    for name, dtype in frame.schema.items():
        # Numeric, not integer: a count that has been through any arithmetic —
        # a rate, a share, a square root — comes back as a float and would
        # otherwise walk straight past the one check meant to catch it.
        if not dtype.is_numeric() or not _is_count(name):
            continue
        small = frame.filter(pl.col(name) < MIN_GROUP_ROWS)
        if small.height == 0:
            continue
        smallest = small[name].min()
        return Caveat(
            code="tiny-groups",
            severity=Severity.WARNING,
            message=(
                f"{small.height} of {frame.height} groups rest on fewer than "
                f"{MIN_GROUP_ROWS} rows (smallest: {smallest!r}). Differences that "
                "size are usually noise."
            ),
        )
    return None


def _correlation_on_outliers(code: str, profile: Profile) -> Caveat | None:
    """§8's Pearson-on-outliered-data check."""
    if not _mentions(code, ("pl.corr", ".corr(", "pearson")):
        return None
    if "spearman" in code.lower():
        return None
    affected = [
        column.name
        for column in profile.columns
        if _named_in(column.name, code) and _outlier_rate(column, profile) >= OUTLIER_RATE
    ]
    if not affected:
        return None
    return Caveat(
        code="correlation-outliers",
        severity=Severity.SERIOUS,
        message=(
            f"Pearson correlation assumes a linear relationship and no heavy tails, but "
            f"{_join(affected)} {_verb(affected)} outliers. A few extreme rows can create "
            'or hide this number entirely — check it against method="spearman".'
        ),
    )


def _mean_on_skewed(code: str, profile: Profile) -> Caveat | None:
    if not _mentions(code, (".mean(", "pl.mean")):
        return None
    skewed = [
        column.name
        for column in profile.columns
        if _named_in(column.name, code)
        and column.numeric is not None
        and column.numeric.skew is not None
        and abs(column.numeric.skew) >= SKEW_LIMIT
    ]
    if not skewed:
        return None
    return Caveat(
        code="mean-on-skewed",
        severity=Severity.WARNING,
        message=(
            f"{_join(skewed)} {_verb(skewed, 'is', 'are')} heavily skewed, so the mean "
            "sits away from any typical value and a few large rows dominate it. The "
            "median describes this better."
        ),
    )


def _unacknowledged_nulls(code: str, profile: Profile) -> Caveat | None:
    """Polars aggregates skip nulls silently, which quietly changes the denominator."""
    if _mentions(code, ("drop_nulls", "fill_null", "is_null", "null_count", "drop_nans")):
        return None
    missing = [
        f"{column.name} ({column.null_rate:.0%})"
        for column in profile.columns
        if _named_in(column.name, code) and column.null_rate >= NULL_RATE
    ]
    if not missing:
        return None
    return Caveat(
        code="unacknowledged-nulls",
        severity=Severity.WARNING,
        message=(
            f"{_join(missing)} {_verb(missing, 'has', 'have')} missing values the code "
            "never addresses. Polars aggregates skip nulls, so those rows left the "
            "denominator without saying so."
        ),
    )


def _multiple_comparisons(frame: pl.DataFrame | None) -> Caveat | None:
    """§8's "a significant p-value out of 40 untracked comparisons"."""
    if frame is None or frame.height < MANY_COMPARISONS:
        return None
    for name, dtype in frame.schema.items():
        if not dtype.is_numeric() or not _is_p_value(name):
            continue
        significant = frame.filter(pl.col(name) < 0.05).height
        expected = frame.height * 0.05
        return Caveat(
            code="multiple-comparisons",
            severity=Severity.SERIOUS,
            message=(
                f"{frame.height} p-values are reported together and {significant} fall "
                f"below 0.05, where chance alone would produce about {expected:.0f}. "
                "Without a correction these cannot be read as individually significant."
            ),
        )
    return None


def _non_finite(frame: pl.DataFrame | None, value: Any) -> Caveat | None:
    """Division by zero survives as inf or NaN and prints like a number."""
    if isinstance(value, float) and not _finite(value):
        return Caveat(
            code="non-finite",
            severity=Severity.SERIOUS,
            message=f"the result is {value}, which means the calculation divided by zero "
            "or lost its data somewhere.",
            fatal=True,
        )
    if frame is None:
        return None
    for name, dtype in frame.schema.items():
        if not dtype.is_float():
            continue
        column = frame[name]
        bad = int(column.is_nan().sum() or 0) + int(column.is_infinite().sum() or 0)
        if bad:
            return Caveat(
                code="non-finite",
                severity=Severity.SERIOUS,
                message=(
                    f"{name} holds {bad} non-finite value(s) — a division by zero or an "
                    "empty group. Those rows are not numbers and must not be summarised."
                ),
                fatal=True,
            )
    return None


def _ungrouped_result(
    question: str,
    profile: Profile,
    frame: pl.DataFrame | None,
    value: Any,
) -> Caveat | None:
    """A question asking for a breakdown, answered with one number.

    "average violations *per year*" against a table with a Year column wants a
    row per year, and a single figure is a different question's answer. This is
    the one wrong-question case that does not need a model to spot, which
    matters: asked to judge these, a small model said "sound" every time.

    The grouping word alone is not enough — "revenue per customer" is often a
    rate, not a breakdown — so the noun after it has to name a real column.
    """
    if frame is not None and frame.height > 1:
        return None
    if frame is None and value is None:
        return None

    columns = {_normalise(column.name): column.name for column in profile.columns}
    for match in _BREAKDOWN.finditer(question):
        words = match.group(1).split()
        for size in range(min(_BREAKDOWN_WORDS, len(words)), 0, -1):
            column = columns.get(_normalise(" ".join(words[:size])))
            if column is not None:
                return Caveat(
                    code="ungrouped-result",
                    severity=Severity.SERIOUS,
                    message=(
                        f"the question asks for a breakdown by {column}, but the result "
                        f"is a single row. This is one overall figure, not one per "
                        f"{column}."
                    ),
                )
    return None


def _normalise(text: str) -> str:
    """Fold the ways a column name and a question refer to the same thing."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _short_trend(question: str, frame: pl.DataFrame | None) -> Caveat | None:
    if not _mentions(question, ("trend", "over time", "growth", "trajectory", "time series")):
        return None
    if frame is None or frame.height >= MIN_TREND_POINTS:
        return None
    return Caveat(
        code="short-trend",
        severity=Severity.WARNING,
        message=(
            f"the question asks about movement over time but the result has "
            f"{frame.height} point(s), which is not enough to show a direction."
        ),
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _mentions(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _named_in(column: str, code: str) -> bool:
    """Whether the code references this column, by the quotes Polars needs.

    Substring matching would let a column called ``n`` match nearly any snippet;
    a column is addressed in Polars as a string literal, so the quotes are the
    reliable signal.
    """
    return f'"{column}"' in code or f"'{column}'" in code


def _outlier_rate(column: ColumnProfile, profile: Profile) -> float:
    rows = profile.sampled_rows or profile.n_rows
    if column.numeric is None or rows <= 0:
        return 0.0
    return max(column.iqr_outliers, column.modified_z_outliers) / rows


def _is_count(name: str) -> bool:
    """Names that can only mean "how many".

    "size" is deliberately absent: it means bytes or dimensions at least as
    often as it means a row count, and a caveat that says "groups rest on fewer
    than 5 rows" about a column of megabytes is worse than no caveat at all.
    """
    lowered = name.lower()
    return lowered in {"count", "n", "len", "rows", "n_rows", "n_obs", "num"} or lowered.endswith(
        ("_count", "_n", "_len", "_rows")
    )


def _is_p_value(name: str) -> bool:
    lowered = name.lower()
    return lowered in {"p", "pval", "pvalue", "p_value"} or lowered.endswith(
        ("_p", "_pval", "_pvalue", "_p_value")
    )


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _verb(names: list[str], singular: str = "carries", plural: str = "carry") -> str:
    """Caveats are read by people; a plural subject takes a plural verb."""
    return singular if len(names) == 1 else plural


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"
