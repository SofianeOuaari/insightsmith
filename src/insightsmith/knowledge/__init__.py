"""Reference material an agent can consult, as opposed to data it may not see.

The coder writes Polars, and a small local model's memory of the Polars API is
the weakest link in the chain — it reaches for ``groupby`` and ``sort_values``
because it has read far more pandas than Polars. §7's answer to a bad snippet is
the retry loop, but a retry is expensive and a wrong API name is cheap to
prevent. So the package ships a Polars reference and retrieves the few sections
that bear on the question at hand.

This is the mirror image of :mod:`insightsmith.profiling.card`. The card is how
the *data* reaches the model, minimised and masked. This is how *documentation*
reaches it, budgeted and ranked. Neither ever sends a raw row.
"""

from __future__ import annotations

from insightsmith.knowledge.guide import GUIDE_FILE, Section, sections
from insightsmith.knowledge.retrieve import (
    CODER_EXCLUDES,
    DEFAULT_BUDGET,
    reference,
    retrieve,
)

__all__ = [
    "CODER_EXCLUDES",
    "DEFAULT_BUDGET",
    "GUIDE_FILE",
    "Section",
    "reference",
    "retrieve",
    "sections",
]
