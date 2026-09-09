"""Data types for temporal/longitudinal MES analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ScoredEntry:
    """A single scored journal entry with timestamp.

    Represents one point in a user's MES trajectory over time.
    This is a stateless data container - the backend handles storage/retrieval.

    Attributes:
        timestamp: UTC datetime when the journal entry was created.
        mes_score: Overall MES score (0-100) for this entry.
        features: Optional full feature dict (ce_*, se_*, ne_*, cle_*, bc_*).
            If provided, enables sub-dimension temporal analysis.
            If None, only mes_score trajectory is analyzed.
    """

    timestamp: datetime
    mes_score: float
    features: dict[str, float] | None = None
