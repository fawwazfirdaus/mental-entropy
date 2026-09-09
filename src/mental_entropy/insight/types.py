"""Data types for the MES Insight Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MentalState(str, Enum):
    """Diagnosed mental state from MES signals.

    Priority order matters for classification:
    1. INSUFFICIENT_DATA — not enough history
    2. ACTIVE_INTEGRATION — improving trajectory (checked first to override level)
    3. OVERWHELM — high + volatile
    4. STUCK — high + flat
    5. RIGIDITY — low + flat
    6. BASELINE — healthy range, normal fluctuation
    """

    INSUFFICIENT_DATA = "insufficient_data"
    ACTIVE_INTEGRATION = "active_integration"
    OVERWHELM = "overwhelm"
    STUCK = "stuck"
    RIGIDITY = "rigidity"
    BASELINE = "baseline"


#: Human-readable display names for each state.
STATE_DISPLAY: dict[MentalState, str] = {
    MentalState.INSUFFICIENT_DATA: "Building Baseline",
    MentalState.ACTIVE_INTEGRATION: "Actively Integrating",
    MentalState.OVERWHELM: "Overwhelmed",
    MentalState.STUCK: "Stuck",
    MentalState.RIGIDITY: "Over-Compressed",
    MentalState.BASELINE: "Balanced",
}

#: Descriptions for each state.
STATE_DESCRIPTION: dict[MentalState, str] = {
    MentalState.INSUFFICIENT_DATA: (
        "Not enough journal history to diagnose a pattern. "
        "Keep journaling to build a personal baseline."
    ),
    MentalState.ACTIVE_INTEGRATION: (
        "Entropy is declining — your mind is actively compressing "
        "experience into a more coherent model. Learning is happening."
    ),
    MentalState.OVERWHELM: (
        "High entropy with significant volatility — too much unintegrated "
        "information. The mind is struggling to process."
    ),
    MentalState.STUCK: (
        "Persistently high entropy without movement — the mind's model "
        "isn't updating despite elevated disorganization."
    ),
    MentalState.RIGIDITY: (
        "Very low entropy that isn't changing — the world model may be "
        "over-compressed, not processing new information."
    ),
    MentalState.BASELINE: (
        "Entropy is in a healthy range with normal fluctuation. "
        "The mind is functioning within its typical patterns."
    ),
}

#: Target direction for MES movement per state.
#: "decrease" = MES should go down, "increase" = up, "stable" = maintain.
STATE_TARGET_DIRECTION: dict[MentalState, str] = {
    MentalState.INSUFFICIENT_DATA: "stable",
    MentalState.ACTIVE_INTEGRATION: "decrease",
    MentalState.OVERWHELM: "decrease",
    MentalState.STUCK: "decrease",
    MentalState.RIGIDITY: "increase",
    MentalState.BASELINE: "stable",
}


@dataclass(frozen=True, slots=True)
class WeakDimension:
    """A subscore dimension identified as below threshold."""

    name: str
    value: float
    display_name: str
    description: str


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Full state classification output."""

    state: MentalState
    state_display: str
    description: str
    confidence: float
    primary_signals: list[str]
    weakest_dimensions: list[WeakDimension]
    directive: str
    avoid: str
    target_direction: str


@dataclass(frozen=True, slots=True)
class InsightPrompt:
    """Ready-to-send LLM prompt payload."""

    system: str
    user: str
    model_recommendation: str
    max_tokens: int
    temperature: float


@dataclass(frozen=True, slots=True)
class FeedbackContext:
    """Snapshot stored by backend to enable the feedback loop."""

    timestamp: str
    state: str
    mes_at_insight: float
    subscores_at_insight: dict[str, float]
    target_direction: str


@dataclass(frozen=True, slots=True)
class SubscoreDelta:
    """Per-dimension change between insight and follow-up."""

    delta: float
    improved: bool


@dataclass(frozen=True, slots=True)
class FeedbackResult:
    """Effectiveness measurement of a previous insight."""

    effective: bool
    mes_delta: float
    target_direction: str
    moved_correctly: bool
    subscore_deltas: dict[str, SubscoreDelta]
    interpretation: str
