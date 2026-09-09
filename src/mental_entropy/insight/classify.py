"""State classification from MES signals.

Priority-ordered decision tree that diagnoses the user's mental state
from their current MES, temporal features, and FEP-aligned subscores.

The classifier is deterministic and stateless — no LLM calls, no side
effects. It returns a ``ClassificationResult`` with the diagnosed state,
confidence, primary signals that triggered the classification, weakest
subscore dimensions, and state-specific directives.
"""

from __future__ import annotations

from mental_entropy.insight.types import (
    ClassificationResult,
    MentalState,
    STATE_DESCRIPTION,
    STATE_DISPLAY,
    STATE_TARGET_DIRECTION,
    WeakDimension,
)

# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

#: Minimum entries before a reliable classification is possible.
MIN_ENTRIES: int = 3

#: Minimum confidence (from the user-state endpoint) to classify.
MIN_CONFIDENCE: float = 0.1

#: Slope threshold (pts/week) for "actively improving" detection.
#: A slope more negative than this means entropy is declining meaningfully.
INTEGRATION_SLOPE: float = -1.0

#: MES threshold above which the user is in a high-entropy zone.
HIGH_MES: float = 55.0

#: MES threshold below which the user is in a low-entropy zone.
LOW_MES: float = 35.0

#: Volatility (coefficient of variation) threshold for overwhelm detection.
VOLATILITY_T: float = 0.25

#: Rolling 7-day standard deviation threshold for overwhelm detection.
ROLLING_STD_T: float = 12.0

#: Absolute slope threshold below which the user is considered "flat".
FLAT_SLOPE: float = 0.5

#: Subscore threshold — dimensions scoring at or below this are flagged.
WEAK_DIM_T: float = 2.5

# ---------------------------------------------------------------------------
# FEP dimension display metadata
# ---------------------------------------------------------------------------

#: Human-readable names for each FEP-aligned subscore dimension.
DIMENSION_DISPLAY_NAMES: dict[str, str] = {
    "prediction_coherence": "Prediction Coherence",
    "model_complexity": "Model Complexity",
    "compression_progress": "Compression Progress",
    "belief_integration": "Belief Integration",
    "precision_weighting": "Precision Weighting",
}

#: Brief descriptions for each FEP-aligned subscore dimension.
DIMENSION_DESCRIPTIONS: dict[str, str] = {
    "prediction_coherence": (
        "How well the mind's predictions align with experience — "
        "high coherence means thoughts flow logically."
    ),
    "model_complexity": (
        "The complexity of the internal world model — "
        "extreme values indicate either over-simplification or overload."
    ),
    "compression_progress": (
        "Active learning and insight formation — "
        "high values indicate the mind is successfully compressing experience."
    ),
    "belief_integration": (
        "How well beliefs are integrated and consistent — "
        "low values indicate unresolved contradictions."
    ),
    "precision_weighting": (
        "Appropriate allocation of attention and certainty — "
        "low values indicate misplaced confidence or excessive doubt."
    ),
}

# ---------------------------------------------------------------------------
# State-specific directives
# ---------------------------------------------------------------------------

#: Therapeutic posture and guidance per state.
STATE_DIRECTIVES: dict[MentalState, str] = {
    MentalState.INSUFFICIENT_DATA: (
        "Encourage continued journaling to build a personal baseline. "
        "Be warm and supportive without making diagnostic claims."
    ),
    MentalState.ACTIVE_INTEGRATION: (
        "Reflect and deepen. Name the insight that is forming. "
        "Help consolidate gains without introducing new complexity."
    ),
    MentalState.OVERWHELM: (
        "Ground, simplify, help find anchors. Reduce cognitive load. "
        "Offer one clear, concrete thing to focus on."
    ),
    MentalState.STUCK: (
        "Challenge gently, surface one specific pattern to examine. "
        "The goal is to create movement, not add more information."
    ),
    MentalState.RIGIDITY: (
        "Introduce novelty carefully. Ask one gentle question that "
        "challenges a single assumption. Avoid overwhelming."
    ),
    MentalState.BASELINE: (
        "Provide balanced reflective support. The user is in a healthy "
        "range — help maintain awareness without over-pathologizing."
    ),
}

#: What to avoid per state.
STATE_AVOID: dict[MentalState, str] = {
    MentalState.INSUFFICIENT_DATA: (
        "Do not diagnose, label, or interpret patterns from too little data."
    ),
    MentalState.ACTIVE_INTEGRATION: (
        "Do not introduce new complexity, new problems, or redirect attention. "
        "Let the integration process complete."
    ),
    MentalState.OVERWHELM: (
        "Do not introduce new complexity, challenge beliefs, or ask "
        "open-ended questions. Avoid long responses."
    ),
    MentalState.STUCK: (
        "Do not pile on more information or repeat what they already know. "
        "Avoid vague encouragement."
    ),
    MentalState.RIGIDITY: (
        "Do not confront or push too hard. Avoid presenting multiple "
        "challenges at once. Do not label their thinking as rigid."
    ),
    MentalState.BASELINE: (
        "Do not over-pathologize normal fluctuations or create problems "
        "where none exist."
    ),
}


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------


def _find_weak_dimensions(
    subscores: dict[str, float],
    threshold: float = WEAK_DIM_T,
    max_dims: int = 2,
) -> list[WeakDimension]:
    """Identify the weakest subscore dimensions below threshold.

    Args:
        subscores: Dict mapping dimension names to scores (1-5 scale).
        threshold: Dimensions at or below this are considered weak.
        max_dims: Maximum number of weak dimensions to return.

    Returns:
        List of WeakDimension objects, sorted by value ascending.
    """
    weak: list[tuple[str, float]] = []
    for name, value in subscores.items():
        if value <= threshold:
            weak.append((name, value))

    # Sort by value ascending (weakest first)
    weak.sort(key=lambda x: x[1])

    return [
        WeakDimension(
            name=name,
            value=round(value, 2),
            display_name=DIMENSION_DISPLAY_NAMES.get(
                name, name.replace("_", " ").title()
            ),
            description=DIMENSION_DESCRIPTIONS.get(name, ""),
        )
        for name, value in weak[:max_dims]
    ]


def _build_signals(
    *,
    current_mes: float,
    confidence: float,
    trend: str,
    trend_slope: float,
    n_entries: int,
    volatility: float | None = None,
    rolling_7d_std: float | None = None,
    state: MentalState,
) -> list[str]:
    """Build a human-readable list of the primary signals that triggered classification."""
    signals: list[str] = []

    if state == MentalState.INSUFFICIENT_DATA:
        if n_entries < MIN_ENTRIES:
            signals.append(f"n_entries={n_entries} (below {MIN_ENTRIES})")
        if confidence < MIN_CONFIDENCE:
            signals.append(f"confidence={confidence:.2f} (below {MIN_CONFIDENCE})")
        return signals

    if state == MentalState.ACTIVE_INTEGRATION:
        signals.append(f"trend={trend}")
        signals.append(f"trend_slope={trend_slope:.1f} pts/week (below {INTEGRATION_SLOPE})")
        return signals

    if state == MentalState.OVERWHELM:
        signals.append(f"current_mes={current_mes:.1f} (above {HIGH_MES})")
        if volatility is not None and volatility > VOLATILITY_T:
            signals.append(f"volatility={volatility:.2f} (above {VOLATILITY_T})")
        if rolling_7d_std is not None and rolling_7d_std > ROLLING_STD_T:
            signals.append(f"rolling_7d_std={rolling_7d_std:.1f} (above {ROLLING_STD_T})")
        return signals

    if state == MentalState.STUCK:
        signals.append(f"current_mes={current_mes:.1f} (above {HIGH_MES})")
        signals.append(f"|trend_slope|={abs(trend_slope):.1f} (below {FLAT_SLOPE})")
        return signals

    if state == MentalState.RIGIDITY:
        signals.append(f"current_mes={current_mes:.1f} (below {LOW_MES})")
        signals.append(f"|trend_slope|={abs(trend_slope):.1f} (below {FLAT_SLOPE})")
        return signals

    # BASELINE
    signals.append(f"current_mes={current_mes:.1f} (in normal range)")
    return signals


def classify_mental_state(
    *,
    current_mes: float,
    confidence: float,
    trend: str,
    trend_slope: float,
    baseline_mes: float,
    n_entries: int,
    volatility: float | None = None,
    rolling_7d_std: float | None = None,
    current_subscores: dict[str, float] | None = None,
) -> ClassificationResult:
    """Classify the user's mental state from MES signals.

    This is a priority-ordered decision tree:
    1. INSUFFICIENT_DATA — not enough history
    2. ACTIVE_INTEGRATION — improving trajectory (checked first to override level)
    3. OVERWHELM — high MES + volatile
    4. STUCK — high MES + flat
    5. RIGIDITY — low MES + flat
    6. BASELINE — healthy range, normal fluctuation

    Args:
        current_mes: Current MES score (0-100), typically from /user-state.
        confidence: Confidence in the current MES (0-1).
        trend: Trend direction ("improving", "worsening", "stable").
        trend_slope: MES slope in points per week (negative = improving).
        baseline_mes: User's all-time baseline MES.
        n_entries: Number of journal entries in history.
        volatility: Coefficient of variation of MES scores (optional).
        rolling_7d_std: Rolling 7-day standard deviation (optional).
        current_subscores: Dict mapping FEP dimension names to scores (1-5).

    Returns:
        ClassificationResult with state, signals, directives, and weak dimensions.
    """
    subscores = current_subscores or {}

    # 1. INSUFFICIENT_DATA
    if n_entries < MIN_ENTRIES or confidence < MIN_CONFIDENCE:
        state = MentalState.INSUFFICIENT_DATA
        return _build_result(
            state=state,
            current_mes=current_mes,
            confidence=confidence,
            trend=trend,
            trend_slope=trend_slope,
            n_entries=n_entries,
            volatility=volatility,
            rolling_7d_std=rolling_7d_std,
            subscores=subscores,
        )

    # 2. ACTIVE_INTEGRATION — improving trajectory overrides level
    if trend == "improving" and trend_slope < INTEGRATION_SLOPE:
        state = MentalState.ACTIVE_INTEGRATION
        return _build_result(
            state=state,
            current_mes=current_mes,
            confidence=confidence,
            trend=trend,
            trend_slope=trend_slope,
            n_entries=n_entries,
            volatility=volatility,
            rolling_7d_std=rolling_7d_std,
            subscores=subscores,
        )

    # 3. OVERWHELM — high + volatile
    is_high = current_mes > HIGH_MES
    is_volatile = (
        (volatility is not None and volatility > VOLATILITY_T)
        or (rolling_7d_std is not None and rolling_7d_std > ROLLING_STD_T)
    )
    if is_high and is_volatile:
        state = MentalState.OVERWHELM
        return _build_result(
            state=state,
            current_mes=current_mes,
            confidence=confidence,
            trend=trend,
            trend_slope=trend_slope,
            n_entries=n_entries,
            volatility=volatility,
            rolling_7d_std=rolling_7d_std,
            subscores=subscores,
        )

    # 4. STUCK — high + flat
    is_flat = abs(trend_slope) < FLAT_SLOPE
    if is_high and is_flat:
        state = MentalState.STUCK
        return _build_result(
            state=state,
            current_mes=current_mes,
            confidence=confidence,
            trend=trend,
            trend_slope=trend_slope,
            n_entries=n_entries,
            volatility=volatility,
            rolling_7d_std=rolling_7d_std,
            subscores=subscores,
        )

    # 5. RIGIDITY — low + flat
    is_low = current_mes < LOW_MES
    if is_low and is_flat:
        state = MentalState.RIGIDITY
        return _build_result(
            state=state,
            current_mes=current_mes,
            confidence=confidence,
            trend=trend,
            trend_slope=trend_slope,
            n_entries=n_entries,
            volatility=volatility,
            rolling_7d_std=rolling_7d_std,
            subscores=subscores,
        )

    # 6. BASELINE — everything else
    state = MentalState.BASELINE
    return _build_result(
        state=state,
        current_mes=current_mes,
        confidence=confidence,
        trend=trend,
        trend_slope=trend_slope,
        n_entries=n_entries,
        volatility=volatility,
        rolling_7d_std=rolling_7d_std,
        subscores=subscores,
    )


def _build_result(
    *,
    state: MentalState,
    current_mes: float,
    confidence: float,
    trend: str,
    trend_slope: float,
    n_entries: int,
    volatility: float | None,
    rolling_7d_std: float | None,
    subscores: dict[str, float],
) -> ClassificationResult:
    """Assemble a ClassificationResult from classification inputs."""
    signals = _build_signals(
        current_mes=current_mes,
        confidence=confidence,
        trend=trend,
        trend_slope=trend_slope,
        n_entries=n_entries,
        volatility=volatility,
        rolling_7d_std=rolling_7d_std,
        state=state,
    )
    weak_dims = _find_weak_dimensions(subscores)

    return ClassificationResult(
        state=state,
        state_display=STATE_DISPLAY[state],
        description=STATE_DESCRIPTION[state],
        confidence=round(confidence, 2),
        primary_signals=signals,
        weakest_dimensions=weak_dims,
        directive=STATE_DIRECTIVES[state],
        avoid=STATE_AVOID[state],
        target_direction=STATE_TARGET_DIRECTION[state],
    )
