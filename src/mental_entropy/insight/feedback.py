"""Feedback loop — measure whether an insight moved MES in the right direction.

Compares MES and subscores at insight time vs follow-up to determine whether
the insight was effective. The target direction depends on the user's state
at insight time:
- overwhelm/stuck → MES should decrease
- rigidity → MES should increase
- active_integration → MES should continue decreasing
- baseline → MES should remain stable
- insufficient_data → MES should remain stable
"""

from __future__ import annotations

from mental_entropy.insight.types import (
    FeedbackContext,
    FeedbackResult,
    SubscoreDelta,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum MES change to count as "moved" (avoids noise).
MES_NOISE_FLOOR: float = 1.0

#: For "stable" target direction, movement within this band is considered stable.
STABLE_BAND: float = 3.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_feedback(
    *,
    feedback_context: FeedbackContext,
    new_mes: float,
    new_subscores: dict[str, float] | None = None,
) -> FeedbackResult:
    """Compute the effectiveness of a previous insight.

    Compares the MES and subscores at insight time (stored in feedback_context)
    with the new values to determine whether the insight moved entropy in the
    right direction.

    Args:
        feedback_context: Snapshot from the original /insight response.
        new_mes: The user's MES score at follow-up.
        new_subscores: Optional updated FEP dimension scores (1-5).

    Returns:
        FeedbackResult with effectiveness assessment.
    """
    mes_delta = new_mes - feedback_context.mes_at_insight
    target = feedback_context.target_direction

    # Determine if MES moved in the correct direction
    moved_correctly = _check_direction(mes_delta, target)

    # Compute per-subscore deltas
    subscore_deltas: dict[str, SubscoreDelta] = {}
    if new_subscores:
        for dim, old_value in feedback_context.subscores_at_insight.items():
            if dim in new_subscores:
                delta = new_subscores[dim] - old_value
                # Higher subscores = better (less entropy contribution)
                improved = delta > 0
                subscore_deltas[dim] = SubscoreDelta(
                    delta=round(delta, 2),
                    improved=improved,
                )

    # Build interpretation
    interpretation = _interpret(
        mes_delta=mes_delta,
        target=target,
        moved_correctly=moved_correctly,
        subscore_deltas=subscore_deltas,
    )

    # Effective = moved in the right direction by a meaningful amount
    if target == "stable":
        effective = abs(mes_delta) <= STABLE_BAND
    else:
        effective = moved_correctly and abs(mes_delta) > MES_NOISE_FLOOR

    return FeedbackResult(
        effective=effective,
        mes_delta=round(mes_delta, 2),
        target_direction=target,
        moved_correctly=moved_correctly,
        subscore_deltas=subscore_deltas,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_direction(mes_delta: float, target: str) -> bool:
    """Check if MES moved in the target direction."""
    if target == "decrease":
        return mes_delta < -MES_NOISE_FLOOR
    elif target == "increase":
        return mes_delta > MES_NOISE_FLOOR
    elif target == "stable":
        return abs(mes_delta) <= STABLE_BAND
    return False


def _interpret(
    *,
    mes_delta: float,
    target: str,
    moved_correctly: bool,
    subscore_deltas: dict[str, SubscoreDelta],
) -> str:
    """Generate a human-readable interpretation of the feedback."""
    direction_word = "decreased" if mes_delta < 0 else "increased" if mes_delta > 0 else "unchanged"
    abs_delta = abs(mes_delta)

    if target == "stable":
        if abs_delta <= STABLE_BAND:
            base = f"MES {direction_word} by {abs_delta:.1f} points — within the stable band. The insight maintained equilibrium."
        else:
            base = f"MES {direction_word} by {abs_delta:.1f} points — outside the stable band ({STABLE_BAND:.0f} pts). The insight may have disrupted balance."
    elif moved_correctly:
        base = f"MES {direction_word} by {abs_delta:.1f} points — insight moved entropy in the right direction (target: {target})."
    else:
        if abs_delta <= MES_NOISE_FLOOR:
            base = f"MES {direction_word} by {abs_delta:.1f} points — no significant movement (target: {target})."
        else:
            base = f"MES {direction_word} by {abs_delta:.1f} points — entropy moved opposite to target direction ({target})."

    # Add subscore summary
    if subscore_deltas:
        improved = [k for k, v in subscore_deltas.items() if v.improved]
        declined = [k for k, v in subscore_deltas.items() if not v.improved and v.delta != 0]
        parts = []
        if improved:
            parts.append(f"{len(improved)} dimension(s) improved")
        if declined:
            parts.append(f"{len(declined)} dimension(s) declined")
        if parts:
            base += f" {', '.join(parts)}."

    return base
