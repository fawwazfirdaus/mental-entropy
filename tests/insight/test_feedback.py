"""Tests for the insight feedback loop."""

from __future__ import annotations

import pytest

from mental_entropy.insight.feedback import (
    MES_NOISE_FLOOR,
    STABLE_BAND,
    compute_feedback,
)
from mental_entropy.insight.types import (
    FeedbackContext,
    FeedbackResult,
    SubscoreDelta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(
    state: str = "overwhelm",
    mes: float = 62.5,
    target: str = "decrease",
    subscores: dict[str, float] | None = None,
) -> FeedbackContext:
    """Create a FeedbackContext for testing."""
    return FeedbackContext(
        timestamp="2026-04-01T10:00:00Z",
        state=state,
        mes_at_insight=mes,
        subscores_at_insight=subscores or {
            "prediction_coherence": 2.8,
            "compression_progress": 1.9,
        },
        target_direction=target,
    )


# ---------------------------------------------------------------------------
# Target direction: decrease (overwhelm, stuck, active_integration)
# ---------------------------------------------------------------------------


class TestDecreaseTarget:
    """Test feedback when target is 'decrease'."""

    def test_effective_decrease(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(target="decrease", mes=62.5),
            new_mes=57.0,
        )
        assert result.effective is True
        assert result.moved_correctly is True
        assert result.mes_delta == -5.5

    def test_ineffective_increase(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(target="decrease", mes=62.5),
            new_mes=68.0,
        )
        assert result.effective is False
        assert result.moved_correctly is False
        assert result.mes_delta == 5.5

    def test_noise_floor_not_effective(self) -> None:
        """Small decrease within noise floor is not effective."""
        result = compute_feedback(
            feedback_context=_ctx(target="decrease", mes=62.5),
            new_mes=62.0,
        )
        assert result.effective is False
        assert result.moved_correctly is False  # delta of -0.5 is within noise

    def test_just_beyond_noise_floor(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(target="decrease", mes=62.5),
            new_mes=62.5 - MES_NOISE_FLOOR - 0.1,
        )
        assert result.effective is True
        assert result.moved_correctly is True


# ---------------------------------------------------------------------------
# Target direction: increase (rigidity)
# ---------------------------------------------------------------------------


class TestIncreaseTarget:
    """Test feedback when target is 'increase'."""

    def test_effective_increase(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(state="rigidity", target="increase", mes=25.0),
            new_mes=30.0,
        )
        assert result.effective is True
        assert result.moved_correctly is True
        assert result.mes_delta == 5.0

    def test_ineffective_decrease(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(state="rigidity", target="increase", mes=25.0),
            new_mes=22.0,
        )
        assert result.effective is False
        assert result.moved_correctly is False

    def test_noise_floor_not_effective(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(state="rigidity", target="increase", mes=25.0),
            new_mes=25.5,
        )
        assert result.effective is False


# ---------------------------------------------------------------------------
# Target direction: stable (baseline, insufficient_data)
# ---------------------------------------------------------------------------


class TestStableTarget:
    """Test feedback when target is 'stable'."""

    def test_stable_within_band(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(state="baseline", target="stable", mes=45.0),
            new_mes=47.0,
        )
        assert result.effective is True
        assert result.moved_correctly is True
        assert result.mes_delta == 2.0

    def test_stable_at_band_edge(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(state="baseline", target="stable", mes=45.0),
            new_mes=45.0 + STABLE_BAND,
        )
        assert result.effective is True

    def test_unstable_beyond_band(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(state="baseline", target="stable", mes=45.0),
            new_mes=45.0 + STABLE_BAND + 0.1,
        )
        assert result.effective is False

    def test_no_change_is_effective(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(state="baseline", target="stable", mes=45.0),
            new_mes=45.0,
        )
        assert result.effective is True
        assert result.mes_delta == 0.0


# ---------------------------------------------------------------------------
# Subscore deltas
# ---------------------------------------------------------------------------


class TestSubscoreDeltas:
    """Test per-dimension subscore tracking."""

    def test_subscore_improvement(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(
                subscores={"prediction_coherence": 2.8, "compression_progress": 1.9},
            ),
            new_mes=57.0,
            new_subscores={"prediction_coherence": 3.2, "compression_progress": 2.5},
        )
        assert "prediction_coherence" in result.subscore_deltas
        assert result.subscore_deltas["prediction_coherence"].delta == 0.4
        assert result.subscore_deltas["prediction_coherence"].improved is True
        assert result.subscore_deltas["compression_progress"].delta == 0.6
        assert result.subscore_deltas["compression_progress"].improved is True

    def test_subscore_decline(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(
                subscores={"prediction_coherence": 3.0},
            ),
            new_mes=57.0,
            new_subscores={"prediction_coherence": 2.5},
        )
        assert result.subscore_deltas["prediction_coherence"].delta == -0.5
        assert result.subscore_deltas["prediction_coherence"].improved is False

    def test_no_new_subscores(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(),
            new_mes=57.0,
            new_subscores=None,
        )
        assert result.subscore_deltas == {}

    def test_partial_subscores(self) -> None:
        """Only dimensions present in both old and new are tracked."""
        result = compute_feedback(
            feedback_context=_ctx(
                subscores={"prediction_coherence": 2.8, "compression_progress": 1.9},
            ),
            new_mes=57.0,
            new_subscores={"prediction_coherence": 3.2},
        )
        assert "prediction_coherence" in result.subscore_deltas
        assert "compression_progress" not in result.subscore_deltas


# ---------------------------------------------------------------------------
# Interpretation text
# ---------------------------------------------------------------------------


class TestInterpretation:
    """Test human-readable interpretation."""

    def test_effective_decrease_interpretation(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(target="decrease", mes=62.5),
            new_mes=57.0,
        )
        assert "decreased" in result.interpretation
        assert "right direction" in result.interpretation

    def test_wrong_direction_interpretation(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(target="decrease", mes=62.5),
            new_mes=68.0,
        )
        assert "opposite" in result.interpretation

    def test_stable_interpretation(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(state="baseline", target="stable", mes=45.0),
            new_mes=46.0,
        )
        assert "stable" in result.interpretation.lower()

    def test_subscore_summary_in_interpretation(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(
                subscores={"prediction_coherence": 2.8, "compression_progress": 1.9},
            ),
            new_mes=57.0,
            new_subscores={"prediction_coherence": 3.2, "compression_progress": 1.5},
        )
        assert "improved" in result.interpretation
        assert "declined" in result.interpretation


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    """Verify FeedbackResult structure."""

    def test_all_fields_present(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(),
            new_mes=57.0,
        )
        assert isinstance(result, FeedbackResult)
        assert isinstance(result.effective, bool)
        assert isinstance(result.mes_delta, float)
        assert isinstance(result.target_direction, str)
        assert isinstance(result.moved_correctly, bool)
        assert isinstance(result.subscore_deltas, dict)
        assert isinstance(result.interpretation, str)

    def test_mes_delta_is_rounded(self) -> None:
        result = compute_feedback(
            feedback_context=_ctx(mes=62.5),
            new_mes=57.123456,
        )
        # Should be rounded to 2 decimal places
        assert result.mes_delta == round(57.123456 - 62.5, 2)
