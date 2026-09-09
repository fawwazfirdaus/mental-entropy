"""Tests for the MES state classification engine."""

from __future__ import annotations

import pytest

from mental_entropy.insight.classify import (
    FLAT_SLOPE,
    HIGH_MES,
    INTEGRATION_SLOPE,
    LOW_MES,
    MIN_CONFIDENCE,
    MIN_ENTRIES,
    ROLLING_STD_T,
    VOLATILITY_T,
    WEAK_DIM_T,
    _find_weak_dimensions,
    classify_mental_state,
)
from mental_entropy.insight.types import (
    ClassificationResult,
    MentalState,
    STATE_DESCRIPTION,
    STATE_DISPLAY,
    STATE_TARGET_DIRECTION,
    WeakDimension,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_kwargs(**overrides: object) -> dict:
    """Return a valid classification input dict with optional overrides."""
    defaults = {
        "current_mes": 45.0,
        "confidence": 0.8,
        "trend": "stable",
        "trend_slope": 0.0,
        "baseline_mes": 45.0,
        "n_entries": 10,
        "volatility": 0.15,
        "rolling_7d_std": 8.0,
        "current_subscores": {
            "prediction_coherence": 3.5,
            "model_complexity": 3.0,
            "compression_progress": 3.2,
            "belief_integration": 3.1,
            "precision_weighting": 3.3,
        },
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# State reachability — every state should be reachable
# ---------------------------------------------------------------------------


class TestStateReachability:
    """Verify that every MentalState is reachable from classify_mental_state."""

    def test_insufficient_data_low_entries(self) -> None:
        result = classify_mental_state(**_base_kwargs(n_entries=1))
        assert result.state == MentalState.INSUFFICIENT_DATA

    def test_insufficient_data_low_confidence(self) -> None:
        result = classify_mental_state(**_base_kwargs(confidence=0.05))
        assert result.state == MentalState.INSUFFICIENT_DATA

    def test_active_integration(self) -> None:
        result = classify_mental_state(**_base_kwargs(
            trend="improving",
            trend_slope=-2.0,
        ))
        assert result.state == MentalState.ACTIVE_INTEGRATION

    def test_overwhelm_via_volatility(self) -> None:
        result = classify_mental_state(**_base_kwargs(
            current_mes=65.0,
            volatility=0.35,
        ))
        assert result.state == MentalState.OVERWHELM

    def test_overwhelm_via_rolling_std(self) -> None:
        result = classify_mental_state(**_base_kwargs(
            current_mes=65.0,
            volatility=0.10,
            rolling_7d_std=15.0,
        ))
        assert result.state == MentalState.OVERWHELM

    def test_stuck(self) -> None:
        result = classify_mental_state(**_base_kwargs(
            current_mes=65.0,
            trend_slope=0.1,
            volatility=0.10,
            rolling_7d_std=5.0,
        ))
        assert result.state == MentalState.STUCK

    def test_rigidity(self) -> None:
        result = classify_mental_state(**_base_kwargs(
            current_mes=25.0,
            trend_slope=0.1,
        ))
        assert result.state == MentalState.RIGIDITY

    def test_baseline(self) -> None:
        result = classify_mental_state(**_base_kwargs())
        assert result.state == MentalState.BASELINE


# ---------------------------------------------------------------------------
# Priority ordering — higher-priority states override lower ones
# ---------------------------------------------------------------------------


class TestPriorityOrder:
    """Verify the decision tree priority order."""

    def test_insufficient_data_overrides_overwhelm(self) -> None:
        """Even with overwhelm signals, insufficient data wins."""
        result = classify_mental_state(**_base_kwargs(
            n_entries=1,
            current_mes=80.0,
            volatility=0.5,
        ))
        assert result.state == MentalState.INSUFFICIENT_DATA

    def test_active_integration_overrides_overwhelm(self) -> None:
        """Improving trajectory at high MES = active integration, not overwhelm."""
        result = classify_mental_state(**_base_kwargs(
            current_mes=70.0,
            trend="improving",
            trend_slope=-3.0,
            volatility=0.35,
        ))
        assert result.state == MentalState.ACTIVE_INTEGRATION

    def test_active_integration_overrides_stuck(self) -> None:
        """Improving trajectory overrides stuck, even with high MES."""
        result = classify_mental_state(**_base_kwargs(
            current_mes=65.0,
            trend="improving",
            trend_slope=-2.0,
            volatility=0.10,
            rolling_7d_std=5.0,
        ))
        assert result.state == MentalState.ACTIVE_INTEGRATION

    def test_overwhelm_overrides_stuck(self) -> None:
        """High MES + volatile = overwhelm, even if slope is flat."""
        result = classify_mental_state(**_base_kwargs(
            current_mes=65.0,
            trend_slope=0.1,
            volatility=0.35,
        ))
        assert result.state == MentalState.OVERWHELM


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    """Test threshold edge cases."""

    def test_mes_at_high_boundary(self) -> None:
        """MES exactly at HIGH_MES (55.0) should NOT trigger high states."""
        result = classify_mental_state(**_base_kwargs(
            current_mes=HIGH_MES,
            volatility=0.35,
        ))
        # 55.0 is not > 55.0, so not overwhelm/stuck
        assert result.state == MentalState.BASELINE

    def test_mes_just_above_high(self) -> None:
        """MES just above HIGH_MES should trigger high states."""
        result = classify_mental_state(**_base_kwargs(
            current_mes=55.1,
            volatility=0.35,
        ))
        assert result.state == MentalState.OVERWHELM

    def test_mes_at_low_boundary(self) -> None:
        """MES exactly at LOW_MES (35.0) should NOT trigger rigidity."""
        result = classify_mental_state(**_base_kwargs(
            current_mes=LOW_MES,
            trend_slope=0.0,
        ))
        assert result.state == MentalState.BASELINE

    def test_mes_just_below_low(self) -> None:
        """MES just below LOW_MES should trigger rigidity."""
        result = classify_mental_state(**_base_kwargs(
            current_mes=34.9,
            trend_slope=0.0,
        ))
        assert result.state == MentalState.RIGIDITY

    def test_n_entries_at_boundary(self) -> None:
        """n_entries exactly at MIN_ENTRIES should NOT trigger insufficient data."""
        result = classify_mental_state(**_base_kwargs(n_entries=MIN_ENTRIES))
        assert result.state != MentalState.INSUFFICIENT_DATA

    def test_n_entries_below_boundary(self) -> None:
        result = classify_mental_state(**_base_kwargs(n_entries=MIN_ENTRIES - 1))
        assert result.state == MentalState.INSUFFICIENT_DATA

    def test_confidence_at_boundary(self) -> None:
        """Confidence exactly at MIN_CONFIDENCE should NOT trigger insufficient data."""
        result = classify_mental_state(**_base_kwargs(confidence=MIN_CONFIDENCE))
        assert result.state != MentalState.INSUFFICIENT_DATA

    def test_confidence_below_boundary(self) -> None:
        result = classify_mental_state(**_base_kwargs(confidence=MIN_CONFIDENCE - 0.01))
        assert result.state == MentalState.INSUFFICIENT_DATA

    def test_integration_slope_at_boundary(self) -> None:
        """Slope exactly at INTEGRATION_SLOPE should NOT trigger active integration."""
        result = classify_mental_state(**_base_kwargs(
            trend="improving",
            trend_slope=INTEGRATION_SLOPE,
        ))
        # -1.0 is not < -1.0
        assert result.state != MentalState.ACTIVE_INTEGRATION

    def test_integration_slope_below_boundary(self) -> None:
        result = classify_mental_state(**_base_kwargs(
            trend="improving",
            trend_slope=INTEGRATION_SLOPE - 0.1,
        ))
        assert result.state == MentalState.ACTIVE_INTEGRATION

    def test_flat_slope_at_boundary(self) -> None:
        """Slope exactly at FLAT_SLOPE should NOT count as flat."""
        result = classify_mental_state(**_base_kwargs(
            current_mes=65.0,
            trend_slope=FLAT_SLOPE,
            volatility=0.10,
            rolling_7d_std=5.0,
        ))
        # |0.5| is not < 0.5, so not stuck
        assert result.state != MentalState.STUCK


# ---------------------------------------------------------------------------
# ClassificationResult structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    """Verify the structure of ClassificationResult."""

    def test_result_has_all_fields(self) -> None:
        result = classify_mental_state(**_base_kwargs())
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.state, MentalState)
        assert isinstance(result.state_display, str)
        assert isinstance(result.description, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.primary_signals, list)
        assert isinstance(result.weakest_dimensions, list)
        assert isinstance(result.directive, str)
        assert isinstance(result.avoid, str)
        assert isinstance(result.target_direction, str)

    def test_state_display_matches(self) -> None:
        for state in MentalState:
            # Find kwargs that produce this state
            kwargs = _kwargs_for_state(state)
            result = classify_mental_state(**kwargs)
            assert result.state_display == STATE_DISPLAY[state]

    def test_description_matches(self) -> None:
        for state in MentalState:
            kwargs = _kwargs_for_state(state)
            result = classify_mental_state(**kwargs)
            assert result.description == STATE_DESCRIPTION[state]

    def test_target_direction_matches(self) -> None:
        for state in MentalState:
            kwargs = _kwargs_for_state(state)
            result = classify_mental_state(**kwargs)
            assert result.target_direction == STATE_TARGET_DIRECTION[state]

    def test_primary_signals_not_empty(self) -> None:
        for state in MentalState:
            kwargs = _kwargs_for_state(state)
            result = classify_mental_state(**kwargs)
            assert len(result.primary_signals) > 0, f"No signals for {state}"

    def test_confidence_preserved(self) -> None:
        result = classify_mental_state(**_base_kwargs(confidence=0.73))
        assert result.confidence == 0.73


# ---------------------------------------------------------------------------
# Weak dimensions
# ---------------------------------------------------------------------------


class TestWeakDimensions:
    """Test weak dimension detection."""

    def test_finds_weak_dims(self) -> None:
        subscores = {
            "prediction_coherence": 1.5,
            "model_complexity": 3.5,
            "compression_progress": 2.0,
            "belief_integration": 3.0,
            "precision_weighting": 3.5,
        }
        weak = _find_weak_dimensions(subscores)
        assert len(weak) == 2
        assert weak[0].name == "prediction_coherence"
        assert weak[0].value == 1.5
        assert weak[1].name == "compression_progress"
        assert weak[1].value == 2.0

    def test_no_weak_dims(self) -> None:
        subscores = {k: 4.0 for k in [
            "prediction_coherence", "model_complexity",
            "compression_progress", "belief_integration",
            "precision_weighting",
        ]}
        weak = _find_weak_dimensions(subscores)
        assert len(weak) == 0

    def test_max_two_dims_returned(self) -> None:
        subscores = {k: 1.0 for k in [
            "prediction_coherence", "model_complexity",
            "compression_progress", "belief_integration",
            "precision_weighting",
        ]}
        weak = _find_weak_dimensions(subscores)
        assert len(weak) == 2

    def test_weak_dim_has_display_info(self) -> None:
        subscores = {"prediction_coherence": 1.5}
        weak = _find_weak_dimensions(subscores)
        assert len(weak) == 1
        assert weak[0].display_name == "Prediction Coherence"
        assert len(weak[0].description) > 0

    def test_empty_subscores(self) -> None:
        weak = _find_weak_dimensions({})
        assert weak == []

    def test_weak_dims_in_classification(self) -> None:
        subscores = {
            "prediction_coherence": 1.5,
            "model_complexity": 3.5,
            "compression_progress": 2.0,
            "belief_integration": 3.0,
            "precision_weighting": 3.5,
        }
        result = classify_mental_state(**_base_kwargs(current_subscores=subscores))
        assert len(result.weakest_dimensions) == 2
        assert result.weakest_dimensions[0].name == "prediction_coherence"


# ---------------------------------------------------------------------------
# Optional parameters
# ---------------------------------------------------------------------------


class TestOptionalParams:
    """Test behavior with missing optional parameters."""

    def test_no_volatility(self) -> None:
        """Classification works without volatility."""
        result = classify_mental_state(**_base_kwargs(
            volatility=None,
            rolling_7d_std=None,
        ))
        assert isinstance(result, ClassificationResult)

    def test_no_subscores(self) -> None:
        """Classification works without subscores."""
        result = classify_mental_state(**_base_kwargs(current_subscores=None))
        assert isinstance(result, ClassificationResult)
        assert result.weakest_dimensions == []

    def test_high_mes_no_volatility_not_overwhelm(self) -> None:
        """High MES without volatility data should not be classified as overwhelm."""
        result = classify_mental_state(**_base_kwargs(
            current_mes=65.0,
            trend_slope=0.1,
            volatility=None,
            rolling_7d_std=None,
        ))
        # Without volatility, can't confirm overwhelm → falls to STUCK
        assert result.state == MentalState.STUCK


# ---------------------------------------------------------------------------
# Helper: produce kwargs for any given state
# ---------------------------------------------------------------------------


def _kwargs_for_state(state: MentalState) -> dict:
    """Return kwargs that will produce the given state."""
    if state == MentalState.INSUFFICIENT_DATA:
        return _base_kwargs(n_entries=1)
    elif state == MentalState.ACTIVE_INTEGRATION:
        return _base_kwargs(trend="improving", trend_slope=-2.0)
    elif state == MentalState.OVERWHELM:
        return _base_kwargs(current_mes=65.0, volatility=0.35)
    elif state == MentalState.STUCK:
        return _base_kwargs(
            current_mes=65.0, trend_slope=0.1,
            volatility=0.10, rolling_7d_std=5.0,
        )
    elif state == MentalState.RIGIDITY:
        return _base_kwargs(current_mes=25.0, trend_slope=0.0)
    else:  # BASELINE
        return _base_kwargs()
