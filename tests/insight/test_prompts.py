"""Tests for insight prompt assembly."""

from __future__ import annotations

import pytest

from mental_entropy.insight.prompts import assemble_insight_prompt
from mental_entropy.insight.types import (
    ClassificationResult,
    FeedbackContext,
    InsightPrompt,
    MentalState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_kwargs(**overrides: object) -> dict:
    """Return valid prompt assembly inputs with optional overrides."""
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
        "journal_text": "Today I felt scattered but managed to focus on one task.",
        "timestamp": "2026-04-01T10:00:00Z",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------


class TestReturnStructure:
    """Verify the return tuple structure."""

    def test_returns_three_items(self) -> None:
        classification, prompt, feedback_ctx = assemble_insight_prompt(**_base_kwargs())
        assert isinstance(classification, ClassificationResult)
        assert isinstance(prompt, InsightPrompt)
        assert isinstance(feedback_ctx, FeedbackContext)

    def test_prompt_has_system(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs())
        assert len(prompt.system) > 100
        assert "Free Energy" in prompt.system

    def test_prompt_has_user(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs())
        assert "Journal Entry" in prompt.user
        assert "scattered" in prompt.user

    def test_prompt_has_model_recommendation(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs())
        assert "claude" in prompt.model_recommendation.lower()

    def test_prompt_has_max_tokens(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs())
        assert prompt.max_tokens > 0

    def test_prompt_has_temperature(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs())
        assert 0.0 <= prompt.temperature <= 1.0


# ---------------------------------------------------------------------------
# State-specific prompt content
# ---------------------------------------------------------------------------


class TestStateSpecificContent:
    """Verify that prompt content changes based on classified state."""

    def test_overwhelm_has_ground_directive(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs(
            current_mes=65.0, volatility=0.35,
        ))
        assert "ground" in prompt.system.lower() or "simplify" in prompt.system.lower()

    def test_overwhelm_shorter_tokens(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs(
            current_mes=65.0, volatility=0.35,
        ))
        assert prompt.max_tokens <= 300

    def test_stuck_has_challenge_directive(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs(
            current_mes=65.0, trend_slope=0.1,
            volatility=0.10, rolling_7d_std=5.0,
        ))
        assert "challenge" in prompt.system.lower() or "surface" in prompt.system.lower()

    def test_rigidity_has_novelty_directive(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs(
            current_mes=25.0, trend_slope=0.0,
        ))
        assert "novelty" in prompt.system.lower() or "question" in prompt.system.lower()

    def test_active_integration_has_reflect_directive(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs(
            trend="improving", trend_slope=-2.0,
        ))
        assert "reflect" in prompt.system.lower() or "consolidate" in prompt.system.lower()

    def test_insufficient_data_has_encourage_directive(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs(n_entries=1))
        assert "journaling" in prompt.system.lower() or "encourage" in prompt.system.lower()

    def test_baseline_has_balanced_directive(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs())
        assert "balanced" in prompt.system.lower() or "reflective" in prompt.system.lower()


# ---------------------------------------------------------------------------
# Dimension targeting in prompts
# ---------------------------------------------------------------------------


class TestDimensionTargeting:
    """Verify weak dimensions are targeted in prompts."""

    def test_weak_dims_included_in_system_prompt(self) -> None:
        subscores = {
            "prediction_coherence": 1.5,
            "model_complexity": 3.5,
            "compression_progress": 2.0,
            "belief_integration": 3.0,
            "precision_weighting": 3.5,
        }
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs(
            current_subscores=subscores,
        ))
        assert "Prediction Coherence" in prompt.system
        assert "Compression Progress" in prompt.system

    def test_no_weak_dims_no_targeting_section(self) -> None:
        subscores = {k: 4.0 for k in [
            "prediction_coherence", "model_complexity",
            "compression_progress", "belief_integration",
            "precision_weighting",
        ]}
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs(
            current_subscores=subscores,
        ))
        assert "Dimension Targeting" not in prompt.system

    def test_weak_dims_shown_in_user_message(self) -> None:
        subscores = {
            "prediction_coherence": 1.5,
            "model_complexity": 3.5,
            "compression_progress": 2.0,
            "belief_integration": 3.0,
            "precision_weighting": 3.5,
        }
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs(
            current_subscores=subscores,
        ))
        assert "Prediction Coherence" in prompt.user
        assert "1.5" in prompt.user


# ---------------------------------------------------------------------------
# Previous insight context
# ---------------------------------------------------------------------------


class TestPreviousInsight:
    """Verify previous insight is included in user message."""

    def test_previous_insight_included(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs(
            previous_insight={
                "text": "Last time we noticed you were juggling many threads.",
                "effectiveness": "effective",
            },
        ))
        assert "juggling many threads" in prompt.user
        assert "effective" in prompt.user

    def test_no_previous_insight(self) -> None:
        _, prompt, _ = assemble_insight_prompt(**_base_kwargs(
            previous_insight=None,
        ))
        assert "Previous Insight" not in prompt.user


# ---------------------------------------------------------------------------
# Feedback context
# ---------------------------------------------------------------------------


class TestFeedbackContext:
    """Verify feedback context is correctly populated."""

    def test_timestamp_preserved(self) -> None:
        _, _, ctx = assemble_insight_prompt(**_base_kwargs(
            timestamp="2026-04-01T10:00:00Z",
        ))
        assert ctx.timestamp == "2026-04-01T10:00:00Z"

    def test_state_preserved(self) -> None:
        _, _, ctx = assemble_insight_prompt(**_base_kwargs())
        assert ctx.state == "baseline"

    def test_mes_preserved(self) -> None:
        _, _, ctx = assemble_insight_prompt(**_base_kwargs(current_mes=62.5))
        assert ctx.mes_at_insight == 62.5

    def test_subscores_preserved(self) -> None:
        subscores = {
            "prediction_coherence": 2.8,
            "model_complexity": 3.1,
        }
        _, _, ctx = assemble_insight_prompt(**_base_kwargs(
            current_subscores=subscores,
        ))
        assert ctx.subscores_at_insight == subscores

    def test_target_direction_for_overwhelm(self) -> None:
        _, _, ctx = assemble_insight_prompt(**_base_kwargs(
            current_mes=65.0, volatility=0.35,
        ))
        assert ctx.target_direction == "decrease"

    def test_target_direction_for_rigidity(self) -> None:
        _, _, ctx = assemble_insight_prompt(**_base_kwargs(
            current_mes=25.0, trend_slope=0.0,
        ))
        assert ctx.target_direction == "increase"

    def test_target_direction_for_baseline(self) -> None:
        _, _, ctx = assemble_insight_prompt(**_base_kwargs())
        assert ctx.target_direction == "stable"

    def test_empty_timestamp_default(self) -> None:
        _, _, ctx = assemble_insight_prompt(**_base_kwargs(timestamp=None))
        assert ctx.timestamp == ""
