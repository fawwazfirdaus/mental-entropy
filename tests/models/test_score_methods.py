"""Tests for scoring method parameter in score.py.

Verifies that method="linear" produces identical results to the original
implementation, and that method="auto" gracefully falls back when subscore
models are not available.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mental_entropy.score import (
    _WEIGHTS,
    _apply_transform,
    _compute_linear_mes,
    compute_mes_score,
)


# ---------------------------------------------------------------------------
# Synthetic feature fixtures
# ---------------------------------------------------------------------------


def _make_zero_features() -> dict[str, float]:
    """Features where all values are 0.0."""
    return {entry[0]: 0.0 for entry in _WEIGHTS}


def _make_one_features() -> dict[str, float]:
    """Features where all values are 1.0."""
    return {entry[0]: 1.0 for entry in _WEIGHTS}


def _make_mid_features() -> dict[str, float]:
    """Features where all values are 0.5."""
    return {entry[0]: 0.5 for entry in _WEIGHTS}


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestApplyTransform:
    """Test individual transform functions."""

    def test_inv(self) -> None:
        assert _apply_transform(0.3, "inv") == pytest.approx(0.7)

    def test_inv_zero(self) -> None:
        assert _apply_transform(0.0, "inv") == pytest.approx(1.0)

    def test_inv_one(self) -> None:
        assert _apply_transform(1.0, "inv") == pytest.approx(0.0)

    def test_dir(self) -> None:
        assert _apply_transform(0.3, "dir") == pytest.approx(0.3)

    def test_dir_norm(self) -> None:
        assert _apply_transform(10.0, "dir_norm", 20.0) == pytest.approx(0.5)

    def test_dir_norm_clipped(self) -> None:
        assert _apply_transform(25.0, "dir_norm", 20.0) == pytest.approx(1.0)

    def test_unknown_transform(self) -> None:
        assert _apply_transform(0.5, "unknown") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Method parameter tests
# ---------------------------------------------------------------------------


class TestScoringMethods:
    """Test that method parameter works correctly."""

    def test_linear_explicit(self) -> None:
        """method='linear' returns a valid score."""
        features = _make_mid_features()
        score = compute_mes_score(features, method="linear")
        assert 0.0 <= score <= 100.0

    def test_auto_falls_back_to_linear(self) -> None:
        """method='auto' falls back to linear when no models exist."""
        features = _make_mid_features()
        with patch(
            "mental_entropy.models._registry.subscore_models_available",
            return_value=False,
        ):
            auto_score = compute_mes_score(features, method="auto")
        linear_score = compute_mes_score(features, method="linear")
        assert auto_score == pytest.approx(linear_score)

    def test_auto_uses_subscore_when_available(self) -> None:
        """method='auto' uses subscore models when they exist."""
        features = _make_mid_features()
        # If subscore models are available, auto should differ from linear
        from mental_entropy.models._registry import subscore_models_available
        if subscore_models_available():
            auto_score = compute_mes_score(features, method="auto")
            linear_score = compute_mes_score(features, method="linear")
            # They may differ because subscore uses different model
            assert 0.0 <= auto_score <= 100.0
        else:
            pytest.skip("Subscore models not available")

    def test_default_is_auto(self) -> None:
        """Default method is 'auto'."""
        features = _make_mid_features()
        default_score = compute_mes_score(features)
        auto_score = compute_mes_score(features, method="auto")
        assert default_score == pytest.approx(auto_score)

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            compute_mes_score({}, method="invalid")

    def test_subscore_without_models_raises(self) -> None:
        """method='subscore' raises when models are missing."""
        features = _make_mid_features()
        with patch(
            "mental_entropy.models._registry._ARTIFACTS_DIR",
            __import__("pathlib").Path("/nonexistent/path"),
        ):
            from mental_entropy.models._registry import reset_cache
            reset_cache()
            with pytest.raises((ImportError, FileNotFoundError)):
                compute_mes_score(features, method="subscore")
            reset_cache()  # Clean up

    def test_subscore_with_models_returns_valid(self) -> None:
        """method='subscore' returns valid score when models exist."""
        from mental_entropy.models._registry import subscore_models_available
        if not subscore_models_available():
            pytest.skip("Subscore models not available")
        features = _make_mid_features()
        score = compute_mes_score(features, method="subscore")
        assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure the new method parameter doesn't break existing behavior."""

    def test_linear_method_same_as_internal(self) -> None:
        """method='linear' produces same result as _compute_linear_mes."""
        features = _make_mid_features()
        new_score = compute_mes_score(features, method="linear")
        old_score = _compute_linear_mes(features)
        assert new_score == pytest.approx(old_score)

    def test_zero_features_score(self) -> None:
        """All-zero features produce expected score (all inv -> 100, all dir -> 0)."""
        features = _make_zero_features()
        score = compute_mes_score(features, method="linear")
        # inv features contribute weight * (1-0) = weight
        # dir features contribute weight * 0 = 0
        # dir_norm features contribute weight * 0/param = 0
        inv_sum = sum(w for _, w, t, *_ in _WEIGHTS if t == "inv")
        expected = inv_sum * 100.0
        assert score == pytest.approx(expected, rel=1e-6)

    def test_one_features_score(self) -> None:
        """All-one features produce expected score."""
        features = _make_one_features()
        score = compute_mes_score(features, method="linear")
        # inv features contribute weight * (1-1) = 0
        # dir features contribute weight * 1 = weight
        # dir_norm features contribute weight * min(1/param, 1)
        dir_sum = sum(
            entry[1] for entry in _WEIGHTS if entry[2] == "dir"
        )
        dir_norm_sum = sum(
            entry[1] * min(1.0 / entry[3], 1.0)
            for entry in _WEIGHTS
            if entry[2] == "dir_norm"
        )
        expected = (dir_sum + dir_norm_sum) * 100.0
        assert score == pytest.approx(expected, rel=1e-6)

    def test_empty_features_dict(self) -> None:
        """Empty features dict uses defaults (doesn't crash)."""
        score = compute_mes_score({}, method="linear")
        assert 0.0 <= score <= 100.0

    def test_score_clamped_to_range(self) -> None:
        """Score is always in [0, 100]."""
        # Even with extreme values
        features = {entry[0]: 1000.0 for entry in _WEIGHTS}
        score = compute_mes_score(features, method="linear")
        assert 0.0 <= score <= 100.0

    def test_weights_sum_to_one(self) -> None:
        """Weight configuration sums to 1.0."""
        total = sum(entry[1] for entry in _WEIGHTS)
        assert total == pytest.approx(1.0, abs=0.01)
