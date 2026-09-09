"""Tests for temporal entropy (TE) feature extraction."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from mental_entropy.temporal import (
    TE_FEATURE_KEYS,
    ScoredEntry,
    te_features,
    te_features_from_entries,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_entries(
    scores: list[float],
    start: datetime = datetime(2024, 1, 1, 12, 0),
    gap_days: float = 1.0,
) -> list[tuple[datetime, float, None]]:
    """Create test entries with regular spacing."""
    return [
        (start + timedelta(days=i * gap_days), score, None)
        for i, score in enumerate(scores)
    ]


# ---------------------------------------------------------------------------
# Test: exact keyset
# ---------------------------------------------------------------------------


def test_te_features_returns_exact_keys():
    """TE features dict must have exactly the expected keys."""
    entries = make_entries([50.0, 60.0])
    features = te_features(entries)
    assert set(features.keys()) == TE_FEATURE_KEYS


def test_te_features_all_values_json_serializable():
    """All returned values must be Python int or float."""
    entries = make_entries([50.0, 60.0, 55.0, 70.0])
    features = te_features(entries)

    for key, value in features.items():
        assert isinstance(value, (int, float)), f"{key} is {type(value)}"
        # Ensure not numpy types
        assert type(value).__module__ == "builtins", f"{key} is numpy type"


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------


def test_te_features_empty_input():
    """Empty list returns all zeros."""
    features = te_features([])
    assert features["te_n_entries"] == 0
    assert features["te_baseline_mean"] == 0.0
    assert features["te_slope"] == 0.0
    assert set(features.keys()) == TE_FEATURE_KEYS


def test_te_features_single_entry():
    """Single entry returns sensible defaults."""
    entries = make_entries([65.0])
    features = te_features(entries)

    assert features["te_n_entries"] == 1
    assert features["te_baseline_mean"] == 65.0
    assert features["te_baseline_std"] == 0.0
    assert features["te_latest_percentile"] == 50.0
    assert features["te_slope"] == 0.0
    assert features["te_rolling_7d_mean"] == 65.0


def test_te_features_two_entries():
    """Two entries: velocity works, acceleration is 0."""
    entries = make_entries([50.0, 60.0])
    features = te_features(entries)

    assert features["te_n_entries"] == 2
    # 10 point increase in 1 day = 10 pts/day velocity
    assert abs(features["te_velocity_mean"] - 10.0) < 0.01
    assert features["te_velocity_std"] == 0.0  # Need 3+ for std
    assert features["te_acceleration_mean"] == 0.0  # Need 3+ entries


# ---------------------------------------------------------------------------
# Test: trajectory metrics
# ---------------------------------------------------------------------------


def test_te_features_positive_slope():
    """Increasing scores => positive slope."""
    entries = make_entries([40.0, 50.0, 60.0, 70.0])
    features = te_features(entries)

    assert features["te_slope"] > 0
    assert features["te_trend_direction"] == 1
    assert features["te_slope_per_week"] == features["te_slope"] * 7


def test_te_features_negative_slope():
    """Decreasing scores => negative slope."""
    entries = make_entries([70.0, 60.0, 50.0, 40.0])
    features = te_features(entries)

    assert features["te_slope"] < 0
    assert features["te_trend_direction"] == -1


def test_te_features_flat_slope():
    """Constant scores => zero slope."""
    entries = make_entries([50.0, 50.0, 50.0])
    features = te_features(entries)

    assert abs(features["te_slope"]) < 0.01
    assert features["te_trend_direction"] == 0


def test_te_features_velocity_calculation():
    """Verify velocity = (score_diff) / (time_diff_days)."""
    # 20 point increase over 2 days = 10 pts/day
    entries = [
        (datetime(2024, 1, 1), 40.0, None),
        (datetime(2024, 1, 3), 60.0, None),
    ]
    features = te_features(entries)

    assert abs(features["te_velocity_mean"] - 10.0) < 0.01


def test_te_features_acceleration():
    """Test acceleration with 3 entries."""
    # Day 1->2: 40->50 = +10 pts/day
    # Day 2->3: 50->70 = +20 pts/day
    # Acceleration = (20-10) / 1 day = +10 pts/day^2
    entries = [
        (datetime(2024, 1, 1), 40.0, None),
        (datetime(2024, 1, 2), 50.0, None),
        (datetime(2024, 1, 3), 70.0, None),
    ]
    features = te_features(entries)

    assert features["te_acceleration_mean"] > 0


def test_te_features_time_span():
    """Test time span calculation."""
    entries = [
        (datetime(2024, 1, 1), 50.0, None),
        (datetime(2024, 1, 11), 60.0, None),  # 10 days later
    ]
    features = te_features(entries)

    assert abs(features["te_time_span_days"] - 10.0) < 0.01


# ---------------------------------------------------------------------------
# Test: baseline comparison
# ---------------------------------------------------------------------------


def test_te_features_zscore():
    """Z-score correctly identifies outliers."""
    # Three normal values, one outlier
    entries = make_entries([50.0, 50.0, 50.0, 90.0])
    features = te_features(entries)

    # Latest (90) is well above mean (~60)
    assert features["te_latest_zscore"] > 1.0


def test_te_features_percentile_max():
    """Latest at maximum => 100th percentile."""
    entries = make_entries([10.0, 20.0, 30.0, 40.0, 50.0])
    features = te_features(entries)

    # Latest (50) is maximum
    assert features["te_latest_percentile"] == 100.0


def test_te_features_percentile_min():
    """Latest at minimum => low percentile."""
    entries = make_entries([50.0, 40.0, 30.0, 20.0, 10.0])
    features = te_features(entries)

    # Latest (10) is minimum => 20th percentile (1/5 entries <= 10)
    assert features["te_latest_percentile"] == 20.0


def test_te_features_days_since_high():
    """Days since max score."""
    entries = [
        (datetime(2024, 1, 1), 90.0, None),  # High
        (datetime(2024, 1, 5), 50.0, None),
        (datetime(2024, 1, 10), 60.0, None),  # Latest
    ]
    features = te_features(entries)

    # High was on Jan 1, latest on Jan 10 => 9 days
    assert abs(features["te_days_since_high"] - 9.0) < 0.01


def test_te_features_days_since_low():
    """Days since min score."""
    entries = [
        (datetime(2024, 1, 1), 50.0, None),
        (datetime(2024, 1, 5), 20.0, None),  # Low
        (datetime(2024, 1, 10), 60.0, None),  # Latest
    ]
    features = te_features(entries)

    # Low was on Jan 5, latest on Jan 10 => 5 days
    assert abs(features["te_days_since_low"] - 5.0) < 0.01


# ---------------------------------------------------------------------------
# Test: stability metrics
# ---------------------------------------------------------------------------


def test_te_features_volatility():
    """Volatility = std / mean (coefficient of variation)."""
    # Scores with known mean and std
    entries = make_entries([40.0, 60.0])  # mean=50, std=10
    features = te_features(entries)

    # CV = 10 / 50 = 0.2
    assert abs(features["te_volatility"] - 0.2) < 0.01


def test_te_features_improving_streak():
    """Count consecutive decreasing scores."""
    # All improving (decreasing entropy)
    entries = make_entries([70.0, 60.0, 50.0, 40.0])
    features = te_features(entries)

    assert features["te_streak_improving"] == 3
    assert features["te_streak_worsening"] == 0


def test_te_features_worsening_streak():
    """Count consecutive increasing scores."""
    # All worsening (increasing entropy)
    entries = make_entries([40.0, 50.0, 60.0, 70.0])
    features = te_features(entries)

    assert features["te_streak_worsening"] == 3
    assert features["te_streak_improving"] == 0


def test_te_features_broken_streak():
    """Streak breaks at direction change."""
    # up, down, down (from perspective of latest)
    entries = make_entries([50.0, 60.0, 55.0, 45.0])
    features = te_features(entries)

    # From latest going back: 45<55 (improving), 55<60 (improving), 60>50 (break)
    assert features["te_streak_improving"] == 2
    assert features["te_streak_worsening"] == 0


def test_te_features_no_streak():
    """Alternating scores => no streak."""
    entries = make_entries([50.0, 60.0, 50.0, 60.0])
    features = te_features(entries)

    # Latest is 60, previous is 50 => worsening
    assert features["te_streak_worsening"] == 1
    # But before that 50<60 was improving, so streak breaks
    assert features["te_streak_improving"] == 0


# ---------------------------------------------------------------------------
# Test: time gaps
# ---------------------------------------------------------------------------


def test_te_features_irregular_gaps():
    """Handle irregular time gaps."""
    entries = [
        (datetime(2024, 1, 1), 50.0, None),
        (datetime(2024, 1, 2), 55.0, None),  # 1 day gap
        (datetime(2024, 1, 10), 60.0, None),  # 8 day gap
    ]
    features = te_features(entries)

    assert features["te_max_gap_days"] == 8.0
    # Mean gap = (1 + 8) / 2 = 4.5
    assert abs(features["te_mean_gap_days"] - 4.5) < 0.01


def test_te_features_rolling_window():
    """Rolling 7-day window captures recent entries only."""
    entries = [
        (datetime(2024, 1, 1), 80.0, None),  # Old, outside window
        (datetime(2024, 1, 20), 50.0, None),  # Recent
        (datetime(2024, 1, 21), 60.0, None),  # Recent (latest)
    ]
    features = te_features(entries)

    # Rolling mean should only include last two entries (within 7 days of latest)
    assert abs(features["te_rolling_7d_mean"] - 55.0) < 0.01


def test_te_features_all_in_rolling_window():
    """All entries within 7 days => rolling stats equal baseline stats."""
    entries = make_entries([40.0, 50.0, 60.0], gap_days=1.0)
    features = te_features(entries)

    assert features["te_rolling_7d_mean"] == features["te_baseline_mean"]


# ---------------------------------------------------------------------------
# Test: sorting behavior
# ---------------------------------------------------------------------------


def test_te_features_sorts_by_default():
    """Entries are sorted by timestamp by default."""
    # Out of order: 3rd, 1st, 2nd
    entries = [
        (datetime(2024, 1, 3), 60.0, None),
        (datetime(2024, 1, 1), 40.0, None),
        (datetime(2024, 1, 2), 50.0, None),
    ]
    features = te_features(entries)

    # After sorting: 40, 50, 60 => positive slope
    assert features["te_slope"] > 0


def test_te_features_no_sort():
    """sort=False preserves input order."""
    # Already sorted
    entries = make_entries([40.0, 50.0, 60.0])
    features_sorted = te_features(entries, sort=True)
    features_unsorted = te_features(entries, sort=False)

    assert features_sorted == features_unsorted


# ---------------------------------------------------------------------------
# Test: duplicate timestamps
# ---------------------------------------------------------------------------


def test_te_features_raises_on_duplicate_timestamp():
    """Duplicate timestamps (within minute) raise ValueError."""
    entries = [
        (datetime(2024, 1, 1, 12, 0, 0), 50.0, None),
        (datetime(2024, 1, 1, 12, 0, 30), 60.0, None),  # Same minute
    ]
    with pytest.raises(ValueError, match="duplicate timestamps"):
        te_features(entries)


def test_te_features_different_minutes_ok():
    """Different minutes are allowed."""
    entries = [
        (datetime(2024, 1, 1, 12, 0, 0), 50.0, None),
        (datetime(2024, 1, 1, 12, 1, 0), 60.0, None),  # Different minute
    ]
    features = te_features(entries)
    assert features["te_n_entries"] == 2


# ---------------------------------------------------------------------------
# Test: wrapper function
# ---------------------------------------------------------------------------


def test_te_features_from_entries():
    """Test ScoredEntry wrapper."""
    entries = [
        ScoredEntry(datetime(2024, 1, 1), 50.0),
        ScoredEntry(datetime(2024, 1, 2), 60.0),
    ]
    features = te_features_from_entries(entries)

    assert features["te_n_entries"] == 2
    assert set(features.keys()) == TE_FEATURE_KEYS


def test_te_features_from_entries_with_features():
    """ScoredEntry with features dict."""
    entries = [
        ScoredEntry(datetime(2024, 1, 1), 50.0, {"ce_break_rate": 0.1}),
        ScoredEntry(datetime(2024, 1, 2), 60.0, {"ce_break_rate": 0.2}),
    ]
    features = te_features_from_entries(entries)

    assert features["te_n_entries"] == 2


# ---------------------------------------------------------------------------
# Test: determinism
# ---------------------------------------------------------------------------


def test_te_features_deterministic():
    """Same input always produces same output."""
    entries = make_entries([50.0, 60.0, 55.0, 70.0, 45.0])

    f1 = te_features(entries)
    f2 = te_features(entries)

    assert f1 == f2


# ---------------------------------------------------------------------------
# Test: realistic scenarios
# ---------------------------------------------------------------------------


def test_te_features_improving_trend():
    """Realistic improving trend over two weeks."""
    # Start high (bad), end low (good)
    entries = [
        (datetime(2024, 1, 1), 70.0, None),
        (datetime(2024, 1, 3), 65.0, None),
        (datetime(2024, 1, 6), 55.0, None),
        (datetime(2024, 1, 10), 45.0, None),
        (datetime(2024, 1, 14), 40.0, None),
    ]
    features = te_features(entries)

    assert features["te_slope"] < 0  # Negative slope = improving
    assert features["te_trend_direction"] == -1
    assert features["te_streak_improving"] > 0


def test_te_features_stable_baseline():
    """User with stable baseline."""
    # Fluctuations around 50, ending same as start
    entries = make_entries([50.0, 52.0, 48.0, 51.0, 50.0])
    features = te_features(entries)

    assert abs(features["te_baseline_mean"] - 50.2) < 1.0
    assert features["te_baseline_std"] < 3.0
    # Small slope due to fluctuations, but low volatility
    assert features["te_volatility"] < 0.1


# ---------------------------------------------------------------------------
# compute_user_state tests
# ---------------------------------------------------------------------------


from mental_entropy.temporal.te import compute_user_state


def test_user_state_empty():
    """No entries → insufficient_data."""
    result = compute_user_state([])
    assert result["status"] == "insufficient_data"
    assert result["confidence"] == 0.0
    assert result["current_mes"] == 0.0
    assert result["n_entries"] == 0


def test_user_state_single_entry():
    """One entry from today → score with moderate confidence."""
    now = datetime(2026, 3, 25, 12, 0)
    entries = [(now, 42.0, None)]
    result = compute_user_state(entries, now=now)

    assert result["status"] == "ok"
    assert result["current_mes"] == 42.0
    assert 0.3 < result["confidence"] < 0.5  # ~0.39 for single entry
    assert result["n_entries"] == 1
    assert result["days_since_last_entry"] == 0.0
    assert result["trend"] == "unknown"  # Can't compute with 1 entry


def test_user_state_recent_entries_high_confidence():
    """3 entries from the past week → high confidence."""
    now = datetime(2026, 3, 25, 12, 0)
    entries = [
        (datetime(2026, 3, 23, 10, 0), 45.0, None),
        (datetime(2026, 3, 24, 9, 0), 50.0, None),
        (datetime(2026, 3, 25, 11, 0), 40.0, None),
    ]
    result = compute_user_state(entries, now=now)

    assert result["status"] == "ok"
    assert result["confidence"] > 0.6
    assert result["n_entries"] == 3
    assert result["n_recent_entries"] == 3
    # Most recent entry (40.0) should pull average down from 45.0
    assert result["current_mes"] < 45.0


def test_user_state_old_entries_stale():
    """Entries from ~3 weeks ago → stale status (still some weight)."""
    now = datetime(2026, 3, 25, 12, 0)
    # 20-25 days ago — old enough to be stale, recent enough to have some weight
    entries = [
        (datetime(2026, 3, 1, 10, 0), 55.0, None),
        (datetime(2026, 3, 3, 9, 0), 60.0, None),
        (datetime(2026, 3, 5, 11, 0), 50.0, None),
    ]
    result = compute_user_state(entries, now=now, half_life_days=7.0)

    assert result["status"] == "stale"
    assert result["confidence"] < 0.3
    assert result["days_since_last_entry"] > 14


def test_user_state_decay_weighting():
    """Recent entries weighted more than old entries."""
    now = datetime(2026, 3, 25, 12, 0)
    entries = [
        # Old entry: high entropy (80) from 30 days ago
        (datetime(2026, 2, 23, 10, 0), 80.0, None),
        # Recent entry: low entropy (20) from today
        (datetime(2026, 3, 25, 10, 0), 20.0, None),
    ]
    result = compute_user_state(entries, now=now, half_life_days=14.0)

    # Current MES should be much closer to 20 (recent) than 80 (old)
    assert result["current_mes"] < 35.0
    assert result["status"] == "ok"


def test_user_state_confidence_increases_with_more_entries():
    """More recent entries → higher confidence."""
    now = datetime(2026, 3, 25, 12, 0)

    one_entry = [(datetime(2026, 3, 25, 10, 0), 50.0, None)]
    three_entries = [
        (datetime(2026, 3, 23, 10, 0), 50.0, None),
        (datetime(2026, 3, 24, 10, 0), 50.0, None),
        (datetime(2026, 3, 25, 10, 0), 50.0, None),
    ]

    r1 = compute_user_state(one_entry, now=now)
    r3 = compute_user_state(three_entries, now=now)

    assert r3["confidence"] > r1["confidence"]


def test_user_state_trend_improving():
    """Declining scores → improving trend."""
    now = datetime(2026, 3, 25, 12, 0)
    entries = [
        (datetime(2026, 3, 20, 10, 0), 70.0, None),
        (datetime(2026, 3, 21, 10, 0), 60.0, None),
        (datetime(2026, 3, 22, 10, 0), 50.0, None),
        (datetime(2026, 3, 23, 10, 0), 40.0, None),
        (datetime(2026, 3, 24, 10, 0), 30.0, None),
    ]
    result = compute_user_state(entries, now=now)

    assert result["trend"] == "improving"
    assert result["trend_slope"] < 0  # Negative = scores declining = improving


def test_user_state_json_serializable():
    """All values must be JSON-serializable."""
    import json
    now = datetime(2026, 3, 25, 12, 0)
    entries = [
        (datetime(2026, 3, 24, 10, 0), 45.0, None),
        (datetime(2026, 3, 25, 10, 0), 50.0, None),
    ]
    result = compute_user_state(entries, now=now)
    serialized = json.dumps(result)
    assert serialized  # No TypeError from numpy types


def test_user_state_interpretation():
    """Interpretation matches score range."""
    now = datetime(2026, 3, 25, 12, 0)
    entries = [(now, 75.0, None)]
    result = compute_user_state(entries, now=now)
    assert "entropy" in result["interpretation"].lower() or "disorg" in result["interpretation"].lower()
