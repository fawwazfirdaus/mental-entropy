"""Temporal Entropy (TE) feature extraction for MES longitudinal analysis.

This module computes deterministic numeric features from a sequence of scored
journal entries to measure trajectory, baseline deviation, and stability over time.

The module is stateless - it accepts a list of (timestamp, mes_score, features)
tuples and returns temporal metrics. The backend handles storage/retrieval.

Key feature categories:
- Trajectory metrics: slope, velocity, acceleration, rolling stats
- Baseline comparison: personal mean/std, z-score, percentile, days since extremes
- Stability metrics: trend direction, volatility, streaks
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mental_entropy.temporal.types import ScoredEntry

# ---------------------------------------------------------------------------
# Expected feature keys (for validation / documentation)
# ---------------------------------------------------------------------------

TE_FEATURE_KEYS: frozenset[str] = frozenset(
    [
        # Trajectory metrics (11)
        "te_n_entries",
        "te_time_span_days",
        "te_mean_gap_days",
        "te_max_gap_days",
        "te_slope",
        "te_slope_per_week",
        "te_velocity_mean",
        "te_velocity_std",
        "te_acceleration_mean",
        "te_rolling_7d_mean",
        "te_rolling_7d_std",
        # Baseline comparison (6)
        "te_baseline_mean",
        "te_baseline_std",
        "te_latest_zscore",
        "te_latest_percentile",
        "te_days_since_high",
        "te_days_since_low",
        # Stability metrics (4)
        "te_trend_direction",
        "te_volatility",
        "te_streak_improving",
        "te_streak_worsening",
    ]
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empty_features() -> dict[str, int | float]:
    """Return feature dict for empty input."""
    return {
        "te_n_entries": 0,
        "te_time_span_days": 0.0,
        "te_mean_gap_days": 0.0,
        "te_max_gap_days": 0.0,
        "te_slope": 0.0,
        "te_slope_per_week": 0.0,
        "te_velocity_mean": 0.0,
        "te_velocity_std": 0.0,
        "te_acceleration_mean": 0.0,
        "te_rolling_7d_mean": 0.0,
        "te_rolling_7d_std": 0.0,
        "te_baseline_mean": 0.0,
        "te_baseline_std": 0.0,
        "te_latest_zscore": 0.0,
        "te_latest_percentile": 0.0,
        "te_days_since_high": 0.0,
        "te_days_since_low": 0.0,
        "te_trend_direction": 0,
        "te_volatility": 0.0,
        "te_streak_improving": 0,
        "te_streak_worsening": 0,
    }


def _single_entry_features(score: float) -> dict[str, int | float]:
    """Return feature dict for single entry."""
    return {
        "te_n_entries": 1,
        "te_time_span_days": 0.0,
        "te_mean_gap_days": 0.0,
        "te_max_gap_days": 0.0,
        "te_slope": 0.0,
        "te_slope_per_week": 0.0,
        "te_velocity_mean": 0.0,
        "te_velocity_std": 0.0,
        "te_acceleration_mean": 0.0,
        "te_rolling_7d_mean": float(score),
        "te_rolling_7d_std": 0.0,
        "te_baseline_mean": float(score),
        "te_baseline_std": 0.0,
        "te_latest_zscore": 0.0,
        "te_latest_percentile": 50.0,
        "te_days_since_high": 0.0,
        "te_days_since_low": 0.0,
        "te_trend_direction": 0,
        "te_volatility": 0.0,
        "te_streak_improving": 0,
        "te_streak_worsening": 0,
    }


def _count_streak(scores: np.ndarray, direction: int) -> int:
    """Count consecutive entries where score moves in the given direction.

    Args:
        scores: Array of MES scores in chronological order.
        direction: -1 for decreasing (improving), +1 for increasing (worsening).

    Returns:
        Count of consecutive entries from the latest moving in the given direction.
    """
    if len(scores) <= 1:
        return 0

    streak = 0
    for i in range(len(scores) - 1, 0, -1):
        diff = scores[i] - scores[i - 1]
        if (direction == -1 and diff < 0) or (direction == 1 and diff > 0):
            streak += 1
        else:
            break
    return streak


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def te_features(
    entries: list[tuple[datetime, float, dict[str, float] | None]],
    *,
    sort: bool = True,
) -> dict[str, int | float]:
    """Compute temporal entropy features from a list of scored entries.

    Args:
        entries: List of (timestamp, mes_score, optional_features) tuples.
            Timestamps should be datetime objects.
            mes_score is the MES score (0-100) for that entry.
            features dict is optional (can be None).
        sort: If True (default), sort entries by timestamp before analysis.
            If False, assume entries are already in chronological order.

    Returns:
        Dict with exactly the keys in TE_FEATURE_KEYS.
        All values are Python int or float (JSON-serializable).

    Raises:
        ValueError: If entries contain duplicate timestamps (within same minute).
    """
    n = len(entries)

    # Edge case: empty
    if n == 0:
        return _empty_features()

    # Sort if requested
    if sort:
        entries = sorted(entries, key=lambda x: x[0])

    # Check for duplicate timestamps (round to minute granularity)
    timestamps_rounded = [
        e[0].replace(second=0, microsecond=0) for e in entries
    ]
    if len(set(timestamps_rounded)) != len(timestamps_rounded):
        raise ValueError(
            "Entries contain duplicate timestamps (within same minute)"
        )

    # Extract arrays
    ts_array = np.array(
        [e[0].timestamp() for e in entries], dtype=np.float64
    )
    scores = np.array([e[1] for e in entries], dtype=np.float64)

    # Edge case: single entry
    if n == 1:
        return _single_entry_features(scores[0])

    # -------------------------------------------------------------------------
    # Time computations (in days)
    # -------------------------------------------------------------------------
    ts_days = (ts_array - ts_array[0]) / 86400.0
    gaps = np.diff(ts_days)

    time_span_days = float(ts_days[-1])
    mean_gap_days = float(np.mean(gaps))
    max_gap_days = float(np.max(gaps))

    # -------------------------------------------------------------------------
    # Trajectory: slope via linear regression
    # -------------------------------------------------------------------------
    slope, _ = np.polyfit(ts_days, scores, 1)
    slope = float(slope)
    slope_per_week = slope * 7

    # -------------------------------------------------------------------------
    # Velocity: points per day change between consecutive entries
    # -------------------------------------------------------------------------
    score_diffs = np.diff(scores)
    # Avoid division by zero for very small gaps
    safe_gaps = np.maximum(gaps, 1e-6)
    velocities = score_diffs / safe_gaps

    velocity_mean = float(np.mean(velocities))
    velocity_std = float(np.std(velocities)) if n > 2 else 0.0

    # -------------------------------------------------------------------------
    # Acceleration: change in velocity over time
    # -------------------------------------------------------------------------
    if n > 2:
        vel_diffs = np.diff(velocities)
        gaps_for_acc = safe_gaps[:-1]
        accelerations = vel_diffs / gaps_for_acc
        acceleration_mean = float(np.mean(accelerations))
    else:
        acceleration_mean = 0.0

    # -------------------------------------------------------------------------
    # Rolling 7-day statistics (from latest entry)
    # -------------------------------------------------------------------------
    latest_ts = ts_array[-1]
    seven_days_ago = latest_ts - 7 * 86400
    in_window = ts_array >= seven_days_ago
    window_scores = scores[in_window]

    rolling_7d_mean = float(np.mean(window_scores))
    rolling_7d_std = (
        float(np.std(window_scores)) if len(window_scores) > 1 else 0.0
    )

    # -------------------------------------------------------------------------
    # Baseline statistics (all-time)
    # -------------------------------------------------------------------------
    baseline_mean = float(np.mean(scores))
    baseline_std = float(np.std(scores))

    # -------------------------------------------------------------------------
    # Latest entry comparison
    # -------------------------------------------------------------------------
    latest_score = float(scores[-1])

    # Z-score: how many std deviations from baseline
    if baseline_std > 1e-9:
        latest_zscore = float((latest_score - baseline_mean) / baseline_std)
    else:
        latest_zscore = 0.0

    # Percentile rank: what fraction of scores are <= latest
    latest_percentile = float(np.sum(scores <= latest_score) / n * 100)

    # Days since high/low
    high_idx = int(np.argmax(scores))
    low_idx = int(np.argmin(scores))
    days_since_high = float((ts_array[-1] - ts_array[high_idx]) / 86400)
    days_since_low = float((ts_array[-1] - ts_array[low_idx]) / 86400)

    # -------------------------------------------------------------------------
    # Stability metrics
    # -------------------------------------------------------------------------

    # Trend direction: -1 (improving), 0 (flat), +1 (worsening)
    slope_threshold = 0.01  # Near-zero threshold
    if abs(slope) < slope_threshold:
        trend_direction = 0
    else:
        trend_direction = 1 if slope > 0 else -1

    # Volatility: coefficient of variation (std / mean)
    if baseline_mean > 1e-9:
        volatility = float(baseline_std / baseline_mean)
    else:
        volatility = 0.0

    # Streaks: consecutive improving/worsening entries from latest
    streak_improving = _count_streak(scores, direction=-1)
    streak_worsening = _count_streak(scores, direction=1)

    return {
        # Trajectory metrics
        "te_n_entries": n,
        "te_time_span_days": time_span_days,
        "te_mean_gap_days": mean_gap_days,
        "te_max_gap_days": max_gap_days,
        "te_slope": slope,
        "te_slope_per_week": slope_per_week,
        "te_velocity_mean": velocity_mean,
        "te_velocity_std": velocity_std,
        "te_acceleration_mean": acceleration_mean,
        "te_rolling_7d_mean": rolling_7d_mean,
        "te_rolling_7d_std": rolling_7d_std,
        # Baseline comparison
        "te_baseline_mean": baseline_mean,
        "te_baseline_std": baseline_std,
        "te_latest_zscore": latest_zscore,
        "te_latest_percentile": latest_percentile,
        "te_days_since_high": days_since_high,
        "te_days_since_low": days_since_low,
        # Stability metrics
        "te_trend_direction": trend_direction,
        "te_volatility": volatility,
        "te_streak_improving": streak_improving,
        "te_streak_worsening": streak_worsening,
    }


def te_features_from_entries(
    entries: list["ScoredEntry"],
    *,
    sort: bool = True,
) -> dict[str, int | float]:
    """Compute temporal entropy features from ScoredEntry objects.

    Convenience wrapper that converts ScoredEntry list to raw tuples.

    Args:
        entries: List of ScoredEntry objects.
        sort: If True (default), sort by timestamp before analysis.

    Returns:
        TE feature dict (same as te_features).
    """
    raw = [(e.timestamp, e.mes_score, e.features) for e in entries]
    return te_features(raw, sort=sort)


# ---------------------------------------------------------------------------
# User state: exponential decay weighted current MES
# ---------------------------------------------------------------------------


def compute_user_state(
    entries: list[tuple[datetime, float, dict[str, float] | None]],
    *,
    half_life_days: float = 14.0,
    now: datetime | None = None,
) -> dict[str, float | int | str]:
    """Compute a user's current mental entropy state from journal history.

    Uses exponential decay weighting so recent entries count more than old
    ones. Returns a current MES score, confidence level, trend, and status.

    Args:
        entries: List of (timestamp, mes_score, optional_features) tuples.
            Same format as te_features().
        half_life_days: How fast old entries fade. At this many days ago,
            an entry's weight is 0.5. Default 14 days (2 weeks).
        now: Reference "current" time. Defaults to datetime.utcnow().
            Useful for testing with deterministic timestamps.

    Returns:
        Dict with keys:
        - current_mes (float): Decay-weighted average score (0-100)
        - confidence (float): How reliable the estimate is (0.0-1.0)
        - trend (str): "improving" | "stable" | "worsening" | "unknown"
        - trend_slope (float): Points per week change
        - baseline_mes (float): All-time unweighted average
        - deviation_from_baseline (float): current_mes - baseline_mes
        - n_entries (int): Total entries provided
        - n_recent_entries (int): Entries within 2 * half_life window
        - days_since_last_entry (float): Days since most recent entry
        - interpretation (str): Human-readable interpretation of current_mes
        - status (str): "ok" | "stale" | "insufficient_data"
    """
    from mental_entropy.score import interpret_mes_score

    if now is None:
        from datetime import timezone
        now = datetime.now(timezone.utc)

    # Ensure consistent timezone handling: strip tzinfo from all timestamps
    # so naive and aware datetimes can be mixed
    if now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    sorted_entries_raw = list(entries)  # preserve for later

    n = len(entries)

    # --- Empty case ---
    if n == 0:
        return {
            "current_mes": 0.0,
            "confidence": 0.0,
            "trend": "unknown",
            "trend_slope": 0.0,
            "baseline_mes": 0.0,
            "deviation_from_baseline": 0.0,
            "n_entries": 0,
            "n_recent_entries": 0,
            "days_since_last_entry": float("inf"),
            "interpretation": "No journal entries available.",
            "status": "insufficient_data",
        }

    # Sort by timestamp
    sorted_entries = sorted(entries, key=lambda e: e[0])
    # Strip timezone info for consistent subtraction
    timestamps = [
        ts.replace(tzinfo=None) if ts.tzinfo is not None else ts
        for ts in (e[0] for e in sorted_entries)
    ]
    scores = np.array([float(e[1]) for e in sorted_entries], dtype=np.float64)

    # --- Compute days ago for each entry ---
    days_ago = np.array(
        [(now - ts).total_seconds() / 86400.0 for ts in timestamps],
        dtype=np.float64,
    )
    # Clamp negative values (entries in the "future") to 0
    days_ago = np.maximum(days_ago, 0.0)

    days_since_last = float(days_ago.min())

    # --- Exponential decay weights ---
    # weight = exp(-days_ago * ln(2) / half_life)
    # This gives weight = 0.5 at exactly half_life_days ago
    decay_rate = np.log(2.0) / max(half_life_days, 1e-6)
    weights = np.exp(-days_ago * decay_rate)
    total_weight = float(weights.sum())

    # --- Decay-weighted current MES ---
    if total_weight > 1e-9:
        current_mes = float(np.dot(weights, scores) / total_weight)
    else:
        current_mes = float(scores[-1])  # Fallback to latest

    current_mes = min(100.0, max(0.0, current_mes))

    # --- Confidence: how much recent signal we have ---
    # confidence = 1 - exp(-total_weight / 2)
    # 0 weight → 0.0, weight=1 → 0.39, weight=3 → 0.78, weight=6 → 0.95
    confidence = float(1.0 - np.exp(-total_weight / 2.0))

    # --- Baseline (unweighted all-time average) ---
    baseline_mes = float(scores.mean())
    deviation = float(current_mes - baseline_mes)

    # --- Recent entries count (within 2 * half_life) ---
    recent_window = 2.0 * half_life_days
    n_recent = int((days_ago <= recent_window).sum())

    # --- Trend from te_features (if enough entries) ---
    trend = "unknown"
    trend_slope = 0.0
    if n >= 3:
        # Build tz-stripped entries for te_features (which also does datetime math)
        stripped_entries = [
            (ts, score, e[2])
            for ts, score, e in zip(timestamps, scores, sorted_entries)
        ]
        te = te_features(stripped_entries, sort=False)
        trend_slope = float(te["te_slope_per_week"])
        td = te["te_trend_direction"]
        if td == -1:
            trend = "improving"
        elif td == 1:
            trend = "worsening"
        else:
            trend = "stable"

    # --- Status ---
    stale_threshold = 2.0 * half_life_days  # Default: 28 days
    if confidence < 0.05:
        status = "insufficient_data"
    elif days_since_last > stale_threshold:
        status = "stale"
    else:
        status = "ok"

    # --- Interpretation ---
    interpretation = interpret_mes_score(current_mes)

    return {
        "current_mes": round(current_mes, 2),
        "confidence": round(confidence, 4),
        "trend": trend,
        "trend_slope": round(trend_slope, 4),
        "baseline_mes": round(baseline_mes, 2),
        "deviation_from_baseline": round(deviation, 2),
        "n_entries": n,
        "n_recent_entries": n_recent,
        "days_since_last_entry": round(days_since_last, 2),
        "interpretation": interpretation,
        "status": status,
    }
