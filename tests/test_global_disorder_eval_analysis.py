from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def test_analyze_global_disorder_features_compares_groups() -> None:
    import analyze_global_disorder_features as analysis

    rows = [
        {
            "id": "high_miss",
            "control_type": "neutral_fragmented_positive",
            "expected_mes": 90.0,
            "model_mes": 45.0,
            "gd_features": {"gd_global_disorder_score": 0.7, "gd_corruption_rate": 0.5},
        },
        {
            "id": "emotion_ok",
            "control_type": "emotional_coherent_negative",
            "expected_mes": 20.0,
            "model_mes": 30.0,
            "gd_features": {"gd_global_disorder_score": 0.1, "gd_corruption_rate": 0.0},
        },
    ]

    result = analysis.build_global_disorder_analysis(rows)

    assert result["group_metrics"]["high_false_lows"]["n"] == 1
    assert result["group_metrics"]["emotional_coherent_negative"]["n"] == 1
    assert result["feature_deltas"][0]["name"] == "gd_global_disorder_score"
    assert result["feature_deltas"][0]["delta_high_false_low_minus_emotional"] == 0.6
    assert result["separation_summary"]["best_feature"] == "gd_global_disorder_score"
