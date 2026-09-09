from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def test_analyze_locked_eval_errors_identifies_high_false_lows() -> None:
    import analyze_locked_eval_errors as analysis

    report = {
        "entries": [
            {
                "id": "missed_high",
                "control_type": "spectrum_9_10",
                "label_overall_entropy": 9.0,
                "expected_mes": 88.89,
                "model_mes": 45.0,
                "error": -43.89,
                "abs_error": 43.89,
                "selected_features": {"ce_adj_mean": 0.7, "ne_fragment_sentence_rate": 0.1},
                "subscores": {"prediction_coherence_subscore": 4.0},
            },
            {
                "id": "detected_high",
                "control_type": "spectrum_7_8",
                "label_overall_entropy": 8.0,
                "expected_mes": 77.78,
                "model_mes": 70.0,
                "error": -7.78,
                "abs_error": 7.78,
                "selected_features": {"ce_adj_mean": 0.4, "ne_fragment_sentence_rate": 0.7},
                "subscores": {"prediction_coherence_subscore": 2.0},
            },
            {
                "id": "low",
                "control_type": "spectrum_1_2",
                "label_overall_entropy": 2.0,
                "expected_mes": 11.11,
                "model_mes": 30.0,
                "error": 18.89,
                "abs_error": 18.89,
                "selected_features": {"ce_adj_mean": 0.6, "ne_fragment_sentence_rate": 0.0},
                "subscores": {"prediction_coherence_subscore": 4.5},
            },
        ],
        "control_checks": {
            "neutral_fragmented_positive_mean_model_mes": 52.0,
            "neutral_fragmented_positive_gte_60": False,
        },
    }

    result = analysis.build_error_analysis(report)

    assert result["summary"]["high_false_low_count"] == 1
    assert result["summary"]["high_labeled_count"] == 2
    assert result["high_false_lows"][0]["id"] == "missed_high"
    assert result["group_metrics"]["high_false_lows"]["mean_model_mes"] == 45.0
    assert result["feature_deltas"][0]["name"] == "ne_fragment_sentence_rate"
    assert result["feature_deltas"][0]["delta_false_low_minus_detected"] == -0.6
    assert result["subscore_deltas"][0]["name"] == "prediction_coherence_subscore"
    assert result["hypotheses"]
