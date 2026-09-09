from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def test_rank_averages_ties() -> None:
    import audit_scoring_components as audit

    assert audit._rank([10.0, 20.0, 20.0, 40.0]) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_detects_inverse_order() -> None:
    import audit_scoring_components as audit

    assert audit._spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_build_component_audit_summarizes_components_and_golden_order() -> None:
    import audit_scoring_components as audit

    locked_rows = [
        {
            "id": "low",
            "control_type": "spectrum_1_2",
            "expected_mes": 10.0,
            "model_mes": 12.0,
            "features": {
                "ce_break_rate": 0.1,
                "gd_global_disorder_score": 0.0,
            },
            "subscores": {"prediction_coherence_subscore": 1.2},
            "text": "low",
        },
        {
            "id": "mid",
            "control_type": "spectrum_5_6",
            "expected_mes": 50.0,
            "model_mes": 40.0,
            "features": {
                "ce_break_rate": 0.5,
                "gd_global_disorder_score": 0.2,
            },
            "subscores": {"prediction_coherence_subscore": 2.5},
            "text": "mid",
        },
        {
            "id": "high",
            "control_type": "spectrum_9_10",
            "expected_mes": 90.0,
            "model_mes": 60.0,
            "features": {
                "ce_break_rate": 0.9,
                "gd_global_disorder_score": 0.8,
            },
            "subscores": {"prediction_coherence_subscore": 4.5},
            "text": "high",
        },
    ]
    golden_rows = [
        {"target": "very_low", "expected_mes": 10.0, "model_mes": 10.0, "subscores": {}},
        {"target": "low", "expected_mes": 20.0, "model_mes": 30.0, "subscores": {}},
        {"target": "medium", "expected_mes": 40.0, "model_mes": 25.0, "subscores": {}},
    ]

    result = audit.build_component_audit(locked_rows, golden_rows)

    assert result["locked_row_count"] == 3
    assert result["component_summaries"]["ce"]["feature_count"] == 1
    assert result["component_summaries"]["gd"]["feature_count"] == 1
    assert result["component_summaries"]["ce"]["top_expected_signals"][0]["name"] == "ce_break_rate"
    assert result["subscore_summary"]["subscores"][0]["name"] == "prediction_coherence_subscore"
    assert result["golden_order"]["model_mes"]["ordered"] is False
    assert result["top_misses"]["false_lows"][0]["id"] == "high"


def test_build_combiner_audit_marks_subscore_and_feature_inputs() -> None:
    import audit_scoring_components as audit

    rows = [
        {
            "expected_mes": 10.0,
            "features": {"ce_break_rate": 0.1},
            "subscores": {"prediction_coherence_subscore": 4.0},
        },
        {
            "expected_mes": 90.0,
            "features": {"ce_break_rate": 0.9},
            "subscores": {"prediction_coherence_subscore": 2.0},
        },
    ]
    combiner = {
        "architecture": "subscore_embedding_v7",
        "dimension_names": ["prediction_coherence"],
        "feature_names": ["ce_break_rate"],
        "coef": [-1.0, 0.5],
        "intercept": 10.0,
    }

    result = audit.build_combiner_audit(rows, combiner)

    assert result["inputs"][0]["kind"] == "subscore"
    assert result["inputs"][0]["spearman_expected"] == pytest.approx(-1.0)
    assert result["inputs"][1]["kind"] == "feature"
    assert result["inputs"][1]["spearman_expected"] == pytest.approx(1.0)
