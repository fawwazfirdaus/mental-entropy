from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def test_rating_stats_records_raw_std_and_spread() -> None:
    import aggregate_multirater as aggregate

    stats = aggregate.rating_stats([4.0, 4.0, 7.0])

    assert stats["raw"] == "4|4|7"
    assert stats["std"] == pytest.approx(1.414, abs=0.001)
    assert stats["spread"] == 3.0
    assert stats["n_raters"] == 3


def test_refresh_locked_eval_labels_updates_from_consensus() -> None:
    import refresh_locked_eval_labels as refresh

    payload = {
        "version": "human_locked_eval_v1",
        "label_source": "old",
        "entries": [
            {
                "id": "row_1",
                "source_index": 1,
                "text": "old text",
                "label_overall_entropy": 2.0,
                "expected_mes": 11.11,
                "control_type": "spectrum_1_2",
                "label_std": 0.0,
                "reasoning": "old",
            },
        ],
    }
    consensus = {
        1: {
            "journal": "new text",
            "overall_entropy": "4.5",
            "label_std": "0.816",
            "overall_entropy_raw": "4|5|4",
            "overall_entropy_spread": "1.0",
            "reasoning": "new reasoning",
        },
    }

    result = refresh.refresh_payload(payload, consensus)
    entry = result["entries"][0]

    assert "raw rater disagreement" in result["label_source"]
    assert entry["text"] == "new text"
    assert entry["label_overall_entropy"] == 4.5
    assert entry["expected_mes"] == pytest.approx(38.89)
    assert entry["label_std"] == 0.816
    assert entry["overall_entropy_raw"] == "4|5|4"
    assert entry["overall_entropy_spread"] == 1.0
    assert entry["reasoning"] == "new reasoning"


def test_refresh_feature_cache_labels_updates_std_and_source_indices() -> None:
    import refresh_feature_cache_labels as refresh

    cache = {
        "features": [{"x": 1.0}, {"x": 2.0}],
        "labels": [3.0, 7.0],
        "dimension_labels": {},
    }
    consensus_rows = [
        (
            4,
            {
                "overall_entropy": "3.0",
                "label_std": "0.471",
                "prediction_coherence": "6",
                "model_complexity": "4",
                "compression_progress": "5",
                "belief_integration": "7",
                "precision_weighting": "8",
            },
        ),
        (
            9,
            {
                "overall_entropy": "7.0",
                "label_std": "1.247",
                "prediction_coherence": "2",
                "model_complexity": "3",
                "compression_progress": "4",
                "belief_integration": "5",
                "precision_weighting": "6",
            },
        ),
    ]

    result = refresh.refresh_cache_payload(cache, consensus_rows)

    assert result["labels"] == [3.0, 7.0]
    assert result["label_stds"] == [0.471, 1.247]
    assert result["source_indices"] == [4, 9]
    assert result["dimension_labels"]["prediction_coherence"] == [6.0, 2.0]
