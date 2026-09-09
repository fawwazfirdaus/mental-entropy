from __future__ import annotations

import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "human_locked_eval_v1.json"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "autoresearch-macos"))


def test_locked_human_eval_schema_and_composition() -> None:
    payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    assert payload["version"] == "human_locked_eval_v1"
    assert payload["locked"] is True
    assert payload["source_dataset"] == "data/multirater/consensus_labels.csv"

    entries = payload["entries"]
    assert len(entries) == 25
    assert len({entry["id"] for entry in entries}) == 25
    assert len({entry["source_index"] for entry in entries}) == 25

    allowed_controls = {
        "spectrum_1_2",
        "spectrum_3_4",
        "spectrum_5_6",
        "spectrum_7_8",
        "spectrum_9_10",
        "emotional_coherent_negative",
        "neutral_fragmented_positive",
    }
    counts = {control_type: 0 for control_type in allowed_controls}

    for entry in entries:
        assert set(entry) == {
            "id",
            "source_index",
            "text",
            "label_overall_entropy",
            "expected_mes",
            "control_type",
            "label_std",
            "overall_entropy_raw",
            "overall_entropy_spread",
            "reasoning",
        }
        assert entry["control_type"] in allowed_controls
        counts[entry["control_type"]] += 1
        assert isinstance(entry["text"], str)
        assert len(entry["text"]) >= 200
        assert 1.0 <= entry["label_overall_entropy"] <= 10.0
        assert 0.0 <= entry["expected_mes"] <= 100.0
        assert math.isclose(
            entry["expected_mes"],
            (entry["label_overall_entropy"] - 1.0) / 9.0 * 100.0,
            abs_tol=0.01,
        )
        assert entry["label_std"] >= 0.0
        assert isinstance(entry["overall_entropy_raw"], str)
        assert entry["overall_entropy_raw"]
        assert entry["overall_entropy_spread"] >= 0.0
        assert isinstance(entry["reasoning"], str)
        assert entry["reasoning"]

    assert counts["emotional_coherent_negative"] == 5
    assert counts["neutral_fragmented_positive"] == 5
    for control_type in [
        "spectrum_1_2",
        "spectrum_3_4",
        "spectrum_5_6",
        "spectrum_7_8",
        "spectrum_9_10",
    ]:
        assert counts[control_type] == 3


def test_evaluate_locked_human_eval_report_helpers() -> None:
    import evaluate_locked_human_eval as eval_script

    rows = [
        {
            "id": "low",
            "control_type": "spectrum_1_2",
            "expected_mes": 10.0,
            "model_mes": 20.0,
            "label_overall_entropy": 1.9,
            "text": "a" * 220,
            "subscores": {},
            "selected_features": {},
        },
        {
            "id": "high",
            "control_type": "spectrum_9_10",
            "expected_mes": 90.0,
            "model_mes": 60.0,
            "label_overall_entropy": 9.1,
            "text": "b" * 220,
            "subscores": {},
            "selected_features": {},
        },
    ]

    report = eval_script.build_report(rows, golden_results=[])

    assert report["metrics"]["mae"] == 20.0
    assert report["metrics"]["rmse"] == round(math.sqrt(500.0), 2)
    assert report["per_band"]["spectrum_1_2"]["mae"] == 10.0
    assert report["per_band"]["spectrum_9_10"]["mae"] == 30.0
    assert report["acceptance_gates"]["overall_mae_lte_12"] is False
    assert report["false_lows"][0]["id"] == "high"
    assert report["false_highs"][0]["id"] == "low"


def test_prepare_filters_locked_eval_source_indices() -> None:
    import prepare

    entries = [
        {"journal": "keep", "source": "human"},
        {"journal": "locked", "source": "human"},
        {"journal": "also keep", "source": "synthetic"},
    ]

    filtered = prepare.filter_locked_eval_entries(entries, {1})

    assert [entry["journal"] for _, entry in filtered] == ["keep", "also keep"]
    assert [source_index for source_index, _ in filtered] == [0, 2]
