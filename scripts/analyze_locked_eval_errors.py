#!/usr/bin/env python3
"""Analyze high false-low patterns in the locked MES eval report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "eval" / "human_locked_eval_v1_report.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "artifacts" / "eval" / "human_locked_eval_v1_error_analysis.json"
)
HIGH_EXPECTED_MES = 66.0
FALSE_LOW_ERROR = -15.0


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _mean_or_none(values: list[float]) -> float | None:
    mean = _mean(values)
    return None if mean is None else _round(mean)


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "mean_expected_mes": _mean_or_none([row["expected_mes"] for row in rows]),
        "mean_model_mes": _mean_or_none([row["model_mes"] for row in rows]),
        "mean_error": _mean_or_none([row["error"] for row in rows]),
        "mean_abs_error": _mean_or_none([row["abs_error"] for row in rows]),
    }


def _mean_map(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    names = sorted({
        name
        for row in rows
        for name in row.get(key, {}).keys()
    })
    result = {}
    for name in names:
        values = [
            float(row[key][name])
            for row in rows
            if name in row.get(key, {})
        ]
        if values:
            result[name] = _round(_mean(values) or 0.0, 4)
    return result


def _delta_table(
    false_low_rows: list[dict[str, Any]],
    detected_high_rows: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    false_means = _mean_map(false_low_rows, key)
    detected_means = _mean_map(detected_high_rows, key)
    names = sorted(set(false_means) | set(detected_means))
    deltas = []
    for name in names:
        false_value = false_means.get(name)
        detected_value = detected_means.get(name)
        if false_value is None or detected_value is None:
            continue
        delta = false_value - detected_value
        deltas.append({
            "name": name,
            "false_low_mean": false_value,
            "detected_high_mean": detected_value,
            "delta_false_low_minus_detected": _round(delta, 4),
        })
    deltas.sort(key=lambda row: abs(row["delta_false_low_minus_detected"]), reverse=True)
    return deltas


def _entry_digest(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "control_type": row["control_type"],
        "label_overall_entropy": row["label_overall_entropy"],
        "expected_mes": row["expected_mes"],
        "model_mes": row["model_mes"],
        "error": row["error"],
        "abs_error": row["abs_error"],
        "preview": row.get("text", "")[:220].replace("\n", " "),
        "selected_features": row.get("selected_features", {}),
        "subscores": row.get("subscores", {}),
    }


def _hypotheses(
    report: dict[str, Any],
    false_low_rows: list[dict[str, Any]],
    feature_deltas: list[dict[str, Any]],
    subscore_deltas: list[dict[str, Any]],
) -> list[str]:
    hypotheses = []
    if not false_low_rows:
        return hypotheses

    ce_values = [
        row.get("selected_features", {}).get("ce_adj_mean")
        for row in false_low_rows
    ]
    ce_values = [float(value) for value in ce_values if value is not None]
    if ce_values and (_mean(ce_values) or 0.0) >= 0.55:
        hypotheses.append(
            "High false-lows often retain decent adjacent coherence, so local sentence flow may be masking global disorganization."
        )

    fragment_delta = next(
        (
            row["delta_false_low_minus_detected"]
            for row in feature_deltas
            if row["name"] == "ne_fragment_sentence_rate"
        ),
        None,
    )
    if fragment_delta is not None and fragment_delta < 0:
        hypotheses.append(
            "The current feature set under-fires on fragmented high-entropy rows when explicit fragment markers are low."
        )

    coherence_delta = next(
        (
            row["delta_false_low_minus_detected"]
            for row in subscore_deltas
            if row["name"] == "prediction_coherence_subscore"
        ),
        None,
    )
    if coherence_delta is not None and coherence_delta > 0:
        hypotheses.append(
            "Prediction-coherence subscores are too generous on some high-entropy rows."
        )

    neutral_check = report.get("control_checks", {}).get(
        "neutral_fragmented_positive_gte_60"
    )
    if neutral_check is False:
        hypotheses.append(
            "Neutral fragmented controls are under-scored, so the model likely relies too much on affective/semantic cues."
        )

    return hypotheses


def build_error_analysis(report: dict[str, Any]) -> dict[str, Any]:
    """Build high false-low diagnostics from an eval report."""
    entries = report.get("entries", [])
    high_rows = [row for row in entries if row["expected_mes"] >= HIGH_EXPECTED_MES]
    false_low_rows = [
        row for row in high_rows
        if row["model_mes"] - row["expected_mes"] <= FALSE_LOW_ERROR
    ]
    detected_high_rows = [
        row for row in high_rows
        if row["model_mes"] >= 60.0
    ]
    low_rows = [row for row in entries if row["expected_mes"] <= 35.0]

    feature_deltas = _delta_table(false_low_rows, detected_high_rows, "selected_features")
    subscore_deltas = _delta_table(false_low_rows, detected_high_rows, "subscores")

    false_low_rows_sorted = sorted(
        false_low_rows,
        key=lambda row: row["model_mes"] - row["expected_mes"],
    )

    return {
        "eval_set_version": report.get("eval_set_version", "human_locked_eval_v1"),
        "summary": {
            "entry_count": len(entries),
            "high_labeled_count": len(high_rows),
            "high_false_low_count": len(false_low_rows),
            "detected_high_count": len(detected_high_rows),
            "low_labeled_count": len(low_rows),
        },
        "group_metrics": {
            "high_false_lows": _group_metrics(false_low_rows),
            "detected_highs": _group_metrics(detected_high_rows),
            "low_labeled": _group_metrics(low_rows),
        },
        "high_false_lows": [
            _entry_digest(row) for row in false_low_rows_sorted
        ],
        "feature_deltas": feature_deltas,
        "subscore_deltas": subscore_deltas,
        "hypotheses": _hypotheses(
            report, false_low_rows, feature_deltas, subscore_deltas,
        ),
        "next_modeling_checks": [
            "Add global disorder features for topic resets, dangling referents, abandoned setup, and no-compression endings.",
            "Calibrate raw predictions with locked-eval MAE and high-band recall, not correlation alone.",
            "Compare neutral fragmented controls against emotional coherent controls after every retrain.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze current MES false-low patterns on locked eval report."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    analysis = build_error_analysis(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Locked Eval Error Analysis")
    print(f"  high_false_lows={analysis['summary']['high_false_low_count']}")
    print(f"  detected_highs={analysis['summary']['detected_high_count']}")
    print(f"  hypotheses={len(analysis['hypotheses'])}")
    print(f"  report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
