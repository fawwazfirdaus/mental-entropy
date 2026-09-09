#!/usr/bin/env python3
"""Analyze GD feature separation on the locked human eval set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mental_entropy.features.gd import GD_FEATURE_KEYS, gd_features
from mental_entropy.embedding.text import split_sentences


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "human_locked_eval_v1.json"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "eval" / "human_locked_eval_v1_report.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "artifacts" / "eval" / "human_locked_eval_v1_gd_analysis.json"
)


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"n": len(rows)}
    for key in sorted(GD_FEATURE_KEYS):
        mean = _mean([
            float(row["gd_features"][key])
            for row in rows
            if key in row.get("gd_features", {})
        ])
        metrics[key] = None if mean is None else _round(mean)
    return metrics


def _feature_deltas(
    high_false_lows: list[dict[str, Any]],
    emotional_controls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deltas = []
    for key in sorted(GD_FEATURE_KEYS):
        high_mean = _mean([
            float(row["gd_features"][key])
            for row in high_false_lows
            if key in row.get("gd_features", {})
        ])
        emotional_mean = _mean([
            float(row["gd_features"][key])
            for row in emotional_controls
            if key in row.get("gd_features", {})
        ])
        if high_mean is None or emotional_mean is None:
            continue
        deltas.append({
            "name": key,
            "high_false_low_mean": _round(high_mean),
            "emotional_mean": _round(emotional_mean),
            "delta_high_false_low_minus_emotional": _round(high_mean - emotional_mean),
        })
    deltas.sort(key=lambda row: abs(row["delta_high_false_low_minus_emotional"]), reverse=True)
    return deltas


def build_global_disorder_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build GD feature separation report from enriched eval rows."""
    high_false_lows = [
        row for row in rows
        if row["expected_mes"] >= 66.0 and row["model_mes"] < 60.0
    ]
    emotional_controls = [
        row for row in rows
        if row["control_type"] == "emotional_coherent_negative"
    ]
    neutral_controls = [
        row for row in rows
        if row["control_type"] == "neutral_fragmented_positive"
    ]
    deltas = _feature_deltas(high_false_lows, emotional_controls)
    best = deltas[0]["name"] if deltas else None

    return {
        "eval_set_version": "human_locked_eval_v1",
        "group_metrics": {
            "high_false_lows": _group_metrics(high_false_lows),
            "emotional_coherent_negative": _group_metrics(emotional_controls),
            "neutral_fragmented_positive": _group_metrics(neutral_controls),
        },
        "feature_deltas": deltas,
        "separation_summary": {
            "best_feature": best,
            "high_false_low_count": len(high_false_lows),
            "emotional_control_count": len(emotional_controls),
            "neutral_control_count": len(neutral_controls),
        },
        "rows": rows,
    }


def _load_model_scores(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"]: row for row in report.get("entries", [])}


def load_enriched_rows(eval_path: Path, report_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    model_rows = _load_model_scores(report_path)
    rows = []
    for entry in payload["entries"]:
        text = entry["text"]
        model_row = model_rows.get(entry["id"], {})
        rows.append({
            "id": entry["id"],
            "source_index": entry["source_index"],
            "control_type": entry["control_type"],
            "label_overall_entropy": entry["label_overall_entropy"],
            "expected_mes": entry["expected_mes"],
            "model_mes": model_row.get("model_mes", 0.0),
            "gd_features": gd_features(split_sentences(text)),
            "preview": text[:220].replace("\n", " "),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GD feature separation on locked human eval rows."
    )
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_enriched_rows(args.eval_path, args.report)
    analysis = build_global_disorder_analysis(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Global Disorder Feature Analysis")
    print(f"  best_feature={analysis['separation_summary']['best_feature']}")
    print(
        "  high_false_lows="
        f"{analysis['separation_summary']['high_false_low_count']}"
    )
    print(f"  report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
