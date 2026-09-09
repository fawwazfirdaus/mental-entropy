#!/usr/bin/env python3
"""Evaluate the current MES model on the locked human eval set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "human_locked_eval_v1.json"
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_benchmark.json"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "artifacts" / "eval" / "human_locked_eval_v1_report.json"
)

SELECTED_FEATURES = (
    "ce_adj_mean",
    "ce_adj_median",
    "ce_adj_p25",
    "ce_adj_p75",
    "ce_inter_block_break_rate",
    "se_dominant_cluster_frac",
    "ne_fragment_sentence_rate",
    "bc_conflict_rate",
    "bc_unresolved_count",
    "cle_hedge_rate",
    "gd_disorder_marker_count_norm",
    "gd_disorder_markers_per_500w",
    "gd_global_disorder_score",
    "gd_block_reset_rate",
    "gd_entity_drift_score",
    "gd_compression_failure_score",
    "gd_corruption_rate",
    "gd_runon_chain_rate",
    "gd_unanchored_question_rate",
)

CONTROL_TYPES = (
    "spectrum_1_2",
    "spectrum_3_4",
    "spectrum_5_6",
    "spectrum_7_8",
    "spectrum_9_10",
    "emotional_coherent_negative",
    "neutral_fragmented_positive",
)


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _mae(rows: list[dict[str, Any]]) -> float | None:
    values = [abs(row["model_mes"] - row["expected_mes"]) for row in rows]
    return _mean(values)


def _rmse(rows: list[dict[str, Any]]) -> float | None:
    values = [(row["model_mes"] - row["expected_mes"]) ** 2 for row in rows]
    mean = _mean(values)
    return math.sqrt(mean) if mean is not None else None


def _bucket_expected_mes(expected_mes: float) -> str:
    lower = int(expected_mes // 20) * 20
    upper = min(lower + 20, 100)
    if lower == 100:
        lower = 80
    return f"{lower:02d}-{upper:03d}"


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mae": None, "rmse": None, "mean_expected": None, "mean_model": None}
    return {
        "n": len(rows),
        "mae": _round(_mae(rows) or 0.0),
        "rmse": _round(_rmse(rows) or 0.0),
        "mean_expected": _round(_mean([row["expected_mes"] for row in rows]) or 0.0),
        "mean_model": _round(_mean([row["model_mes"] for row in rows]) or 0.0),
    }


def _ranking_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    spectrum_order = [
        "spectrum_1_2",
        "spectrum_3_4",
        "spectrum_5_6",
        "spectrum_7_8",
        "spectrum_9_10",
    ]
    means: dict[str, float | None] = {}
    for control_type in spectrum_order:
        control_rows = [row for row in rows if row["control_type"] == control_type]
        means[control_type] = _mean([row["model_mes"] for row in control_rows])

    ordered = True
    previous: float | None = None
    for control_type in spectrum_order:
        current = means[control_type]
        if current is None:
            ordered = False
            continue
        if previous is not None and current <= previous:
            ordered = False
        previous = current

    return {
        "spectrum_mean_model_mes": {
            key: None if value is None else _round(value)
            for key, value in means.items()
        },
        "strictly_increasing": ordered,
    }


def _control_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    emotional = [
        row for row in rows if row["control_type"] == "emotional_coherent_negative"
    ]
    neutral = [
        row for row in rows if row["control_type"] == "neutral_fragmented_positive"
    ]
    emotional_mean = _mean([row["model_mes"] for row in emotional])
    neutral_mean = _mean([row["model_mes"] for row in neutral])

    return {
        "emotional_coherent_negative_mean_model_mes": (
            None if emotional_mean is None else _round(emotional_mean)
        ),
        "emotional_coherent_negative_lte_40": (
            emotional_mean is not None and emotional_mean <= 40.0
        ),
        "neutral_fragmented_positive_mean_model_mes": (
            None if neutral_mean is None else _round(neutral_mean)
        ),
        "neutral_fragmented_positive_gte_60": (
            neutral_mean is not None and neutral_mean >= 60.0
        ),
    }


def _calibration_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(_bucket_expected_mes(row["expected_mes"]), []).append(row)

    table = []
    for bucket in sorted(buckets):
        bucket_rows = buckets[bucket]
        table.append({
            "expected_mes_bucket": bucket,
            **_summarize_rows(bucket_rows),
        })
    return table


def _acceptance_gates(
    rows: list[dict[str, Any]],
    ranking: dict[str, Any],
    controls: dict[str, Any],
    golden_results: list[dict[str, Any]],
) -> dict[str, bool]:
    overall_mae = _mae(rows)
    high_rows = [
        row for row in rows
        if row["control_type"] in {"spectrum_7_8", "spectrum_9_10"}
    ]
    high_mae = _mae(high_rows)
    golden_ordered = True
    if golden_results:
        golden_ordered = all(
            golden_results[i]["model_mes"] < golden_results[i + 1]["model_mes"]
            for i in range(len(golden_results) - 1)
        )

    return {
        "overall_mae_lte_12": overall_mae is not None and overall_mae <= 12.0,
        "high_band_mae_lte_15": high_mae is not None and high_mae <= 15.0,
        "emotional_coherent_mean_lte_40": bool(
            controls["emotional_coherent_negative_lte_40"]
        ),
        "neutral_fragmented_mean_gte_60": bool(
            controls["neutral_fragmented_positive_gte_60"]
        ),
        "locked_spectrum_ordered": bool(ranking["strictly_increasing"]),
        "golden_5_order_preserved": golden_ordered,
    }


def build_report(
    scored_rows: list[dict[str, Any]],
    golden_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic evaluation report from scored rows."""
    metrics = _summarize_rows(scored_rows)
    per_band = {
        control_type: _summarize_rows([
            row for row in scored_rows if row["control_type"] == control_type
        ])
        for control_type in CONTROL_TYPES
    }
    ranking = _ranking_checks(scored_rows)
    controls = _control_checks(scored_rows)

    rows_with_error = []
    for row in scored_rows:
        error = row["model_mes"] - row["expected_mes"]
        enriched = {
            **row,
            "error": _round(error),
            "abs_error": _round(abs(error)),
        }
        rows_with_error.append(enriched)

    false_lows = sorted(
        rows_with_error,
        key=lambda row: row["expected_mes"] - row["model_mes"],
        reverse=True,
    )[:5]
    false_highs = sorted(
        rows_with_error,
        key=lambda row: row["model_mes"] - row["expected_mes"],
        reverse=True,
    )[:5]

    gates = _acceptance_gates(scored_rows, ranking, controls, golden_results)

    return {
        "eval_set_version": "human_locked_eval_v1",
        "metrics": metrics,
        "per_band": per_band,
        "calibration": _calibration_table(scored_rows),
        "ranking_checks": ranking,
        "control_checks": controls,
        "acceptance_gates": gates,
        "false_lows": false_lows,
        "false_highs": false_highs,
        "golden_5": {
            "n": len(golden_results),
            "order_preserved": gates["golden_5_order_preserved"],
            "results": golden_results,
        },
        "entries": rows_with_error,
    }


def load_eval_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No eval entries found in {path}")
    return entries


def score_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from mental_entropy.score import compute_mes_from_text

    scored = []
    for entry in entries:
        result = compute_mes_from_text(entry["text"], method="auto")
        scored.append({
            "id": entry["id"],
            "source_index": entry["source_index"],
            "control_type": entry["control_type"],
            "label_overall_entropy": float(entry["label_overall_entropy"]),
            "expected_mes": float(entry["expected_mes"]),
            "label_std": float(entry.get("label_std", 0.0)),
            "overall_entropy_raw": entry.get("overall_entropy_raw", ""),
            "overall_entropy_spread": float(entry.get("overall_entropy_spread", 0.0)),
            "model_mes": _round(float(result["mes_score"])),
            "text": entry["text"],
            "subscores": {
                key: _round(float(value))
                for key, value in result.items()
                if key.endswith("_subscore")
            },
            "selected_features": {
                key: _round(float(result.get(key, 0.0)), 4)
                for key in SELECTED_FEATURES
            },
        })
    return scored


def score_golden(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    from mental_entropy.score import compute_mes_from_text

    payload = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for anchor in payload.get("anchors", []):
        target = anchor["target"]
        text = payload["anchor_texts"][target]
        result = compute_mes_from_text(text, method="auto")
        label_oe = float(anchor["overall_entropy"])
        expected_mes = (label_oe - 1.0) / 9.0 * 100.0
        results.append({
            "target": target,
            "source_index": anchor.get("idx"),
            "label_overall_entropy": label_oe,
            "expected_mes": _round(expected_mes),
            "model_mes": _round(float(result["mes_score"])),
        })
    return results


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_summary(report: dict[str, Any], output_path: Path) -> None:
    metrics = report["metrics"]
    controls = report["control_checks"]
    gates = report["acceptance_gates"]

    print("Locked Human Eval v1")
    print(f"  n={metrics['n']} mae={metrics['mae']} rmse={metrics['rmse']}")
    print("  controls:")
    print(
        "    emotional coherent mean MES="
        f"{controls['emotional_coherent_negative_mean_model_mes']}"
    )
    print(
        "    neutral fragmented mean MES="
        f"{controls['neutral_fragmented_positive_mean_model_mes']}"
    )
    print("  gates:")
    for key, value in gates.items():
        print(f"    {key}: {value}")
    print(f"  report: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate current MES model on locked human eval set."
    )
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--fail-on-gates",
        action="store_true",
        help="Exit non-zero when any acceptance gate fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = load_eval_entries(args.eval_path)
    scored_rows = score_entries(entries)
    golden_results = score_golden(args.golden_path)
    report = build_report(scored_rows, golden_results)
    write_report(report, args.output)
    print_summary(report, args.output)

    if args.fail_on_gates and not all(report["acceptance_gates"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
