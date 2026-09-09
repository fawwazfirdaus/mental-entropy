#!/usr/bin/env python3
"""Audit MES scoring components against locked and golden eval sets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_locked_human_eval import (  # noqa: E402
    DEFAULT_EVAL_PATH,
    DEFAULT_GOLDEN_PATH,
    _round,
    load_eval_entries,
)
from mental_entropy.score import compute_mes_from_text  # noqa: E402


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "artifacts" / "eval" / "scoring_component_audit.json"
)
DEFAULT_COMBINER_PATH = (
    PROJECT_ROOT / "src" / "mental_entropy" / "models" / "_artifacts" / "combiner.json"
)

COMPONENT_PREFIXES: dict[str, tuple[str, ...]] = {
    "ce": ("ce_",),
    "se": ("se_",),
    "ne": ("ne_",),
    "cle": ("cle_",),
    "bc": ("bc_",),
    "gd": ("gd_",),
}
SUBSCORE_SUFFIX = "_subscore"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        average_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = average_rank
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    if x_mean is None or y_mean is None:
        return None
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    x_norm = math.sqrt(sum(x * x for x in x_centered))
    y_norm = math.sqrt(sum(y * y for y in y_centered))
    if x_norm == 0.0 or y_norm == 0.0:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered)) / (x_norm * y_norm)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _feature_names(rows: list[dict[str, Any]], prefixes: tuple[str, ...]) -> list[str]:
    names = {
        key
        for row in rows
        for key in row.get("features", {})
        if key.startswith(prefixes)
    }
    return sorted(names)


def _feature_values(rows: list[dict[str, Any]], name: str) -> list[float]:
    return [float(row.get("features", {}).get(name, 0.0)) for row in rows]


def _row_errors(rows: list[dict[str, Any]]) -> list[float]:
    return [float(row["model_mes"]) - float(row["expected_mes"]) for row in rows]


def _row_abs_errors(rows: list[dict[str, Any]]) -> list[float]:
    return [abs(error) for error in _row_errors(rows)]


def _correlation_rows(
    rows: list[dict[str, Any]],
    names: list[str],
) -> list[dict[str, Any]]:
    expected = [float(row["expected_mes"]) for row in rows]
    model = [float(row["model_mes"]) for row in rows]
    abs_error = _row_abs_errors(rows)
    result = []
    for name in names:
        values = _feature_values(rows, name)
        expected_corr = _spearman(values, expected)
        model_corr = _spearman(values, model)
        abs_error_corr = _spearman(values, abs_error)
        if expected_corr is None:
            continue
        result.append({
            "name": name,
            "spearman_expected": _round(expected_corr, 4),
            "spearman_model": None if model_corr is None else _round(model_corr, 4),
            "spearman_abs_error": (
                None if abs_error_corr is None else _round(abs_error_corr, 4)
            ),
        })
    result.sort(key=lambda row: abs(row["spearman_expected"]), reverse=True)
    return result


def _component_summary(
    rows: list[dict[str, Any]],
    component: str,
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    names = _feature_names(rows, prefixes)
    correlations = _correlation_rows(rows, names)
    signed = [
        abs(float(row["spearman_expected"]))
        for row in correlations
        if row["spearman_expected"] is not None
    ]
    return {
        "component": component,
        "feature_count": len(names),
        "mean_abs_spearman_expected": (
            None if not signed else _round(_mean(signed) or 0.0, 4)
        ),
        "top_expected_signals": correlations[:8],
        "top_inverse_signals": sorted(
            correlations,
            key=lambda row: row["spearman_expected"],
        )[:5],
        "top_error_associated_signals": sorted(
            [
                row for row in correlations
                if row["spearman_abs_error"] is not None
            ],
            key=lambda row: abs(row["spearman_abs_error"]),
            reverse=True,
        )[:5],
    }


def _subscore_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({
        key
        for row in rows
        for key in row.get("subscores", {})
        if key.endswith(SUBSCORE_SUFFIX)
    })
    expected = [float(row["expected_mes"]) for row in rows]
    model = [float(row["model_mes"]) for row in rows]
    summaries = []
    for name in names:
        values = [float(row.get("subscores", {}).get(name, 0.0)) for row in rows]
        summaries.append({
            "name": name,
            "spearman_expected": _round(_spearman(values, expected) or 0.0, 4),
            "spearman_model": _round(_spearman(values, model) or 0.0, 4),
            "mean": _round(_mean(values) or 0.0, 4),
        })
    summaries.sort(key=lambda row: abs(row["spearman_expected"]), reverse=True)
    return {"subscores": summaries}


def _control_component_means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    controls = sorted({row["control_type"] for row in rows})
    output: dict[str, Any] = {}
    for component, prefixes in COMPONENT_PREFIXES.items():
        names = _feature_names(rows, prefixes)
        output[component] = {}
        for control in controls:
            control_rows = [row for row in rows if row["control_type"] == control]
            if not control_rows:
                continue
            values = []
            for row in control_rows:
                feature_values = [
                    float(row.get("features", {}).get(name, 0.0))
                    for name in names
                ]
                if feature_values:
                    values.append(_mean(feature_values) or 0.0)
            output[component][control] = None if not values else _round(_mean(values) or 0.0, 4)
    return output


def _golden_order(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"model_mes": {"ordered": False, "values": []}, "subscores": {}}
    model_values = [
        {
            "target": row["target"],
            "expected_mes": row["expected_mes"],
            "value": row["model_mes"],
        }
        for row in rows
    ]
    subscore_names = sorted({
        key
        for row in rows
        for key in row.get("subscores", {})
        if key.endswith(SUBSCORE_SUFFIX)
    })
    subscore_orders = {}
    for name in subscore_names:
        values = [
            {
                "target": row["target"],
                "expected_mes": row["expected_mes"],
                "value": row.get("subscores", {}).get(name, 0.0),
            }
            for row in rows
        ]
        subscore_orders[name] = {
            "ordered": all(values[i]["value"] < values[i + 1]["value"] for i in range(len(values) - 1)),
            "values": values,
        }
    return {
        "model_mes": {
            "ordered": all(
                model_values[i]["value"] < model_values[i + 1]["value"]
                for i in range(len(model_values) - 1)
            ),
            "values": model_values,
        },
        "subscores": subscore_orders,
    }


def _top_misses(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    enriched = []
    for row in rows:
        error = float(row["model_mes"]) - float(row["expected_mes"])
        enriched.append({
            "id": row["id"],
            "control_type": row["control_type"],
            "expected_mes": _round(row["expected_mes"]),
            "model_mes": _round(row["model_mes"]),
            "error": _round(error),
            "preview": row.get("text", "")[:220].replace("\n", " "),
            "subscores": row.get("subscores", {}),
        })
    return {
        "false_lows": sorted(enriched, key=lambda row: row["error"])[:8],
        "false_highs": sorted(enriched, key=lambda row: row["error"], reverse=True)[:8],
    }


def _combiner_input_value(row: dict[str, Any], name: str) -> float:
    subscore_name = f"{name}{SUBSCORE_SUFFIX}"
    if subscore_name in row.get("subscores", {}):
        return float(row["subscores"][subscore_name])
    return float(row.get("features", {}).get(name, 0.0))


def build_combiner_audit(
    rows: list[dict[str, Any]],
    combiner: dict[str, Any],
) -> dict[str, Any]:
    """Summarize v7 combiner inputs, coefficients, and label alignment."""
    dimension_names = list(combiner.get("dimension_names", []))
    feature_names = list(combiner.get("feature_names", []))
    input_names = dimension_names + feature_names
    coefs = [float(value) for value in combiner.get("coef", [])]
    expected = [float(row["expected_mes"]) for row in rows]
    entries = []
    for name, coef in zip(input_names, coefs):
        values = [_combiner_input_value(row, name) for row in rows]
        corr = _spearman(values, expected)
        entries.append({
            "name": name,
            "kind": "subscore" if name in dimension_names else "feature",
            "coef": _round(coef, 6),
            "spearman_expected": None if corr is None else _round(corr, 4),
            "coef_times_corr": None if corr is None else _round(coef * corr, 4),
        })
    return {
        "architecture": combiner.get("architecture"),
        "calibration": combiner.get("calibration"),
        "intercept": _round(float(combiner.get("intercept", 0.0)), 6),
        "inputs": entries,
    }


def build_component_audit(
    locked_rows: list[dict[str, Any]],
    golden_rows: list[dict[str, Any]],
    combiner: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a component-level scoring audit from pre-scored rows."""
    components = {
        name: _component_summary(locked_rows, name, prefixes)
        for name, prefixes in COMPONENT_PREFIXES.items()
    }
    return {
        "eval_set_version": "human_locked_eval_v1",
        "locked_row_count": len(locked_rows),
        "component_summaries": components,
        "subscore_summary": _subscore_summary(locked_rows),
        "control_component_means": _control_component_means(locked_rows),
        "golden_order": _golden_order(golden_rows),
        "combiner_audit": None if combiner is None else build_combiner_audit(
            locked_rows, combiner,
        ),
        "top_misses": _top_misses(locked_rows),
        "interpretation_hints": [
            "High abs correlation with expected MES means the component is label-aligned on locked eval.",
            "High correlation with abs error means the signal may identify where the scorer breaks, not necessarily useful direction.",
            "Golden subscore ordering reveals whether a dimension carries the anchor ordering before final combiner calibration.",
        ],
    }


def _score_entry(entry: dict[str, Any]) -> dict[str, Any]:
    result = compute_mes_from_text(entry["text"], method="auto")
    return {
        "id": entry["id"],
        "source_index": entry.get("source_index"),
        "control_type": entry["control_type"],
        "label_overall_entropy": float(entry["label_overall_entropy"]),
        "expected_mes": float(entry["expected_mes"]),
        "model_mes": float(result["mes_score"]),
        "text": entry["text"],
        "features": {
            key: float(value)
            for key, value in result.items()
            if key != "mes_score" and not key.endswith(SUBSCORE_SUFFIX)
        },
        "subscores": {
            key: float(value)
            for key, value in result.items()
            if key.endswith(SUBSCORE_SUFFIX)
        },
    }


def score_locked_rows(path: Path) -> list[dict[str, Any]]:
    return [_score_entry(entry) for entry in load_eval_entries(path)]


def score_golden_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for anchor in payload.get("anchors", []):
        target = anchor["target"]
        text = payload["anchor_texts"][target]
        label_oe = float(anchor["overall_entropy"])
        expected_mes = (label_oe - 1.0) / 9.0 * 100.0
        rows.append(_score_entry({
            "id": f"golden_{target}",
            "source_index": anchor.get("idx"),
            "control_type": "golden_5",
            "label_overall_entropy": label_oe,
            "expected_mes": expected_mes,
            "text": text,
            "target": target,
        }) | {"target": target})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MES feature families and subscores on locked eval."
    )
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--combiner", type=Path, default=DEFAULT_COMBINER_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    locked_rows = score_locked_rows(args.eval_path)
    golden_rows = score_golden_rows(args.golden_path)
    combiner = json.loads(args.combiner.read_text(encoding="utf-8"))
    audit = build_component_audit(locked_rows, golden_rows, combiner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("MES Scoring Component Audit")
    print(f"  locked_rows={audit['locked_row_count']}")
    print("  top components by mean abs Spearman(expected):")
    ranked = sorted(
        audit["component_summaries"].values(),
        key=lambda row: row["mean_abs_spearman_expected"] or 0.0,
        reverse=True,
    )
    for row in ranked:
        print(
            f"    {row['component']}: "
            f"{row['mean_abs_spearman_expected']} ({row['feature_count']} features)"
        )
    print(
        "  golden_order_model_mes="
        f"{audit['golden_order']['model_mes']['ordered']}"
    )
    print(f"  report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
