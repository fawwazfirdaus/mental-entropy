#!/usr/bin/env python3
"""Train a candidate sign-constrained v10 MES combiner.

This is diagnostic-only. It keeps the current v7 dimension models fixed and
fits only the final combiner with explicit coefficient signs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare import SOURCE_HUMAN, load_cached_data, load_cached_embeddings  # noqa: E402
from evaluate_locked_human_eval import (  # noqa: E402
    DEFAULT_EVAL_PATH,
    DEFAULT_GOLDEN_PATH,
    DEFAULT_OUTPUT_PATH,
    SELECTED_FEATURES,
    build_report,
    load_eval_entries,
    write_report,
)
from mental_entropy.models import _registry  # noqa: E402
from mental_entropy.score import compute_mes_from_text  # noqa: E402


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_COMBINER_PATH = (
    PROJECT_ROOT / "src" / "mental_entropy" / "models" / "_artifacts" / "combiner.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "models" / "v10_constrained_combiner"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "eval" / "human_locked_eval_v1_report_v10_constrained_combiner.json"
)

FEATURE_SIGN_CONTRACT: dict[str, int] = {
    "bc_belief_sentence_count": 1,
    "bc_conflict_pair_count": 1,
    "bc_conflict_rate": 1,
    "bc_unresolved_count": 1,
    "ce_adj_mean": -1,
    "ce_adj_median": -1,
    "ce_adj_p25": -1,
    "ne_fragment_sentence_rate": 1,
    "se_dominant_cluster_frac": -1,
    "se_intra_mean": -1,
}


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _mes_from_overall_entropy(value: float) -> float:
    return min(100.0, max(0.0, (float(value) - 1.0) / 9.0 * 100.0))


def _input_names(combiner: dict[str, Any]) -> list[str]:
    return list(combiner.get("dimension_names", [])) + list(combiner.get("feature_names", []))


def _sign_for_input(name: str, combiner: dict[str, Any]) -> int:
    if name in combiner.get("dimension_names", []):
        return -1
    return FEATURE_SIGN_CONTRACT.get(name, 0)


def coefficient_bounds(
    input_names: list[str],
    combiner: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return coefficient bounds plus an unbounded intercept slot."""
    lower: list[float] = []
    upper: list[float] = []
    for name in input_names:
        sign = _sign_for_input(name, combiner)
        if sign < 0:
            lower.append(-np.inf)
            upper.append(0.0)
        elif sign > 0:
            lower.append(0.0)
            upper.append(np.inf)
        else:
            lower.append(-np.inf)
            upper.append(np.inf)
    lower.append(-np.inf)
    upper.append(np.inf)
    return np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)


def standardize_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    scales = X.std(axis=0)
    scales[scales == 0.0] = 1.0
    return (X - means) / scales, means, scales


def _training_subscores(embeddings: np.ndarray, features: list[dict[str, float]]) -> list[dict[str, float]]:
    _registry.reset_cache()
    return [
        _registry.predict_subscores_from_embedding(embedding.tolist(), row)
        for embedding, row in zip(embeddings, features)
    ]


def build_input_matrix(
    features: list[dict[str, float]],
    subscores: list[dict[str, float]],
    input_names: list[str],
    dimension_names: list[str],
) -> np.ndarray:
    rows = []
    dimension_set = set(dimension_names)
    for feature_row, subscore_row in zip(features, subscores):
        values = []
        for name in input_names:
            if name in dimension_set:
                values.append(float(subscore_row.get(f"{name}_subscore", 0.0)))
            else:
                values.append(float(feature_row.get(name, 0.0)))
        rows.append(values)
    return np.array(rows, dtype=np.float64)


def fit_constrained_combiner(
    X: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    input_names: list[str],
    combiner: dict[str, Any],
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Fit a sign-constrained linear combiner on human rows only."""
    human_mask = sources == SOURCE_HUMAN
    X_human = X[human_mask]
    y_human = y[human_mask]
    X_std, means, scales = standardize_matrix(X_human)
    design = np.hstack([X_std, np.ones((len(X_std), 1), dtype=np.float64)])

    if alpha > 0.0:
        penalty = np.zeros((len(input_names), len(input_names) + 1), dtype=np.float64)
        penalty[:, :len(input_names)] = np.eye(len(input_names)) * np.sqrt(alpha)
        design = np.vstack([design, penalty])
        y_human = np.concatenate([y_human, np.zeros(len(input_names), dtype=np.float64)])

    lower, upper = coefficient_bounds(input_names, combiner)
    result = lsq_linear(design, y_human, bounds=(lower, upper), lsmr_tol="auto")

    coef = result.x[:len(input_names)]
    intercept = float(result.x[-1])
    return {
        "model_version": "v10_constrained_combiner",
        "base_model": "current_v7_dimension_models",
        "architecture": "fixed_subscores_sign_constrained_combiner",
        "dimension_names": list(combiner.get("dimension_names", [])),
        "feature_names": list(combiner.get("feature_names", [])),
        "input_names": input_names,
        "coef_standardized": [float(value) for value in coef],
        "intercept": intercept,
        "feature_means": [float(value) for value in means],
        "feature_scales": [float(value) for value in scales],
        "alpha": float(alpha),
        "training_rows": int(human_mask.sum()),
        "sign_contract": {
            name: _sign_for_input(name, combiner)
            for name in input_names
        },
        "solver_status": int(result.status),
        "solver_cost": float(result.cost),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def predict_overall_entropy(
    values: list[float],
    artifact: dict[str, Any],
) -> float:
    x = np.array(values, dtype=np.float64)
    means = np.array(artifact["feature_means"], dtype=np.float64)
    scales = np.array(artifact["feature_scales"], dtype=np.float64)
    coef = np.array(artifact["coef_standardized"], dtype=np.float64)
    x_std = (x - means) / scales
    return float(np.dot(coef, x_std)) + float(artifact["intercept"])


def _score_result_with_artifact(result: dict[str, float], artifact: dict[str, Any]) -> float:
    dimension_names = set(artifact["dimension_names"])
    values = []
    for name in artifact["input_names"]:
        if name in dimension_names:
            values.append(float(result.get(f"{name}_subscore", 0.0)))
        else:
            values.append(float(result.get(name, 0.0)))
    return _mes_from_overall_entropy(predict_overall_entropy(values, artifact))


def score_entries_with_artifact(
    entries: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        result = compute_mes_from_text(entry["text"], method="auto")
        model_mes = _score_result_with_artifact(result, artifact)
        rows.append({
            "id": entry["id"],
            "source_index": entry.get("source_index"),
            "control_type": entry["control_type"],
            "label_overall_entropy": float(entry["label_overall_entropy"]),
            "expected_mes": float(entry["expected_mes"]),
            "model_mes": _round(model_mes),
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
    return rows


def score_golden_with_artifact(path: Path, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for anchor in payload.get("anchors", []):
        target = anchor["target"]
        text = payload["anchor_texts"][target]
        result = compute_mes_from_text(text, method="auto")
        label_oe = float(anchor["overall_entropy"])
        expected_mes = _mes_from_overall_entropy(label_oe)
        rows.append({
            "target": target,
            "source_index": anchor.get("idx"),
            "label_overall_entropy": label_oe,
            "expected_mes": _round(expected_mes),
            "model_mes": _round(_score_result_with_artifact(result, artifact)),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v10 constrained combiner candidate.")
    parser.add_argument("--combiner", type=Path, default=DEFAULT_COMBINER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--alpha", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_combiner = json.loads(args.combiner.read_text(encoding="utf-8"))
    input_names = _input_names(base_combiner)
    dimension_names = list(base_combiner.get("dimension_names", []))

    features, labels, _, sources = load_cached_data()
    embeddings = load_cached_embeddings()
    subscores = _training_subscores(embeddings, features)
    X = build_input_matrix(features, subscores, input_names, dimension_names)
    artifact = fit_constrained_combiner(
        X,
        labels,
        sources,
        input_names,
        base_combiner,
        alpha=args.alpha,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "combiner.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    entries = load_eval_entries(args.eval_path)
    report = build_report(
        score_entries_with_artifact(entries, artifact),
        score_golden_with_artifact(args.golden_path, artifact),
    )
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    report["baseline_metrics"] = baseline["metrics"]
    report["candidate_artifact"] = _project_relative(artifact_path)
    report["constrained_combiner_summary"] = {
        "baseline_mae": baseline["metrics"]["mae"],
        "candidate_mae": report["metrics"]["mae"],
        "mae_improved": report["metrics"]["mae"] < baseline["metrics"]["mae"],
    }
    write_report(report, args.report)

    print("MES v10 constrained combiner candidate")
    print(f"  artifact: {artifact_path}")
    print(f"  report: {args.report}")
    print(f"  mae: {report['metrics']['mae']}")
    print(f"  golden_order: {report['golden_5']['order_preserved']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
