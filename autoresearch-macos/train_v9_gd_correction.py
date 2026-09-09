#!/usr/bin/env python3
"""Train MES v9 candidate: post-v7 GD uplift correction.

This does not replace the production scorer. It writes a candidate correction
artifact and locked-eval report under artifacts/ so promotion remains manual.
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
from mental_entropy.models import _registry  # noqa: E402
from evaluate_locked_human_eval import (  # noqa: E402
    DEFAULT_EVAL_PATH,
    DEFAULT_GOLDEN_PATH,
    DEFAULT_OUTPUT_PATH,
    build_report,
    load_eval_entries,
    score_golden,
    write_report,
)
from mental_entropy.score import compute_mes_from_text  # noqa: E402


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_BASELINE_REPORT = DEFAULT_OUTPUT_PATH
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "models" / "v9_gd_correction"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "eval" / "human_locked_eval_v1_report_v9_gd_correction.json"
)
DEFAULT_HIGH_LABEL_WEIGHT = 50.0
DEFAULT_FALSE_LOW_WEIGHT = 2.0

GD_UPLIFT_FEATURES: tuple[str, ...] = (
    "gd_disorder_marker_count_norm",
    "gd_disorder_markers_per_500w",
    "gd_global_disorder_score",
    "gd_loop_repetition_rate",
    "gd_runon_chain_rate",
    "gd_unanchored_question_rate",
    "gd_corruption_rate",
    "gd_compression_failure_score",
    "gd_marker_x_global",
    "gd_density_cap",
    "gd_marker_or_density",
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def correction_feature_values(features: dict[str, float]) -> dict[str, float]:
    """Return raw and derived GD correction features."""
    values = {key: float(value) for key, value in features.items()}
    marker_norm = float(values.get("gd_disorder_marker_count_norm", 0.0))
    density_cap = min(float(values.get("gd_disorder_markers_per_500w", 0.0)) / 20.0, 1.0)
    global_score = float(values.get("gd_global_disorder_score", 0.0))
    values["gd_marker_x_global"] = marker_norm * global_score
    values["gd_density_cap"] = density_cap
    values["gd_marker_or_density"] = max(marker_norm, density_cap)
    return values


def apply_gd_correction(
    base_mes: float,
    features: dict[str, float],
    artifact: dict[str, Any],
) -> float:
    """Apply a nonnegative GD uplift correction to a base MES score."""
    feature_names = artifact["feature_names"]
    coef = artifact["coef"]
    correction_features = correction_feature_values(features)
    raw_uplift = sum(
        float(weight) * float(correction_features.get(name, 0.0))
        for name, weight in zip(feature_names, coef)
    )
    uplift = _clamp(raw_uplift, 0.0, float(artifact.get("max_uplift", 40.0)))
    return _clamp(float(base_mes) + uplift, 0.0, 100.0)


def acceptance_summary(
    candidate_report: dict[str, Any],
    baseline_report: dict[str, Any],
) -> dict[str, bool]:
    """Return locked-eval promotion checks for a v9 candidate."""
    candidate_controls = candidate_report["control_checks"]
    baseline_controls = baseline_report["control_checks"]

    overall_mae_improved = (
        float(candidate_report["metrics"]["mae"])
        < float(baseline_report["metrics"]["mae"])
    )
    neutral_fragmented_improved = (
        float(candidate_controls["neutral_fragmented_positive_mean_model_mes"])
        > float(baseline_controls["neutral_fragmented_positive_mean_model_mes"])
    )
    neutral_fragmented_gte_60 = (
        float(candidate_controls["neutral_fragmented_positive_mean_model_mes"]) >= 60.0
    )
    emotional_coherent_not_regressed = (
        float(candidate_controls["emotional_coherent_negative_mean_model_mes"])
        <= max(40.0, float(baseline_controls["emotional_coherent_negative_mean_model_mes"]) + 5.0)
    )
    high_band_mae_improved = (
        float(candidate_report["per_band"]["spectrum_7_8"]["mae"])
        < float(baseline_report["per_band"]["spectrum_7_8"]["mae"])
        and float(candidate_report["per_band"]["spectrum_9_10"]["mae"])
        < float(baseline_report["per_band"]["spectrum_9_10"]["mae"])
    )

    return {
        "overall_mae_improved": overall_mae_improved,
        "neutral_fragmented_improved": neutral_fragmented_improved,
        "neutral_fragmented_gte_60": neutral_fragmented_gte_60,
        "emotional_coherent_not_regressed": emotional_coherent_not_regressed,
        "high_band_mae_improved": high_band_mae_improved,
        "promotable": (
            overall_mae_improved
            and neutral_fragmented_improved
            and neutral_fragmented_gte_60
            and emotional_coherent_not_regressed
            and high_band_mae_improved
        ),
    }


def _feature_matrix(
    features: list[dict[str, float]],
    feature_names: tuple[str, ...],
) -> np.ndarray:
    return np.array(
        [
            [
                float(correction_feature_values(row).get(name, 0.0))
                for name in feature_names
            ]
            for row in features
        ],
        dtype=np.float64,
    )


def _current_v7_mes_scores(
    embeddings: np.ndarray,
    features: list[dict[str, float]],
) -> np.ndarray:
    scores = []
    _registry.reset_cache()
    for emb, row in zip(embeddings, features):
        score, _ = _registry.predict_mes_v7(emb.tolist(), row)
        scores.append(score)
    return np.array(scores, dtype=np.float64)


def train_correction(
    features: list[dict[str, float]],
    labels_oe: np.ndarray,
    sources: np.ndarray,
    base_mes: np.ndarray,
    feature_names: tuple[str, ...] = GD_UPLIFT_FEATURES,
    high_label_weight: float = DEFAULT_HIGH_LABEL_WEIGHT,
    false_low_weight: float = DEFAULT_FALSE_LOW_WEIGHT,
) -> dict[str, Any]:
    """Fit a nonnegative linear uplift to positive residuals."""
    human_mask = sources == SOURCE_HUMAN
    target_mes = np.clip((labels_oe - 1.0) / 9.0 * 100.0, 0.0, 100.0)
    residual = np.maximum(0.0, target_mes - base_mes)
    X = _feature_matrix(features, feature_names)
    weights = np.ones(len(labels_oe), dtype=np.float64)
    weights[(target_mes >= 66.0) & human_mask] *= high_label_weight
    weights[(residual >= 20.0) & human_mask] *= false_low_weight
    sqrt_weights = np.sqrt(weights[human_mask])

    result = lsq_linear(
        X[human_mask] * sqrt_weights[:, None],
        residual[human_mask] * sqrt_weights,
        bounds=(0.0, np.inf),
        lsmr_tol="auto",
    )

    return {
        "model_version": "v9_gd_correction",
        "base_model": "current_subscore_embedding_v7",
        "correction_kind": "nonnegative_linear_uplift",
        "feature_names": list(feature_names),
        "coef": [float(value) for value in result.x],
        "max_uplift": 40.0,
        "training_weights": {
            "high_label_weight": float(high_label_weight),
            "false_low_weight": float(false_low_weight),
            "high_label_threshold_mes": 66.0,
            "false_low_residual_threshold_mes": 20.0,
        },
        "training_rows": int(human_mask.sum()),
        "solver_status": int(result.status),
        "solver_cost": float(result.cost),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _score_entries_with_correction(
    entries: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    scored = []
    for entry in entries:
        result = compute_mes_from_text(entry["text"], method="auto")
        correction_features = correction_feature_values({
            key: float(value) for key, value in result.items() if key.startswith("gd_")
        })
        corrected_mes = apply_gd_correction(
            float(result["mes_score"]),
            correction_features,
            artifact,
        )
        scored.append({
            "id": entry["id"],
            "source_index": entry["source_index"],
            "control_type": entry["control_type"],
            "label_overall_entropy": float(entry["label_overall_entropy"]),
            "expected_mes": float(entry["expected_mes"]),
            "model_mes": round(corrected_mes, 2),
            "text": entry["text"],
            "subscores": {
                key: round(float(value), 2)
                for key, value in result.items()
                if key.endswith("_subscore")
            },
            "selected_features": {
                key: round(float(correction_features.get(key, 0.0)), 4)
                for key in artifact["feature_names"]
            },
        })
    return scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v9 GD correction candidate.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--high-label-weight", type=float, default=DEFAULT_HIGH_LABEL_WEIGHT)
    parser.add_argument("--false-low-weight", type=float, default=DEFAULT_FALSE_LOW_WEIGHT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features, labels, _, sources = load_cached_data()
    embeddings = load_cached_embeddings()
    base_mes = _current_v7_mes_scores(embeddings, features)
    artifact = train_correction(
        features,
        labels,
        sources,
        base_mes,
        high_label_weight=args.high_label_weight,
        false_low_weight=args.false_low_weight,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "correction.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    entries = load_eval_entries(args.eval_path)
    scored_rows = _score_entries_with_correction(entries, artifact)
    report = build_report(scored_rows, score_golden(args.golden_path))
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    report["v9_acceptance"] = acceptance_summary(report, baseline)
    try:
        report["correction_artifact"] = str(artifact_path.relative_to(PROJECT_ROOT))
    except ValueError:
        report["correction_artifact"] = str(artifact_path)
    write_report(report, args.report)

    print("MES v9 GD correction candidate")
    print(f"  artifact: {artifact_path}")
    print(f"  report: {args.report}")
    print(f"  mae: {report['metrics']['mae']}")
    print(f"  promotable: {report['v9_acceptance']['promotable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
