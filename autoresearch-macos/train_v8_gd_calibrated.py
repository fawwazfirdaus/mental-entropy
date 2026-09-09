#!/usr/bin/env python3
"""Train MES v8: GD-aware exact-value calibrated combiner.

v8 keeps the v7 inference shape:
  Stage 1: 5 Ridge models, embedding -> FEP dimension labels.
  Stage 2: Ridge combiner, [5 subscores + selected features] -> OE.

Differences from v7:
  - candidate combiners are selected by human MES MAE, not correlation;
  - GD v2 features are forced into the candidate feature sets;
  - optional isotonic MES calibration is saved in combiner.json.

By default artifacts are written to artifacts/models/subscore_embedding_v8.
Use --output-dir src/mental_entropy/models/_artifacts to promote manually.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prepare import (  # noqa: E402
    DIMENSION_NAMES,
    SOURCE_HUMAN,
    SOURCE_SYNTHETIC,
    load_cached_data,
    load_cached_embeddings,
    load_dimension_labels,
    pearson_correlation,
)
from train_embedding_regression import (  # noqa: E402
    compute_feature_human_correlations,
    make_sample_weights_fn,
)
from train_subscores import build_feature_matrix  # noqa: E402
from train_subscores_v4 import get_stratified_folds  # noqa: E402


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "models" / "subscore_embedding_v8"
W = 78

DIMENSION_DISPLAY = {
    "prediction_coherence": ("Thought Flow", "How smoothly your thoughts connect"),
    "model_complexity": ("Focus", "How well you organized around key themes"),
    "compression_progress": ("Insight", "Whether you moved toward understanding"),
    "belief_integration": ("Integration", "How well you hold contradictions with awareness"),
    "precision_weighting": ("Clarity", "How decisively you express yourself"),
}

GD_CORE_FEATURES = (
    "gd_disorder_markers_per_500w",
    "gd_disorder_marker_count_norm",
    "gd_global_disorder_score",
    "gd_entity_drift_score",
    "gd_compression_failure_score",
    "gd_abandoned_setup_count_norm",
    "gd_corruption_rate",
    "gd_unanchored_question_rate",
)

LEGACY_ANCHOR_FEATURES = (
    "ne_fragment_sentence_rate",
    "ce_inter_block_break_rate",
    "bc_unresolved_count",
    "bc_conflict_rate",
    "cle_hedge_rate",
    "se_dominant_cluster_frac",
    "ce_adj_mean",
    "ce_adj_median",
    "ce_adj_p25",
)


def _mes_from_oe(values: np.ndarray) -> np.ndarray:
    return np.clip((values - 1.0) / 9.0 * 100.0, 0.0, 100.0)


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean((pred - target) ** 2))))


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def _metrics(pred_oe: np.ndarray, labels_oe: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pred_mes = _mes_from_oe(pred_oe[mask])
    target_mes = _mes_from_oe(labels_oe[mask])
    return {
        "mae": _mae(pred_mes, target_mes),
        "rmse": _rmse(pred_mes, target_mes),
        "corr": pearson_correlation(pred_oe[mask], labels_oe[mask]),
    }


def _fit_scaled_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y, sample_weight=sample_weight)
    return model, scaler


def _raw_linear_params(model: Ridge, scaler: StandardScaler) -> tuple[np.ndarray, float]:
    scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    coef = model.coef_ / scale
    intercept = float(model.intercept_ - np.dot(coef, scaler.mean_))
    return coef.astype(np.float64), intercept


def _predict_scaled(model: Ridge, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return model.predict(scaler.transform(X))


def _ordered_unique(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def _candidate_feature_sets(
    feature_names: list[str],
    feat_corrs: list[tuple[str, float]],
) -> dict[str, list[str]]:
    top_10 = [name for name, _ in feat_corrs[:10]]
    top_20 = [name for name, _ in feat_corrs[:20]]
    top_35 = [name for name, _ in feat_corrs[:35]]
    gd_present = [name for name in GD_CORE_FEATURES if name in feature_names]
    anchors_present = [name for name in LEGACY_ANCHOR_FEATURES if name in feature_names]

    return {
        "subscores_only": [],
        "top10_plus_gd": _ordered_unique(top_10 + gd_present),
        "top20_plus_gd": _ordered_unique(top_20 + gd_present),
        "anchors_plus_gd": _ordered_unique(anchors_present + gd_present),
        "top35_plus_gd": _ordered_unique(top_35 + gd_present),
    }


def _fit_oof_dimensions(
    E: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    sw_fn: Any,
) -> tuple[np.ndarray, float, dict[str, dict[str, float]]]:
    n = E.shape[0]
    n_dims = len(DIMENSION_NAMES)
    best_alpha = 0.1
    best_score = float("inf")

    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        maes = []
        for dim_name in DIMENSION_NAMES:
            d_labels = dim_labels[dim_name]
            oof = np.full(n, np.nan)
            for train_idx, val_idx in folds:
                model = Ridge(alpha=alpha)
                model.fit(
                    E[train_idx],
                    d_labels[train_idx],
                    sample_weight=sw_fn(sources[train_idx]),
                )
                oof[val_idx] = model.predict(E[val_idx])
            human_mask = sources == SOURCE_HUMAN
            maes.append(_mae(oof[human_mask], d_labels[human_mask]))
        avg_mae = float(np.mean(maes))
        print(f"  dim_alpha={alpha:<6}: avg_human_dim_mae={avg_mae:.4f}")
        if avg_mae < best_score:
            best_score = avg_mae
            best_alpha = alpha

    dim_oof = np.zeros((n, n_dims), dtype=np.float64)
    dim_metrics: dict[str, dict[str, float]] = {}
    human_mask = sources == SOURCE_HUMAN

    print(f"\n  Dimension OOF results (alpha={best_alpha}):")
    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        d_labels = dim_labels[dim_name]
        for train_idx, val_idx in folds:
            model = Ridge(alpha=best_alpha)
            model.fit(
                E[train_idx],
                d_labels[train_idx],
                sample_weight=sw_fn(sources[train_idx]),
            )
            dim_oof[val_idx, d_idx] = model.predict(E[val_idx])
        dim_metrics[dim_name] = {
            "human_mae": _mae(dim_oof[human_mask, d_idx], d_labels[human_mask]),
            "human_corr": pearson_correlation(dim_oof[human_mask, d_idx], d_labels[human_mask]),
            "cv_corr": pearson_correlation(dim_oof[:, d_idx], d_labels),
        }
        print(
            f"    {dim_name:<24} "
            f"human_mae={dim_metrics[dim_name]['human_mae']:.4f} "
            f"human_corr={dim_metrics[dim_name]['human_corr']:.4f}"
        )

    return dim_oof, best_alpha, dim_metrics


def _fit_isotonic_calibration(
    raw_pred_oe: np.ndarray,
    labels_oe: np.ndarray,
    mask: np.ndarray,
) -> tuple[IsotonicRegression, dict[str, Any]]:
    raw_mes = _mes_from_oe(raw_pred_oe[mask])
    target_mes = _mes_from_oe(labels_oe[mask])
    iso = IsotonicRegression(y_min=0.0, y_max=100.0, out_of_bounds="clip")
    iso.fit(raw_mes, target_mes)
    calibration = {
        "kind": "isotonic_mes",
        "trained_on": "human_oof_predictions",
        "x_thresholds": [float(x) for x in iso.X_thresholds_],
        "y_thresholds": [float(y) for y in iso.y_thresholds_],
    }
    return iso, calibration


def _evaluate_combiner_candidates(
    dim_oof: np.ndarray,
    features: list[dict[str, float]],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_names: list[str],
    candidate_sets: dict[str, list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    human_mask = sources == SOURCE_HUMAN
    all_results = []
    best: dict[str, Any] | None = None

    for set_name, selected_features in candidate_sets.items():
        X_feat = build_feature_matrix(features, selected_features) if selected_features else None
        X = dim_oof if X_feat is None else np.hstack([dim_oof, X_feat])
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            raw_oof = np.full(len(labels), np.nan)
            for train_idx, val_idx in folds:
                model, scaler = _fit_scaled_ridge(X[train_idx], labels[train_idx], alpha)
                raw_oof[val_idx] = _predict_scaled(model, scaler, X[val_idx])

            iso, calibration = _fit_isotonic_calibration(raw_oof, labels, human_mask)
            calibrated_mes = raw_oof.copy()
            calibrated_mes = iso.predict(_mes_from_oe(calibrated_mes))
            target_mes = _mes_from_oe(labels)
            human_mae = _mae(calibrated_mes[human_mask], target_mes[human_mask])
            human_rmse = _rmse(calibrated_mes[human_mask], target_mes[human_mask])
            raw_human = _metrics(raw_oof, labels, human_mask)
            result = {
                "feature_set": set_name,
                "alpha": alpha,
                "feature_names": selected_features,
                "human_mae": human_mae,
                "human_rmse": human_rmse,
                "human_corr": pearson_correlation(calibrated_mes[human_mask], target_mes[human_mask]),
                "raw_human_mae": raw_human["mae"],
                "raw_human_corr": raw_human["corr"],
                "oof_raw_oe": raw_oof,
                "calibration": calibration,
            }
            all_results.append(result)
            print(
                f"  {set_name:<16} alpha={alpha:<6} "
                f"human_mae={human_mae:>6.2f} raw_mae={raw_human['mae']:>6.2f} "
                f"human_corr={result['human_corr']:.4f}"
            )
            if best is None or human_mae < best["human_mae"]:
                best = result

    if best is None:
        raise RuntimeError("No combiner candidate was evaluated")
    return best, all_results


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _save_artifacts(
    output_dir: Path,
    E: np.ndarray,
    features: list[dict[str, float]],
    labels: np.ndarray,
    sources: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    best_dim_alpha: float,
    best_combiner: dict[str, Any],
    dim_metrics: dict[str, dict[str, float]],
    all_candidate_results: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sw_fn = make_sample_weights_fn(0.3)

    dim_models = {}
    for dim_name in DIMENSION_NAMES:
        model = Ridge(alpha=best_dim_alpha)
        model.fit(E, dim_labels[dim_name], sample_weight=sw_fn(sources))
        dim_models[dim_name] = model
        _write_json(
            output_dir / f"dim_{dim_name}.json",
            {
                "coef": [float(c) for c in model.coef_],
                "intercept": float(model.intercept_),
                "alpha": float(best_dim_alpha),
                "embedding_dim": int(E.shape[1]),
                "dimension": dim_name,
                "display_name": DIMENSION_DISPLAY[dim_name][0],
                "description": DIMENSION_DISPLAY[dim_name][1],
            },
        )

    full_subscores = np.column_stack([
        dim_models[dim_name].predict(E) for dim_name in DIMENSION_NAMES
    ])
    selected_features = best_combiner["feature_names"]
    X_feat = build_feature_matrix(features, selected_features) if selected_features else None
    X_full = full_subscores if X_feat is None else np.hstack([full_subscores, X_feat])
    model, scaler = _fit_scaled_ridge(X_full, labels, float(best_combiner["alpha"]))
    coef, intercept = _raw_linear_params(model, scaler)

    combiner_data = {
        "architecture": "subscore_embedding_v7",
        "model_version": "subscore_embedding_v8_gd_calibrated",
        "coef": [float(c) for c in coef],
        "intercept": float(intercept),
        "alpha": float(best_combiner["alpha"]),
        "dimension_names": list(DIMENSION_NAMES),
        "n_dims": len(DIMENSION_NAMES),
        "feature_names": selected_features,
        "uses_features": bool(selected_features),
        "calibration": best_combiner["calibration"],
        "dimension_display": {
            dim: {
                "name": DIMENSION_DISPLAY[dim][0],
                "description": DIMENSION_DISPLAY[dim][1],
            }
            for dim in DIMENSION_NAMES
        },
    }
    _write_json(output_dir / "combiner.json", combiner_data)

    manifest = {
        "architecture": "subscore_embedding_v7",
        "model_version": "subscore_embedding_v8_gd_calibrated",
        "training_script": "train_v8_gd_calibrated.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_entries": int(len(labels)),
        "n_human": int((sources == SOURCE_HUMAN).sum()),
        "n_synthetic": int((sources == SOURCE_SYNTHETIC).sum()),
        "embedding_dim": int(E.shape[1]),
        "dimension_names": list(DIMENSION_NAMES),
        "dim_alpha": float(best_dim_alpha),
        "combiner_alpha": float(best_combiner["alpha"]),
        "combiner_feature_set": best_combiner["feature_set"],
        "combiner_features": selected_features,
        "selection_metric": "human_oof_mes_mae_after_isotonic_calibration",
        "metrics": {
            "v8_human_mae": float(best_combiner["human_mae"]),
            "v8_human_rmse": float(best_combiner["human_rmse"]),
            "v8_human_corr": float(best_combiner["human_corr"]),
            "raw_human_mae": float(best_combiner["raw_human_mae"]),
            "raw_human_corr": float(best_combiner["raw_human_corr"]),
            "per_dimension": dim_metrics,
        },
        "candidate_results": [
            {
                key: value
                for key, value in result.items()
                if key not in {"oof_raw_oe", "calibration"}
            }
            for result in all_candidate_results
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GD-aware calibrated MES v8.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    print("=" * W)
    print("MES v8: GD-aware calibrated training")
    print("=" * W)

    features, labels, feature_names, sources = load_cached_data()
    E = load_cached_embeddings()
    dim_labels = load_dimension_labels()

    print(f"  Entries: {len(labels)}")
    print(f"  Human: {int((sources == SOURCE_HUMAN).sum())}")
    print(f"  Synthetic: {int((sources == SOURCE_SYNTHETIC).sum())}")
    print(f"  Features: {len(feature_names)} ({sum(1 for f in feature_names if f.startswith('gd_'))} GD)")
    print(f"  Output: {args.output_dir}")

    folds = get_stratified_folds(sources)
    sw_fn = make_sample_weights_fn(0.3)

    print("\nStage 1: dimension models")
    dim_oof, best_dim_alpha, dim_metrics = _fit_oof_dimensions(
        E, dim_labels, sources, folds, sw_fn,
    )

    print("\nStage 2: GD-aware combiner candidates")
    feat_corrs = compute_feature_human_correlations(features, labels, sources, feature_names)
    candidate_sets = _candidate_feature_sets(feature_names, feat_corrs)
    best_combiner, all_candidate_results = _evaluate_combiner_candidates(
        dim_oof, features, labels, sources, folds, feature_names, candidate_sets,
    )

    print("\nBest combiner")
    print(f"  feature_set={best_combiner['feature_set']}")
    print(f"  alpha={best_combiner['alpha']}")
    print(f"  human_mae={best_combiner['human_mae']:.2f}")
    print(f"  raw_human_mae={best_combiner['raw_human_mae']:.2f}")
    print("  features:")
    for name in best_combiner["feature_names"]:
        print(f"    {name}")

    print("\nSaving artifacts")
    _save_artifacts(
        args.output_dir,
        E,
        features,
        labels,
        sources,
        dim_labels,
        best_dim_alpha,
        best_combiner,
        dim_metrics,
        all_candidate_results,
    )

    print(f"  elapsed={time.time() - t0:.1f}s")
    print(f"  saved={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
