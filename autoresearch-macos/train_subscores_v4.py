"""
MES Subscore Model Training Pipeline v4 — Human-Prioritized.

Keeps the v3 dimension-aligned architecture (5 XGBoost per rubric dimension +
Ridge combiner) but grid-searches hyperparameters to maximize human_corr.

Key improvements over v3:
1. Stratified folds (proportional human/synthetic per fold)
2. Per-fold human_corr tracking with stability metric
3. Human-primary objective function (60% human, 25% cv, 15% stability)
4. Variable synthetic weight (instead of fixed equal-source contribution)
5. Ridge alpha sweep
6. XGB hyperparameter sweep

Usage:
    cd autoresearch-macos && PYTHONPATH=../src ../.venv/bin/python -u train_subscores_v4.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

from prepare import (
    DIMENSION_NAMES,
    N_FOLDS,
    SOURCE_HUMAN,
    SOURCE_SYNTHETIC,
    load_cached_data,
    load_dimension_labels,
    pearson_correlation,
)
from train_subscores import build_feature_matrix
from train_subscores_v3 import ALL_FEATURE_KEYS, ARTIFACTS_DIR

W = 72  # print width


# ---------------------------------------------------------------------------
# Stratified CV folds
# ---------------------------------------------------------------------------


def get_stratified_folds(
    sources: np.ndarray,
    n_folds: int = N_FOLDS,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create CV folds stratified by source (human vs synthetic)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in skf.split(np.zeros(len(sources)), sources):
        folds.append((train_idx, val_idx))
    return folds


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------


def compute_v4_objective(result: dict, cv_floor: float = 0.5) -> float:
    """Human-corr primary objective with cv_corr floor and stability bonus.

    Returns:
        Score in [0, 1] range, or -1.0 if cv_corr below floor.
    """
    cv = result["cv_corr"]
    human = result["oof_human_corr"]
    fold_std = result.get("fold_human_corr_std", 0.5)

    # Hard floor: disqualify configs where cv_corr is too low
    if cv < cv_floor:
        return -1.0

    cv_norm = max(0.0, min(1.0, cv))
    human_norm = max(0.0, min(1.0, human))
    # Stability: low fold-level human_corr variance is better
    stability = max(0.0, 1.0 - fold_std / 0.3)

    return 0.60 * human_norm + 0.25 * cv_norm + 0.15 * stability


# ---------------------------------------------------------------------------
# Parameterized training pipeline
# ---------------------------------------------------------------------------


def train_v4_pipeline(
    X: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    xgb_params: dict,
    synth_weight: float,
    ridge_alpha: float,
) -> dict:
    """Train full dimension-aligned pipeline with given hyperparameters.

    Returns dict with cv_corr, oof_human_corr, oof_synth_corr,
    fold_human_corrs, fold_human_corr_std, and per-dimension stats.
    """
    n = len(labels)
    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC

    # Build sample weights: human=1.0, synthetic=synth_weight
    w = np.ones(n, dtype=np.float64)
    w[synth_mask] = synth_weight

    # Stage 1: Per-dimension XGBoost OOF predictions
    oof_subscores = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)
    dim_cv_corrs: dict[str, float] = {}

    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        y_dim = dim_labels[dim_name]
        fold_corrs = []

        for train_idx, val_idx in folds:
            model = XGBRegressor(**xgb_params)
            model.fit(X[train_idx], y_dim[train_idx], sample_weight=w[train_idx])
            preds = model.predict(X[val_idx])
            oof_subscores[val_idx, d_idx] = preds
            fold_corrs.append(pearson_correlation(preds, y_dim[val_idx]))

        dim_cv_corrs[dim_name] = float(np.mean(fold_corrs))

    # Stage 2: Ridge combiner CV with per-fold human_corr tracking
    fold_cv_corrs = []
    fold_human_corrs = []

    for train_idx, val_idx in folds:
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(oof_subscores[train_idx], labels[train_idx], sample_weight=w[train_idx])
        preds = ridge.predict(oof_subscores[val_idx])

        fold_cv_corrs.append(pearson_correlation(preds, labels[val_idx]))

        # Human-only correlation for this fold
        human_val = human_mask[val_idx]
        if human_val.sum() > 5:
            fold_human_corrs.append(
                pearson_correlation(preds[human_val], labels[val_idx][human_val])
            )

    cv_corr = float(np.mean(fold_cv_corrs))
    cv_human_corr = float(np.mean(fold_human_corrs)) if fold_human_corrs else 0.0
    fold_human_std = float(np.std(fold_human_corrs)) if fold_human_corrs else 0.5

    # Full OOF combined for global per-source analysis
    ridge_full = Ridge(alpha=ridge_alpha)
    ridge_full.fit(oof_subscores, labels, sample_weight=w)
    combined = ridge_full.predict(oof_subscores)

    oof_human_corr = pearson_correlation(combined[human_mask], labels[human_mask])
    oof_synth_corr = pearson_correlation(combined[synth_mask], labels[synth_mask])

    return {
        "cv_corr": cv_corr,
        "cv_human_corr": cv_human_corr,
        "oof_human_corr": oof_human_corr,
        "oof_synth_corr": oof_synth_corr,
        "fold_human_corrs": fold_human_corrs,
        "fold_human_corr_std": fold_human_std,
        "dim_cv_corrs": dim_cv_corrs,
        "combiner_coef": ridge_full.coef_.tolist(),
        "combiner_intercept": float(ridge_full.intercept_),
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def run_grid_search(
    X: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    """Run grid search over synth_weight, XGB params, and ridge_alpha."""

    synth_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    max_depths = [2, 3, 4]
    n_estimators_list = [50, 100, 200]
    learning_rates = [0.05, 0.1]
    ridge_alphas = [0.1, 1.0, 10.0]

    xgb_grid = list(itertools.product(max_depths, n_estimators_list, learning_rates))

    total = len(synth_weights) * len(xgb_grid) * len(ridge_alphas)
    print(f"\n  Grid: {len(synth_weights)} synth_weights × {len(xgb_grid)} XGB combos "
          f"× {len(ridge_alphas)} ridge_alphas = {total} configurations")

    results: list[dict] = []
    best_score = -1.0
    best_human = -1.0
    i = 0

    for sw in synth_weights:
        for max_d, n_est, lr in xgb_grid:
            for r_alpha in ridge_alphas:
                i += 1

                xgb_params = dict(
                    n_estimators=n_est,
                    max_depth=max_d,
                    learning_rate=lr,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0,
                )

                result = train_v4_pipeline(
                    X, dim_labels, labels, sources, folds,
                    xgb_params=xgb_params,
                    synth_weight=sw,
                    ridge_alpha=r_alpha,
                )

                # Build compact result (drop large arrays)
                compact = {
                    "cv_corr": result["cv_corr"],
                    "cv_human_corr": result["cv_human_corr"],
                    "oof_human_corr": result["oof_human_corr"],
                    "oof_synth_corr": result["oof_synth_corr"],
                    "fold_human_corr_std": result["fold_human_corr_std"],
                    "dim_cv_corrs": result["dim_cv_corrs"],
                    "synth_weight": sw,
                    "max_depth": max_d,
                    "n_estimators": n_est,
                    "learning_rate": lr,
                    "ridge_alpha": r_alpha,
                }
                compact["v4_score"] = compute_v4_objective(compact)
                results.append(compact)

                if compact["v4_score"] > best_score:
                    best_score = compact["v4_score"]
                    best_human = compact["oof_human_corr"]

                if i % 25 == 0 or i == total:
                    print(f"  [{i:>4d}/{total}] best_obj={best_score:.4f} "
                          f"best_human={best_human:.4f} "
                          f"(current: cv={result['cv_corr']:.3f} "
                          f"human={result['oof_human_corr']:.3f} "
                          f"sw={sw} md={max_d} ne={n_est} lr={lr} ra={r_alpha})")

    return results


# ---------------------------------------------------------------------------
# Retrain best config and save artifacts
# ---------------------------------------------------------------------------


def train_and_save_best(
    X: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    best: dict,
    n_human: int,
    n_synth: int,
) -> dict:
    """Retrain best config on all data, save artifacts, return final metrics."""

    n = len(labels)
    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC

    sw = best["synth_weight"]
    r_alpha = best["ridge_alpha"]

    xgb_params = dict(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        learning_rate=best["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    # Build sample weights
    w = np.ones(n, dtype=np.float64)
    w[synth_mask] = sw

    # Generate OOF subscores for Ridge fitting (must use CV, not train predictions)
    oof_subscores = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)
    dim_models: dict[str, XGBRegressor] = {}

    print(f"\n  Retraining with: synth_weight={sw}, max_depth={best['max_depth']}, "
          f"n_estimators={best['n_estimators']}, lr={best['learning_rate']}, "
          f"ridge_alpha={r_alpha}")

    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        y_dim = dim_labels[dim_name]

        # OOF predictions for Ridge
        for train_idx, val_idx in folds:
            model = XGBRegressor(**xgb_params)
            model.fit(X[train_idx], y_dim[train_idx], sample_weight=w[train_idx])
            oof_subscores[val_idx, d_idx] = model.predict(X[val_idx])

        # Final model on all data
        final_model = XGBRegressor(**xgb_params)
        final_model.fit(X, y_dim, sample_weight=w)
        dim_models[dim_name] = final_model

        h_corr = pearson_correlation(oof_subscores[human_mask, d_idx], y_dim[human_mask])
        s_corr = pearson_correlation(oof_subscores[synth_mask, d_idx], y_dim[synth_mask])
        print(f"    {dim_name:<28s} human={h_corr:.4f}  synth={s_corr:.4f}")

    # Ridge combiner
    combiner = Ridge(alpha=r_alpha)
    combiner.fit(oof_subscores, labels, sample_weight=w)
    oof_combined = combiner.predict(oof_subscores)

    oof_human_corr = pearson_correlation(oof_combined[human_mask], labels[human_mask])
    oof_synth_corr = pearson_correlation(oof_combined[synth_mask], labels[synth_mask])

    print(f"\n  Final OOF human_corr: {oof_human_corr:.4f}")
    print(f"  Final OOF synth_corr: {oof_synth_corr:.4f}")

    coef_dict = dict(zip(DIMENSION_NAMES, combiner.coef_.round(4)))
    print(f"  Coefficients: {coef_dict}")
    print(f"  Intercept: {combiner.intercept_:.4f}")

    # Save artifacts (same format as v3 for _registry.py compatibility)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old model files
    for old_name in ["ce_model.json", "se_model.json", "ne_model.json",
                     "cle_model.json", "bc_model.json", "feature_filter.json"]:
        old_path = ARTIFACTS_DIR / old_name
        if old_path.exists():
            old_path.unlink()
            print(f"  Removed old {old_name}")

    # Save per-dimension XGBoost models
    for dim_name, model in dim_models.items():
        path = ARTIFACTS_DIR / f"{dim_name}_model.json"
        model.save_model(str(path))
        print(f"  Saved {path.name}")

    # Save combiner
    combiner_data = {
        "coef": combiner.coef_.tolist(),
        "intercept": float(combiner.intercept_),
        "dimension_names": list(DIMENSION_NAMES),
        "architecture": "dimension_aligned_v3",
    }
    combiner_path = ARTIFACTS_DIR / "combiner.json"
    with open(combiner_path, "w") as f:
        json.dump(combiner_data, f, indent=2)
    print(f"  Saved {combiner_path.name}")

    # Save manifest with v4 training details
    metrics = {
        "combiner_cv_corr": best["cv_corr"],
        "oof_human_corr": oof_human_corr,
        "oof_synth_corr": oof_synth_corr,
        "fold_human_corr_std": best["fold_human_corr_std"],
        "v4_objective_score": best["v4_score"],
    }

    manifest = {
        "architecture": "dimension_aligned_v3",
        "training_script": "v4",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_entries": n,
        "n_human": n_human,
        "n_synthetic": n_synth,
        "n_folds": N_FOLDS,
        "xgb_params": xgb_params,
        "synth_weight": sw,
        "ridge_alpha": r_alpha,
        "dimension_names": list(DIMENSION_NAMES),
        "n_features_per_dimension": len(ALL_FEATURE_KEYS),
        "feature_names": ALL_FEATURE_KEYS,
        "combiner_coef": combiner.coef_.tolist(),
        "combiner_intercept": float(combiner.intercept_),
        "metrics": metrics,
    }
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved {manifest_path.name}")

    return {
        "oof_human_corr": oof_human_corr,
        "oof_synth_corr": oof_synth_corr,
        "cv_corr": best["cv_corr"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()

    # Load data
    features, labels, feature_names, sources = load_cached_data()
    dim_labels = load_dimension_labels()
    n = len(features)
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())

    print("=" * W)
    print("MES SUBSCORE MODEL TRAINING v4 — HUMAN-PRIORITIZED")
    print("=" * W)
    print(f"\nDataset: {n} entries ({n_human} human + {n_synth} synthetic)")
    print(f"Dimensions: {len(DIMENSION_NAMES)} ({', '.join(DIMENSION_NAMES)})")
    print(f"Features: {len(ALL_FEATURE_KEYS)}")

    # Build feature matrix
    X = build_feature_matrix(features, ALL_FEATURE_KEYS)

    # Stratified folds (proportional human/synthetic per fold)
    folds = get_stratified_folds(sources)

    # Verify stratification
    print(f"\nFold stratification check:")
    human_mask = sources == SOURCE_HUMAN
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        n_h = int(human_mask[val_idx].sum())
        n_s = len(val_idx) - n_h
        print(f"  Fold {fold_idx}: {len(val_idx)} entries ({n_h} human, {n_s} synthetic)")

    # Grid search
    print()
    print("=" * W)
    print("GRID SEARCH")
    print("=" * W)

    results = run_grid_search(X, dim_labels, labels, sources, folds)

    # Sort by objective score
    results.sort(key=lambda r: r["v4_score"], reverse=True)

    # Top-10 table
    print()
    print("=" * W)
    print("TOP-10 CONFIGURATIONS")
    print("=" * W)
    print(f"\n{'#':>3s} {'obj':>6s} {'cv':>6s} {'human':>7s} {'synth':>7s} "
          f"{'h_std':>6s} {'sw':>5s} {'md':>3s} {'ne':>4s} {'lr':>5s} {'ra':>5s}")
    print("-" * W)

    for rank, r in enumerate(results[:10]):
        print(f"  {rank + 1:>2d} {r['v4_score']:6.4f} {r['cv_corr']:6.3f} "
              f"{r['oof_human_corr']:7.4f} {r['oof_synth_corr']:7.4f} "
              f"{r['fold_human_corr_std']:6.3f} {r['synth_weight']:5.2f} "
              f"{r['max_depth']:>3d} {r['n_estimators']:>4d} "
              f"{r['learning_rate']:5.2f} {r['ridge_alpha']:5.1f}")

    best = results[0]

    # Retrain best config and save artifacts
    print()
    print("=" * W)
    print("RETRAINING BEST CONFIG")
    print("=" * W)

    final = train_and_save_best(
        X, dim_labels, labels, sources, folds, best,
        n_human=n_human, n_synth=n_synth,
    )

    # Load v3 metrics from existing manifest for comparison
    v3_human = 0.2815  # from diagnosis
    v3_cv = 0.7930

    # Linear baseline
    from prepare import evaluate_weights
    from train import WEIGHTS as LINEAR_WEIGHTS
    linear = evaluate_weights(LINEAR_WEIGHTS, features, labels, sources=sources)

    # Comparison table
    print()
    print("=" * W)
    print("COMPARISON")
    print("=" * W)
    print(f"\n{'Model':<30s} {'cv_corr':>10s} {'human_corr':>12s} {'synth_corr':>12s}")
    print("-" * W)
    print(f"  {'Linear (13 feat)':<28s} {linear['cv_corr']:>10.4f} "
          f"{linear.get('human_corr', 0):>12.4f} {linear.get('synth_corr', 0):>12.4f}")
    print(f"  {'Dim-aligned v3':<28s} {v3_cv:>10.4f} "
          f"{v3_human:>12.4f} {'0.9246':>12s}")
    print(f"  {'Dim-aligned v4 (human-opt)':<28s} {final['cv_corr']:>10.4f} "
          f"{final['oof_human_corr']:>12.4f} {final['oof_synth_corr']:>12.4f}")

    h_gap = final["oof_human_corr"] - v3_human
    print(f"\n  human_corr change vs v3: {h_gap:+.4f}")
    print(f"  human_corr change vs linear: {final['oof_human_corr'] - linear.get('human_corr', 0):+.4f}")

    elapsed = time.time() - start
    print()
    print("=" * W)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Artifacts saved to {ARTIFACTS_DIR}")
    print("=" * W)


if __name__ == "__main__":
    main()
