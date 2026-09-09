"""
MES Subscore Model Training Pipeline v2 — Human-Aware.

Addresses the human correlation regression from v1 (0.301 → 0.235) by:
1. Feature filtering: Removes features with high human/synth correlation divergence
2. Stratified CV folds: Ensures proportional human/synthetic in each fold
3. Human multiplier sweep: Tests range of upweighting values
4. XGB hyperparameter sweep: Grid over max_depth, n_estimators, learning_rate
5. Dual-objective selection: Balances cv_corr + human_corr + source balance

Usage:
    cd autoresearch-macos && uv run python train_subscores_v2.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

from prepare import (
    N_FOLDS,
    SOURCE_HUMAN,
    SOURCE_SYNTHETIC,
    load_cached_data,
    pearson_correlation,
)
from train_subscores import (
    MODULE_FEATURE_KEYS,
    MODULE_NAMES,
    build_feature_matrix,
    save_artifacts,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path(__file__).parent.parent / "src" / "mental_entropy" / "models" / "_artifacts"

# ---------------------------------------------------------------------------
# Feature filtering
# ---------------------------------------------------------------------------


def compute_feature_divergence(
    features: list[dict[str, float]],
    labels: np.ndarray,
    sources: np.ndarray,
    feature_keys: list[str],
) -> dict[str, float]:
    """Compute |r_human - r_synth| for each feature.

    Returns dict mapping feature_name -> divergence score.
    """
    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC
    human_labels = labels[human_mask]
    synth_labels = labels[synth_mask]

    divergences = {}
    for fname in feature_keys:
        vals = np.array([f.get(fname, 0.0) for f in features], dtype=np.float64)
        r_human = pearson_correlation(vals[human_mask], human_labels)
        r_synth = pearson_correlation(vals[synth_mask], synth_labels)
        divergences[fname] = abs(r_human - r_synth)

    return divergences


def filter_features_by_divergence(
    module_keys: dict[str, list[str]],
    divergences: dict[str, float],
    threshold: float,
    min_features: int = 3,
) -> dict[str, list[str]]:
    """Filter features per module, removing those with divergence > threshold.

    Always keeps at least min_features per module (the least divergent ones).
    """
    filtered = {}
    for module, keys in module_keys.items():
        # Sort by divergence ascending (least divergent first)
        sorted_keys = sorted(keys, key=lambda k: divergences.get(k, 0.0))

        # Keep features below threshold, with minimum floor
        kept = [k for k in sorted_keys if divergences.get(k, 0.0) <= threshold]
        if len(kept) < min_features:
            kept = sorted_keys[:min_features]

        filtered[module] = kept

    return filtered


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
    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(sources)), sources):
        folds.append((train_idx, val_idx))
    return folds


# ---------------------------------------------------------------------------
# Core training (parameterized)
# ---------------------------------------------------------------------------


def train_pipeline(
    module_matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    human_multiplier: float,
    xgb_params: dict,
    ridge_alpha: float = 1.0,
) -> dict:
    """Train full stacked pipeline and return metrics.

    Returns dict with:
        cv_corr, human_corr, synth_corr, per-module stats,
        oof_subscores, combiner coefficients
    """
    n = len(labels)
    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC

    # Build sample weights
    w = np.ones(n, dtype=np.float64)
    w[sources == SOURCE_HUMAN] = human_multiplier

    # Stage 1: Per-module OOF
    oof_subscores = np.zeros((n, len(MODULE_NAMES)), dtype=np.float64)
    module_cv_corrs = {}

    for m_idx, module in enumerate(MODULE_NAMES):
        X = module_matrices[module]
        fold_corrs = []

        for train_idx, val_idx in folds:
            model = XGBRegressor(**xgb_params)
            model.fit(X[train_idx], labels[train_idx], sample_weight=w[train_idx])
            preds = model.predict(X[val_idx])
            oof_subscores[val_idx, m_idx] = preds
            fold_corrs.append(pearson_correlation(preds, labels[val_idx], weights=w[val_idx]))

        module_cv_corrs[module] = float(np.mean(fold_corrs))

    # Stage 2: Ridge combiner CV
    fold_corrs = []
    fold_human_corrs = []

    for train_idx, val_idx in folds:
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(oof_subscores[train_idx], labels[train_idx], sample_weight=w[train_idx])
        preds = ridge.predict(oof_subscores[val_idx])

        fold_corrs.append(pearson_correlation(preds, labels[val_idx], weights=w[val_idx]))

        # Human-only correlation for this fold
        human_val_mask = human_mask[val_idx]
        if human_val_mask.sum() > 5:
            fold_human_corrs.append(
                pearson_correlation(preds[human_val_mask], labels[val_idx][human_val_mask])
            )

    cv_corr = float(np.mean(fold_corrs))
    cv_human_corr = float(np.mean(fold_human_corrs)) if fold_human_corrs else 0.0

    # Full OOF combined for source-level analysis
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
        "module_cv_corrs": module_cv_corrs,
        "combiner_coef": ridge_full.coef_.tolist(),
        "combiner_intercept": float(ridge_full.intercept_),
        "oof_subscores": oof_subscores,
    }


def compute_dual_objective(result: dict, max_gap: float = 1.0) -> float:
    """Compute composite selection score.

    Balances overall accuracy, human accuracy, and source balance.
    """
    cv = result["cv_corr"]
    human = result["oof_human_corr"]
    synth = result["oof_synth_corr"]
    gap = abs(human - synth)

    # Normalize to [0, 1] range (approximate)
    cv_norm = max(0.0, min(1.0, cv))
    human_norm = max(0.0, min(1.0, human))
    balance = max(0.0, 1.0 - gap / max_gap)

    score = 0.5 * cv_norm + 0.3 * human_norm + 0.2 * balance
    return score


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def run_grid_search(
    features: list[dict[str, float]],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    divergences: dict[str, float],
) -> list[dict]:
    """Run the full grid search over multipliers, XGB params, and feature filters."""

    # Sweep parameters
    human_multipliers = [1.0, 1.29, 2.0, 3.0, 5.0, 8.0, 10.0]
    max_depths = [2, 3, 4]
    n_estimators_list = [50, 100, 200]
    learning_rates = [0.05, 0.1]
    divergence_thresholds = [0.3, 0.5, 0.7, 1.0]  # 1.0 = no filtering

    xgb_grid = list(itertools.product(max_depths, n_estimators_list, learning_rates))

    total = len(human_multipliers) * len(xgb_grid) * len(divergence_thresholds)
    print(f"\n  Grid: {len(human_multipliers)} multipliers × {len(xgb_grid)} XGB combos "
          f"× {len(divergence_thresholds)} filter thresholds = {total} configurations")

    results = []
    best_score = -1.0
    i = 0

    for div_thresh in divergence_thresholds:
        # Filter features for this threshold
        if div_thresh < 1.0:
            filtered_keys = filter_features_by_divergence(
                MODULE_FEATURE_KEYS, divergences, div_thresh
            )
        else:
            filtered_keys = MODULE_FEATURE_KEYS.copy()

        # Build matrices with filtered features
        module_matrices = {}
        for module in MODULE_NAMES:
            module_matrices[module] = build_feature_matrix(
                features, filtered_keys[module]
            )

        n_features = sum(len(v) for v in filtered_keys.values())

        for mult in human_multipliers:
            for max_d, n_est, lr in xgb_grid:
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

                result = train_pipeline(
                    module_matrices, labels, sources, folds,
                    human_multiplier=mult,
                    xgb_params=xgb_params,
                )

                # Don't store oof_subscores in results (memory)
                result_compact = {
                    k: v for k, v in result.items()
                    if k != "oof_subscores"
                }
                result_compact["human_multiplier"] = mult
                result_compact["max_depth"] = max_d
                result_compact["n_estimators"] = n_est
                result_compact["learning_rate"] = lr
                result_compact["div_threshold"] = div_thresh
                result_compact["n_features"] = n_features
                result_compact["dual_score"] = compute_dual_objective(result)

                results.append(result_compact)

                if result_compact["dual_score"] > best_score:
                    best_score = result_compact["dual_score"]

                if i % 50 == 0 or i == total:
                    print(f"  [{i}/{total}] best_dual={best_score:.4f} "
                          f"(current: cv={result['cv_corr']:.3f} "
                          f"human={result['oof_human_corr']:.3f})")

    return results


# ---------------------------------------------------------------------------
# Save v2 artifacts
# ---------------------------------------------------------------------------


def train_and_save_best(
    features: list[dict[str, float]],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    best_config: dict,
    divergences: dict[str, float],
) -> dict:
    """Retrain with best config on all data and save artifacts."""

    div_thresh = best_config["div_threshold"]
    if div_thresh < 1.0:
        filtered_keys = filter_features_by_divergence(
            MODULE_FEATURE_KEYS, divergences, div_thresh
        )
    else:
        filtered_keys = MODULE_FEATURE_KEYS.copy()

    module_matrices = {}
    for module in MODULE_NAMES:
        module_matrices[module] = build_feature_matrix(
            features, filtered_keys[module]
        )

    xgb_params = dict(
        n_estimators=best_config["n_estimators"],
        max_depth=best_config["max_depth"],
        learning_rate=best_config["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    human_mult = best_config["human_multiplier"]
    n = len(labels)
    w = np.ones(n, dtype=np.float64)
    w[sources == SOURCE_HUMAN] = human_mult

    # Generate OOF subscores for Ridge fitting
    oof_subscores = np.zeros((n, len(MODULE_NAMES)), dtype=np.float64)
    module_models: dict[str, XGBRegressor] = {}

    for m_idx, module in enumerate(MODULE_NAMES):
        X = module_matrices[module]

        # OOF predictions
        for train_idx, val_idx in folds:
            model = XGBRegressor(**xgb_params)
            model.fit(X[train_idx], labels[train_idx], sample_weight=w[train_idx])
            oof_subscores[val_idx, m_idx] = model.predict(X[val_idx])

        # Train final model on all data
        final_model = XGBRegressor(**xgb_params)
        final_model.fit(X, labels, sample_weight=w)
        module_models[module] = final_model

    # Train Ridge combiner
    combiner = Ridge(alpha=1.0)
    combiner.fit(oof_subscores, labels, sample_weight=w)

    # Evaluate
    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC

    combined = combiner.predict(oof_subscores)
    oof_human_corr = pearson_correlation(combined[human_mask], labels[human_mask])
    oof_synth_corr = pearson_correlation(combined[synth_mask], labels[synth_mask])

    # CV corr of combiner
    fold_corrs = []
    for train_idx, val_idx in folds:
        ridge_fold = Ridge(alpha=1.0)
        ridge_fold.fit(oof_subscores[train_idx], labels[train_idx], sample_weight=w[train_idx])
        p = ridge_fold.predict(oof_subscores[val_idx])
        fold_corrs.append(pearson_correlation(p, labels[val_idx], weights=w[val_idx]))
    cv_corr = float(np.mean(fold_corrs))

    n_human = int(human_mask.sum())
    n_synth = int(synth_mask.sum())

    metrics = {
        "combiner_cv_corr": cv_corr,
        "oof_human_corr": oof_human_corr,
        "oof_synth_corr": oof_synth_corr,
        "human_multiplier": human_mult,
        "div_threshold": div_thresh,
        "xgb_max_depth": best_config["max_depth"],
        "xgb_n_estimators": best_config["n_estimators"],
        "xgb_learning_rate": best_config["learning_rate"],
        "n_features_total": best_config["n_features"],
        "version": "v2",
    }

    # Update MODULE_FEATURE_KEYS in the registry if features were filtered
    # We need to save which features each model expects
    filtered_feature_lists = {m: filtered_keys[m] for m in MODULE_NAMES}

    # Save artifacts
    save_artifacts(
        module_models, combiner, metrics,
        n_entries=n, n_human=n_human, n_synthetic=n_synth,
    )

    # Also save feature filter info
    filter_info = {
        "div_threshold": div_thresh,
        "filtered_keys": filtered_feature_lists,
        "feature_counts": {m: len(filtered_keys[m]) for m in MODULE_NAMES},
        "original_counts": {m: len(MODULE_FEATURE_KEYS[m]) for m in MODULE_NAMES},
    }
    filter_path = ARTIFACTS_DIR / "feature_filter.json"
    with open(filter_path, "w") as f:
        json.dump(filter_info, f, indent=2)
    print(f"  Saved {filter_path.name}")

    return {
        "cv_corr": cv_corr,
        "oof_human_corr": oof_human_corr,
        "oof_synth_corr": oof_synth_corr,
        "combiner_coef": combiner.coef_.tolist(),
        "combiner_intercept": float(combiner.intercept_),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()

    features, labels, feature_names, sources = load_cached_data()
    n = len(features)
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())

    print("=" * 70)
    print("MES SUBSCORE MODEL TRAINING v2 — HUMAN-AWARE")
    print("=" * 70)
    print(f"\nDataset: {n} entries ({n_human} human + {n_synth} synthetic)")

    # Stratified folds
    print("\nUsing stratified CV folds (proportional human/synthetic per fold)")
    folds = get_stratified_folds(sources)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        n_human_val = int((sources[val_idx] == SOURCE_HUMAN).sum())
        n_synth_val = int((sources[val_idx] == SOURCE_SYNTHETIC).sum())
        print(f"  Fold {fold_idx}: val={len(val_idx)} "
              f"(human={n_human_val}, synth={n_synth_val})")

    # Compute feature divergences
    print("\nComputing feature-label correlation divergences...")
    all_feature_keys = []
    for keys in MODULE_FEATURE_KEYS.values():
        all_feature_keys.extend(keys)

    divergences = compute_feature_divergence(
        features, labels, sources, all_feature_keys
    )

    # Show filter impact at different thresholds
    print("\n  Feature counts at different divergence thresholds:")
    for thresh in [0.3, 0.5, 0.7, 1.0]:
        if thresh < 1.0:
            fk = filter_features_by_divergence(MODULE_FEATURE_KEYS, divergences, thresh)
        else:
            fk = MODULE_FEATURE_KEYS
        counts = {m: len(fk[m]) for m in MODULE_NAMES}
        total = sum(counts.values())
        print(f"    thresh={thresh:.1f}: {total} features "
              f"({', '.join(f'{m.upper()}={c}' for m, c in counts.items())})")

    # Get linear baseline for comparison
    from prepare import evaluate_weights
    from train import WEIGHTS as LINEAR_WEIGHTS
    linear = evaluate_weights(LINEAR_WEIGHTS, features, labels, sources=sources)
    print(f"\n  Linear baseline: cv_corr={linear['cv_corr']:.4f}  "
          f"human_corr={linear.get('human_corr', 0):.4f}")

    # Load v1 metrics for comparison
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            v1_manifest = json.load(f)
        v1_cv = v1_manifest["metrics"]["combiner_cv_corr"]
        v1_human = v1_manifest["metrics"]["oof_human_corr"]
        print(f"  v1 subscore:   cv_corr={v1_cv:.4f}  human_corr={v1_human:.4f}")
    else:
        v1_cv = 0.0
        v1_human = 0.0

    # Run grid search
    print("\n" + "=" * 70)
    print("GRID SEARCH")
    print("=" * 70)

    results = run_grid_search(features, labels, sources, folds, divergences)

    # Sort by dual objective
    results.sort(key=lambda r: r["dual_score"], reverse=True)

    # Print top 10
    print(f"\n  Top 10 configurations (by dual objective):")
    print(f"  {'#':>3s}  {'Dual':>6s}  {'CV':>7s}  {'Human':>7s}  {'Synth':>7s}  "
          f"{'Mult':>5s}  {'Depth':>5s}  {'nEst':>5s}  {'LR':>5s}  {'Div':>4s}  {'Feat':>4s}")
    print("  " + "-" * 75)

    for rank, r in enumerate(results[:10]):
        print(f"  {rank+1:>3d}  {r['dual_score']:>6.4f}  {r['cv_corr']:>7.4f}  "
              f"{r['oof_human_corr']:>7.4f}  {r['oof_synth_corr']:>7.4f}  "
              f"{r['human_multiplier']:>5.1f}  {r['max_depth']:>5d}  "
              f"{r['n_estimators']:>5d}  {r['learning_rate']:>5.2f}  "
              f"{r['div_threshold']:>4.1f}  {r['n_features']:>4d}")

    # Also show top 5 by human corr (for context)
    results_by_human = sorted(results, key=lambda r: r["oof_human_corr"], reverse=True)
    print(f"\n  Top 5 by human correlation:")
    print(f"  {'#':>3s}  {'Dual':>6s}  {'CV':>7s}  {'Human':>7s}  {'Synth':>7s}  "
          f"{'Mult':>5s}  {'Depth':>5s}  {'nEst':>5s}  {'LR':>5s}  {'Div':>4s}")
    print("  " + "-" * 65)
    for rank, r in enumerate(results_by_human[:5]):
        print(f"  {rank+1:>3d}  {r['dual_score']:>6.4f}  {r['cv_corr']:>7.4f}  "
              f"{r['oof_human_corr']:>7.4f}  {r['oof_synth_corr']:>7.4f}  "
              f"{r['human_multiplier']:>5.1f}  {r['max_depth']:>5d}  "
              f"{r['n_estimators']:>5d}  {r['learning_rate']:>5.2f}  "
              f"{r['div_threshold']:>4.1f}")

    # Also show top 5 by CV corr
    results_by_cv = sorted(results, key=lambda r: r["cv_corr"], reverse=True)
    print(f"\n  Top 5 by CV correlation:")
    print(f"  {'#':>3s}  {'Dual':>6s}  {'CV':>7s}  {'Human':>7s}  {'Synth':>7s}  "
          f"{'Mult':>5s}  {'Depth':>5s}  {'nEst':>5s}  {'LR':>5s}  {'Div':>4s}")
    print("  " + "-" * 65)
    for rank, r in enumerate(results_by_cv[:5]):
        print(f"  {rank+1:>3d}  {r['dual_score']:>6.4f}  {r['cv_corr']:>7.4f}  "
              f"{r['oof_human_corr']:>7.4f}  {r['oof_synth_corr']:>7.4f}  "
              f"{r['human_multiplier']:>5.1f}  {r['max_depth']:>5d}  "
              f"{r['n_estimators']:>5d}  {r['learning_rate']:>5.2f}  "
              f"{r['div_threshold']:>4.1f}")

    # Select best
    best = results[0]
    print(f"\n  Selected best config:")
    print(f"    Human multiplier: {best['human_multiplier']}")
    print(f"    XGB: max_depth={best['max_depth']}, n_estimators={best['n_estimators']}, "
          f"lr={best['learning_rate']}")
    print(f"    Div threshold: {best['div_threshold']} ({best['n_features']} features)")
    print(f"    Dual score: {best['dual_score']:.4f}")
    print(f"    CV corr: {best['cv_corr']:.4f}")
    print(f"    Human corr: {best['oof_human_corr']:.4f}")
    print(f"    Synth corr: {best['oof_synth_corr']:.4f}")

    # Retrain and save
    print("\n" + "=" * 70)
    print("RETRAINING BEST CONFIG AND SAVING ARTIFACTS")
    print("=" * 70)

    final = train_and_save_best(
        features, labels, sources, folds, best, divergences
    )

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\n  {'Method':<25s}  {'CV_corr':>8s}  {'Human_corr':>9s}  {'Synth_corr':>10s}")
    print("  " + "-" * 55)
    print(f"  {'Linear (13 feat)':<25s}  {linear['cv_corr']:>8.4f}  "
          f"{linear.get('human_corr', 0):>9.4f}  {linear.get('synth_corr', 0):>10.4f}")
    if v1_cv > 0:
        print(f"  {'Subscore v1':<25s}  {v1_cv:>8.4f}  {v1_human:>9.4f}  "
              f"{'---':>10s}")
    print(f"  {'Subscore v2 (new)':<25s}  {final['cv_corr']:>8.4f}  "
          f"{final['oof_human_corr']:>9.4f}  {final['oof_synth_corr']:>10.4f}")

    # Deltas
    print(f"\n  v2 vs linear:  cv {final['cv_corr'] - linear['cv_corr']:+.4f}  "
          f"human {final['oof_human_corr'] - linear.get('human_corr', 0):+.4f}")
    if v1_cv > 0:
        print(f"  v2 vs v1:      cv {final['cv_corr'] - v1_cv:+.4f}  "
              f"human {final['oof_human_corr'] - v1_human:+.4f}")

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")

    # Machine-readable output
    print("\n---")
    print(f"v2_cv_corr:     {final['cv_corr']:.6f}")
    print(f"v2_human_corr:  {final['oof_human_corr']:.6f}")
    print(f"v2_synth_corr:  {final['oof_synth_corr']:.6f}")
    print(f"linear_cv_corr: {linear['cv_corr']:.6f}")
    print(f"linear_human_corr: {linear.get('human_corr', 0):.6f}")
    if v1_cv > 0:
        print(f"v1_cv_corr:     {v1_cv:.6f}")
        print(f"v1_human_corr:    {v1_human:.6f}")


if __name__ == "__main__":
    main()
