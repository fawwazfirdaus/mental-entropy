#!/usr/bin/env python3
"""Ordinal regression experiment for MES scoring.

Instead of treating labels as continuous (Ridge regression), treat them as
ordinal categories where 1 < 2 < 3 < ... < 10. This respects the ordinal
structure and may handle compressed label distributions better.

Approaches tested:
1. Ordinal logistic regression (cumulative link model via mord)
2. Binary decomposition (K-1 binary classifiers: "is label > k?")
3. Soft-label Ridge (label smoothing + continuous regression)
4. Quantile-binned Ridge (bin labels into quantiles, predict bin center)
5. Weighted Ridge with label-uncertainty weighting (use label_std from consensus)

Baseline: Hybrid v6 Ridge (human_corr=0.683)

Usage:
    cd mental-entropy
    PYTHONPATH=src .venv/bin/python -u autoresearch-macos/train_ordinal.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prepare import (
    load_cached_data,
    load_cached_embeddings,
    load_label_stds,
    pearson_correlation,
)
from train_subscores_v4 import get_stratified_folds, compute_v4_objective
from train_embedding_regression import (
    oof_evaluate,
    make_sample_weights_fn,
    compute_feature_human_correlations,
)
from train_subscores import build_feature_matrix
from train_subscores_v3 import ALL_FEATURE_KEYS


# ---------------------------------------------------------------------------
# Approach 1: Binary decomposition ordinal regression
# ---------------------------------------------------------------------------

def oof_ordinal_binary(
    E: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    sources: np.ndarray,
    thresholds: list[float],
    alpha: float = 1.0,
    sample_weight_fn=None,
) -> dict:
    """Ordinal regression via K-1 binary Ridge classifiers.

    For each threshold k in thresholds, train a Ridge to predict P(label > k).
    Final prediction = 1 + sum of P(label > k) for each k.

    This naturally handles ordinal structure: a journal rated 7 must also
    be rated > 3, > 4, > 5, > 6.
    """
    n = len(labels)
    oof_preds = np.zeros(n, dtype=np.float64)
    fold_corrs = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        E_train, E_val = E[train_idx], E[val_idx]
        y_train = labels[train_idx]

        # Sample weights
        sw = None
        if sample_weight_fn is not None:
            sw = sample_weight_fn(sources[train_idx])

        # For each threshold, train binary Ridge
        probs = np.zeros((len(val_idx), len(thresholds)), dtype=np.float64)

        for t_idx, thresh in enumerate(thresholds):
            # Binary target: is label > threshold?
            y_binary = (y_train > thresh).astype(np.float64)

            # Skip if all same class
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                probs[:, t_idx] = y_binary[0]
                continue

            model = Ridge(alpha=alpha)
            model.fit(E_train, y_binary, sample_weight=sw)
            pred = model.predict(E_val)
            # Clip to [0, 1]
            probs[:, t_idx] = np.clip(pred, 0, 1)

        # Sum probabilities: E[Y] = 1 + sum P(Y > k) for k=1..K-1
        # (assuming thresholds are 1.5, 2.5, ..., 9.5)
        val_preds = 1.0 + probs.sum(axis=1)
        oof_preds[val_idx] = val_preds

        # Fold correlation on human entries
        human_val = sources[val_idx] == 0
        if human_val.sum() > 10:
            fold_corr = pearson_correlation(
                val_preds[human_val], labels[val_idx][human_val]
            )
            fold_corrs.append(fold_corr)

    # Overall metrics
    human_mask = sources == 0
    synth_mask = sources == 1
    cv_corr = float(pearson_correlation(oof_preds, labels))
    human_corr = float(pearson_correlation(oof_preds[human_mask], labels[human_mask]))
    synth_corr = float(pearson_correlation(oof_preds[synth_mask], labels[synth_mask]))
    fold_std = float(np.std(fold_corrs)) if fold_corrs else 0.0

    return {
        "cv_corr": cv_corr,
        "oof_human_corr": human_corr,
        "oof_synth_corr": synth_corr,
        "fold_human_corr_std": fold_std,
        "oof_preds": oof_preds,
    }


# ---------------------------------------------------------------------------
# Approach 2: Soft-label Ridge with Gaussian smoothing
# ---------------------------------------------------------------------------

def oof_soft_label_ridge(
    E: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    sources: np.ndarray,
    sigma: float = 0.5,
    alpha: float = 0.1,
    n_augment: int = 3,
    sample_weight_fn=None,
) -> dict:
    """Ridge with Gaussian noise augmentation on labels.

    Adds noise to labels during training to regularize and spread
    the effective label range. Tests if label smoothing helps with
    compressed distributions.
    """
    n = len(labels)
    oof_preds = np.zeros(n, dtype=np.float64)
    fold_corrs = []

    rng = np.random.RandomState(42)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        E_train, E_val = E[train_idx], E[val_idx]
        y_train = labels[train_idx]

        # Augment: replicate training data with noisy labels
        E_aug = np.tile(E_train, (n_augment, 1))
        y_aug = np.tile(y_train, n_augment)
        # Add Gaussian noise to labels (not to the first copy)
        noise = rng.normal(0, sigma, len(y_aug))
        noise[:len(y_train)] = 0  # Keep original labels clean
        y_aug = np.clip(y_aug + noise, 1.0, 10.0)

        sw = None
        if sample_weight_fn is not None:
            sw_base = sample_weight_fn(sources[train_idx])
            sw = np.tile(sw_base, n_augment)

        model = Ridge(alpha=alpha)
        model.fit(E_aug, y_aug, sample_weight=sw)
        val_preds = model.predict(E_val)
        oof_preds[val_idx] = val_preds

        human_val = sources[val_idx] == 0
        if human_val.sum() > 10:
            fold_corr = pearson_correlation(
                val_preds[human_val], labels[val_idx][human_val]
            )
            fold_corrs.append(fold_corr)

    human_mask = sources == 0
    synth_mask = sources == 1
    return {
        "cv_corr": float(pearson_correlation(oof_preds, labels)),
        "oof_human_corr": float(pearson_correlation(oof_preds[human_mask], labels[human_mask])),
        "oof_synth_corr": float(pearson_correlation(oof_preds[synth_mask], labels[synth_mask])),
        "fold_human_corr_std": float(np.std(fold_corrs)) if fold_corrs else 0.0,
        "oof_preds": oof_preds,
    }


# ---------------------------------------------------------------------------
# Approach 3: Uncertainty-weighted Ridge (use label_std from consensus)
# ---------------------------------------------------------------------------

def oof_uncertainty_weighted_ridge(
    E: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    sources: np.ndarray,
    label_stds: np.ndarray,
    alpha: float = 0.1,
    sample_weight_fn=None,
) -> dict:
    """Ridge with per-sample weights inversely proportional to label uncertainty.

    Entries where all 3 raters agreed (low std) get higher weight.
    Entries with high disagreement (high std) get lower weight.
    This focuses the model on the most reliable labels.
    """
    n = len(labels)
    oof_preds = np.zeros(n, dtype=np.float64)
    fold_corrs = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        E_train, E_val = E[train_idx], E[val_idx]
        y_train = labels[train_idx]
        stds_train = label_stds[train_idx]

        # Uncertainty weights: 1 / (std + epsilon)
        eps = 0.1
        uncertainty_weights = 1.0 / (stds_train + eps)
        # Normalize to mean=1
        uncertainty_weights = uncertainty_weights / uncertainty_weights.mean()

        # Combine with sample weights if provided
        if sample_weight_fn is not None:
            sw = sample_weight_fn(sources[train_idx])
            sw = sw * uncertainty_weights
        else:
            sw = uncertainty_weights

        model = Ridge(alpha=alpha)
        model.fit(E_train, y_train, sample_weight=sw)
        val_preds = model.predict(E_val)
        oof_preds[val_idx] = val_preds

        human_val = sources[val_idx] == 0
        if human_val.sum() > 10:
            fold_corr = pearson_correlation(
                val_preds[human_val], labels[val_idx][human_val]
            )
            fold_corrs.append(fold_corr)

    human_mask = sources == 0
    synth_mask = sources == 1
    return {
        "cv_corr": float(pearson_correlation(oof_preds, labels)),
        "oof_human_corr": float(pearson_correlation(oof_preds[human_mask], labels[human_mask])),
        "oof_synth_corr": float(pearson_correlation(oof_preds[synth_mask], labels[synth_mask])),
        "fold_human_corr_std": float(np.std(fold_corrs)) if fold_corrs else 0.0,
        "oof_preds": oof_preds,
    }


# ---------------------------------------------------------------------------
# Approach 4: Rank-based regression (predict rank instead of score)
# ---------------------------------------------------------------------------

def oof_rank_regression(
    E: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    sources: np.ndarray,
    alpha: float = 0.1,
    sample_weight_fn=None,
) -> dict:
    """Ridge regression on percentile ranks instead of raw labels.

    Transforms labels to their percentile rank [0, 1] within the training
    set, regresses on rank, then converts back via inverse transform.
    This handles non-uniform label distributions by equalizing the target.
    """
    from scipy.stats import rankdata

    n = len(labels)
    oof_preds = np.zeros(n, dtype=np.float64)
    fold_corrs = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        E_train, E_val = E[train_idx], E[val_idx]
        y_train = labels[train_idx]

        # Convert labels to percentile ranks [0, 1]
        ranks = rankdata(y_train) / len(y_train)

        sw = None
        if sample_weight_fn is not None:
            sw = sample_weight_fn(sources[train_idx])

        model = Ridge(alpha=alpha)
        model.fit(E_train, ranks, sample_weight=sw)
        rank_preds = model.predict(E_val)

        # Convert predicted ranks back to label scale via interpolation
        sorted_labels = np.sort(y_train)
        sorted_ranks = np.linspace(0, 1, len(sorted_labels))
        val_preds = np.interp(rank_preds, sorted_ranks, sorted_labels)
        oof_preds[val_idx] = val_preds

        human_val = sources[val_idx] == 0
        if human_val.sum() > 10:
            fold_corr = pearson_correlation(
                val_preds[human_val], labels[val_idx][human_val]
            )
            fold_corrs.append(fold_corr)

    human_mask = sources == 0
    synth_mask = sources == 1
    return {
        "cv_corr": float(pearson_correlation(oof_preds, labels)),
        "oof_human_corr": float(pearson_correlation(oof_preds[human_mask], labels[human_mask])),
        "oof_synth_corr": float(pearson_correlation(oof_preds[synth_mask], labels[synth_mask])),
        "fold_human_corr_std": float(np.std(fold_corrs)) if fold_corrs else 0.0,
        "oof_preds": oof_preds,
    }


# ---------------------------------------------------------------------------
# Approach 5: Huber-loss Ridge (robust to label outliers)
# ---------------------------------------------------------------------------

def oof_huber_ridge(
    E: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    sources: np.ndarray,
    alpha: float = 0.1,
    epsilon: float = 1.35,
    sample_weight_fn=None,
) -> dict:
    """Ridge with Huber loss — less sensitive to label outliers.

    Huber loss is quadratic for small errors and linear for large errors,
    making the model more robust to mislabeled entries.
    """
    from sklearn.linear_model import HuberRegressor

    n = len(labels)
    oof_preds = np.zeros(n, dtype=np.float64)
    fold_corrs = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        E_train, E_val = E[train_idx], E[val_idx]
        y_train = labels[train_idx]

        sw = None
        if sample_weight_fn is not None:
            sw = sample_weight_fn(sources[train_idx])

        model = HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=500)
        model.fit(E_train, y_train, sample_weight=sw)
        val_preds = model.predict(E_val)
        oof_preds[val_idx] = val_preds

        human_val = sources[val_idx] == 0
        if human_val.sum() > 10:
            fold_corr = pearson_correlation(
                val_preds[human_val], labels[val_idx][human_val]
            )
            fold_corrs.append(fold_corr)

    human_mask = sources == 0
    synth_mask = sources == 1
    return {
        "cv_corr": float(pearson_correlation(oof_preds, labels)),
        "oof_human_corr": float(pearson_correlation(oof_preds[human_mask], labels[human_mask])),
        "oof_synth_corr": float(pearson_correlation(oof_preds[synth_mask], labels[synth_mask])),
        "fold_human_corr_std": float(np.std(fold_corrs)) if fold_corrs else 0.0,
        "oof_preds": oof_preds,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("ORDINAL REGRESSION EXPERIMENT")
    print("=" * 72)

    # Load data
    print("\n--- Loading data ---")
    features, labels, feature_names, sources = load_cached_data()
    E = load_cached_embeddings()
    label_stds = load_label_stds()
    n = len(labels)
    n_human = int((sources == 0).sum())
    n_synth = int((sources == 1).sum())
    print(f"Entries: {n} ({n_human} human, {n_synth} synthetic)")
    print(f"Embedding dim: {E.shape[1]}")
    print(f"Label stds available: {label_stds is not None}")

    # Build feature matrix for hybrid
    X_feat = build_feature_matrix(features, ALL_FEATURE_KEYS)
    top_k = 10
    human_mask = sources == 0
    corrs = compute_feature_human_correlations(features, labels, human_mask, ALL_FEATURE_KEYS)
    top_feat_names = [name for name, _ in corrs[:top_k]]
    top_feat_idx = [ALL_FEATURE_KEYS.index(f) for f in top_feat_names]
    X_sel = X_feat[:, top_feat_idx]
    E_hybrid = np.hstack([E, X_sel])
    print(f"Hybrid input: {E_hybrid.shape[1]}-dim (1024 + {top_k} features)")

    # Folds
    folds = get_stratified_folds(sources)
    print(f"Folds: {len(folds)}")

    # ---------------------------------------------------------------------------
    # Baseline: standard Ridge
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("BASELINE: Standard Ridge (current v6)")
    print("=" * 72)

    sw_fn = make_sample_weights_fn(0.3)
    baseline = oof_evaluate(Ridge, {"alpha": 0.1}, E_hybrid, labels, folds, sources, sw_fn)
    print(f"  human_corr={baseline['oof_human_corr']:.4f}  cv={baseline['cv_corr']:.4f}  fold_std={baseline['fold_human_corr_std']:.4f}")

    results: list[dict] = []
    results.append({
        "approach": "baseline_ridge",
        "config": "alpha=0.1, human_heavy",
        "human_corr": baseline["oof_human_corr"],
        "cv_corr": baseline["cv_corr"],
        "synth_corr": baseline["oof_synth_corr"],
        "fold_std": baseline["fold_human_corr_std"],
    })

    # ---------------------------------------------------------------------------
    # Approach 1: Binary decomposition ordinal regression
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("APPROACH 1: Binary Decomposition Ordinal Regression")
    print("=" * 72)

    thresholds = [k + 0.5 for k in range(1, 10)]  # 1.5, 2.5, ..., 9.5
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        for wname, wfn in [("equal", None), ("human_heavy", make_sample_weights_fn(0.3))]:
            result = oof_ordinal_binary(E_hybrid, labels, folds, sources, thresholds, alpha=alpha, sample_weight_fn=wfn)
            v4 = compute_v4_objective(result)
            print(f"  α={alpha:<6} w={wname:<12} human={result['oof_human_corr']:.4f}  cv={result['cv_corr']:.4f}  v4={v4:.4f}")
            results.append({
                "approach": "binary_ordinal",
                "config": f"alpha={alpha}, {wname}",
                "human_corr": result["oof_human_corr"],
                "cv_corr": result["cv_corr"],
                "synth_corr": result["oof_synth_corr"],
                "fold_std": result["fold_human_corr_std"],
                "v4_score": v4,
            })

    # ---------------------------------------------------------------------------
    # Approach 2: Soft-label Ridge
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("APPROACH 2: Soft-Label Ridge (label noise augmentation)")
    print("=" * 72)

    for sigma in [0.25, 0.5, 1.0, 1.5]:
        for n_aug in [2, 5]:
            for alpha in [0.1, 1.0]:
                sw_fn = make_sample_weights_fn(0.3)
                result = oof_soft_label_ridge(E_hybrid, labels, folds, sources, sigma=sigma, alpha=alpha, n_augment=n_aug, sample_weight_fn=sw_fn)
                v4 = compute_v4_objective(result)
                print(f"  σ={sigma:<4} n_aug={n_aug} α={alpha:<4} human={result['oof_human_corr']:.4f}  cv={result['cv_corr']:.4f}  v4={v4:.4f}")
                results.append({
                    "approach": "soft_label_ridge",
                    "config": f"sigma={sigma}, n_aug={n_aug}, alpha={alpha}",
                    "human_corr": result["oof_human_corr"],
                    "cv_corr": result["cv_corr"],
                    "synth_corr": result["oof_synth_corr"],
                    "fold_std": result["fold_human_corr_std"],
                    "v4_score": v4,
                })

    # ---------------------------------------------------------------------------
    # Approach 3: Uncertainty-weighted Ridge
    # ---------------------------------------------------------------------------
    if label_stds is not None:
        print("\n" + "=" * 72)
        print("APPROACH 3: Uncertainty-Weighted Ridge")
        print("=" * 72)

        print(f"  Label std stats: mean={label_stds.mean():.2f}, min={label_stds.min():.2f}, max={label_stds.max():.2f}")
        for alpha in [0.01, 0.1, 1.0]:
            for wname, wfn in [("equal", None), ("human_heavy", make_sample_weights_fn(0.3))]:
                result = oof_uncertainty_weighted_ridge(E_hybrid, labels, folds, sources, label_stds, alpha=alpha, sample_weight_fn=wfn)
                v4 = compute_v4_objective(result)
                print(f"  α={alpha:<6} w={wname:<12} human={result['oof_human_corr']:.4f}  cv={result['cv_corr']:.4f}  v4={v4:.4f}")
                results.append({
                    "approach": "uncertainty_weighted",
                    "config": f"alpha={alpha}, {wname}",
                    "human_corr": result["oof_human_corr"],
                    "cv_corr": result["cv_corr"],
                    "synth_corr": result["oof_synth_corr"],
                    "fold_std": result["fold_human_corr_std"],
                    "v4_score": v4,
                })

    # ---------------------------------------------------------------------------
    # Approach 4: Rank-based regression
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("APPROACH 4: Rank-Based Regression")
    print("=" * 72)

    for alpha in [0.01, 0.1, 1.0, 10.0]:
        for wname, wfn in [("equal", None), ("human_heavy", make_sample_weights_fn(0.3))]:
            result = oof_rank_regression(E_hybrid, labels, folds, sources, alpha=alpha, sample_weight_fn=wfn)
            v4 = compute_v4_objective(result)
            print(f"  α={alpha:<6} w={wname:<12} human={result['oof_human_corr']:.4f}  cv={result['cv_corr']:.4f}  v4={v4:.4f}")
            results.append({
                "approach": "rank_regression",
                "config": f"alpha={alpha}, {wname}",
                "human_corr": result["oof_human_corr"],
                "cv_corr": result["cv_corr"],
                "synth_corr": result["oof_synth_corr"],
                "fold_std": result["fold_human_corr_std"],
                "v4_score": v4,
            })

    # ---------------------------------------------------------------------------
    # Approach 5: Huber-loss Ridge
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("APPROACH 5: Huber-Loss Ridge (robust to outliers)")
    print("=" * 72)

    for alpha in [0.001, 0.01, 0.1]:
        for epsilon in [1.1, 1.35, 2.0]:
            sw_fn = make_sample_weights_fn(0.3)
            try:
                result = oof_huber_ridge(E_hybrid, labels, folds, sources, alpha=alpha, epsilon=epsilon, sample_weight_fn=sw_fn)
                v4 = compute_v4_objective(result)
                print(f"  α={alpha:<6} ε={epsilon:<4} human={result['oof_human_corr']:.4f}  cv={result['cv_corr']:.4f}  v4={v4:.4f}")
                results.append({
                    "approach": "huber_ridge",
                    "config": f"alpha={alpha}, epsilon={epsilon}",
                    "human_corr": result["oof_human_corr"],
                    "cv_corr": result["cv_corr"],
                    "synth_corr": result["oof_synth_corr"],
                    "fold_std": result["fold_human_corr_std"],
                    "v4_score": v4,
                })
            except Exception as e:
                print(f"  α={alpha:<6} ε={epsilon:<4} FAILED: {e}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY — Best config per approach")
    print("=" * 72)

    approaches = set(r["approach"] for r in results)
    print(f"\n  {'Approach':<25} {'Config':<35} {'human_corr':>11} {'cv_corr':>9} {'v4':>7}")
    print("  " + "-" * 90)

    summary = []
    for approach in sorted(approaches):
        app_results = [r for r in results if r["approach"] == approach]
        best = max(app_results, key=lambda x: x["human_corr"])
        delta = best["human_corr"] - baseline["oof_human_corr"]
        sign = "+" if delta >= 0 else ""
        print(f"  {approach:<25} {best['config']:<35} {best['human_corr']:>11.4f} {best['cv_corr']:>9.4f} {best.get('v4_score', 0):>7.4f}  ({sign}{delta:.4f})")
        summary.append(best)

    # Save results
    out_path = Path(__file__).parent / "ordinal_regression_results.json"
    out_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_entries": n,
        "n_human": n_human,
        "n_synth": n_synth,
        "baseline_human_corr": baseline["oof_human_corr"],
        "baseline_cv_corr": baseline["cv_corr"],
        "n_configs": len(results),
        "all_results": [{k: v for k, v in r.items() if k != "oof_preds"} for r in results],
        "best_per_approach": [{k: v for k, v in r.items() if k != "oof_preds"} for r in summary],
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path.name}")


if __name__ == "__main__":
    main()
