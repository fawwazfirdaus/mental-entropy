"""
MES Diagnostic: Model vs Data Bottleneck.

Trains XGBoost and Ridge regression on cached features to determine
whether the MES scoring bottleneck is our linear model or our data.

Usage: cd autoresearch-macos && uv run python diagnose.py
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from prepare import (
    load_cached_data,
    pearson_correlation,
    compute_mes_scores,
    SOURCE_HUMAN,
    SOURCE_SYNTHETIC,
    N_FOLDS,
)

# Current production weights (from train.py)
LINEAR_WEIGHTS = [
    ("ce_adj_p75",                0.17, "inv"),
    ("ne_fragment_sentence_rate", 0.14, "dir"),
    ("ce_inter_block_break_rate", 0.11, "dir"),
    ("bc_belief_sentence_count",  0.11, "dir_norm", 18.0),
    ("ce_n_blocks",               0.10, "dir_norm", 20.0),
    ("cle_length_cv",             0.08, "dir"),
    ("ce_skip_mean",              0.07, "inv"),
    ("ne_start_to_centroid",      0.06, "inv"),
    ("se_dominant_cluster_frac",  0.05, "inv"),
    ("ne_arc_linearity",          0.03, "inv"),
    ("ne_end_to_centroid",        0.03, "inv"),
    ("cle_hedge_rate",            0.03, "dir"),
    ("ne_start_end_sim",          0.02, "inv"),
]

LINEAR_FEATURE_NAMES = [w[0] for w in LINEAR_WEIGHTS]


def build_feature_matrix(
    features: list[dict[str, float]],
    feature_names: list[str],
) -> np.ndarray:
    """Convert list of feature dicts to a 2D numpy array."""
    n = len(features)
    m = len(feature_names)
    X = np.zeros((n, m), dtype=np.float64)
    for i, feat_dict in enumerate(features):
        for j, name in enumerate(feature_names):
            X[i, j] = feat_dict.get(name, 0.0)
    return X


def build_sample_weights(sources: np.ndarray) -> np.ndarray:
    """Build sample weights so human and synthetic sources contribute equally."""
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())
    w = np.ones(len(sources), dtype=np.float64)
    if n_human > 0 and n_synth > 0:
        w[sources == SOURCE_HUMAN] = n_synth / n_human
    return w


def cv_evaluate(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    n_folds: int = N_FOLDS,
) -> dict[str, float]:
    """5-fold CV with the same split as prepare.py (seed 42)."""
    n = len(X)
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)

    fold_size = n // n_folds
    fold_corrs = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        w_train = sample_weights[train_idx]
        w_val = sample_weights[val_idx]

        model = model_fn()
        model.fit(X_train, y_train, sample_weight=w_train)
        preds = model.predict(X_val)

        fold_corr = pearson_correlation(preds, y_val, weights=w_val)
        fold_corrs.append(fold_corr)

    val_model = model_fn()
    val_model.fit(X, y, sample_weight=sample_weights)
    val_preds = val_model.predict(X)
    val_corr = pearson_correlation(val_preds, y, weights=sample_weights)

    return {
        "cv_corr": float(np.mean(fold_corrs)),
        "cv_std": float(np.std(fold_corrs)),
        "val_corr": val_corr,
        "model": val_model,
    }


def main() -> None:
    features, labels, feature_names, sources = load_cached_data()
    sample_weights = build_sample_weights(sources)

    X_all = build_feature_matrix(features, feature_names)
    linear_idx = [feature_names.index(n) for n in LINEAR_FEATURE_NAMES]
    X_linear = X_all[:, linear_idx]

    print("=" * 60)
    print("MES DIAGNOSTIC: Model vs Data Bottleneck")
    print("=" * 60)
    print(f"\nDataset: {len(features)} entries ({int((sources == SOURCE_HUMAN).sum())} human + {int((sources == SOURCE_SYNTHETIC).sum())} synthetic)")
    print(f"Features: {len(feature_names)} total, {len(LINEAR_FEATURE_NAMES)} used by linear model")
    print()

    # 1. Linear baseline (current production model)
    scores = compute_mes_scores(LINEAR_WEIGHTS, features)
    linear_cv_corrs = []
    n = len(features)
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    fold_size = n // N_FOLDS
    for fold in range(N_FOLDS):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < N_FOLDS - 1 else n
        val_idx = indices[val_start:val_end]
        w_val = sample_weights[val_idx]
        fold_corr = pearson_correlation(scores[val_idx], labels[val_idx], weights=w_val)
        linear_cv_corrs.append(fold_corr)
    linear_cv = float(np.mean(linear_cv_corrs))
    linear_std = float(np.std(linear_cv_corrs))
    linear_val = pearson_correlation(scores, labels, weights=sample_weights)

    # 2. XGBoost on all 77 features
    xgb_all = cv_evaluate(
        lambda: XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        ),
        X_all, labels, sample_weights,
    )

    # 3. XGBoost on same 10 features as linear model
    xgb_10 = cv_evaluate(
        lambda: XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        ),
        X_linear, labels, sample_weights,
    )

    # 4. Ridge regression on all 77 features
    ridge_all = cv_evaluate(
        lambda: Ridge(alpha=1.0),
        X_all, labels, sample_weights,
    )

    # Print results
    print("-" * 60)
    print(f"{'Model':<35s}  {'cv_corr':>8s}  {'±std':>6s}  {'val_corr':>8s}")
    print("-" * 60)
    print(f"{'1. Linear weighted (10 feat)':<35s}  {linear_cv:>8.4f}  {linear_std:>5.3f}  {linear_val:>8.4f}")
    print(f"{'2. XGBoost (all 77 feat)':<35s}  {xgb_all['cv_corr']:>8.4f}  {xgb_all['cv_std']:>5.3f}  {xgb_all['val_corr']:>8.4f}")
    print(f"{'3. XGBoost (same 10 feat)':<35s}  {xgb_10['cv_corr']:>8.4f}  {xgb_10['cv_std']:>5.3f}  {xgb_10['val_corr']:>8.4f}")
    print(f"{'4. Ridge regression (all 77 feat)':<35s}  {ridge_all['cv_corr']:>8.4f}  {ridge_all['cv_std']:>5.3f}  {ridge_all['val_corr']:>8.4f}")
    print("-" * 60)

    # Feature importance from XGBoost (all 77)
    importances = xgb_all["model"].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\nXGBoost feature importance (top 15 by gain):")
    for rank, idx in enumerate(sorted_idx[:15], 1):
        name = feature_names[idx]
        imp = importances[idx]
        in_linear = "  ✓ (in linear)" if name in LINEAR_FEATURE_NAMES else ""
        print(f"  {rank:2d}. {name:<35s}  importance={imp:.4f}{in_linear}")

    # Interpretation
    gap = xgb_all["cv_corr"] - linear_cv
    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print(f"{'=' * 60}")
    print(f"\nXGBoost advantage over linear: {gap:+.4f} ({gap/linear_cv*100:+.1f}%)")

    if gap > 0.05:
        print("\n→ MODEL IS THE BOTTLENECK")
        print("  XGBoost significantly outperforms the linear model.")
        print("  Next step: Build subscore models (GAM/XGBoost per feature group)")
        print("  to capture non-linear relationships the linear formula misses.")
    elif gap > 0.02:
        print("\n→ MODERATE HEADROOM")
        print("  Some signal is being missed by the linear model.")
        print("  Consider: non-linear subscore models OR more data (or both).")
    else:
        print("\n→ DATA IS THE BOTTLENECK")
        print("  XGBoost barely improves over the linear model.")
        print("  The linear formula is already near-optimal for this data.")
        print("  Next step: Get more/better labeled journals to improve all models.")

    # Ridge vs XGBoost comparison
    ridge_gap = xgb_all["cv_corr"] - ridge_all["cv_corr"]
    if ridge_gap > 0.02:
        print(f"\n  Non-linearity matters: XGBoost beats Ridge by {ridge_gap:+.4f}")
    else:
        print(f"\n  Non-linearity doesn't help much: XGBoost vs Ridge gap is only {ridge_gap:+.4f}")
        print("  More features help, but interactions between them don't add much.")


if __name__ == "__main__":
    main()
