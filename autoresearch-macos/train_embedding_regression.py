"""
MES Embedding Regression Experiment — Skip the Feature Bottleneck.

Tests 5 architectures that predict MES directly from 1024-dim document
embeddings, bypassing the 82 hand-crafted features entirely.

Architectures:
A. Direct Ridge on embeddings -> overall_entropy
B. Direct Lasso (sparse embedding selection)
C. ElasticNet (blend Ridge + Lasso)
D. Dimension-aligned embedding Ridge (per-dimension -> combiner)
E. Hybrid (embedding + top-k features by human label correlation)

Usage:
    cd autoresearch-macos && PYTHONPATH=../src ../.venv/bin/python -u train_embedding_regression.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from prepare import (
    DIMENSION_NAMES,
    N_FOLDS,
    SOURCE_HUMAN,
    SOURCE_SYNTHETIC,
    load_cached_data,
    load_cached_embeddings,
    load_dimension_labels,
    pearson_correlation,
)
from train_subscores_v4 import get_stratified_folds, compute_v4_objective
from train_subscores import build_feature_matrix
from train_subscores_v3 import ALL_FEATURE_KEYS

W = 72  # print width
ARTIFACTS_DIR = Path(__file__).parent.parent / "src" / "mental_entropy" / "models" / "_artifacts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def oof_evaluate(
    model_class,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    sources: np.ndarray,
    sample_weights_fn=None,
) -> dict:
    """Out-of-fold evaluation for a single sklearn model.

    Args:
        model_class: sklearn model class (Ridge, Lasso, ElasticNet).
        model_kwargs: kwargs for model constructor.
        X: Feature matrix (n, d).
        y: Labels (n,).
        folds: List of (train_idx, val_idx) from stratified split.
        sources: Source array (0=human, 1=synth).
        sample_weights_fn: Optional callable(sources[train]) -> weights array.

    Returns:
        Dict with oof predictions and metrics.
    """
    n = len(y)
    oof_preds = np.full(n, np.nan)
    fold_human_corrs = []

    for train_idx, val_idx in folds:
        model = model_class(**model_kwargs)

        sw = None
        if sample_weights_fn is not None:
            sw = sample_weights_fn(sources[train_idx])

        model.fit(X[train_idx], y[train_idx], sample_weight=sw)
        oof_preds[val_idx] = model.predict(X[val_idx])

        # Per-fold human correlation
        val_human = sources[val_idx] == SOURCE_HUMAN
        if val_human.sum() > 2:
            fold_human_corrs.append(
                pearson_correlation(oof_preds[val_idx][val_human], y[val_idx][val_human])
            )

    # Overall metrics
    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC

    cv_corr = pearson_correlation(oof_preds, y)
    oof_human_corr = pearson_correlation(oof_preds[human_mask], y[human_mask]) if human_mask.sum() > 2 else 0.0
    oof_synth_corr = pearson_correlation(oof_preds[synth_mask], y[synth_mask]) if synth_mask.sum() > 2 else 0.0
    fold_human_corr_std = float(np.std(fold_human_corrs)) if fold_human_corrs else 0.5

    return {
        "cv_corr": cv_corr,
        "oof_human_corr": oof_human_corr,
        "oof_synth_corr": oof_synth_corr,
        "fold_human_corr_std": fold_human_corr_std,
        "oof_preds": oof_preds,
    }


def oof_evaluate_dimension_aligned(
    dim_alpha: float,
    combiner_alpha: float,
    E: np.ndarray,
    overall_labels: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray]],
    sources: np.ndarray,
    sample_weights_fn=None,
) -> dict:
    """Out-of-fold evaluation for dimension-aligned embedding Ridge.

    Stage 1: Per-dimension Ridge(dim_alpha) on embeddings -> dimension subscores
    Stage 2: Ridge(combiner_alpha) on OOF subscores -> overall_entropy
    """
    n = len(overall_labels)
    n_dims = len(DIMENSION_NAMES)

    # Stage 1: Per-dimension OOF subscores
    dim_oof = np.zeros((n, n_dims))
    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        d_labels = dim_labels[dim_name]
        for train_idx, val_idx in folds:
            model = Ridge(alpha=dim_alpha)
            sw = sample_weights_fn(sources[train_idx]) if sample_weights_fn else None
            model.fit(E[train_idx], d_labels[train_idx], sample_weight=sw)
            dim_oof[val_idx, d_idx] = model.predict(E[val_idx])

    # Stage 2: Combiner Ridge on OOF subscores -> overall_entropy
    oof_preds = np.full(n, np.nan)
    fold_human_corrs = []

    for train_idx, val_idx in folds:
        combiner = Ridge(alpha=combiner_alpha)
        combiner.fit(dim_oof[train_idx], overall_labels[train_idx])
        oof_preds[val_idx] = combiner.predict(dim_oof[val_idx])

        val_human = sources[val_idx] == SOURCE_HUMAN
        if val_human.sum() > 2:
            fold_human_corrs.append(
                pearson_correlation(oof_preds[val_idx][val_human], overall_labels[val_idx][val_human])
            )

    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC

    return {
        "cv_corr": pearson_correlation(oof_preds, overall_labels),
        "oof_human_corr": pearson_correlation(oof_preds[human_mask], overall_labels[human_mask]) if human_mask.sum() > 2 else 0.0,
        "oof_synth_corr": pearson_correlation(oof_preds[synth_mask], overall_labels[synth_mask]) if synth_mask.sum() > 2 else 0.0,
        "fold_human_corr_std": float(np.std(fold_human_corrs)) if fold_human_corrs else 0.5,
        "oof_preds": oof_preds,
    }


def make_sample_weights_fn(synth_weight: float = 1.0):
    """Create a sample weight function for training."""
    def fn(train_sources: np.ndarray) -> np.ndarray:
        w = np.ones(len(train_sources), dtype=np.float64)
        n_h = int((train_sources == SOURCE_HUMAN).sum())
        n_s = int((train_sources == SOURCE_SYNTHETIC).sum())
        if n_h > 0 and n_s > 0:
            # Scale synthetic entries relative to human
            w[train_sources == SOURCE_SYNTHETIC] = synth_weight * n_h / n_s
        return w
    return fn


def compute_feature_human_correlations(
    features: list[dict[str, float]],
    labels: np.ndarray,
    sources: np.ndarray,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """Compute absolute correlation of each feature with labels on human entries."""
    human_mask = sources == SOURCE_HUMAN
    human_labels = labels[human_mask]
    corrs = []
    for fname in feature_names:
        vals = np.array([f.get(fname, 0.0) for f in features], dtype=np.float64)
        r = abs(pearson_correlation(vals[human_mask], human_labels))
        corrs.append((fname, r))
    corrs.sort(key=lambda x: -x[1])
    return corrs


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def save_hybrid_artifacts(
    E: np.ndarray,
    labels: np.ndarray,
    features: list[dict[str, float]],
    sources: np.ndarray,
    feat_corrs: list[tuple[str, float]],
    k: int,
    alpha: float,
    metrics: dict,
) -> None:
    """Train final hybrid model on full data and save artifacts.

    Args:
        E: Embedding matrix (n, 1024).
        labels: Overall entropy labels (n,).
        features: Feature dicts for all entries.
        sources: Source array.
        feat_corrs: Feature-label correlations sorted by |corr|.
        k: Number of top features to include.
        alpha: Ridge alpha.
        metrics: Dict with cv_corr, oof_human_corr, etc.
    """
    from train_subscores import build_feature_matrix
    from train_subscores_v3 import ALL_FEATURE_KEYS

    print("\n" + "=" * W)
    print("SAVING HYBRID v6 ARTIFACTS")
    print("=" * W)

    # Select top-k feature names
    top_k_names = [fname for fname, _ in feat_corrs[:k]]
    top_k_idx = [ALL_FEATURE_KEYS.index(f) for f in top_k_names if f in ALL_FEATURE_KEYS]

    # Build full-data input matrix
    X_feat = build_feature_matrix(features, ALL_FEATURE_KEYS)
    X_sel = X_feat[:, top_k_idx]
    X_hybrid = np.hstack([E, X_sel])
    print(f"  Input dim: {X_hybrid.shape[1]} (1024 emb + {len(top_k_names)} features)")
    print(f"  Features: {top_k_names}")

    # Fit Ridge on full data
    model = Ridge(alpha=alpha)
    model.fit(X_hybrid, labels)
    print(f"  Ridge trained: coef shape={model.coef_.shape}, intercept={model.intercept_:.6f}")

    # Save embedding_model.json
    embedding_model = {
        "coef": [float(c) for c in model.coef_],
        "intercept": float(model.intercept_),
        "embedding_dim": 1024,
        "feature_names": top_k_names,
        "alpha": alpha,
        "input_dim": int(X_hybrid.shape[1]),
    }
    emb_path = ARTIFACTS_DIR / "embedding_model.json"
    with open(emb_path, "w") as f:
        json.dump(embedding_model, f, indent=2)
    print(f"  Saved: {emb_path}")

    # Update combiner.json
    combiner_path = ARTIFACTS_DIR / "combiner.json"
    if combiner_path.exists():
        with open(combiner_path) as f:
            combiner = json.load(f)
    else:
        combiner = {}
    combiner["architecture"] = "hybrid_v6"
    with open(combiner_path, "w") as f:
        json.dump(combiner, f, indent=2)
    print(f"  Updated: {combiner_path} (architecture=hybrid_v6)")

    # Update manifest.json
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())
    manifest = {
        "architecture": "hybrid_v6",
        "training_script": "train_embedding_regression.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_entries": int(len(labels)),
        "n_human": n_human,
        "n_synthetic": n_synth,
        "embedding_dim": 1024,
        "n_features": len(top_k_names),
        "feature_names": top_k_names,
        "alpha": alpha,
        "k": k,
        "metrics": {
            "cv_corr": float(metrics.get("cv_corr", 0)),
            "oof_human_corr": float(metrics.get("oof_human_corr", 0)),
            "oof_synth_corr": float(metrics.get("oof_synth_corr", 0)),
            "fold_human_corr_std": float(metrics.get("fold_human_corr_std", 0)),
            "v4_score": float(metrics.get("v4_score", 0)),
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Updated: {manifest_path}")
    print(f"\n  Done! Deploy with hybrid_v6 architecture.")


def main() -> None:
    print("=" * W)
    print("MES Embedding Regression Experiment")
    print("=" * W)

    # Load data
    print("\nLoading cached data...")
    features, labels, feature_names, sources = load_cached_data()
    E = load_cached_embeddings()
    dim_labels = load_dimension_labels()

    n = len(features)
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())
    emb_dim = E.shape[1]

    print(f"  Entries: {n} ({n_human} human, {n_synth} synthetic)")
    print(f"  Embedding dim: {emb_dim}")
    print(f"  Features: {len(feature_names)}")

    # Verify alignment
    assert E.shape[0] == n, f"Embedding count {E.shape[0]} != entry count {n}"

    # Folds
    folds = get_stratified_folds(sources)
    print(f"\n  Folds: {len(folds)} (stratified)")
    human_mask = sources == SOURCE_HUMAN
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        n_h = int(human_mask[val_idx].sum())
        n_s = len(val_idx) - n_h
        print(f"    Fold {fold_idx}: {len(val_idx)} val ({n_h} human, {n_s} synthetic)")

    # Sample weight strategies
    sw_options = [
        ("equal", None),
        ("balanced", make_sample_weights_fn(1.0)),
        ("human_heavy", make_sample_weights_fn(0.3)),
    ]

    all_results: list[dict] = []
    t0 = time.time()

    # ===================================================================
    # A. Direct Ridge on embeddings
    # ===================================================================
    print("\n" + "=" * W)
    print("A. Direct Ridge on Embeddings")
    print("=" * W)

    ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    n_configs = len(ridge_alphas) * len(sw_options)
    i = 0
    best_a = {"score": -1.0}

    for alpha, (sw_name, sw_fn) in itertools.product(ridge_alphas, sw_options):
        i += 1
        result = oof_evaluate(
            Ridge, {"alpha": alpha}, E, labels, folds, sources, sw_fn
        )
        result["v4_score"] = compute_v4_objective(result)
        result["arch"] = "ridge"
        result["alpha"] = alpha
        result["sw"] = sw_name
        all_results.append(result)

        if result["v4_score"] > best_a["score"]:
            best_a = {"score": result["v4_score"], **result}

        if i % 7 == 0 or i == n_configs:
            print(f"  [{i:>3d}/{n_configs}] best: human={best_a.get('oof_human_corr', 0):.4f} "
                  f"cv={best_a.get('cv_corr', 0):.4f} obj={best_a['score']:.4f}")

    print(f"\n  Best Ridge: alpha={best_a.get('alpha')} sw={best_a.get('sw')}")
    print(f"    human_corr={best_a.get('oof_human_corr', 0):.4f} "
          f"cv_corr={best_a.get('cv_corr', 0):.4f} "
          f"synth_corr={best_a.get('oof_synth_corr', 0):.4f}")

    # ===================================================================
    # B. Direct Lasso (sparse embedding selection)
    # ===================================================================
    print("\n" + "=" * W)
    print("B. Direct Lasso on Embeddings")
    print("=" * W)

    lasso_alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]
    n_configs = len(lasso_alphas) * len(sw_options)
    i = 0
    best_b = {"score": -1.0}

    for alpha, (sw_name, sw_fn) in itertools.product(lasso_alphas, sw_options):
        i += 1
        result = oof_evaluate(
            Lasso, {"alpha": alpha, "max_iter": 5000}, E, labels, folds, sources, sw_fn
        )
        result["v4_score"] = compute_v4_objective(result)
        result["arch"] = "lasso"
        result["alpha"] = alpha
        result["sw"] = sw_name

        # Count nonzero dims from a full-data fit for reporting
        lasso_full = Lasso(alpha=alpha, max_iter=5000)
        lasso_full.fit(E, labels)
        result["n_nonzero"] = int(np.sum(np.abs(lasso_full.coef_) > 1e-10))

        all_results.append(result)

        if result["v4_score"] > best_b["score"]:
            best_b = {"score": result["v4_score"], **result}

        if i % 5 == 0 or i == n_configs:
            print(f"  [{i:>3d}/{n_configs}] best: human={best_b.get('oof_human_corr', 0):.4f} "
                  f"cv={best_b.get('cv_corr', 0):.4f} "
                  f"nonzero={best_b.get('n_nonzero', '?')}")

    print(f"\n  Best Lasso: alpha={best_b.get('alpha')} sw={best_b.get('sw')} "
          f"nonzero={best_b.get('n_nonzero')}")
    print(f"    human_corr={best_b.get('oof_human_corr', 0):.4f} "
          f"cv_corr={best_b.get('cv_corr', 0):.4f}")

    # ===================================================================
    # C. ElasticNet
    # ===================================================================
    print("\n" + "=" * W)
    print("C. ElasticNet on Embeddings")
    print("=" * W)

    en_alphas = [0.001, 0.01, 0.1, 1.0]
    en_l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_configs = len(en_alphas) * len(en_l1_ratios) * len(sw_options)
    i = 0
    best_c = {"score": -1.0}

    for alpha, l1_ratio, (sw_name, sw_fn) in itertools.product(en_alphas, en_l1_ratios, sw_options):
        i += 1
        result = oof_evaluate(
            ElasticNet, {"alpha": alpha, "l1_ratio": l1_ratio, "max_iter": 5000},
            E, labels, folds, sources, sw_fn,
        )
        result["v4_score"] = compute_v4_objective(result)
        result["arch"] = "elasticnet"
        result["alpha"] = alpha
        result["l1_ratio"] = l1_ratio
        result["sw"] = sw_name
        all_results.append(result)

        if result["v4_score"] > best_c["score"]:
            best_c = {"score": result["v4_score"], **result}

        if i % 20 == 0 or i == n_configs:
            print(f"  [{i:>3d}/{n_configs}] best: human={best_c.get('oof_human_corr', 0):.4f} "
                  f"cv={best_c.get('cv_corr', 0):.4f} obj={best_c['score']:.4f}")

    print(f"\n  Best ElasticNet: alpha={best_c.get('alpha')} l1={best_c.get('l1_ratio')} "
          f"sw={best_c.get('sw')}")
    print(f"    human_corr={best_c.get('oof_human_corr', 0):.4f} "
          f"cv_corr={best_c.get('cv_corr', 0):.4f}")

    # ===================================================================
    # D. Dimension-aligned embedding Ridge
    # ===================================================================
    print("\n" + "=" * W)
    print("D. Dimension-Aligned Embedding Ridge")
    print("=" * W)

    dim_alphas = [0.1, 1.0, 10.0, 100.0]
    comb_alphas = [0.1, 1.0, 10.0]
    n_configs = len(dim_alphas) * len(comb_alphas) * len(sw_options)
    i = 0
    best_d = {"score": -1.0}

    for d_alpha, c_alpha, (sw_name, sw_fn) in itertools.product(dim_alphas, comb_alphas, sw_options):
        i += 1
        result = oof_evaluate_dimension_aligned(
            d_alpha, c_alpha, E, labels, dim_labels, folds, sources, sw_fn,
        )
        result["v4_score"] = compute_v4_objective(result)
        result["arch"] = "dim_ridge"
        result["dim_alpha"] = d_alpha
        result["combiner_alpha"] = c_alpha
        result["sw"] = sw_name
        all_results.append(result)

        if result["v4_score"] > best_d["score"]:
            best_d = {"score": result["v4_score"], **result}

        if i % 12 == 0 or i == n_configs:
            print(f"  [{i:>3d}/{n_configs}] best: human={best_d.get('oof_human_corr', 0):.4f} "
                  f"cv={best_d.get('cv_corr', 0):.4f} obj={best_d['score']:.4f}")

    print(f"\n  Best Dim Ridge: d_alpha={best_d.get('dim_alpha')} c_alpha={best_d.get('combiner_alpha')} "
          f"sw={best_d.get('sw')}")
    print(f"    human_corr={best_d.get('oof_human_corr', 0):.4f} "
          f"cv_corr={best_d.get('cv_corr', 0):.4f}")

    # ===================================================================
    # E. Hybrid (embedding + top-k features)
    # ===================================================================
    print("\n" + "=" * W)
    print("E. Hybrid (Embedding + Top-k Features)")
    print("=" * W)

    # Compute feature-label correlations on human entries for ranking
    feat_corrs = compute_feature_human_correlations(features, labels, sources, feature_names)
    print(f"  Top 10 features by human |corr|:")
    for fname, r in feat_corrs[:10]:
        print(f"    {fname}: {r:.4f}")

    # Build full feature matrix
    X_feat = build_feature_matrix(features, ALL_FEATURE_KEYS)

    k_values = [5, 10, 15, 20, len(ALL_FEATURE_KEYS)]
    hybrid_alphas = [0.1, 1.0, 10.0, 100.0]
    n_configs = len(k_values) * len(hybrid_alphas) * len(sw_options)
    i = 0
    best_e = {"score": -1.0}

    for k, alpha, (sw_name, sw_fn) in itertools.product(k_values, hybrid_alphas, sw_options):
        i += 1
        # Select top-k features by human label correlation
        top_k_names = [fname for fname, _ in feat_corrs[:k]]
        top_k_idx = [ALL_FEATURE_KEYS.index(f) for f in top_k_names if f in ALL_FEATURE_KEYS]
        X_sel = X_feat[:, top_k_idx] if top_k_idx else np.zeros((n, 0))

        # Concatenate embeddings + selected features
        X_hybrid = np.hstack([E, X_sel])

        result = oof_evaluate(
            Ridge, {"alpha": alpha}, X_hybrid, labels, folds, sources, sw_fn,
        )
        result["v4_score"] = compute_v4_objective(result)
        result["arch"] = "hybrid"
        result["k"] = k
        result["alpha"] = alpha
        result["sw"] = sw_name
        result["input_dim"] = X_hybrid.shape[1]
        all_results.append(result)

        if result["v4_score"] > best_e["score"]:
            best_e = {"score": result["v4_score"], **result}

        if i % 20 == 0 or i == n_configs:
            print(f"  [{i:>3d}/{n_configs}] best: human={best_e.get('oof_human_corr', 0):.4f} "
                  f"cv={best_e.get('cv_corr', 0):.4f} k={best_e.get('k')} "
                  f"dim={best_e.get('input_dim')}")

    print(f"\n  Best Hybrid: k={best_e.get('k')} alpha={best_e.get('alpha')} sw={best_e.get('sw')}")
    print(f"    human_corr={best_e.get('oof_human_corr', 0):.4f} "
          f"cv_corr={best_e.get('cv_corr', 0):.4f} "
          f"input_dim={best_e.get('input_dim')}")

    # ===================================================================
    # Summary comparison
    # ===================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * W)
    print("SUMMARY COMPARISON")
    print("=" * W)

    # Sort all results by v4 objective
    ranked = sorted(
        [r for r in all_results if r.get("v4_score", -1) > 0],
        key=lambda r: -r["v4_score"],
    )

    # Top 20 overall
    print(f"\nTop 20 configs (of {len(all_results)} total):")
    print(f"{'Rank':>4s}  {'Arch':<12s}  {'human':>7s}  {'cv':>7s}  {'synth':>7s}  {'obj':>7s}  Details")
    print("-" * W)
    for idx, r in enumerate(ranked[:20]):
        details = ""
        if r["arch"] == "ridge":
            details = f"alpha={r['alpha']} sw={r['sw']}"
        elif r["arch"] == "lasso":
            details = f"alpha={r['alpha']} nz={r.get('n_nonzero', '?')} sw={r['sw']}"
        elif r["arch"] == "elasticnet":
            details = f"alpha={r['alpha']} l1={r.get('l1_ratio')} sw={r['sw']}"
        elif r["arch"] == "dim_ridge":
            details = f"da={r.get('dim_alpha')} ca={r.get('combiner_alpha')} sw={r['sw']}"
        elif r["arch"] == "hybrid":
            details = f"k={r.get('k')} alpha={r['alpha']} dim={r.get('input_dim')} sw={r['sw']}"

        print(f"{idx+1:>4d}  {r['arch']:<12s}  "
              f"{r['oof_human_corr']:>7.4f}  {r['cv_corr']:>7.4f}  "
              f"{r['oof_synth_corr']:>7.4f}  {r['v4_score']:>7.4f}  {details}")

    # Best per architecture
    print(f"\nBest per architecture:")
    print(f"{'Arch':<15s}  {'human':>7s}  {'cv':>7s}  {'synth':>7s}  {'obj':>7s}")
    print("-" * 50)
    baselines = [
        ("Linear (13f)", 0.301, 0.627, 0.0),
        ("Ensemble v5", 0.324, 0.822, 0.933),
    ]
    for name, hc, cc, sc in baselines:
        print(f"{name:<15s}  {hc:>7.3f}  {cc:>7.3f}  {sc:>7.3f}  {'--':>7s}")

    for arch_name, best in [("Ridge", best_a), ("Lasso", best_b),
                             ("ElasticNet", best_c), ("DimRidge", best_d),
                             ("Hybrid", best_e)]:
        hc = best.get("oof_human_corr", 0)
        cc = best.get("cv_corr", 0)
        sc = best.get("oof_synth_corr", 0)
        obj = best.get("score", 0) if isinstance(best.get("score"), float) else best.get("v4_score", 0)
        print(f"{arch_name:<15s}  {hc:>7.4f}  {cc:>7.4f}  {sc:>7.4f}  {obj:>7.4f}")

    print(f"\nTotal configs tested: {len(all_results)}")
    print(f"Total time: {elapsed:.1f}s")

    # ===================================================================
    # Save results
    # ===================================================================
    results_path = Path(__file__).parent / "embedding_regression_results.json"
    save_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_entries": n,
        "n_human": n_human,
        "n_synth": n_synth,
        "embedding_dim": emb_dim,
        "n_configs": len(all_results),
        "elapsed_seconds": round(elapsed, 1),
        "best_per_arch": {
            "ridge": {k: v for k, v in best_a.items() if k not in ("oof_preds", "score")},
            "lasso": {k: v for k, v in best_b.items() if k not in ("oof_preds", "score")},
            "elasticnet": {k: v for k, v in best_c.items() if k not in ("oof_preds", "score")},
            "dim_ridge": {k: v for k, v in best_d.items() if k not in ("oof_preds", "score")},
            "hybrid": {k: v for k, v in best_e.items() if k not in ("oof_preds", "score")},
        },
        "top_20": [
            {k: v for k, v in r.items() if k != "oof_preds"}
            for r in ranked[:20]
        ],
    }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ===================================================================
    # Winner analysis & artifact saving
    # ===================================================================
    if ranked:
        winner = ranked[0]
        wh = winner["oof_human_corr"]
        print(f"\n{'=' * W}")
        if wh > 0.324:
            print(f"WINNER: {winner['arch']} with human_corr={wh:.4f} (beats v5 ensemble 0.324)")

            # Auto-save artifacts for hybrid winner
            if winner["arch"] == "hybrid":
                save_hybrid_artifacts(
                    E=E,
                    labels=labels,
                    features=features,
                    sources=sources,
                    feat_corrs=feat_corrs,
                    k=winner["k"],
                    alpha=winner["alpha"],
                    metrics=winner,
                )
            else:
                print(f"  Winner is {winner['arch']} — manual deployment needed.")
        else:
            print(f"BEST: {winner['arch']} with human_corr={wh:.4f}")
            if wh <= 0.324:
                print(f"  Does NOT beat v5 ensemble (0.324). Feature bottleneck may not be the issue.")
                print(f"  Consider: label quality improvement (human re-labeling) as next step.")
        print("=" * W)


if __name__ == "__main__":
    main()
