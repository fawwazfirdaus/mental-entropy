#!/usr/bin/env python3
"""Train MES v7: FEP-aligned subscore architecture.

Architecture:
  Stage 1: 5 Ridge models — embedding (1024) → each FEP dimension (1-5)
  Stage 2: Combiner Ridge — [5 subscores + top-10 features] → overall_entropy (1-10)

Also saves the direct hybrid model (embedding + features → MES) for comparison.

Usage:
    cd mental-entropy
    PYTHONPATH=src .venv/bin/python -u autoresearch-macos/train_v7_fep.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
from train_subscores_v4 import get_stratified_folds
from train_embedding_regression import (
    oof_evaluate,
    make_sample_weights_fn,
    compute_feature_human_correlations,
)
from train_subscores import build_feature_matrix
from train_subscores_v3 import ALL_FEATURE_KEYS

W = 72
ARTIFACTS_DIR = Path(__file__).parent.parent / "src" / "mental_entropy" / "models" / "_artifacts"

# Display names for the API
DIMENSION_DISPLAY = {
    "prediction_coherence": ("Thought Flow", "How smoothly your thoughts connect"),
    "model_complexity": ("Focus", "How well you organized around key themes"),
    "compression_progress": ("Insight", "Whether you moved toward understanding"),
    "belief_integration": ("Integration", "How well you hold contradictions with awareness"),
    "precision_weighting": ("Clarity", "How decisively you express yourself"),
}


def main() -> None:
    print("=" * W)
    print("MES v7: FEP-Aligned Subscore Architecture")
    print("=" * W)

    # Load data
    print("\nLoading cached data...")
    features, labels, feature_names, sources = load_cached_data()
    E = load_cached_embeddings()
    dim_labels = load_dimension_labels()

    n = len(features)
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())
    human_mask = sources == SOURCE_HUMAN

    print(f"  Entries: {n} ({n_human} human, {n_synth} synthetic)")
    print(f"  Embedding dim: {E.shape[1]}")
    print(f"  Dimensions: {list(DIMENSION_NAMES)}")

    folds = get_stratified_folds(sources)
    sw_fn = make_sample_weights_fn(0.3)  # human_heavy

    # Feature correlations for top-k selection
    feat_corrs = compute_feature_human_correlations(features, labels, sources, feature_names)
    top_10_names = [fname for fname, _ in feat_corrs[:10]]
    print(f"\n  Top 10 features by human |corr|:")
    for fname, r in feat_corrs[:10]:
        print(f"    {fname}: {r:.4f}")

    X_feat = build_feature_matrix(features, ALL_FEATURE_KEYS)
    top_10_idx = [ALL_FEATURE_KEYS.index(f) for f in top_10_names if f in ALL_FEATURE_KEYS]
    X_sel = X_feat[:, top_10_idx]

    t0 = time.time()

    # =================================================================
    # Baseline: Direct hybrid Ridge (embedding + 10 features → MES)
    # =================================================================
    print("\n" + "=" * W)
    print("BASELINE: Direct Hybrid Ridge (v6 equivalent)")
    print("=" * W)

    X_hybrid = np.hstack([E, X_sel])
    baseline = oof_evaluate(Ridge, {"alpha": 0.1}, X_hybrid, labels, folds, sources, sw_fn)
    print(f"  human_corr={baseline['oof_human_corr']:.4f}, cv_corr={baseline['cv_corr']:.4f}")

    # =================================================================
    # Stage 1: Per-dimension Ridge models
    # =================================================================
    print("\n" + "=" * W)
    print("STAGE 1: Per-Dimension Ridge (embedding → dim label)")
    print("=" * W)

    n_dims = len(DIMENSION_NAMES)
    dim_oof = np.zeros((n, n_dims))
    dim_corrs = {}

    # Sweep dim_alpha
    best_dim_alpha = 0.1
    best_dim_score = -1.0

    for dim_alpha in [0.01, 0.1, 1.0, 10.0]:
        total_h = 0.0
        for d_idx, dim_name in enumerate(DIMENSION_NAMES):
            d_labels = dim_labels[dim_name]
            oof_tmp = np.full(n, np.nan)
            for train_idx, val_idx in folds:
                model = Ridge(alpha=dim_alpha)
                sw = sw_fn(sources[train_idx])
                model.fit(E[train_idx], d_labels[train_idx], sample_weight=sw)
                oof_tmp[val_idx] = model.predict(E[val_idx])
            h_corr = pearson_correlation(oof_tmp[human_mask], d_labels[human_mask])
            total_h += h_corr
        avg_h = total_h / n_dims
        if avg_h > best_dim_score:
            best_dim_score = avg_h
            best_dim_alpha = dim_alpha
        print(f"  dim_alpha={dim_alpha}: avg_human_corr={avg_h:.4f}")

    print(f"  → Best dim_alpha={best_dim_alpha}")

    # Generate OOF subscores with best alpha
    print(f"\n  Per-dimension results (alpha={best_dim_alpha}):")
    print(f"  {'Dimension':<25} {'human_corr':>11} {'cv_corr':>9}")
    print("  " + "-" * 50)

    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        d_labels = dim_labels[dim_name]
        for train_idx, val_idx in folds:
            model = Ridge(alpha=best_dim_alpha)
            sw = sw_fn(sources[train_idx])
            model.fit(E[train_idx], d_labels[train_idx], sample_weight=sw)
            dim_oof[val_idx, d_idx] = model.predict(E[val_idx])
        h_corr = pearson_correlation(dim_oof[human_mask, d_idx], d_labels[human_mask])
        cv_corr = pearson_correlation(dim_oof[:, d_idx], d_labels)
        dim_corrs[dim_name] = {"human_corr": h_corr, "cv_corr": cv_corr}
        print(f"  {dim_name:<25} {h_corr:>11.4f} {cv_corr:>9.4f}")

    # =================================================================
    # Stage 2: Combiner (subscores + features → overall_entropy)
    # =================================================================
    print("\n" + "=" * W)
    print("STAGE 2: Combiner (subscores + features → MES)")
    print("=" * W)

    # Test multiple combiner configurations
    configs = [
        ("subscores_only", dim_oof),
        ("subscores+10feat", np.hstack([dim_oof, X_sel])),
    ]

    best_config_name = None
    best_config_h = -1.0
    combiner_results = {}

    for config_name, X_comb in configs:
        oof_preds = np.full(n, np.nan)
        fold_coefs = []

        for comb_alpha in [0.1, 1.0, 10.0]:
            oof_tmp = np.full(n, np.nan)
            for train_idx, val_idx in folds:
                combiner = Ridge(alpha=comb_alpha)
                combiner.fit(X_comb[train_idx], labels[train_idx])
                oof_tmp[val_idx] = combiner.predict(X_comb[val_idx])
            h_corr = pearson_correlation(oof_tmp[human_mask], labels[human_mask])
            cv_corr = pearson_correlation(oof_tmp, labels)
            print(f"  {config_name} (alpha={comb_alpha}): human={h_corr:.4f} cv={cv_corr:.4f}")

            if h_corr > best_config_h:
                best_config_h = h_corr
                best_config_name = config_name
                best_comb_alpha = comb_alpha
                oof_preds = oof_tmp.copy()

        combiner_results[config_name] = {
            "human_corr": best_config_h,
        }

    print(f"\n  → Best: {best_config_name} (alpha={best_comb_alpha})")
    print(f"    human_corr={best_config_h:.4f}")

    # =================================================================
    # Final comparison
    # =================================================================
    print("\n" + "=" * W)
    print("COMPARISON")
    print("=" * W)

    gap = best_config_h - baseline["oof_human_corr"]
    print(f"  Direct Ridge (v6):       human_corr={baseline['oof_human_corr']:.4f}")
    print(f"  FEP subscores (v7):      human_corr={best_config_h:.4f} (gap: {gap:+.4f})")
    print(f"  Architecture: {best_config_name}")

    # =================================================================
    # SAVE ARTIFACTS
    # =================================================================
    print("\n" + "=" * W)
    print("SAVING v7 ARTIFACTS")
    print("=" * W)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Per-dimension Ridge models (trained on full data) ---
    dim_models = {}
    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        d_labels_full = dim_labels[dim_name]
        model = Ridge(alpha=best_dim_alpha)
        sw = sw_fn(sources)
        model.fit(E, d_labels_full, sample_weight=sw)

        dim_model_data = {
            "coef": [float(c) for c in model.coef_],
            "intercept": float(model.intercept_),
            "alpha": best_dim_alpha,
            "embedding_dim": int(E.shape[1]),
            "dimension": dim_name,
            "display_name": DIMENSION_DISPLAY[dim_name][0],
            "description": DIMENSION_DISPLAY[dim_name][1],
        }
        dim_path = ARTIFACTS_DIR / f"dim_{dim_name}.json"
        with open(dim_path, "w") as f:
            json.dump(dim_model_data, f, indent=2)
        print(f"  Saved: {dim_path.name}")
        dim_models[dim_name] = model

    # --- Generate full-data subscores for combiner training ---
    full_subscores = np.zeros((n, n_dims))
    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        full_subscores[:, d_idx] = dim_models[dim_name].predict(E)

    # --- Combiner (trained on full data) ---
    use_features = "10feat" in best_config_name
    if use_features:
        X_comb_full = np.hstack([full_subscores, X_sel])
        combiner_feature_names = top_10_names
    else:
        X_comb_full = full_subscores
        combiner_feature_names = []

    combiner_model = Ridge(alpha=best_comb_alpha)
    combiner_model.fit(X_comb_full, labels)

    combiner_data = {
        "architecture": "subscore_embedding_v7",
        "coef": [float(c) for c in combiner_model.coef_],
        "intercept": float(combiner_model.intercept_),
        "alpha": best_comb_alpha,
        "dimension_names": list(DIMENSION_NAMES),
        "n_dims": n_dims,
        "feature_names": combiner_feature_names,
        "uses_features": use_features,
        "dimension_display": {
            dim: {"name": DIMENSION_DISPLAY[dim][0], "description": DIMENSION_DISPLAY[dim][1]}
            for dim in DIMENSION_NAMES
        },
    }
    combiner_path = ARTIFACTS_DIR / "combiner.json"
    with open(combiner_path, "w") as f:
        json.dump(combiner_data, f, indent=2)
    print(f"  Saved: {combiner_path.name}")

    # Print combiner coefficients for interpretation
    print(f"\n  Combiner coefficients:")
    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        bar = "█" * int(abs(combiner_model.coef_[d_idx]) * 20)
        print(f"    {dim_name:<25} {combiner_model.coef_[d_idx]:>8.4f}  {bar}")
    if use_features:
        print(f"  Feature coefficients:")
        for f_idx, fname in enumerate(combiner_feature_names):
            coef_idx = n_dims + f_idx
            print(f"    {fname:<25} {combiner_model.coef_[coef_idx]:>8.4f}")

    # --- Also save direct hybrid model (embedding_model.json) for fallback ---
    hybrid_model = Ridge(alpha=0.1)
    sw_full = sw_fn(sources)
    hybrid_model.fit(X_hybrid, labels, sample_weight=sw_full)

    embedding_model_data = {
        "coef": [float(c) for c in hybrid_model.coef_],
        "intercept": float(hybrid_model.intercept_),
        "embedding_dim": 1024,
        "feature_names": top_10_names,
        "alpha": 0.1,
        "input_dim": int(X_hybrid.shape[1]),
    }
    emb_path = ARTIFACTS_DIR / "embedding_model.json"
    with open(emb_path, "w") as f:
        json.dump(embedding_model_data, f, indent=2)
    print(f"  Saved: {emb_path.name} (direct hybrid fallback)")

    # --- Manifest ---
    manifest = {
        "architecture": "subscore_embedding_v7",
        "training_script": "train_v7_fep.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_entries": n,
        "n_human": n_human,
        "n_synthetic": n_synth,
        "embedding_dim": 1024,
        "dimension_names": list(DIMENSION_NAMES),
        "dimension_display": {
            dim: {"name": DIMENSION_DISPLAY[dim][0], "description": DIMENSION_DISPLAY[dim][1]}
            for dim in DIMENSION_NAMES
        },
        "dim_alpha": best_dim_alpha,
        "combiner_alpha": best_comb_alpha,
        "combiner_features": combiner_feature_names,
        "metrics": {
            "direct_hybrid_human_corr": float(baseline["oof_human_corr"]),
            "direct_hybrid_cv_corr": float(baseline["cv_corr"]),
            "v7_human_corr": float(best_config_h),
            "per_dimension": {
                dim: {
                    "human_corr": float(dim_corrs[dim]["human_corr"]),
                    "cv_corr": float(dim_corrs[dim]["cv_corr"]),
                }
                for dim in DIMENSION_NAMES
            },
        },
    }
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {manifest_path.name}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n{'=' * W}")
    print(f"  v7 artifacts saved. Ready to deploy!")
    print(f"{'=' * W}")


if __name__ == "__main__":
    main()
