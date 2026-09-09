#!/usr/bin/env python3
"""Diagnostic analysis of the 5 rubric-defined MES subscores.

Analyzes whether the current 5 dimensions (continuity, topic_focus,
contradiction_integration, cognitive_clarity, narrative_closure) are
all contributing meaningfully, identifies redundancy, and tests whether
a tuned subscore architecture can close the gap with direct Ridge.

Usage:
    cd mental-entropy
    PYTHONPATH=src .venv/bin/python -u autoresearch-macos/diagnose_subscores.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
import itertools
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prepare import (
    load_cached_data,
    load_cached_embeddings,
    load_dimension_labels,
    pearson_correlation,
    SOURCE_HUMAN,
    SOURCE_SYNTHETIC,
)
from train_subscores_v4 import get_stratified_folds, compute_v4_objective
from train_embedding_regression import (
    oof_evaluate,
    oof_evaluate_dimension_aligned,
    make_sample_weights_fn,
    DIMENSION_NAMES,
)


DATA_DIR = Path(__file__).parent.parent / "data" / "multirater"


# ---------------------------------------------------------------------------
# Analysis A: Inter-dimension correlation matrix
# ---------------------------------------------------------------------------

def analysis_a_correlation_matrix(
    dim_labels: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
) -> None:
    """Correlation matrix between 5 dimensions + overall_entropy."""
    print("\n" + "=" * 72)
    print("ANALYSIS A: Inter-Dimension Correlation Matrix (human entries only)")
    print("=" * 72)

    human = sources == SOURCE_HUMAN
    names = list(DIMENSION_NAMES) + ["overall_entropy"]
    arrays = [dim_labels[d][human] for d in DIMENSION_NAMES] + [labels[human]]

    n = len(names)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = pearson_correlation(arrays[i], arrays[j])

    # Print matrix
    header = "                     " + "  ".join(f"{name[:8]:>8}" for name in names)
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(names):
        row = f"  {name:<20}" + "  ".join(f"{corr_matrix[i, j]:>8.3f}" for j in range(n))
        print(row)

    # Identify redundant pairs
    print("\n  Redundant pairs (|r| > 0.8):")
    found = False
    for i in range(n - 1):
        for j in range(i + 1, n - 1):  # Skip overall_entropy
            if abs(corr_matrix[i, j]) > 0.8:
                print(f"    {names[i]} <-> {names[j]}: r={corr_matrix[i, j]:.3f}")
                found = True
    if not found:
        print("    None found — all dimensions are reasonably independent")

    # Identify independent dimensions
    print("\n  Most independent pairs (|r| < 0.4):")
    for i in range(n - 1):
        for j in range(i + 1, n - 1):
            if abs(corr_matrix[i, j]) < 0.4:
                print(f"    {names[i]} <-> {names[j]}: r={corr_matrix[i, j]:.3f}")

    # Correlation with overall_entropy
    print("\n  Correlation with overall_entropy:")
    for i, name in enumerate(names[:-1]):
        print(f"    {name:<30} r={corr_matrix[i, -1]:.3f}")


# ---------------------------------------------------------------------------
# Analysis B: Per-dimension label reliability
# ---------------------------------------------------------------------------

def analysis_b_label_reliability(
    dim_labels: dict[str, np.ndarray],
    sources: np.ndarray,
) -> dict[str, dict]:
    """Check inter-rater agreement per dimension across 3 labeling passes."""
    print("\n" + "=" * 72)
    print("ANALYSIS B: Per-Dimension Label Reliability (inter-rater agreement)")
    print("=" * 72)

    # Load raw ratings from each pass
    passes = {}
    for pass_name in ["a", "b", "c"]:
        path = DATA_DIR / f"pass_{pass_name}_labeled.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping reliability analysis")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        passes[pass_name] = rows

    # Align to minimum length
    min_len = min(len(passes[p]) for p in passes)
    human = sources[:min_len] == SOURCE_HUMAN

    dims = list(DIMENSION_NAMES) + ["overall_entropy"]
    reliability = {}

    print(f"\n  {'Dimension':<30} {'a_b_corr':>8} {'a_c_corr':>8} {'b_c_corr':>8} {'mean_corr':>9} {'mean_std':>8} {'high_disagree':>13}")
    print("  " + "-" * 100)

    for dim in dims:
        try:
            a_vals = np.array([float(passes["a"][i].get(dim, 0)) for i in range(min_len)])
            b_vals = np.array([float(passes["b"][i].get(dim, 0)) for i in range(min_len)])
            c_vals = np.array([float(passes["c"][i].get(dim, 0)) for i in range(min_len)])
        except (ValueError, KeyError):
            continue

        # Human entries only
        a_h, b_h, c_h = a_vals[human], b_vals[human], c_vals[human]

        ab = pearson_correlation(a_h, b_h)
        ac = pearson_correlation(a_h, c_h)
        bc = pearson_correlation(b_h, c_h)
        mean_corr = (ab + ac + bc) / 3

        # Per-entry std across 3 raters
        stds = np.std(np.column_stack([a_h, b_h, c_h]), axis=1)
        mean_std = float(stds.mean())

        # High disagreement: entries where std > 1.5
        high_disagree = float((stds > 1.5).mean() * 100)

        print(f"  {dim:<30} {ab:>8.3f} {ac:>8.3f} {bc:>8.3f} {mean_corr:>9.3f} {mean_std:>8.2f} {high_disagree:>12.1f}%")

        reliability[dim] = {
            "ab_corr": ab, "ac_corr": ac, "bc_corr": bc,
            "mean_corr": mean_corr, "mean_std": mean_std,
            "high_disagree_pct": high_disagree,
        }

    return reliability


# ---------------------------------------------------------------------------
# Analysis C: Per-dimension predictive value
# ---------------------------------------------------------------------------

def analysis_c_predictive_value(
    E: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    labels: np.ndarray,
    folds: list,
    sources: np.ndarray,
    sw_fn,
) -> dict[str, float]:
    """How well can embeddings predict each dimension? Which dimensions matter for MES?"""
    print("\n" + "=" * 72)
    print("ANALYSIS C: Per-Dimension Predictive Value")
    print("=" * 72)

    human_mask = sources == SOURCE_HUMAN
    n = len(labels)
    n_dims = len(DIMENSION_NAMES)

    # Stage 1: Per-dimension OOF predictions
    dim_oof = np.zeros((n, n_dims))
    dim_corrs = {}

    print(f"\n  Per-dimension embedding → dimension_label (Ridge α=0.1, human_heavy):")
    print(f"  {'Dimension':<30} {'human_corr':>11} {'cv_corr':>9}")
    print("  " + "-" * 55)

    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        d_labels = dim_labels[dim_name]
        for train_idx, val_idx in folds:
            model = Ridge(alpha=0.1)
            sw = sw_fn(sources[train_idx]) if sw_fn else None
            model.fit(E[train_idx], d_labels[train_idx], sample_weight=sw)
            dim_oof[val_idx, d_idx] = model.predict(E[val_idx])

        h_corr = pearson_correlation(dim_oof[human_mask, d_idx], d_labels[human_mask])
        cv_corr = pearson_correlation(dim_oof[:, d_idx], d_labels)
        dim_corrs[dim_name] = h_corr
        print(f"  {dim_name:<30} {h_corr:>11.4f} {cv_corr:>9.4f}")

    # Stage 2: Combiner — which dimensions matter most?
    print(f"\n  Combiner: 5 OOF subscores → overall_entropy")

    oof_preds = np.full(n, np.nan)
    combiner_coefs = []

    for train_idx, val_idx in folds:
        combiner = Ridge(alpha=1.0)
        combiner.fit(dim_oof[train_idx], labels[train_idx])
        oof_preds[val_idx] = combiner.predict(dim_oof[val_idx])
        combiner_coefs.append(combiner.coef_)

    mean_coefs = np.mean(combiner_coefs, axis=0)
    h_corr = pearson_correlation(oof_preds[human_mask], labels[human_mask])
    cv_corr = pearson_correlation(oof_preds, labels)

    print(f"  Combined: human_corr={h_corr:.4f}, cv_corr={cv_corr:.4f}")
    print(f"\n  Combiner coefficients (mean across folds):")
    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        bar = "█" * int(abs(mean_coefs[d_idx]) * 20)
        print(f"    {dim_name:<30} {mean_coefs[d_idx]:>8.4f}  {bar}")

    return dim_corrs, dim_oof


# ---------------------------------------------------------------------------
# Analysis D: Ablation study
# ---------------------------------------------------------------------------

def analysis_d_ablation(
    dim_oof: np.ndarray,
    labels: np.ndarray,
    folds: list,
    sources: np.ndarray,
) -> None:
    """Remove each dimension one at a time. Test reduced sets."""
    print("\n" + "=" * 72)
    print("ANALYSIS D: Dimension Ablation Study")
    print("=" * 72)

    human_mask = sources == SOURCE_HUMAN
    n = len(labels)
    dims = list(DIMENSION_NAMES)

    # Full 5-dimension baseline
    def eval_dims(dim_indices, label):
        oof_preds = np.full(n, np.nan)
        for train_idx, val_idx in folds:
            combiner = Ridge(alpha=1.0)
            combiner.fit(dim_oof[train_idx][:, dim_indices], labels[train_idx])
            oof_preds[val_idx] = combiner.predict(dim_oof[val_idx][:, dim_indices])
        h_corr = pearson_correlation(oof_preds[human_mask], labels[human_mask])
        cv_corr = pearson_correlation(oof_preds, labels)
        return h_corr, cv_corr

    all_5_h, all_5_cv = eval_dims(list(range(5)), "all 5")
    print(f"\n  All 5 dimensions: human_corr={all_5_h:.4f}, cv_corr={all_5_cv:.4f}")

    # Remove each one
    print(f"\n  Leave-one-out:")
    print(f"  {'Removed':<30} {'human_corr':>11} {'cv_corr':>9} {'delta':>8}")
    print("  " + "-" * 65)

    for i, dim_name in enumerate(dims):
        remaining = [j for j in range(5) if j != i]
        h_corr, cv_corr = eval_dims(remaining, f"w/o {dim_name}")
        delta = h_corr - all_5_h
        sign = "+" if delta >= 0 else ""
        print(f"  w/o {dim_name:<25} {h_corr:>11.4f} {cv_corr:>9.4f} {sign}{delta:>7.4f}")

    # Best subsets of size 2, 3, 4
    print(f"\n  Best subsets by size:")
    for size in [2, 3, 4]:
        best_h = -1
        best_combo = None
        for combo in itertools.combinations(range(5), size):
            h_corr, cv_corr = eval_dims(list(combo), "")
            if h_corr > best_h:
                best_h = h_corr
                best_cv = cv_corr
                best_combo = combo
        combo_names = [dims[i] for i in best_combo]
        delta = best_h - all_5_h
        sign = "+" if delta >= 0 else ""
        print(f"  Best {size}: {', '.join(combo_names)}")
        print(f"         human_corr={best_h:.4f}, cv_corr={best_cv:.4f} ({sign}{delta:.4f})")


# ---------------------------------------------------------------------------
# Analysis E: Tuned dimension-aligned Ridge
# ---------------------------------------------------------------------------

def analysis_e_tuned_architecture(
    E: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    labels: np.ndarray,
    folds: list,
    sources: np.ndarray,
    features: list[dict],
    feature_names: list[str],
) -> None:
    """Test whether a better-tuned subscore architecture closes the gap."""
    print("\n" + "=" * 72)
    print("ANALYSIS E: Tuned Dimension-Aligned Ridge Architectures")
    print("=" * 72)

    human_mask = sources == SOURCE_HUMAN
    n = len(labels)
    n_dims = len(DIMENSION_NAMES)

    results = []

    # E1: Sweep dim_alpha × combiner_alpha × weights
    print("\n  --- E1: Alpha sweep ---")
    print(f"  {'dim_α':>6} {'comb_α':>7} {'weights':>12} {'human_corr':>11} {'cv_corr':>9}")
    print("  " + "-" * 55)

    for dim_alpha in [0.01, 0.1, 1.0, 10.0]:
        for comb_alpha in [0.1, 1.0, 10.0]:
            for wname, wval in [("equal", None), ("human_heavy", 0.3)]:
                sw_fn = make_sample_weights_fn(wval) if wval else None
                result = oof_evaluate_dimension_aligned(
                    dim_alpha, comb_alpha, E, labels, dim_labels, folds, sources, sw_fn
                )
                h_corr = result["oof_human_corr"]
                cv_corr = result["cv_corr"]
                print(f"  {dim_alpha:>6} {comb_alpha:>7} {wname:>12} {h_corr:>11.4f} {cv_corr:>9.4f}")
                results.append({
                    "type": "alpha_sweep",
                    "dim_alpha": dim_alpha,
                    "comb_alpha": comb_alpha,
                    "weights": wname,
                    "human_corr": h_corr,
                    "cv_corr": cv_corr,
                })

    # E2: Subscores + embedding in combiner
    print("\n  --- E2: Subscores + raw embedding in combiner ---")
    best_alpha = max(results, key=lambda x: x["human_corr"])
    da, ca = best_alpha["dim_alpha"], best_alpha["comb_alpha"]
    sw_fn = make_sample_weights_fn(0.3)

    # Generate per-dimension OOF subscores with best alpha
    dim_oof = np.zeros((n, n_dims))
    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        d_labels = dim_labels[dim_name]
        for train_idx, val_idx in folds:
            model = Ridge(alpha=da)
            sw = sw_fn(sources[train_idx])
            model.fit(E[train_idx], d_labels[train_idx], sample_weight=sw)
            dim_oof[val_idx, d_idx] = model.predict(E[val_idx])

    # Combiner: subscores only
    oof_preds = np.full(n, np.nan)
    for train_idx, val_idx in folds:
        combiner = Ridge(alpha=ca)
        combiner.fit(dim_oof[train_idx], labels[train_idx])
        oof_preds[val_idx] = combiner.predict(dim_oof[val_idx])
    base_h = pearson_correlation(oof_preds[human_mask], labels[human_mask])
    base_cv = pearson_correlation(oof_preds, labels)
    print(f"  Subscores only:         human_corr={base_h:.4f}, cv_corr={base_cv:.4f}")

    # Combiner: subscores + embedding
    X_combined = np.hstack([dim_oof, E])
    oof_preds2 = np.full(n, np.nan)
    for train_idx, val_idx in folds:
        combiner = Ridge(alpha=1.0)
        combiner.fit(X_combined[train_idx], labels[train_idx])
        oof_preds2[val_idx] = combiner.predict(X_combined[val_idx])
    emb_h = pearson_correlation(oof_preds2[human_mask], labels[human_mask])
    emb_cv = pearson_correlation(oof_preds2, labels)
    print(f"  Subscores + embedding:  human_corr={emb_h:.4f}, cv_corr={emb_cv:.4f}")

    # E3: Subscores + top-10 features in combiner
    from train_embedding_regression import compute_feature_human_correlations
    from train_subscores import build_feature_matrix
    from train_subscores_v3 import ALL_FEATURE_KEYS

    X_feat = build_feature_matrix(features, ALL_FEATURE_KEYS)
    corrs = compute_feature_human_correlations(features, labels, human_mask, ALL_FEATURE_KEYS)
    top_10 = [name for name, _ in corrs[:10]]
    top_idx = [ALL_FEATURE_KEYS.index(f) for f in top_10]
    X_sel = X_feat[:, top_idx]

    X_combined_feat = np.hstack([dim_oof, X_sel])
    oof_preds3 = np.full(n, np.nan)
    for train_idx, val_idx in folds:
        combiner = Ridge(alpha=1.0)
        combiner.fit(X_combined_feat[train_idx], labels[train_idx])
        oof_preds3[val_idx] = combiner.predict(X_combined_feat[val_idx])
    feat_h = pearson_correlation(oof_preds3[human_mask], labels[human_mask])
    feat_cv = pearson_correlation(oof_preds3, labels)
    print(f"  Subscores + 10 feats:   human_corr={feat_h:.4f}, cv_corr={feat_cv:.4f}")

    # E4: Subscores + embedding + features (kitchen sink)
    X_all = np.hstack([dim_oof, E, X_sel])
    oof_preds4 = np.full(n, np.nan)
    for train_idx, val_idx in folds:
        combiner = Ridge(alpha=1.0)
        combiner.fit(X_all[train_idx], labels[train_idx])
        oof_preds4[val_idx] = combiner.predict(X_all[val_idx])
    all_h = pearson_correlation(oof_preds4[human_mask], labels[human_mask])
    all_cv = pearson_correlation(oof_preds4, labels)
    print(f"  Subscores + emb + feat: human_corr={all_h:.4f}, cv_corr={all_cv:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("MES SUBSCORE DIAGNOSTIC")
    print("=" * 72)

    # Load data
    print("\n--- Loading data ---")
    features, labels, feature_names, sources = load_cached_data()
    E = load_cached_embeddings()
    dim_labels = load_dimension_labels()
    n = len(labels)
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())
    print(f"Entries: {n} ({n_human} human, {n_synth} synthetic)")
    print(f"Dimensions: {list(DIMENSION_NAMES)}")

    folds = get_stratified_folds(sources)
    sw_fn = make_sample_weights_fn(0.3)

    # Direct Ridge baseline
    print("\n--- Direct Ridge Baseline ---")
    from train_subscores import build_feature_matrix
    from train_subscores_v3 import ALL_FEATURE_KEYS
    from train_embedding_regression import compute_feature_human_correlations

    human_mask = sources == SOURCE_HUMAN
    X_feat = build_feature_matrix(features, ALL_FEATURE_KEYS)
    corrs = compute_feature_human_correlations(features, labels, human_mask, ALL_FEATURE_KEYS)
    top_10 = [name for name, _ in corrs[:10]]
    top_idx = [ALL_FEATURE_KEYS.index(f) for f in top_10]
    X_sel = X_feat[:, top_idx]
    E_hybrid = np.hstack([E, X_sel])

    baseline = oof_evaluate(Ridge, {"alpha": 0.1}, E_hybrid, labels, folds, sources, sw_fn)
    print(f"Direct hybrid Ridge: human_corr={baseline['oof_human_corr']:.4f}, cv_corr={baseline['cv_corr']:.4f}")

    t0 = time.time()

    # Run analyses
    analysis_a_correlation_matrix(dim_labels, labels, sources)
    reliability = analysis_b_label_reliability(dim_labels, sources)
    dim_corrs, dim_oof = analysis_c_predictive_value(E, dim_labels, labels, folds, sources, sw_fn)
    analysis_d_ablation(dim_oof, labels, folds, sources)
    e_results = analysis_e_tuned_architecture(E, dim_labels, labels, folds, sources, features, feature_names)

    elapsed = time.time() - t0

    # Final summary
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    best_tuned = max(e_results, key=lambda x: x["human_corr"])
    direct_h = baseline["oof_human_corr"]

    print(f"\n  Direct Ridge (baseline):     human_corr={direct_h:.4f}")
    print(f"  Best tuned subscores:        human_corr={best_tuned['human_corr']:.4f} (gap: {best_tuned['human_corr'] - direct_h:+.4f})")
    print(f"\n  Total time: {elapsed:.1f}s")

    # Save results
    out_path = Path(__file__).parent / "subscore_diagnostic_results.json"
    out_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_entries": n,
        "n_human": n_human,
        "direct_ridge_human_corr": direct_h,
        "direct_ridge_cv_corr": baseline["cv_corr"],
        "best_tuned_human_corr": best_tuned["human_corr"],
        "best_tuned_config": best_tuned,
        "reliability": reliability,
        "dim_embedding_corrs": {k: float(v) for k, v in dim_corrs.items()},
        "all_tuned_results": e_results,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n  Results saved to {out_path.name}")


if __name__ == "__main__":
    main()
