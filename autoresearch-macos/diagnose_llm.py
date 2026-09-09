"""
Human Correlation Diagnostic for MES Subscore Models.

Analyzes why the subscore models regressed on human-written journals
(human_corr dropped from 0.301 to 0.235) while improving overall (0.627→0.769).

Sections:
  1. Label distributions (human vs synthetic)
  2. Per-module human journal performance
  3. Feature-label correlation divergence
  4. Combiner analysis
  5. Prediction error patterns
  6. Human weight sensitivity sweep

Usage:
    cd autoresearch-macos && uv run python diagnose_llm.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
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
    XGB_PARAMS,
    build_feature_matrix,
    get_cv_indices,
)

ARTIFACTS_DIR = Path(__file__).parent.parent / "src" / "mental_entropy" / "models" / "_artifacts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def text_histogram(values: np.ndarray, bins: int = 10, width: int = 40) -> str:
    """Create a text-based histogram."""
    counts, edges = np.histogram(values, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1
    lines = []
    for i, count in enumerate(counts):
        bar_len = int(count / max_count * width)
        bar = "█" * bar_len
        lines.append(f"    {edges[i]:5.1f}-{edges[i+1]:5.1f} | {bar} {count}")
    return "\n".join(lines)


def build_sample_weights_with_multiplier(
    sources: np.ndarray, human_multiplier: float
) -> np.ndarray:
    """Build sample weights with a custom human entry multiplier."""
    w = np.ones(len(sources), dtype=np.float64)
    w[sources == SOURCE_HUMAN] = human_multiplier
    return w


# ---------------------------------------------------------------------------
# Section 1: Label distributions
# ---------------------------------------------------------------------------


def section_label_distributions(
    labels: np.ndarray, sources: np.ndarray
) -> None:
    """Print label distribution analysis for human vs synthetic."""
    print("\n" + "=" * 70)
    print("SECTION 1: Label Distributions")
    print("=" * 70)

    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC
    human_labels = labels[human_mask]
    synth_labels = labels[synth_mask]

    print(f"\nHuman-written entries (n={len(human_labels)}):")
    print(f"  Mean: {human_labels.mean():.2f}  Std: {human_labels.std():.2f}  "
          f"Median: {float(np.median(human_labels)):.1f}  "
          f"Min: {human_labels.min():.0f}  Max: {human_labels.max():.0f}")
    print(text_histogram(human_labels, bins=10))

    print(f"\nSynthetic entries (n={len(synth_labels)}):")
    print(f"  Mean: {synth_labels.mean():.2f}  Std: {synth_labels.std():.2f}  "
          f"Median: {float(np.median(synth_labels)):.1f}  "
          f"Min: {synth_labels.min():.0f}  Max: {synth_labels.max():.0f}")
    print(text_histogram(synth_labels, bins=10))

    # Per-value counts
    print("\n  Per-value counts:")
    print(f"  {'Value':>5s}  {'Human':>5s}  {'Synth':>5s}  {'Total':>5s}")
    for v in range(1, 11):
        n_human = int((human_labels == v).sum())
        n_synth = int((synth_labels == v).sum())
        print(f"  {v:>5d}  {n_human:>5d}  {n_synth:>5d}  {n_human + n_synth:>5d}")


# ---------------------------------------------------------------------------
# Section 2: Per-module human journal performance
# ---------------------------------------------------------------------------


def section_per_module_performance(
    module_matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    sample_weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Analyze per-module OOF performance split by source."""
    print("\n" + "=" * 70)
    print("SECTION 2: Per-Module Performance (Human vs Synthetic)")
    print("=" * 70)

    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC
    n = len(labels)

    print(f"\n  {'Module':>6s}  {'CV_corr':>8s}  {'Human_corr':>10s}  {'Synth_corr':>10s}  "
          f"{'Gap':>8s}  {'Features':>8s}")
    print("  " + "-" * 60)

    for module in MODULE_NAMES:
        X = module_matrices[module]
        oof = np.zeros(n, dtype=np.float64)

        for train_idx, val_idx in folds:
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X[train_idx], labels[train_idx],
                      sample_weight=sample_weights[train_idx])
            oof[val_idx] = model.predict(X[val_idx])

        cv_corr = pearson_correlation(oof, labels, weights=sample_weights)
        human_corr = pearson_correlation(oof[human_mask], labels[human_mask])
        synth_corr = pearson_correlation(oof[synth_mask], labels[synth_mask])
        gap = human_corr - synth_corr

        print(f"  {module.upper():>6s}  {cv_corr:>8.4f}  {human_corr:>10.4f}  "
              f"{synth_corr:>10.4f}  {gap:>+8.4f}  {X.shape[1]:>8d}")


# ---------------------------------------------------------------------------
# Section 3: Feature-label correlation divergence
# ---------------------------------------------------------------------------


def section_feature_correlation_divergence(
    features: list[dict[str, float]],
    labels: np.ndarray,
    sources: np.ndarray,
    feature_names: list[str],
) -> None:
    """Find features where human and synthetic correlations diverge most."""
    print("\n" + "=" * 70)
    print("SECTION 3: Feature-Label Correlation Divergence")
    print("=" * 70)

    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC
    human_labels = labels[human_mask]
    synth_labels = labels[synth_mask]

    divergences = []

    for fname in feature_names:
        vals = np.array([f.get(fname, 0.0) for f in features], dtype=np.float64)

        human_vals = vals[human_mask]
        synth_vals = vals[synth_mask]

        r_human = pearson_correlation(human_vals, human_labels)
        r_synth = pearson_correlation(synth_vals, synth_labels)
        gap = abs(r_human - r_synth)

        # Determine module
        module = fname.split("_")[0]

        divergences.append({
            "feature": fname,
            "module": module,
            "r_human": r_human,
            "r_synth": r_synth,
            "gap": gap,
            "human_mean": float(human_vals.mean()),
            "synth_mean": float(synth_vals.mean()),
        })

    # Sort by gap descending
    divergences.sort(key=lambda x: x["gap"], reverse=True)

    print(f"\n  Top 25 most divergent features (|r_human - r_synth|):")
    print(f"  {'Feature':<35s}  {'Module':>6s}  {'r_Human':>8s}  {'r_Synth':>8s}  "
          f"{'Gap':>7s}  {'Hum_μ':>7s}  {'Syn_μ':>7s}")
    print("  " + "-" * 85)

    for d in divergences[:25]:
        print(f"  {d['feature']:<35s}  {d['module']:>6s}  {d['r_human']:>+7.3f}  "
              f"{d['r_synth']:>+8.3f}  {d['gap']:>7.3f}  "
              f"{d['human_mean']:>7.3f}  {d['synth_mean']:>7.3f}")

    # Per-module summary
    print(f"\n  Per-module mean divergence:")
    for module in MODULE_NAMES:
        mod_divs = [d["gap"] for d in divergences if d["module"] == module]
        if mod_divs:
            mean_gap = np.mean(mod_divs)
            max_gap = max(mod_divs)
            print(f"    {module.upper():>4s}: mean_gap={mean_gap:.3f}  "
                  f"max_gap={max_gap:.3f}  ({len(mod_divs)} features)")


# ---------------------------------------------------------------------------
# Section 4: Combiner analysis
# ---------------------------------------------------------------------------


def section_combiner_analysis(
    module_matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    sample_weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Analyze how the Ridge combiner interacts with per-source predictions."""
    print("\n" + "=" * 70)
    print("SECTION 4: Combiner Analysis")
    print("=" * 70)

    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC
    n = len(labels)

    # Build OOF subscores
    oof_subscores = np.zeros((n, len(MODULE_NAMES)), dtype=np.float64)

    for m_idx, module in enumerate(MODULE_NAMES):
        X = module_matrices[module]
        for train_idx, val_idx in folds:
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X[train_idx], labels[train_idx],
                      sample_weight=sample_weights[train_idx])
            oof_subscores[val_idx, m_idx] = model.predict(X[val_idx])

    # Train Ridge combiner on OOF subscores
    ridge = Ridge(alpha=1.0)
    ridge.fit(oof_subscores, labels, sample_weight=sample_weights)

    combined = ridge.predict(oof_subscores)

    print(f"\n  Current Ridge coefficients:")
    for m_idx, module in enumerate(MODULE_NAMES):
        print(f"    {module.upper():>4s}: {ridge.coef_[m_idx]:>+8.4f}")
    print(f"    {'INT':>4s}: {ridge.intercept_:>+8.4f}")

    # Leave-one-out module analysis
    print(f"\n  Leave-one-module-out analysis:")
    print(f"  {'Excluded':>10s}  {'CV_corr':>8s}  {'Human_corr':>10s}  {'Synth_corr':>10s}  "
          f"{'ΔHuman':>8s}")

    base_human = pearson_correlation(combined[human_mask], labels[human_mask])
    base_synth = pearson_correlation(combined[synth_mask], labels[synth_mask])
    base_cv = pearson_correlation(combined, labels, weights=sample_weights)

    print(f"  {'(none)':>10s}  {base_cv:>8.4f}  {base_human:>9.4f}  "
          f"{base_synth:>10.4f}  {'---':>8s}")

    for m_idx, module in enumerate(MODULE_NAMES):
        # Zero out this module's contribution
        modified = oof_subscores.copy()
        modified[:, m_idx] = 0.0
        ridge_loo = Ridge(alpha=1.0)
        ridge_loo.fit(modified, labels, sample_weight=sample_weights)
        preds_loo = ridge_loo.predict(modified)

        cv_loo = pearson_correlation(preds_loo, labels, weights=sample_weights)
        human_loo = pearson_correlation(preds_loo[human_mask], labels[human_mask])
        synth_loo = pearson_correlation(preds_loo[synth_mask], labels[synth_mask])
        delta_human = human_loo - base_human

        print(f"  {module.upper():>10s}  {cv_loo:>8.4f}  {human_loo:>9.4f}  "
              f"{synth_loo:>10.4f}  {delta_human:>+8.4f}")

    # Per-module subscore correlation with labels by source
    print(f"\n  Raw subscore correlation with labels:")
    print(f"  {'Module':>6s}  {'OOF_corr':>9s}  {'Human_corr':>10s}  {'Synth_corr':>10s}")
    for m_idx, module in enumerate(MODULE_NAMES):
        ss = oof_subscores[:, m_idx]
        cv = pearson_correlation(ss, labels, weights=sample_weights)
        human_c = pearson_correlation(ss[human_mask], labels[human_mask])
        synth_c = pearson_correlation(ss[synth_mask], labels[synth_mask])
        print(f"  {module.upper():>6s}  {cv:>9.4f}  {human_c:>9.4f}  {synth_c:>10.4f}")


# ---------------------------------------------------------------------------
# Section 5: Prediction error patterns
# ---------------------------------------------------------------------------


def section_prediction_errors(
    module_matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    sample_weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Analyze prediction error patterns for human-written entries."""
    print("\n" + "=" * 70)
    print("SECTION 5: Prediction Error Patterns (Human-written entries)")
    print("=" * 70)

    human_mask = sources == SOURCE_HUMAN
    n = len(labels)

    # Build full OOF pipeline
    oof_subscores = np.zeros((n, len(MODULE_NAMES)), dtype=np.float64)
    for m_idx, module in enumerate(MODULE_NAMES):
        X = module_matrices[module]
        for train_idx, val_idx in folds:
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X[train_idx], labels[train_idx],
                      sample_weight=sample_weights[train_idx])
            oof_subscores[val_idx, m_idx] = model.predict(X[val_idx])

    ridge = Ridge(alpha=1.0)
    ridge.fit(oof_subscores, labels, sample_weight=sample_weights)
    preds = ridge.predict(oof_subscores)

    # Human-only analysis
    human_preds = preds[human_mask]
    human_labels = labels[human_mask]
    human_errors = human_preds - human_labels

    print(f"\n  Human prediction error stats (n={len(human_labels)}):")
    print(f"    Mean error:  {human_errors.mean():>+.3f} (bias)")
    print(f"    Std error:   {human_errors.std():>.3f}")
    print(f"    MAE:         {np.abs(human_errors).mean():>.3f}")
    print(f"    RMSE:        {np.sqrt((human_errors ** 2).mean()):>.3f}")

    # By label bucket
    buckets = [(1, 3, "Low (1-3)"), (4, 6, "Mid (4-6)"), (7, 10, "High (7-10)")]
    print(f"\n  Error by label bucket:")
    print(f"  {'Bucket':<12s}  {'N':>4s}  {'Mean_err':>9s}  {'MAE':>7s}  "
          f"{'Pred_mean':>9s}  {'Label_mean':>10s}")
    print("  " + "-" * 60)

    for lo, hi, name in buckets:
        mask = (human_labels >= lo) & (human_labels <= hi)
        if mask.sum() == 0:
            continue
        bucket_errs = human_errors[mask]
        bucket_preds = human_preds[mask]
        bucket_labels = human_labels[mask]
        print(f"  {name:<12s}  {int(mask.sum()):>4d}  {bucket_errs.mean():>+9.3f}  "
              f"{np.abs(bucket_errs).mean():>7.3f}  {bucket_preds.mean():>9.2f}  "
              f"{bucket_labels.mean():>10.2f}")

    # Prediction range comparison
    print(f"\n  Prediction range:")
    print(f"    Human preds:  [{human_preds.min():.2f}, {human_preds.max():.2f}]  "
          f"mean={human_preds.mean():.2f}  std={human_preds.std():.2f}")
    print(f"    Human labels: [{human_labels.min():.0f}, {human_labels.max():.0f}]  "
          f"mean={human_labels.mean():.2f}  std={human_labels.std():.2f}")


# ---------------------------------------------------------------------------
# Section 6: Human weight sensitivity
# ---------------------------------------------------------------------------


def section_human_weight_sensitivity(
    module_matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
) -> None:
    """Sweep human multiplier values and show cv_corr vs human_corr tradeoff."""
    print("\n" + "=" * 70)
    print("SECTION 6: Human Weight Sensitivity Sweep")
    print("=" * 70)

    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC
    n = len(labels)

    multipliers = [1.0, 1.29, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]

    print(f"\n  {'Multiplier':>10s}  {'CV_corr':>8s}  {'Human_corr':>10s}  "
          f"{'Synth_corr':>10s}  {'Gap':>8s}")
    print("  " + "-" * 55)

    for mult in multipliers:
        w = np.ones(n, dtype=np.float64)
        w[sources == SOURCE_HUMAN] = mult

        folds = get_cv_indices(n)

        # Full pipeline: per-module OOF → Ridge
        oof_subscores = np.zeros((n, len(MODULE_NAMES)), dtype=np.float64)
        for m_idx, module in enumerate(MODULE_NAMES):
            X = module_matrices[module]
            for train_idx, val_idx in folds:
                model = XGBRegressor(**XGB_PARAMS)
                model.fit(X[train_idx], labels[train_idx], sample_weight=w[train_idx])
                oof_subscores[val_idx, m_idx] = model.predict(X[val_idx])

        # Ridge combiner
        fold_corrs = []
        for train_idx, val_idx in folds:
            ridge = Ridge(alpha=1.0)
            ridge.fit(oof_subscores[train_idx], labels[train_idx],
                      sample_weight=w[train_idx])
            p = ridge.predict(oof_subscores[val_idx])
            fold_corrs.append(pearson_correlation(p, labels[val_idx], weights=w[val_idx]))

        cv_corr = float(np.mean(fold_corrs))

        # OOF combined predictions for per-source analysis
        ridge_full = Ridge(alpha=1.0)
        ridge_full.fit(oof_subscores, labels, sample_weight=w)
        combined = ridge_full.predict(oof_subscores)

        human_corr = pearson_correlation(combined[human_mask], labels[human_mask])
        synth_corr = pearson_correlation(combined[synth_mask], labels[synth_mask])
        gap = human_corr - synth_corr

        marker = " ◄ current" if abs(mult - 1.29) < 0.05 else ""
        print(f"  {mult:>10.1f}  {cv_corr:>8.4f}  {human_corr:>9.4f}  "
              f"{synth_corr:>10.4f}  {gap:>+8.4f}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("MES SUBSCORE MODEL — HUMAN CORRELATION DIAGNOSTIC")
    print("=" * 70)

    features, labels, feature_names, sources = load_cached_data()
    n = len(features)
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())

    print(f"\nDataset: {n} entries ({n_human} human + {n_synth} synthetic)")

    # Build per-module matrices
    module_matrices: dict[str, np.ndarray] = {}
    for module in MODULE_NAMES:
        module_matrices[module] = build_feature_matrix(
            features, MODULE_FEATURE_KEYS[module]
        )

    # Current sample weights (n_synth/n_human ≈ 1.29)
    sample_weights = np.ones(n, dtype=np.float64)
    sample_weights[sources == SOURCE_HUMAN] = n_synth / n_human

    folds = get_cv_indices(n)

    # Run all sections
    section_label_distributions(labels, sources)
    section_per_module_performance(module_matrices, labels, sources, sample_weights, folds)
    section_feature_correlation_divergence(features, labels, sources, feature_names)
    section_combiner_analysis(module_matrices, labels, sources, sample_weights, folds)
    section_prediction_errors(module_matrices, labels, sources, sample_weights, folds)
    section_human_weight_sensitivity(module_matrices, labels, sources)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
