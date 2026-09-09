"""
Diagnose Human Error Patterns in MES Subscore Models.

Analyzes exactly where and why the current v4 model fails on human entries.
Identifies misleading features, missing signals, and label variance constraints.

Usage:
    cd autoresearch-macos && PYTHONPATH=../src ../.venv/bin/python -u diagnose_human_errors.py
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
    DIMENSION_NAMES,
    SOURCE_HUMAN,
    SOURCE_SYNTHETIC,
    TRANSFORM_DIR,
    TRANSFORM_DIR_NORM,
    TRANSFORM_INV,
    TRANSFORM_INV_CLIP,
    TRANSFORM_INV_NZ,
    apply_transform,
    compute_mes_scores,
    load_cached_data,
    load_dimension_labels,
    pearson_correlation,
)
from train_subscores import build_feature_matrix, get_cv_indices
from train_subscores_v3 import ALL_FEATURE_KEYS, XGB_PARAMS
from train_subscores_v4 import get_stratified_folds

# Linear baseline weights (from train.py)
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

W = 72  # print width


def section(title: str) -> None:
    print()
    print("=" * W)
    print(title)
    print("=" * W)


# ---------------------------------------------------------------------------
# Load data & generate OOF predictions
# ---------------------------------------------------------------------------

features, labels, feature_names, sources = load_cached_data()
dim_labels = load_dimension_labels()
X = build_feature_matrix(features, ALL_FEATURE_KEYS)
n = len(features)
human_mask = sources == SOURCE_HUMAN
synth_mask = sources == SOURCE_SYNTHETIC
n_human = int(human_mask.sum())
n_synth = int(synth_mask.sum())

print(f"Dataset: {n} entries ({n_human} human + {n_synth} synthetic)")
print(f"Features: {len(ALL_FEATURE_KEYS)}")

# Generate XGBoost v4 OOF predictions (same pipeline as v4 best config)
V4_XGB = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
}
V4_SYNTH_WEIGHT = 1.0
V4_RIDGE_ALPHA = 10.0

folds = get_stratified_folds(sources)

# Sample weights
w = np.ones(n, dtype=np.float64)
w[synth_mask] = V4_SYNTH_WEIGHT

# Stage 1: dimension OOF
print("\nGenerating OOF predictions (v4 config)...")
oof_subscores = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)
for d_idx, dim_name in enumerate(DIMENSION_NAMES):
    y_dim = dim_labels[dim_name]
    for train_idx, val_idx in folds:
        model = XGBRegressor(**V4_XGB)
        model.fit(X[train_idx], y_dim[train_idx], sample_weight=w[train_idx])
        oof_subscores[val_idx, d_idx] = model.predict(X[val_idx])

# Stage 2: Ridge combiner OOF
oof_combined = np.zeros(n, dtype=np.float64)
for train_idx, val_idx in folds:
    ridge = Ridge(alpha=V4_RIDGE_ALPHA)
    ridge.fit(oof_subscores[train_idx], labels[train_idx], sample_weight=w[train_idx])
    oof_combined[val_idx] = ridge.predict(oof_subscores[val_idx])

xgb_error = oof_combined - labels  # signed error
xgb_abs_error = np.abs(xgb_error)

# Linear baseline predictions (0-100 scale, convert to 1-10)
linear_scores = compute_mes_scores(LINEAR_WEIGHTS, features)
linear_pred_10 = linear_scores / 100.0 * 9.0 + 1.0  # scale 0-100 -> 1-10
linear_error = linear_pred_10 - labels
linear_abs_error = np.abs(linear_error)

print(f"XGBoost human_corr: {pearson_correlation(oof_combined[human_mask], labels[human_mask]):.4f}")
print(f"Linear  human_corr: {pearson_correlation(linear_pred_10[human_mask], labels[human_mask]):.4f}")


# =========================================================================
# Section 1: Label Distribution
# =========================================================================

section("1. LABEL DISTRIBUTION (human vs synthetic)")

for name, mask in [("Human", human_mask), ("Synthetic", synth_mask), ("All", np.ones(n, dtype=bool))]:
    y = labels[mask]
    pcts = np.percentile(y, [10, 25, 50, 75, 90])
    print(f"\n  {name} (n={len(y)}):")
    print(f"    mean={y.mean():.3f}  std={y.std():.3f}  min={y.min():.2f}  max={y.max():.2f}")
    print(f"    p10={pcts[0]:.2f}  p25={pcts[1]:.2f}  p50={pcts[2]:.2f}  p75={pcts[3]:.2f}  p90={pcts[4]:.2f}")
    print(f"    effective_range (p90-p10) = {pcts[4] - pcts[0]:.2f}")

# Per-dimension distributions
print(f"\n  Per-dimension label stats (human only):")
print(f"  {'Dimension':<30s} {'mean':>6s} {'std':>6s} {'range':>6s}")
print(f"  {'-'*52}")
for dim_name in DIMENSION_NAMES:
    y = dim_labels[dim_name][human_mask]
    print(f"  {dim_name:<30s} {y.mean():6.3f} {y.std():6.3f} {y.max()-y.min():6.2f}")


# =========================================================================
# Section 2: Worst Predictions
# =========================================================================

section("2. WORST PREDICTIONS (human entries)")

h_indices = np.where(human_mask)[0]
h_err = xgb_abs_error[h_indices]
worst_order = np.argsort(-h_err)

KEY_FEATS = ["ce_adj_p75", "ne_fragment_sentence_rate", "ce_n_blocks",
             "se_dominant_cluster_frac", "cle_length_cv"]

print(f"\n  Top 25 human entries by |error|:")
print(f"  {'Idx':>5s} {'Label':>6s} {'XGB':>6s} {'Err':>7s} {'Lin':>6s} {'LinErr':>7s}  {' | '.join(f[:8] for f in KEY_FEATS)}")
print(f"  {'-'*85}")

for rank in range(25):
    h_pos = worst_order[rank]
    idx = h_indices[h_pos]
    lab = labels[idx]
    xpred = oof_combined[idx]
    xerr = xgb_error[idx]
    lpred = linear_pred_10[idx]
    lerr = linear_error[idx]
    feat_vals = [features[idx].get(f, 0.0) for f in KEY_FEATS]
    feat_str = " | ".join(f"{v:8.3f}" for v in feat_vals)
    print(f"  {idx:>5d} {lab:6.2f} {xpred:6.2f} {xerr:+7.3f} {lpred:6.2f} {lerr:+7.3f}  {feat_str}")

# Bias analysis
h_top_quartile = h_indices[worst_order[:n_human // 4]]
bias_top = np.mean(xgb_error[h_top_quartile])
bias_all = np.mean(xgb_error[h_indices])
print(f"\n  Mean signed error (all human): {bias_all:+.4f}")
print(f"  Mean signed error (worst 25%): {bias_top:+.4f}")
print(f"  {'→ Systematically OVER-predicting' if bias_top > 0.3 else '→ Systematically UNDER-predicting' if bias_top < -0.3 else '→ No systematic bias'}")

# Error distribution
for threshold in [0.5, 1.0, 1.5, 2.0, 3.0]:
    frac = np.mean(xgb_abs_error[h_indices] > threshold)
    print(f"  |error| > {threshold:.1f}: {frac*100:.1f}% of human entries")


# =========================================================================
# Section 3: Feature × Error Correlation
# =========================================================================

section("3. FEATURE × ERROR CORRELATION (human entries)")

print(f"\n  {'Feature':<35s} {'|err|_corr':>10s} {'err_corr':>10s} {'lab_corr':>10s} {'Flag':>10s}")
print(f"  {'-'*75}")

feat_error_analysis = []
for j, feat_name in enumerate(ALL_FEATURE_KEYS):
    h_feat = X[h_indices, j]
    h_abs_err = xgb_abs_error[h_indices]
    h_signed_err = xgb_error[h_indices]
    h_lab = labels[h_indices]

    corr_abs = pearson_correlation(h_feat, h_abs_err)
    corr_signed = pearson_correlation(h_feat, h_signed_err)
    corr_label = pearson_correlation(h_feat, h_lab)

    # MISLEADING: error correlation stronger than label correlation
    is_misleading = abs(corr_abs) > abs(corr_label) and abs(corr_abs) > 0.05
    # MISSING SIGNAL: decent label correlation but low XGB use
    is_missing = abs(corr_label) > 0.15

    feat_error_analysis.append((feat_name, corr_abs, corr_signed, corr_label, is_misleading, is_missing))

# Sort by |error| correlation (most misleading first)
feat_error_analysis.sort(key=lambda x: -abs(x[1]))

for feat_name, corr_abs, corr_signed, corr_label, is_misleading, is_missing in feat_error_analysis:
    flag = "MISLEADING" if is_misleading else "signal" if is_missing else ""
    print(f"  {feat_name:<35s} {corr_abs:+10.4f} {corr_signed:+10.4f} {corr_label:+10.4f} {flag:>10s}")

n_misleading = sum(1 for x in feat_error_analysis if x[4])
n_signal = sum(1 for x in feat_error_analysis if x[5])
print(f"\n  {n_misleading} MISLEADING features (error corr > label corr)")
print(f"  {n_signal} features with |label_corr| > 0.15 (useful signal)")

# Top misleading
misleading = [(f, ca, cs, cl) for f, ca, cs, cl, m, _ in feat_error_analysis if m]
misleading.sort(key=lambda x: -abs(x[1]))
if misleading:
    print(f"\n  Top 10 MISLEADING features:")
    for f, ca, cs, cl in misleading[:10]:
        print(f"    {f:<33s} |err|_corr={ca:+.4f}  label_corr={cl:+.4f}")


# =========================================================================
# Section 4: Linear vs XGBoost Error Comparison
# =========================================================================

section("4. LINEAR vs XGBOOST ERROR COMPARISON (human entries)")

# Per-entry comparison
xgb_wins = xgb_abs_error[h_indices] < linear_abs_error[h_indices]
linear_wins = linear_abs_error[h_indices] < xgb_abs_error[h_indices]
ties = ~xgb_wins & ~linear_wins

print(f"\n  XGBoost wins: {xgb_wins.sum():>5d} ({xgb_wins.mean()*100:.1f}%)")
print(f"  Linear wins:  {linear_wins.sum():>5d} ({linear_wins.mean()*100:.1f}%)")
print(f"  Ties:         {ties.sum():>5d} ({ties.mean()*100:.1f}%)")

print(f"\n  Mean |error|: XGBoost={xgb_abs_error[h_indices].mean():.4f}  Linear={linear_abs_error[h_indices].mean():.4f}")

# Entries where XGBoost is much worse than linear ("damaged" entries)
damage = xgb_abs_error[h_indices] - linear_abs_error[h_indices]
damage_order = np.argsort(-damage)
damaged_h_indices = h_indices[damage_order[:n_human // 10]]  # top 10% most damaged

print(f"\n  'XGBoost-damaged' entries (top 10% where XGB >> Linear):")
print(f"  n={len(damaged_h_indices)}, mean XGB error={xgb_abs_error[damaged_h_indices].mean():.3f}, mean Linear error={linear_abs_error[damaged_h_indices].mean():.3f}")

# Feature patterns of damaged entries vs healthy entries
healthy_h_indices = h_indices[damage_order[-n_human // 4:]]  # bottom 25% (XGB does well)

print(f"\n  Feature differences (damaged vs healthy):")
print(f"  {'Feature':<35s} {'Damaged':>10s} {'Healthy':>10s} {'Δ':>10s}")
print(f"  {'-'*65}")

feature_diffs = []
for j, feat_name in enumerate(ALL_FEATURE_KEYS):
    dam_mean = X[damaged_h_indices, j].mean()
    hlt_mean = X[healthy_h_indices, j].mean()
    delta = dam_mean - hlt_mean
    feature_diffs.append((feat_name, dam_mean, hlt_mean, delta))

feature_diffs.sort(key=lambda x: -abs(x[3]))
for feat_name, dam, hlt, delta in feature_diffs[:15]:
    print(f"  {feat_name:<35s} {dam:10.4f} {hlt:10.4f} {delta:+10.4f}")


# =========================================================================
# Section 5: Per-Dimension Breakdown
# =========================================================================

section("5. PER-DIMENSION HUMAN CORRELATION")

print(f"\n  How well does each dimension predict human labels?")
print(f"  {'Dimension':<30s} {'OOF_corr':>10s} {'Human_corr':>12s} {'Synth_corr':>12s} {'Gap':>8s}")
print(f"  {'-'*72}")

for d_idx, dim_name in enumerate(DIMENSION_NAMES):
    y_dim = dim_labels[dim_name]
    oof_dim = oof_subscores[:, d_idx]

    all_corr = pearson_correlation(oof_dim, y_dim)
    h_corr = pearson_correlation(oof_dim[human_mask], y_dim[human_mask])
    s_corr = pearson_correlation(oof_dim[synth_mask], y_dim[synth_mask])
    gap = h_corr - s_corr

    print(f"  {dim_name:<30s} {all_corr:10.4f} {h_corr:12.4f} {s_corr:12.4f} {gap:+8.4f}")


# =========================================================================
# Section 6: Summary
# =========================================================================

section("6. SUMMARY & RECOMMENDATIONS")

h_labels = labels[human_mask]
h_eff_range = float(np.percentile(h_labels, 90) - np.percentile(h_labels, 10))
s_labels = labels[synth_mask]
s_eff_range = float(np.percentile(s_labels, 90) - np.percentile(s_labels, 10))

print(f"\n  Label variance constraint:")
print(f"    Human effective range (p90-p10):     {h_eff_range:.2f} / 9.0")
print(f"    Synthetic effective range (p90-p10):  {s_eff_range:.2f} / 9.0")
if h_eff_range < 3.0:
    print(f"    ⚠ Human labels cluster in narrow band — correlation mathematically bounded")
    print(f"    → Maximum achievable corr is constrained by low label variance")
else:
    print(f"    ✓ Human labels have reasonable spread")

print(f"\n  Top misleading features (remove or dampen):")
for f, ca, cs, cl in misleading[:5]:
    print(f"    {f:<33s} |err|_corr={ca:+.4f}")

top_signal = [(f, cl) for f, _, _, cl, _, ms in feat_error_analysis if ms]
top_signal.sort(key=lambda x: -abs(x[1]))
print(f"\n  Top signal features (human label_corr):")
for f, cl in top_signal[:10]:
    print(f"    {f:<33s} label_corr={cl:+.4f}")

# Damaged entry characterization
print(f"\n  XGBoost-damaged entries are characterized by:")
for feat_name, dam, hlt, delta in feature_diffs[:5]:
    direction = "higher" if delta > 0 else "lower"
    print(f"    {direction} {feat_name} (Δ={delta:+.4f})")

print()
print("=" * W)
print("DIAGNOSIS COMPLETE")
print("=" * W)
