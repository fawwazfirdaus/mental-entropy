"""
Diagnose Human vs Synthetic Gap in MES Subscore Models.

Analyzes why XGBoost dimension-aligned models have lower human_corr (0.282)
than the simple 13-feature linear model (0.301), despite much higher cv_corr.

Sections:
1. Feature distribution divergence (human vs synthetic)
2. XGBoost feature importance vs actual human correlation
3. Per-fold human_corr analysis (is it a few bad folds?)
4. Human-only model experiment (what if we train on human data only?)
5. Feature ablation (which features hurt human_corr when included?)
6. Synthetic weight sensitivity (does reducing synthetic influence help?)

Usage:
    cd autoresearch-macos && PYTHONPATH=../src ../.venv/bin/python diagnose_human_synth_gap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
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
from train_subscores import (
    build_feature_matrix,
    build_sample_weights,
    get_cv_indices,
)
from train_subscores_v3 import ALL_FEATURE_KEYS, XGB_PARAMS

W = 72  # print width


def section(title: str) -> None:
    print()
    print("=" * W)
    print(title)
    print("=" * W)


# ---------------------------------------------------------------------------
# Load data
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

# ---------------------------------------------------------------------------
# Section 1: Feature distribution divergence
# ---------------------------------------------------------------------------

section("1. FEATURE DISTRIBUTION DIVERGENCE (Human vs Synthetic)")

print(f"\n{'Feature':<35s} {'H_mean':>8s} {'S_mean':>8s} {'Δ_mean':>8s} {'KS_stat':>8s} {'KS_p':>10s}")
print("-" * W)

divergences = []
for j, feat_name in enumerate(ALL_FEATURE_KEYS):
    h_vals = X[human_mask, j]
    s_vals = X[synth_mask, j]

    h_mean = np.mean(h_vals)
    s_mean = np.mean(s_vals)
    delta = s_mean - h_mean

    # KS test for distribution difference
    ks_stat, ks_p = stats.ks_2samp(h_vals, s_vals)

    divergences.append((feat_name, h_mean, s_mean, delta, ks_stat, ks_p))

# Sort by KS statistic (most divergent first)
divergences.sort(key=lambda x: -x[4])

for feat_name, h_mean, s_mean, delta, ks_stat, ks_p in divergences[:25]:
    sig = "***" if ks_p < 0.001 else "**" if ks_p < 0.01 else "*" if ks_p < 0.05 else ""
    print(f"  {feat_name:<33s} {h_mean:8.4f} {s_mean:8.4f} {delta:+8.4f} {ks_stat:8.4f} {ks_p:10.2e} {sig}")

print(f"\n  (showing top 25 of {len(divergences)} by KS statistic)")
n_sig = sum(1 for _, _, _, _, _, p in divergences if p < 0.001)
print(f"  {n_sig} features have p < 0.001 (highly divergent distributions)")

# ---------------------------------------------------------------------------
# Section 2: XGBoost feature importance vs human correlation
# ---------------------------------------------------------------------------

section("2. XGBOOST IMPORTANCE vs HUMAN LABEL CORRELATION")

# Train full model on all data for each dimension, get feature importances
print(f"\n{'Feature':<35s} {'XGB_imp':>8s} {'H_corr':>8s} {'S_corr':>8s} {'Gap':>8s} {'KS':>8s}")
print("-" * W)

# Aggregate importances across all 5 dimension models
total_importance = np.zeros(len(ALL_FEATURE_KEYS))
dim_importances = {}

for dim_name in DIMENSION_NAMES:
    y_dim = dim_labels[dim_name]
    model = XGBRegressor(**XGB_PARAMS)
    w = build_sample_weights(sources)
    model.fit(X, y_dim, sample_weight=w)
    imp = model.feature_importances_
    total_importance += imp
    dim_importances[dim_name] = imp

total_importance /= len(DIMENSION_NAMES)  # average importance

# Compute per-feature correlation with overall label for human vs synthetic
feat_analysis = []
for j, feat_name in enumerate(ALL_FEATURE_KEYS):
    xgb_imp = total_importance[j]
    h_corr = pearson_correlation(X[human_mask, j], labels[human_mask])
    s_corr = pearson_correlation(X[synth_mask, j], labels[synth_mask])
    gap = abs(h_corr) - abs(s_corr)  # positive = more useful for humans

    # Get KS divergence for this feature
    ks_stat = [d[4] for d in divergences if d[0] == feat_name][0]

    feat_analysis.append((feat_name, xgb_imp, h_corr, s_corr, gap, ks_stat))

# Sort by XGBoost importance (highest first)
feat_analysis.sort(key=lambda x: -x[1])

for feat_name, xgb_imp, h_corr, s_corr, gap, ks_stat in feat_analysis[:30]:
    flag = " ← SUSPECT" if xgb_imp > 0.01 and ks_stat > 0.15 and gap < -0.05 else ""
    print(f"  {feat_name:<33s} {xgb_imp:8.4f} {h_corr:+8.4f} {s_corr:+8.4f} {gap:+8.4f} {ks_stat:8.3f}{flag}")

print(f"\n  SUSPECT = high XGB importance + high distribution divergence + weaker on humans")

# Identify problematic features: high importance, divergent, weaker for humans
suspects = [(f, imp, hc, sc, g, ks) for f, imp, hc, sc, g, ks in feat_analysis
            if imp > 0.01 and ks > 0.10 and g < -0.03]
suspects.sort(key=lambda x: x[5], reverse=True)

if suspects:
    print(f"\n  {len(suspects)} suspect features (high importance + divergent + weaker on humans):")
    for f, imp, hc, sc, g, ks in suspects:
        print(f"    {f:<33s} imp={imp:.4f} h_corr={hc:+.4f} s_corr={sc:+.4f} KS={ks:.3f}")

# ---------------------------------------------------------------------------
# Section 3: Per-fold human_corr analysis
# ---------------------------------------------------------------------------

section("3. PER-FOLD HUMAN CORRELATION ANALYSIS")

folds = get_cv_indices(n)
sample_weights = build_sample_weights(sources)

# Train full v3 pipeline per fold, report human_corr per fold
oof_subscores = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)

for d_idx, dim_name in enumerate(DIMENSION_NAMES):
    y_dim = dim_labels[dim_name]
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X[train_idx], y_dim[train_idx], sample_weight=sample_weights[train_idx])
        oof_subscores[val_idx, d_idx] = model.predict(X[val_idx])

# Ridge combiner
combiner = Ridge(alpha=1.0)
combiner.fit(oof_subscores, labels, sample_weight=sample_weights)
oof_combined = combiner.predict(oof_subscores)

print(f"\n{'Fold':<8s} {'n_total':>8s} {'n_human':>8s} {'human_corr':>12s} {'synth_corr':>12s} {'all_corr':>10s}")
print("-" * W)

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    val_h = human_mask[val_idx]
    val_s = synth_mask[val_idx]
    n_h = int(val_h.sum())
    n_s = int(val_s.sum())

    h_corr = pearson_correlation(oof_combined[val_idx][val_h], labels[val_idx][val_h]) if n_h > 2 else 0.0
    s_corr = pearson_correlation(oof_combined[val_idx][val_s], labels[val_idx][val_s]) if n_s > 2 else 0.0
    a_corr = pearson_correlation(oof_combined[val_idx], labels[val_idx])

    print(f"  {fold_idx:<6d} {len(val_idx):>8d} {n_h:>8d} {h_corr:>12.4f} {s_corr:>12.4f} {a_corr:>10.4f}")

overall_h = pearson_correlation(oof_combined[human_mask], labels[human_mask])
overall_s = pearson_correlation(oof_combined[synth_mask], labels[synth_mask])
print(f"\n  Overall: human_corr={overall_h:.4f}  synth_corr={overall_s:.4f}")

# ---------------------------------------------------------------------------
# Section 4: Human-only model experiment
# ---------------------------------------------------------------------------

section("4. HUMAN-ONLY MODEL EXPERIMENT")

print("\nTraining XGBoost on human data only (964 entries, 5-fold CV):")

X_human = X[human_mask]
y_human = labels[human_mask]
dim_labels_human = {k: v[human_mask] for k, v in dim_labels.items()}

# Human-only folds
human_folds = get_cv_indices(n_human, seed=42)

oof_sub_human = np.zeros((n_human, len(DIMENSION_NAMES)), dtype=np.float64)

print(f"\n{'Dimension':<30s} {'cv_corr':>10s}")
print("-" * 50)

for d_idx, dim_name in enumerate(DIMENSION_NAMES):
    y_dim = dim_labels_human[dim_name]
    fold_corrs = []
    for train_idx, val_idx in human_folds:
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_human[train_idx], y_dim[train_idx])
        preds = model.predict(X_human[val_idx])
        oof_sub_human[val_idx, d_idx] = preds
        fold_corrs.append(pearson_correlation(preds, y_dim[val_idx]))
    cv = float(np.mean(fold_corrs))
    print(f"  {dim_name:<28s} {cv:>10.4f}")

# Ridge combiner on human-only OOF
combiner_h = Ridge(alpha=1.0)
oof_combined_h = np.zeros(n_human, dtype=np.float64)
fold_corrs_h = []
for train_idx, val_idx in human_folds:
    ridge = Ridge(alpha=1.0)
    ridge.fit(oof_sub_human[train_idx], y_human[train_idx])
    preds = ridge.predict(oof_sub_human[val_idx])
    oof_combined_h[val_idx] = preds
    fold_corrs_h.append(pearson_correlation(preds, y_human[val_idx]))

human_only_cv = float(np.mean(fold_corrs_h))
print(f"\n  Human-only combiner cv_corr: {human_only_cv:.4f}")
print(f"  Mixed (human+synth) human_corr: {overall_h:.4f}")
print(f"  Linear baseline human_corr:     0.3013")
print(f"\n  Δ human-only vs mixed: {human_only_cv - overall_h:+.4f}")

# ---------------------------------------------------------------------------
# Section 5: Synthetic weight sensitivity
# ---------------------------------------------------------------------------

section("5. SYNTHETIC WEIGHT SENSITIVITY")

print("\nTesting different synthetic-to-human weight ratios:")
print(f"\n{'Synth_weight':>14s} {'cv_corr':>10s} {'human_corr':>12s} {'synth_corr':>12s}")
print("-" * W)

for synth_w in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.29, 2.0]:
    # Custom weights
    w = np.ones(n, dtype=np.float64)
    w[human_mask] = 1.0
    w[synth_mask] = synth_w

    oof_sub = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)

    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        y_dim = dim_labels[dim_name]
        for train_idx, val_idx in folds:
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X[train_idx], y_dim[train_idx], sample_weight=w[train_idx])
            oof_sub[val_idx, d_idx] = model.predict(X[val_idx])

    # Ridge combiner
    ridge = Ridge(alpha=1.0)
    ridge.fit(oof_sub, labels, sample_weight=w)
    oof_pred = ridge.predict(oof_sub)

    cv = pearson_correlation(oof_pred, labels, weights=w)
    hc = pearson_correlation(oof_pred[human_mask], labels[human_mask])
    sc = pearson_correlation(oof_pred[synth_mask], labels[synth_mask]) if synth_w > 0 else 0.0

    marker = " ← current" if abs(synth_w - 1.29) < 0.05 else ""
    print(f"  {synth_w:>12.2f} {cv:>10.4f} {hc:>12.4f} {sc:>12.4f}{marker}")

# ---------------------------------------------------------------------------
# Section 6: Feature group ablation
# ---------------------------------------------------------------------------

section("6. FEATURE GROUP ABLATION (impact on human_corr)")

print("\nRemoving one feature module at a time to measure human_corr impact:")
print(f"\n{'Removed module':<20s} {'n_feat':>8s} {'human_corr':>12s} {'Δ':>8s}")
print("-" * 60)

modules = {
    "CE": [f for f in ALL_FEATURE_KEYS if f.startswith("ce_")],
    "SE": [f for f in ALL_FEATURE_KEYS if f.startswith("se_")],
    "NE": [f for f in ALL_FEATURE_KEYS if f.startswith("ne_")],
    "CLE": [f for f in ALL_FEATURE_KEYS if f.startswith("cle_")],
    "BC": [f for f in ALL_FEATURE_KEYS if f.startswith("bc_")],
}

baseline_hcorr = overall_h

for mod_name, mod_feats in modules.items():
    # Features without this module
    kept = [f for f in ALL_FEATURE_KEYS if f not in mod_feats]
    X_abl = build_feature_matrix(features, kept)

    oof_sub_abl = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)
    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        y_dim = dim_labels[dim_name]
        for train_idx, val_idx in folds:
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X_abl[train_idx], y_dim[train_idx], sample_weight=sample_weights[train_idx])
            oof_sub_abl[val_idx, d_idx] = model.predict(X_abl[val_idx])

    ridge = Ridge(alpha=1.0)
    ridge.fit(oof_sub_abl, labels, sample_weight=sample_weights)
    oof_pred_abl = ridge.predict(oof_sub_abl)
    hc_abl = pearson_correlation(oof_pred_abl[human_mask], labels[human_mask])
    delta = hc_abl - baseline_hcorr

    flag = " ← IMPROVES" if delta > 0.005 else ""
    print(f"  -{mod_name:<18s} {len(kept):>8d} {hc_abl:>12.4f} {delta:+8.4f}{flag}")

# Also test removing most-divergent features
print(f"\n  Baseline (all 82):  human_corr={baseline_hcorr:.4f}")

# Remove top 10 most divergent features
top_divergent = [d[0] for d in divergences[:10]]
kept = [f for f in ALL_FEATURE_KEYS if f not in top_divergent]
X_abl = build_feature_matrix(features, kept)

oof_sub_abl = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)
for d_idx, dim_name in enumerate(DIMENSION_NAMES):
    y_dim = dim_labels[dim_name]
    for train_idx, val_idx in folds:
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_abl[train_idx], y_dim[train_idx], sample_weight=sample_weights[train_idx])
        oof_sub_abl[val_idx, d_idx] = model.predict(X_abl[val_idx])

ridge = Ridge(alpha=1.0)
ridge.fit(oof_sub_abl, labels, sample_weight=sample_weights)
oof_pred_abl = ridge.predict(oof_sub_abl)
hc_abl = pearson_correlation(oof_pred_abl[human_mask], labels[human_mask])
delta = hc_abl - baseline_hcorr

print(f"  -top10_divergent:   human_corr={hc_abl:.4f}  Δ={delta:+.4f}")
print(f"  Removed: {', '.join(top_divergent)}")

# Remove suspect features
if suspects:
    suspect_names = [s[0] for s in suspects]
    kept = [f for f in ALL_FEATURE_KEYS if f not in suspect_names]
    X_abl = build_feature_matrix(features, kept)

    oof_sub_abl = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)
    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        y_dim = dim_labels[dim_name]
        for train_idx, val_idx in folds:
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X_abl[train_idx], y_dim[train_idx], sample_weight=sample_weights[train_idx])
            oof_sub_abl[val_idx, d_idx] = model.predict(X_abl[val_idx])

    ridge = Ridge(alpha=1.0)
    ridge.fit(oof_sub_abl, labels, sample_weight=sample_weights)
    oof_pred_abl = ridge.predict(oof_sub_abl)
    hc_abl = pearson_correlation(oof_pred_abl[human_mask], labels[human_mask])
    delta = hc_abl - baseline_hcorr

    print(f"  -suspects ({len(suspect_names)}):      human_corr={hc_abl:.4f}  Δ={delta:+.4f}")

print()
print("=" * W)
print("DIAGNOSIS COMPLETE")
print("=" * W)
