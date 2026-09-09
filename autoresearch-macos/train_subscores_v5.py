"""
MES Subscore Model Training Pipeline v5 — Multi-Architecture Search.

Tests 5 architectures to find the best approach for human_corr:
  A. Ridge-82 (all features, no interactions)
  B. Lasso → XGBoost (human-selected features)
  C. Human-validated XGBoost (early stopping on human entries)
  D. Ensemble (linear baseline + XGBoost blend)
  E. Shallow XGBoost (max_depth 1-2, minimal interactions)

Usage:
    cd autoresearch-macos && PYTHONPATH=../src ../.venv/bin/python -u train_subscores_v5.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor

from prepare import (
    DIMENSION_NAMES,
    N_FOLDS,
    SOURCE_HUMAN,
    SOURCE_SYNTHETIC,
    compute_mes_scores,
    load_cached_data,
    load_dimension_labels,
    pearson_correlation,
)
from train_subscores import build_feature_matrix
from train_subscores_v3 import ALL_FEATURE_KEYS, ARTIFACTS_DIR
from train_subscores_v4 import compute_v4_objective, get_stratified_folds

# Linear baseline weights (copied from train.py to avoid circular import)
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

W = 72


def section(title: str) -> None:
    print()
    print("=" * W)
    print(title)
    print("=" * W)


def make_result(
    arch: str,
    cv_corr: float,
    oof_human_corr: float,
    oof_synth_corr: float,
    fold_human_corr_std: float,
    **params: float | int | str,
) -> dict:
    """Create a standardized result dict."""
    r = {
        "architecture": arch,
        "cv_corr": cv_corr,
        "oof_human_corr": oof_human_corr,
        "oof_synth_corr": oof_synth_corr,
        "fold_human_corr_std": fold_human_corr_std,
    }
    r.update(params)
    r["v4_score"] = compute_v4_objective(r)
    return r


# ---------------------------------------------------------------------------
# Architecture A: Ridge-82
# ---------------------------------------------------------------------------


def sweep_ridge82(
    X: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    human_mask: np.ndarray,
    synth_mask: np.ndarray,
) -> list[dict]:
    """Ridge regression on all 82 features per dimension → Ridge combiner."""
    results = []
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    combiner_alphas = [0.1, 1.0, 10.0]

    for alpha, c_alpha in itertools.product(alphas, combiner_alphas):
        n = len(labels)
        oof_subscores = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)

        for d_idx, dim_name in enumerate(DIMENSION_NAMES):
            y_dim = dim_labels[dim_name]
            for train_idx, val_idx in folds:
                model = Ridge(alpha=alpha)
                model.fit(X[train_idx], y_dim[train_idx])
                oof_subscores[val_idx, d_idx] = model.predict(X[val_idx])

        # Ridge combiner with per-fold human_corr tracking
        fold_cv = []
        fold_human = []
        for train_idx, val_idx in folds:
            ridge = Ridge(alpha=c_alpha)
            ridge.fit(oof_subscores[train_idx], labels[train_idx])
            preds = ridge.predict(oof_subscores[val_idx])
            fold_cv.append(pearson_correlation(preds, labels[val_idx]))
            h_val = human_mask[val_idx]
            if h_val.sum() > 5:
                fold_human.append(pearson_correlation(preds[h_val], labels[val_idx][h_val]))

        # Full OOF for global metrics
        ridge_full = Ridge(alpha=c_alpha)
        ridge_full.fit(oof_subscores, labels)
        combined = ridge_full.predict(oof_subscores)

        r = make_result(
            "ridge82",
            cv_corr=float(np.mean(fold_cv)),
            oof_human_corr=pearson_correlation(combined[human_mask], labels[human_mask]),
            oof_synth_corr=pearson_correlation(combined[synth_mask], labels[synth_mask]),
            fold_human_corr_std=float(np.std(fold_human)) if fold_human else 0.5,
            dim_alpha=alpha,
            combiner_alpha=c_alpha,
        )
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Architecture B: Lasso → XGBoost
# ---------------------------------------------------------------------------


def sweep_lasso_xgb(
    X: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    human_mask: np.ndarray,
    synth_mask: np.ndarray,
) -> list[dict]:
    """Lasso feature selection on human data → XGBoost on selected features."""
    results = []

    lasso_alphas = [0.005, 0.01, 0.05, 0.1]
    xgb_configs = [
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.05},
        {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.05},
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.05},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.1},
    ]
    synth_weights = [0.0, 0.25, 0.5]

    for la in lasso_alphas:
        # Feature selection using Lasso on human data
        selected_features = set()
        for dim_name in DIMENSION_NAMES:
            y_dim = dim_labels[dim_name][human_mask]
            lasso = Lasso(alpha=la, max_iter=5000)
            lasso.fit(X[human_mask], y_dim)
            nonzero = np.where(np.abs(lasso.coef_) > 1e-6)[0]
            selected_features.update(nonzero)

        n_sel = len(selected_features)
        if n_sel < 3:
            continue

        sel_idx = sorted(selected_features)
        X_sel = X[:, sel_idx]

        for xgb_cfg in xgb_configs:
            for sw in synth_weights:
                n = len(labels)
                w = np.ones(n, dtype=np.float64)
                w[synth_mask] = sw

                oof_subscores = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)
                for d_idx, dim_name in enumerate(DIMENSION_NAMES):
                    y_dim = dim_labels[dim_name]
                    for train_idx, val_idx in folds:
                        xgb_params = {**xgb_cfg, "subsample": 0.8,
                                      "colsample_bytree": 0.8, "random_state": 42, "verbosity": 0}
                        model = XGBRegressor(**xgb_params)
                        model.fit(X_sel[train_idx], y_dim[train_idx], sample_weight=w[train_idx])
                        oof_subscores[val_idx, d_idx] = model.predict(X_sel[val_idx])

                fold_cv = []
                fold_human = []
                for train_idx, val_idx in folds:
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(oof_subscores[train_idx], labels[train_idx], sample_weight=w[train_idx])
                    preds = ridge.predict(oof_subscores[val_idx])
                    fold_cv.append(pearson_correlation(preds, labels[val_idx]))
                    h_val = human_mask[val_idx]
                    if h_val.sum() > 5:
                        fold_human.append(pearson_correlation(preds[h_val], labels[val_idx][h_val]))

                ridge_full = Ridge(alpha=1.0)
                ridge_full.fit(oof_subscores, labels, sample_weight=w)
                combined = ridge_full.predict(oof_subscores)

                r = make_result(
                    "lasso_xgb",
                    cv_corr=float(np.mean(fold_cv)),
                    oof_human_corr=pearson_correlation(combined[human_mask], labels[human_mask]),
                    oof_synth_corr=pearson_correlation(combined[synth_mask], labels[synth_mask]),
                    fold_human_corr_std=float(np.std(fold_human)) if fold_human else 0.5,
                    lasso_alpha=la,
                    n_selected=n_sel,
                    synth_weight=sw,
                    **xgb_cfg,
                )
                results.append(r)

    return results


# ---------------------------------------------------------------------------
# Architecture C: Human-validated XGBoost
# ---------------------------------------------------------------------------


def sweep_human_validated(
    X: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    human_mask: np.ndarray,
    synth_mask: np.ndarray,
) -> list[dict]:
    """XGBoost with early stopping on human-only validation set."""
    results = []

    configs = list(itertools.product(
        [2, 3],                # max_depth
        [200, 500],            # n_estimators
        [0.01, 0.05],         # learning_rate
        [0.0, 0.25, 0.5],    # synth_weight
    ))

    for max_d, n_est, lr, sw in configs:
        n = len(labels)
        w = np.ones(n, dtype=np.float64)
        w[synth_mask] = sw

        oof_subscores = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)

        for d_idx, dim_name in enumerate(DIMENSION_NAMES):
            y_dim = dim_labels[dim_name]
            for train_idx, val_idx in folds:
                xgb_params = {
                    "n_estimators": n_est,
                    "max_depth": max_d,
                    "learning_rate": lr,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "verbosity": 0,
                    "early_stopping_rounds": 20,
                }
                model = XGBRegressor(**xgb_params)

                # Human-only eval set from validation fold
                val_human = val_idx[human_mask[val_idx]]
                if len(val_human) > 5:
                    model.fit(
                        X[train_idx], y_dim[train_idx],
                        sample_weight=w[train_idx],
                        eval_set=[(X[val_human], y_dim[val_human])],
                        verbose=False,
                    )
                else:
                    # Fallback to regular training
                    model.set_params(early_stopping_rounds=None)
                    model.fit(X[train_idx], y_dim[train_idx], sample_weight=w[train_idx])

                oof_subscores[val_idx, d_idx] = model.predict(X[val_idx])

        fold_cv = []
        fold_human = []
        for train_idx, val_idx in folds:
            ridge = Ridge(alpha=1.0)
            ridge.fit(oof_subscores[train_idx], labels[train_idx], sample_weight=w[train_idx])
            preds = ridge.predict(oof_subscores[val_idx])
            fold_cv.append(pearson_correlation(preds, labels[val_idx]))
            h_val = human_mask[val_idx]
            if h_val.sum() > 5:
                fold_human.append(pearson_correlation(preds[h_val], labels[val_idx][h_val]))

        ridge_full = Ridge(alpha=1.0)
        ridge_full.fit(oof_subscores, labels, sample_weight=w)
        combined = ridge_full.predict(oof_subscores)

        r = make_result(
            "human_validated",
            cv_corr=float(np.mean(fold_cv)),
            oof_human_corr=pearson_correlation(combined[human_mask], labels[human_mask]),
            oof_synth_corr=pearson_correlation(combined[synth_mask], labels[synth_mask]),
            fold_human_corr_std=float(np.std(fold_human)) if fold_human else 0.5,
            max_depth=max_d,
            n_estimators=n_est,
            learning_rate=lr,
            synth_weight=sw,
        )
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Architecture D: Ensemble (linear + XGBoost blend)
# ---------------------------------------------------------------------------


def sweep_ensemble(
    X: np.ndarray,
    features: list[dict[str, float]],
    dim_labels: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    human_mask: np.ndarray,
    synth_mask: np.ndarray,
) -> list[dict]:
    """Blend linear baseline with XGBoost predictions."""
    results = []

    # Linear baseline predictions (0-100 → 1-10)
    linear_scores = compute_mes_scores(LINEAR_WEIGHTS, features)
    linear_pred = linear_scores / 100.0 * 9.0 + 1.0

    # XGBoost v4 config OOF predictions
    xgb_configs = [
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.05},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.1},
        {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.1},
    ]
    synth_weights = [0.5, 0.75, 1.0]
    blend_alphas = [0.3, 0.4, 0.5, 0.6, 0.7]

    for xgb_cfg, sw in itertools.product(xgb_configs, synth_weights):
        n = len(labels)
        w = np.ones(n, dtype=np.float64)
        w[synth_mask] = sw

        # Generate XGBoost OOF
        oof_subscores = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)
        for d_idx, dim_name in enumerate(DIMENSION_NAMES):
            y_dim = dim_labels[dim_name]
            for train_idx, val_idx in folds:
                xgb_params = {**xgb_cfg, "subsample": 0.8,
                              "colsample_bytree": 0.8, "random_state": 42, "verbosity": 0}
                model = XGBRegressor(**xgb_params)
                model.fit(X[train_idx], y_dim[train_idx], sample_weight=w[train_idx])
                oof_subscores[val_idx, d_idx] = model.predict(X[val_idx])

        # Ridge combiner for XGBoost
        ridge = Ridge(alpha=1.0)
        ridge.fit(oof_subscores, labels, sample_weight=w)
        xgb_pred = ridge.predict(oof_subscores)

        # Blend with different alphas
        for alpha in blend_alphas:
            blended = alpha * linear_pred + (1 - alpha) * xgb_pred

            # Per-fold metrics (use pre-defined folds for consistency)
            fold_cv = []
            fold_human = []
            for _, val_idx in folds:
                fold_cv.append(pearson_correlation(blended[val_idx], labels[val_idx]))
                h_val = human_mask[val_idx]
                if h_val.sum() > 5:
                    fold_human.append(pearson_correlation(blended[val_idx][h_val], labels[val_idx][h_val]))

            r = make_result(
                "ensemble",
                cv_corr=float(np.mean(fold_cv)),
                oof_human_corr=pearson_correlation(blended[human_mask], labels[human_mask]),
                oof_synth_corr=pearson_correlation(blended[synth_mask], labels[synth_mask]),
                fold_human_corr_std=float(np.std(fold_human)) if fold_human else 0.5,
                blend_alpha=alpha,
                synth_weight=sw,
                **xgb_cfg,
            )
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Architecture E: Shallow XGBoost
# ---------------------------------------------------------------------------


def sweep_shallow_xgb(
    X: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    human_mask: np.ndarray,
    synth_mask: np.ndarray,
) -> list[dict]:
    """Shallow XGBoost (depth 1-2) to minimize interaction overfitting."""
    results = []

    configs = list(itertools.product(
        [1, 2],                   # max_depth
        [100, 200, 500],          # n_estimators
        [0.01, 0.05, 0.1],      # learning_rate
        [0.0, 0.25, 0.5, 1.0],  # synth_weight
        [0.1, 1.0, 10.0],       # ridge_alpha
    ))

    for max_d, n_est, lr, sw, r_alpha in configs:
        n = len(labels)
        w = np.ones(n, dtype=np.float64)
        w[synth_mask] = sw

        oof_subscores = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)
        for d_idx, dim_name in enumerate(DIMENSION_NAMES):
            y_dim = dim_labels[dim_name]
            for train_idx, val_idx in folds:
                xgb_params = {
                    "n_estimators": n_est, "max_depth": max_d,
                    "learning_rate": lr, "subsample": 0.8,
                    "colsample_bytree": 0.8, "random_state": 42, "verbosity": 0,
                }
                model = XGBRegressor(**xgb_params)
                model.fit(X[train_idx], y_dim[train_idx], sample_weight=w[train_idx])
                oof_subscores[val_idx, d_idx] = model.predict(X[val_idx])

        fold_cv = []
        fold_human = []
        for train_idx, val_idx in folds:
            ridge = Ridge(alpha=r_alpha)
            ridge.fit(oof_subscores[train_idx], labels[train_idx], sample_weight=w[train_idx])
            preds = ridge.predict(oof_subscores[val_idx])
            fold_cv.append(pearson_correlation(preds, labels[val_idx]))
            h_val = human_mask[val_idx]
            if h_val.sum() > 5:
                fold_human.append(pearson_correlation(preds[h_val], labels[val_idx][h_val]))

        ridge_full = Ridge(alpha=r_alpha)
        ridge_full.fit(oof_subscores, labels, sample_weight=w)
        combined = ridge_full.predict(oof_subscores)

        r = make_result(
            "shallow_xgb",
            cv_corr=float(np.mean(fold_cv)),
            oof_human_corr=pearson_correlation(combined[human_mask], labels[human_mask]),
            oof_synth_corr=pearson_correlation(combined[synth_mask], labels[synth_mask]),
            fold_human_corr_std=float(np.std(fold_human)) if fold_human else 0.5,
            max_depth=max_d,
            n_estimators=n_est,
            learning_rate=lr,
            synth_weight=sw,
            ridge_alpha=r_alpha,
        )
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Save best artifacts (if XGBoost-based)
# ---------------------------------------------------------------------------


def save_best_artifacts(
    X: np.ndarray,
    dim_labels: dict[str, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    best: dict,
    n_human: int,
    n_synth: int,
) -> dict:
    """Retrain best config on all data, save artifacts."""
    arch = best["architecture"]
    n = len(labels)
    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC

    # Only save XGBoost-based or ensemble architectures (registry compatible)
    if arch not in ("shallow_xgb", "human_validated", "lasso_xgb", "ensemble"):
        print(f"\n  Architecture '{arch}' not registry-compatible — skipping artifact save.")
        print(f"  (Would need _registry.py update to deploy)")
        return {"oof_human_corr": best["oof_human_corr"], "oof_synth_corr": best["oof_synth_corr"],
                "cv_corr": best["cv_corr"]}

    sw = best.get("synth_weight", 1.0)
    r_alpha = best.get("ridge_alpha", 1.0)
    max_d = best.get("max_depth", 3)
    n_est = best.get("n_estimators", 200)
    lr = best.get("learning_rate", 0.1)

    xgb_params = {
        "n_estimators": n_est, "max_depth": max_d,
        "learning_rate": lr, "subsample": 0.8,
        "colsample_bytree": 0.8, "random_state": 42, "verbosity": 0,
    }

    w = np.ones(n, dtype=np.float64)
    w[synth_mask] = sw

    # Handle Lasso feature selection
    if arch == "lasso_xgb":
        la = best.get("lasso_alpha", 0.01)
        selected = set()
        for dim_name in DIMENSION_NAMES:
            y_dim = dim_labels[dim_name][human_mask]
            lasso = Lasso(alpha=la, max_iter=5000)
            lasso.fit(X[human_mask], y_dim)
            nonzero = np.where(np.abs(lasso.coef_) > 1e-6)[0]
            selected.update(nonzero)
        sel_idx = sorted(selected)
        X_train = X[:, sel_idx]
        sel_feature_names = [ALL_FEATURE_KEYS[i] for i in sel_idx]
        print(f"\n  Lasso selected {len(sel_idx)} features (alpha={la})")
    else:
        X_train = X
        sel_feature_names = ALL_FEATURE_KEYS

    print(f"\n  Retraining: arch={arch}, synth_weight={sw}, max_depth={max_d}, "
          f"n_estimators={n_est}, lr={lr}, ridge_alpha={r_alpha}")

    # OOF + final models
    oof_subscores = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)
    dim_models: dict[str, XGBRegressor] = {}

    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        y_dim = dim_labels[dim_name]

        for train_idx, val_idx in folds:
            model = XGBRegressor(**xgb_params)
            if arch == "human_validated":
                val_human = val_idx[human_mask[val_idx]]
                if len(val_human) > 5:
                    model.set_params(early_stopping_rounds=20)
                    model.fit(
                        X_train[train_idx], y_dim[train_idx],
                        sample_weight=w[train_idx],
                        eval_set=[(X_train[val_human], y_dim[val_human])],
                        verbose=False,
                    )
                else:
                    model.fit(X_train[train_idx], y_dim[train_idx], sample_weight=w[train_idx])
            else:
                model.fit(X_train[train_idx], y_dim[train_idx], sample_weight=w[train_idx])
            oof_subscores[val_idx, d_idx] = model.predict(X_train[val_idx])

        final_model = XGBRegressor(**xgb_params)
        if arch == "human_validated":
            # For final model, use all human data as eval set
            final_model.set_params(early_stopping_rounds=20)
            final_model.fit(
                X_train, y_dim, sample_weight=w,
                eval_set=[(X_train[human_mask], y_dim[human_mask])],
                verbose=False,
            )
        else:
            final_model.fit(X_train, y_dim, sample_weight=w)
        dim_models[dim_name] = final_model

        h_corr = pearson_correlation(oof_subscores[human_mask, d_idx], y_dim[human_mask])
        s_corr = pearson_correlation(oof_subscores[synth_mask, d_idx], y_dim[synth_mask])
        print(f"    {dim_name:<28s} human={h_corr:.4f}  synth={s_corr:.4f}")

    combiner = Ridge(alpha=r_alpha)
    combiner.fit(oof_subscores, labels, sample_weight=w)
    oof_combined = combiner.predict(oof_subscores)

    oof_human_corr = pearson_correlation(oof_combined[human_mask], labels[human_mask])
    oof_synth_corr = pearson_correlation(oof_combined[synth_mask], labels[synth_mask])
    print(f"\n  Final OOF human_corr: {oof_human_corr:.4f}")
    print(f"  Final OOF synth_corr: {oof_synth_corr:.4f}")

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for dim_name, model in dim_models.items():
        path = ARTIFACTS_DIR / f"{dim_name}_model.json"
        model.save_model(str(path))
        print(f"  Saved {path.name}")

    # Architecture marker depends on type
    if arch == "ensemble":
        arch_marker = "ensemble_v5"
    else:
        arch_marker = "dimension_aligned_v3"

    combiner_data = {
        "coef": combiner.coef_.tolist(),
        "intercept": float(combiner.intercept_),
        "dimension_names": list(DIMENSION_NAMES),
        "architecture": arch_marker,
    }
    if arch == "ensemble":
        combiner_data["blend_alpha"] = best.get("blend_alpha", 0.4)
    with open(ARTIFACTS_DIR / "combiner.json", "w") as f:
        json.dump(combiner_data, f, indent=2)
    print(f"  Saved combiner.json (architecture={arch_marker})")

    manifest = {
        "architecture": arch_marker,
        "training_script": f"v5_{arch}",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_entries": n,
        "n_human": n_human,
        "n_synthetic": n_synth,
        "n_folds": N_FOLDS,
        "xgb_params": xgb_params,
        "synth_weight": sw,
        "ridge_alpha": r_alpha,
        "dimension_names": list(DIMENSION_NAMES),
        "n_features_per_dimension": len(sel_feature_names),
        "feature_names": sel_feature_names,
        "combiner_coef": combiner.coef_.tolist(),
        "combiner_intercept": float(combiner.intercept_),
        "blend_alpha": best.get("blend_alpha") if arch == "ensemble" else None,
        "metrics": {
            "combiner_cv_corr": best["cv_corr"],
            "oof_human_corr": oof_human_corr,
            "oof_synth_corr": oof_synth_corr,
            "fold_human_corr_std": best["fold_human_corr_std"],
            "v4_objective_score": best["v4_score"],
        },
    }
    with open(ARTIFACTS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest.json")

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

    features, labels, feature_names, sources = load_cached_data()
    dim_labels = load_dimension_labels()
    n = len(features)
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())
    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC

    print("=" * W)
    print("MES SUBSCORE MODEL TRAINING v5 — MULTI-ARCHITECTURE")
    print("=" * W)
    print(f"\nDataset: {n} entries ({n_human} human + {n_synth} synthetic)")
    print(f"Features: {len(ALL_FEATURE_KEYS)}")

    X = build_feature_matrix(features, ALL_FEATURE_KEYS)
    folds = get_stratified_folds(sources)

    common = (X, dim_labels, labels, sources, folds, human_mask, synth_mask)

    all_results: list[dict] = []

    # Architecture A: Ridge-82
    section("A. RIDGE-82 (all features, no interactions)")
    t0 = time.time()
    results_a = sweep_ridge82(*common)
    all_results.extend(results_a)
    best_a = max(results_a, key=lambda r: r["v4_score"])
    print(f"  {len(results_a)} configs in {time.time()-t0:.1f}s")
    print(f"  Best: human={best_a['oof_human_corr']:.4f} cv={best_a['cv_corr']:.4f} "
          f"obj={best_a['v4_score']:.4f}")

    # Architecture B: Lasso → XGBoost
    section("B. LASSO → XGBOOST (human-selected features)")
    t0 = time.time()
    results_b = sweep_lasso_xgb(*common)
    all_results.extend(results_b)
    if results_b:
        best_b = max(results_b, key=lambda r: r["v4_score"])
        print(f"  {len(results_b)} configs in {time.time()-t0:.1f}s")
        print(f"  Best: human={best_b['oof_human_corr']:.4f} cv={best_b['cv_corr']:.4f} "
              f"obj={best_b['v4_score']:.4f} n_feat={best_b['n_selected']}")
    else:
        print(f"  No valid configs (Lasso selected too few features)")

    # Architecture C: Human-validated XGBoost
    section("C. HUMAN-VALIDATED XGBOOST (early stop on human)")
    t0 = time.time()
    results_c = sweep_human_validated(*common)
    all_results.extend(results_c)
    best_c = max(results_c, key=lambda r: r["v4_score"])
    print(f"  {len(results_c)} configs in {time.time()-t0:.1f}s")
    print(f"  Best: human={best_c['oof_human_corr']:.4f} cv={best_c['cv_corr']:.4f} "
          f"obj={best_c['v4_score']:.4f}")

    # Architecture D: Ensemble
    section("D. ENSEMBLE (linear + XGBoost blend)")
    t0 = time.time()
    results_d = sweep_ensemble(X, features, dim_labels, labels, sources, folds, human_mask, synth_mask)
    all_results.extend(results_d)
    best_d = max(results_d, key=lambda r: r["v4_score"])
    print(f"  {len(results_d)} configs in {time.time()-t0:.1f}s")
    print(f"  Best: human={best_d['oof_human_corr']:.4f} cv={best_d['cv_corr']:.4f} "
          f"obj={best_d['v4_score']:.4f} blend={best_d.get('blend_alpha', '?')}")

    # Architecture E: Shallow XGBoost
    section("E. SHALLOW XGBOOST (depth 1-2)")
    t0 = time.time()
    results_e = sweep_shallow_xgb(*common)
    all_results.extend(results_e)
    best_e = max(results_e, key=lambda r: r["v4_score"])
    print(f"  {len(results_e)} configs in {time.time()-t0:.1f}s")
    print(f"  Best: human={best_e['oof_human_corr']:.4f} cv={best_e['cv_corr']:.4f} "
          f"obj={best_e['v4_score']:.4f}")

    # Overall comparison
    section("OVERALL TOP-15 (all architectures)")

    all_results.sort(key=lambda r: r["v4_score"], reverse=True)

    print(f"\n{'#':>3s} {'arch':<18s} {'obj':>6s} {'cv':>6s} {'human':>7s} {'synth':>7s} {'h_std':>6s}  params")
    print("-" * W)

    for rank, r in enumerate(all_results[:15]):
        arch = r["architecture"]
        params_str = ""
        if arch == "ridge82":
            params_str = f"α={r.get('dim_alpha')},cα={r.get('combiner_alpha')}"
        elif arch == "lasso_xgb":
            params_str = f"lα={r.get('lasso_alpha')},n={r.get('n_selected')},sw={r.get('synth_weight')}"
        elif arch == "ensemble":
            params_str = f"blend={r.get('blend_alpha')},md={r.get('max_depth')},sw={r.get('synth_weight')}"
        elif arch in ("shallow_xgb", "human_validated"):
            params_str = f"md={r.get('max_depth')},ne={r.get('n_estimators')},sw={r.get('synth_weight')}"

        print(f"  {rank+1:>2d} {arch:<18s} {r['v4_score']:6.4f} {r['cv_corr']:6.3f} "
              f"{r['oof_human_corr']:7.4f} {r['oof_synth_corr']:7.4f} "
              f"{r['fold_human_corr_std']:6.3f}  {params_str}")

    # Per-architecture best summary
    section("BEST PER ARCHITECTURE")

    v4_human = 0.2962  # from v4 manifest
    v4_cv = 0.8237
    linear_human = 0.3013

    print(f"\n{'Architecture':<22s} {'human_corr':>12s} {'cv_corr':>10s} {'Δ_vs_v4':>10s} {'Δ_vs_lin':>10s}")
    print("-" * W)
    print(f"  {'Linear (13 feat)':<20s} {linear_human:12.4f} {'0.6270':>10s} {'':>10s} {'baseline':>10s}")
    print(f"  {'XGBoost v4':<20s} {v4_human:12.4f} {v4_cv:10.4f} {'baseline':>10s} {v4_human-linear_human:+10.4f}")

    arch_bests = {}
    for r in all_results:
        a = r["architecture"]
        if a not in arch_bests or r["v4_score"] > arch_bests[a]["v4_score"]:
            arch_bests[a] = r

    for arch_name in ["ridge82", "lasso_xgb", "human_validated", "ensemble", "shallow_xgb"]:
        if arch_name in arch_bests:
            b = arch_bests[arch_name]
            d_v4 = b["oof_human_corr"] - v4_human
            d_lin = b["oof_human_corr"] - linear_human
            print(f"  {arch_name:<20s} {b['oof_human_corr']:12.4f} {b['cv_corr']:10.4f} "
                  f"{d_v4:+10.4f} {d_lin:+10.4f}")

    # Best overall
    best = all_results[0]
    print(f"\n  🏆 Best overall: {best['architecture']} "
          f"(human_corr={best['oof_human_corr']:.4f}, cv_corr={best['cv_corr']:.4f})")

    # Save if better than v4
    if best["oof_human_corr"] > v4_human + 0.005:
        section("SAVING BEST ARTIFACTS")
        save_best_artifacts(X, dim_labels, labels, sources, folds, best, n_human, n_synth)
    else:
        print(f"\n  Best human_corr ({best['oof_human_corr']:.4f}) not meaningfully better "
              f"than v4 ({v4_human:.4f}) — NOT saving artifacts.")

    elapsed = time.time() - start
    print()
    print("=" * W)
    print(f"v5 search complete in {elapsed:.1f}s ({len(all_results)} total configs)")
    print("=" * W)


if __name__ == "__main__":
    main()
