"""
MES Subscore Model Training Pipeline v3 — Dimension-Aligned.

Restructures the subscore architecture from module-aligned (CE/SE/NE/CLE/BC)
to dimension-aligned (continuity/topic_focus/contradiction_integration/
cognitive_clarity/narrative_closure).

Key differences from v1/v2:
- Each dimension model uses ALL 77 features (not just one module's features)
- Each dimension model is trained against its rubric dimension label (1-5 scale)
- The Ridge combiner maps 5 dimension subscores → overall_entropy (1-10 scale)

This fixes the architecture misalignment discovered in diagnose_dimensions.py:
- 3/5 modules were predicting the wrong rubric dimension
- Cognitive clarity had almost no feature coverage under module-aligned approach

Usage:
    cd autoresearch-macos && uv run python train_subscores_v3.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path(__file__).parent.parent / "src" / "mental_entropy" / "models" / "_artifacts"

# All 82 feature names in canonical sorted order
ALL_FEATURE_KEYS = sorted(
    [
        # CE (29)
        "ce_adj_max", "ce_adj_mean", "ce_adj_median", "ce_adj_min", "ce_adj_p10",
        "ce_adj_p25", "ce_adj_p75", "ce_adj_p90", "ce_adj_std", "ce_break_count",
        "ce_break_rate", "ce_drop_mean", "ce_drop_min", "ce_drop_std",
        "ce_inter_block_break_count", "ce_inter_block_break_rate",
        "ce_intra_block_break_count", "ce_intra_block_break_rate",
        "ce_longest_coherent_run", "ce_low_mass", "ce_low_mass_sum", "ce_n_adj",
        "ce_n_blocks", "ce_n_sentences", "ce_sharp_drop_count", "ce_sharp_drop_rate",
        "ce_skip_break_count", "ce_skip_mean", "ce_skip_min",
        # SE (9)
        "se_cluster_entropy", "se_dominant_cluster_frac", "se_inter_mean",
        "se_intra_mean", "se_n_clusters", "se_switch_count", "se_switch_mean_jump",
        "se_switch_rate", "se_switch_weighted",
        # NE (18)
        "ne_arc_linearity", "ne_connectors_rate", "ne_consolidation_delta",
        "ne_end_abandonment_cue", "ne_end_closure_cue", "ne_end_fragment",
        "ne_end_open_question", "ne_end_similarity", "ne_end_to_centroid",
        "ne_fragment_sentence_rate", "ne_last_sentence_reflective",
        "ne_punct_break_rate", "ne_reflection_rate", "ne_semantic_wander",
        "ne_start_end_sim", "ne_start_to_centroid", "ne_temporal_jump_rate",
        "ne_temporal_markers_rate",
        # CLE (14)
        "cle_adj_low_frac", "cle_adj_sim_cv", "cle_adj_sim_range", "cle_adj_sim_std",
        "cle_fragment_rate", "cle_hedge_rate", "cle_length_cv",
        "cle_punctuation_noise", "cle_repetition_score", "cle_restart_rate",
        "cle_resume_rate", "cle_semantic_break_rate", "cle_semantic_isolated_rate",
        "cle_zigzag_rate",
        # BC (12)
        "bc_belief_sentence_count", "bc_belief_sentence_rate",
        "bc_conflict_pair_count", "bc_conflict_rate", "bc_conflict_span_max",
        "bc_conflict_span_mean", "bc_integrated_count", "bc_integration_rate",
        "bc_max_conflict_sim", "bc_mean_conflict_sim", "bc_unresolved_count",
        "bc_unresolved_rate",
    ]
)

# XGBoost hyperparameters — conservative for 82 features
XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)


# ---------------------------------------------------------------------------
# Stage 1: Per-dimension OOF predictions
# ---------------------------------------------------------------------------


def train_dimension_oof(
    dim_name: str,
    X: np.ndarray,
    y_dim: np.ndarray,
    sample_weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    sources: np.ndarray,
) -> tuple[np.ndarray, XGBRegressor]:
    """Train a per-dimension XGBoost and return OOF predictions + final model.

    Args:
        dim_name: Dimension name (for logging).
        X: Full feature matrix (n_samples, 77).
        y_dim: Dimension labels (1-5 scale).
        sample_weights: Per-sample weights.
        folds: List of (train_idx, val_idx) tuples.
        sources: Source array for per-source correlation reporting.

    Returns:
        (oof_preds, final_model)
    """
    n = len(y_dim)
    oof_preds = np.zeros(n, dtype=np.float64)
    fold_corrs = []

    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X[train_idx], y_dim[train_idx]
        X_val, y_val = X[val_idx], y_dim[val_idx]
        w_train = sample_weights[train_idx]
        w_val = sample_weights[val_idx]

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=w_train)

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        fold_corr = pearson_correlation(preds, y_val, weights=w_val)
        fold_corrs.append(fold_corr)

    cv_corr = float(np.mean(fold_corrs))
    cv_std = float(np.std(fold_corrs))

    # Per-source correlations on OOF predictions
    h_corr = pearson_correlation(oof_preds[human_mask], y_dim[human_mask]) if human_mask.sum() > 2 else 0.0
    s_corr = pearson_correlation(oof_preds[synth_mask], y_dim[synth_mask]) if synth_mask.sum() > 2 else 0.0

    # Final model trained on all data
    final_model = XGBRegressor(**XGB_PARAMS)
    final_model.fit(X, y_dim, sample_weight=sample_weights)

    print(f"  {dim_name:<28s} cv={cv_corr:.4f}±{cv_std:.3f}  human={h_corr:.4f}  synth={s_corr:.4f}")

    return oof_preds, final_model


# ---------------------------------------------------------------------------
# Stage 2: Ridge combiner (dimension subscores → overall_entropy)
# ---------------------------------------------------------------------------


def train_combiner(
    oof_subscores: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[Ridge, float, float]:
    """Train Ridge combiner on OOF dimension subscores → overall_entropy."""
    fold_corrs = []

    for train_idx, val_idx in folds:
        X_train, y_train = oof_subscores[train_idx], y[train_idx]
        X_val, y_val = oof_subscores[val_idx], y[val_idx]
        w_train = sample_weights[train_idx]
        w_val = sample_weights[val_idx]

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train, sample_weight=w_train)

        preds = ridge.predict(X_val)
        fold_corr = pearson_correlation(preds, y_val, weights=w_val)
        fold_corrs.append(fold_corr)

    cv_corr = float(np.mean(fold_corrs))
    cv_std = float(np.std(fold_corrs))

    # Final combiner on all data
    final_ridge = Ridge(alpha=1.0)
    final_ridge.fit(oof_subscores, y, sample_weight=sample_weights)

    return final_ridge, cv_corr, cv_std


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------


def save_artifacts(
    dim_models: dict[str, XGBRegressor],
    combiner: Ridge,
    metrics: dict[str, float],
    n_entries: int,
    n_human: int,
    n_synthetic: int,
) -> None:
    """Save dimension-aligned models and combiner."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old module-aligned model files
    for old_name in ["ce_model.json", "se_model.json", "ne_model.json",
                     "cle_model.json", "bc_model.json", "feature_filter.json"]:
        old_path = ARTIFACTS_DIR / old_name
        if old_path.exists():
            old_path.unlink()
            print(f"  Removed old {old_name}")

    # Save per-dimension XGBoost models
    for dim_name, model in dim_models.items():
        path = ARTIFACTS_DIR / f"{dim_name}_model.json"
        model.save_model(str(path))
        print(f"  Saved {path.name}")

    # Save combiner
    combiner_data = {
        "coef": combiner.coef_.tolist(),
        "intercept": float(combiner.intercept_),
        "dimension_names": list(DIMENSION_NAMES),
        "architecture": "dimension_aligned_v3",
    }
    combiner_path = ARTIFACTS_DIR / "combiner.json"
    with open(combiner_path, "w") as f:
        json.dump(combiner_data, f, indent=2)
    print(f"  Saved {combiner_path.name}")

    # Save manifest
    manifest = {
        "architecture": "dimension_aligned_v3",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_entries": n_entries,
        "n_human": n_human,
        "n_synthetic": n_synthetic,
        "n_folds": N_FOLDS,
        "xgb_params": XGB_PARAMS,
        "ridge_alpha": 1.0,
        "dimension_names": list(DIMENSION_NAMES),
        "n_features_per_dimension": len(ALL_FEATURE_KEYS),
        "feature_names": ALL_FEATURE_KEYS,
        "combiner_coef": combiner.coef_.tolist(),
        "combiner_intercept": float(combiner.intercept_),
        "metrics": metrics,
    }
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved {manifest_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()

    # Load data
    features, labels, feature_names, sources = load_cached_data()
    dim_labels = load_dimension_labels()
    sample_weights = build_sample_weights(sources)
    n = len(features)
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())

    print("=" * 72)
    print("MES SUBSCORE MODEL TRAINING v3 — DIMENSION-ALIGNED")
    print("=" * 72)
    print(f"\nDataset: {n} entries ({n_human} human + {n_synth} synthetic)")
    print(f"Dimensions: {len(DIMENSION_NAMES)} ({', '.join(DIMENSION_NAMES)})")
    print(f"Features per dimension: {len(ALL_FEATURE_KEYS)} (all features)")
    print()

    # Build full feature matrix (n_samples × 77)
    X = build_feature_matrix(features, ALL_FEATURE_KEYS)

    # CV folds
    folds = get_cv_indices(n)

    # Stage 1: Per-dimension XGBoost models
    print("Stage 1: Per-dimension XGBoost models (5-fold OOF)")
    print("-" * 72)

    oof_subscores = np.zeros((n, len(DIMENSION_NAMES)), dtype=np.float64)
    dim_models: dict[str, XGBRegressor] = {}

    for d_idx, dim_name in enumerate(DIMENSION_NAMES):
        y_dim = dim_labels[dim_name]
        oof_preds, final_model = train_dimension_oof(
            dim_name, X, y_dim, sample_weights, folds, sources,
        )
        oof_subscores[:, d_idx] = oof_preds
        dim_models[dim_name] = final_model

    # Stage 2: Ridge combiner (dimension subscores → overall_entropy)
    print()
    print("Stage 2: Ridge combiner (dimension subscores → overall_entropy)")
    print("-" * 72)

    combiner, combiner_cv, combiner_std = train_combiner(
        oof_subscores, labels, sample_weights, folds,
    )

    print(f"  Combiner cv_corr: {combiner_cv:.4f} ± {combiner_std:.3f}")
    coef_dict = dict(zip(DIMENSION_NAMES, combiner.coef_.round(4)))
    print(f"  Coefficients: {coef_dict}")
    print(f"  Intercept: {combiner.intercept_:.4f}")

    # OOF per-source correlations
    oof_combined = combiner.predict(oof_subscores)
    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC
    oof_human_corr = pearson_correlation(oof_combined[human_mask], labels[human_mask]) if human_mask.sum() > 2 else 0.0
    oof_synth_corr = pearson_correlation(oof_combined[synth_mask], labels[synth_mask]) if synth_mask.sum() > 2 else 0.0

    print(f"\n  OOF human_corr: {oof_human_corr:.4f}  (n={n_human})")
    print(f"  OOF synth_corr: {oof_synth_corr:.4f}  (n={n_synth})")

    # Compare with v1 (module-aligned) and linear baseline
    print()
    print("Comparison")
    print("-" * 72)

    from prepare import evaluate_weights
    from train import WEIGHTS as LINEAR_WEIGHTS
    linear_results = evaluate_weights(LINEAR_WEIGHTS, features, labels, sources=sources)

    print(f"  Linear (13 feat):       cv_corr={linear_results['cv_corr']:.4f}  "
          f"human={linear_results.get('human_corr', 0):.4f}  "
          f"synth={linear_results.get('synth_corr', 0):.4f}")
    print(f"  Dimension-aligned (v3): cv_corr={combiner_cv:.4f}  "
          f"human={oof_human_corr:.4f}  "
          f"synth={oof_synth_corr:.4f}")

    cv_gap = combiner_cv - linear_results["cv_corr"]
    human_gap = oof_human_corr - linear_results.get("human_corr", 0)
    print(f"\n  cv_corr improvement: {cv_gap:+.4f} ({cv_gap / linear_results['cv_corr'] * 100:+.1f}%)")
    print(f"  human_corr change:  {human_gap:+.4f}")

    # Save artifacts
    print()
    print("Saving artifacts")
    print("-" * 72)

    metrics = {
        "combiner_cv_corr": combiner_cv,
        "combiner_cv_std": combiner_std,
        "oof_human_corr": oof_human_corr,
        "oof_synth_corr": oof_synth_corr,
        "linear_cv_corr": linear_results["cv_corr"],
        "linear_human_corr": linear_results.get("human_corr", 0),
        "cv_corr_improvement": cv_gap,
        "human_corr_change": human_gap,
    }

    save_artifacts(
        dim_models, combiner, metrics,
        n_entries=n, n_human=n_human, n_synthetic=n_synth,
    )

    elapsed = time.time() - start
    print()
    print("=" * 72)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Artifacts saved to {ARTIFACTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
