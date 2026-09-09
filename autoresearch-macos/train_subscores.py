"""
MES Subscore Model Training Pipeline.

Trains per-module XGBoost regressors (CE, SE, NE, CLE, BC) with stacked
generalization and a Ridge combiner. Saves model artifacts to
src/mental_entropy/models/_artifacts/ for production inference.

Usage:
    cd autoresearch-macos && uv run python train_subscores.py

Requires: xgboost, scikit-learn (install with `pip install mental-entropy[models]`)
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
    N_FOLDS,
    SOURCE_HUMAN,
    SOURCE_SYNTHETIC,
    load_cached_data,
    pearson_correlation,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path(__file__).parent.parent / "src" / "mental_entropy" / "models" / "_artifacts"

# ---------------------------------------------------------------------------
# Feature-group mapping (must match _registry.py)
# ---------------------------------------------------------------------------

# Feature key sets per module (canonical sorted order, must match _registry.py).
# Hardcoded here to avoid importing mental_entropy (which pulls in torch).
# These are the same frozensets from features/{ce,se,ne,cle,bc}.py.
_CE_KEYS = [
    "ce_adj_max", "ce_adj_mean", "ce_adj_median", "ce_adj_min", "ce_adj_p10",
    "ce_adj_p25", "ce_adj_p75", "ce_adj_p90", "ce_adj_std", "ce_break_count",
    "ce_break_rate", "ce_drop_mean", "ce_drop_min", "ce_drop_std",
    "ce_inter_block_break_count", "ce_inter_block_break_rate",
    "ce_intra_block_break_count", "ce_intra_block_break_rate",
    "ce_longest_coherent_run", "ce_low_mass", "ce_low_mass_sum", "ce_n_adj",
    "ce_n_blocks", "ce_n_sentences", "ce_sharp_drop_count", "ce_sharp_drop_rate",
    "ce_skip_break_count", "ce_skip_mean", "ce_skip_min",
]
_SE_KEYS = [
    "se_cluster_entropy", "se_dominant_cluster_frac", "se_inter_mean",
    "se_intra_mean", "se_n_clusters", "se_switch_count", "se_switch_mean_jump",
    "se_switch_rate", "se_switch_weighted",
]
_NE_KEYS = [
    "ne_arc_linearity", "ne_connectors_rate", "ne_consolidation_delta",
    "ne_end_abandonment_cue", "ne_end_closure_cue", "ne_end_fragment",
    "ne_end_open_question", "ne_end_similarity", "ne_end_to_centroid",
    "ne_fragment_sentence_rate", "ne_last_sentence_reflective",
    "ne_punct_break_rate", "ne_reflection_rate", "ne_semantic_wander",
    "ne_start_end_sim", "ne_start_to_centroid", "ne_temporal_jump_rate",
    "ne_temporal_markers_rate",
]
_CLE_KEYS = [
    "cle_adj_low_frac", "cle_adj_sim_cv", "cle_adj_sim_range", "cle_adj_sim_std",
    "cle_fragment_rate", "cle_hedge_rate", "cle_length_cv",
    "cle_punctuation_noise", "cle_repetition_score", "cle_restart_rate",
    "cle_resume_rate", "cle_semantic_break_rate", "cle_semantic_isolated_rate",
    "cle_zigzag_rate",
]
_BC_KEYS = [
    "bc_belief_sentence_count", "bc_belief_sentence_rate",
    "bc_conflict_pair_count", "bc_conflict_rate", "bc_conflict_span_max",
    "bc_conflict_span_mean", "bc_integrated_count", "bc_integration_rate",
    "bc_max_conflict_sim", "bc_mean_conflict_sim", "bc_unresolved_count",
    "bc_unresolved_rate",
]

MODULE_NAMES = ("ce", "se", "ne", "cle", "bc")

MODULE_FEATURE_KEYS: dict[str, list[str]] = {
    "ce": _CE_KEYS,
    "se": _SE_KEYS,
    "ne": _NE_KEYS,
    "cle": _CLE_KEYS,
    "bc": _BC_KEYS,
}

# ---------------------------------------------------------------------------
# XGBoost hyperparameters (conservative to avoid overfitting)
# ---------------------------------------------------------------------------

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
# Helpers
# ---------------------------------------------------------------------------


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
    """Build sample weights so human and synthetic contribute equally."""
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())
    w = np.ones(len(sources), dtype=np.float64)
    if n_human > 0 and n_synth > 0:
        w[sources == SOURCE_HUMAN] = n_synth / n_human
    return w


def get_cv_indices(n: int, n_folds: int = N_FOLDS, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Get train/val indices for each fold (same splits as prepare.py)."""
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    fold_size = n // n_folds
    folds = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        folds.append((train_idx, val_idx))

    return folds


# ---------------------------------------------------------------------------
# Stage 1: Per-module OOF predictions
# ---------------------------------------------------------------------------


def train_module_oof(
    module: str,
    X_module: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, XGBRegressor]:
    """Train a per-module XGBoost and return OOF predictions + final model.

    Args:
        module: Module name (for logging).
        X_module: Feature matrix for this module (n_samples, n_module_features).
        y: Labels (1-10 scale).
        sample_weights: Per-sample weights.
        folds: List of (train_idx, val_idx) tuples.

    Returns:
        (oof_preds, final_model) where oof_preds has shape (n_samples,).
    """
    n = len(y)
    oof_preds = np.zeros(n, dtype=np.float64)
    fold_corrs = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X_module[train_idx], y[train_idx]
        X_val, y_val = X_module[val_idx], y[val_idx]
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

    # Train final model on all data (for inference artifact)
    final_model = XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_module, y, sample_weight=sample_weights)

    n_features = X_module.shape[1]
    print(f"  {module.upper():>4s}: cv_corr={cv_corr:.4f} +/- {cv_std:.3f}  ({n_features} features)")

    return oof_preds, final_model


# ---------------------------------------------------------------------------
# Stage 2: Ridge combiner on OOF subscores
# ---------------------------------------------------------------------------


def train_combiner(
    oof_subscores: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[Ridge, float, float]:
    """Train Ridge combiner on OOF subscores.

    Args:
        oof_subscores: Shape (n_samples, 5) OOF predictions from each module.
        y: Labels (1-10 scale).
        sample_weights: Per-sample weights.
        folds: Same CV folds used for Stage 1.

    Returns:
        (ridge_model, cv_corr, cv_std)
    """
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

    # Train final combiner on all OOF subscores
    final_ridge = Ridge(alpha=1.0)
    final_ridge.fit(oof_subscores, y, sample_weight=sample_weights)

    return final_ridge, cv_corr, cv_std


# ---------------------------------------------------------------------------
# End-to-end evaluation using trained models
# ---------------------------------------------------------------------------


def evaluate_trained_models(
    module_models: dict[str, XGBRegressor],
    combiner: Ridge,
    module_matrices: dict[str, np.ndarray],
    y: np.ndarray,
    sample_weights: np.ndarray,
    sources: np.ndarray,
) -> dict[str, float]:
    """Evaluate the final trained models on the full dataset.

    This uses the final models (trained on all data) to predict on all data.
    It's optimistic (train=test) but useful for checking model behavior.
    The primary metric remains the Stage 2 combiner_cv_corr.
    """
    n = len(y)
    subscores = np.zeros((n, len(MODULE_NAMES)), dtype=np.float64)

    for m_idx, module in enumerate(MODULE_NAMES):
        subscores[:, m_idx] = module_models[module].predict(module_matrices[module])

    # Apply combiner
    preds = combiner.predict(subscores)

    # Scale to 0-100 for MES comparison
    mes_preds = np.clip((preds - 1) / 9 * 100, 0, 100)
    labels_scaled = (y - 1) / 9 * 100

    val_corr = pearson_correlation(preds, y, weights=sample_weights)
    mse_raw = float(((preds - y) ** 2).mean())
    mse_scaled = float(((mes_preds - labels_scaled) ** 2).mean())

    result = {
        "val_corr": val_corr,
        "mse_1_10": mse_raw,
        "mse_0_100": mse_scaled,
        "mes_mean": float(mes_preds.mean()),
        "mes_std": float(mes_preds.std()),
    }

    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC
    if human_mask.sum() > 2:
        result["human_corr"] = pearson_correlation(preds[human_mask], y[human_mask])
        result["human_n"] = int(human_mask.sum())
    if synth_mask.sum() > 2:
        result["synth_corr"] = pearson_correlation(preds[synth_mask], y[synth_mask])
        result["synth_n"] = int(synth_mask.sum())

    return result


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------


def save_artifacts(
    module_models: dict[str, XGBRegressor],
    combiner: Ridge,
    metrics: dict[str, float],
    n_entries: int,
    n_human: int,
    n_synthetic: int,
) -> None:
    """Save trained models and combiner to _artifacts directory."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save per-module XGBoost models
    for module, model in module_models.items():
        path = ARTIFACTS_DIR / f"{module}_model.json"
        model.save_model(str(path))
        print(f"  Saved {path.name}")

    # Save Ridge combiner as JSON
    combiner_data = {
        "coef": combiner.coef_.tolist(),
        "intercept": float(combiner.intercept_),
        "module_names": list(MODULE_NAMES),
    }
    combiner_path = ARTIFACTS_DIR / "combiner.json"
    with open(combiner_path, "w") as f:
        json.dump(combiner_data, f, indent=2)
    print(f"  Saved {combiner_path.name}")

    # Save manifest
    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_entries": n_entries,
        "n_human": n_human,
        "n_synthetic": n_synthetic,
        "n_folds": N_FOLDS,
        "xgb_params": XGB_PARAMS,
        "ridge_alpha": 1.0,
        "module_names": list(MODULE_NAMES),
        "module_feature_counts": {m: len(MODULE_FEATURE_KEYS[m]) for m in MODULE_NAMES},
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

    features, labels, feature_names, sources = load_cached_data()
    sample_weights = build_sample_weights(sources)
    n = len(features)
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())

    print("=" * 60)
    print("MES SUBSCORE MODEL TRAINING")
    print("=" * 60)
    print(f"\nDataset: {n} entries ({n_human} human + {n_synth} synthetic)")
    print(f"Modules: {len(MODULE_NAMES)} ({', '.join(MODULE_NAMES)})")
    total_features = sum(len(MODULE_FEATURE_KEYS[m]) for m in MODULE_NAMES)
    print(f"Features: {total_features} total across all modules")
    print()

    # Build per-module feature matrices
    module_matrices: dict[str, np.ndarray] = {}
    for module in MODULE_NAMES:
        module_matrices[module] = build_feature_matrix(
            features, MODULE_FEATURE_KEYS[module]
        )

    # Stage 1: Per-module OOF predictions
    folds = get_cv_indices(n)

    print("Stage 1: Per-module XGBoost models (5-fold OOF)")
    print("-" * 60)

    oof_subscores = np.zeros((n, len(MODULE_NAMES)), dtype=np.float64)
    module_models: dict[str, XGBRegressor] = {}

    for m_idx, module in enumerate(MODULE_NAMES):
        oof_preds, final_model = train_module_oof(
            module, module_matrices[module], labels,
            sample_weights, folds,
        )
        oof_subscores[:, m_idx] = oof_preds
        module_models[module] = final_model

    # Stage 2: Ridge combiner on OOF subscores
    print()
    print("Stage 2: Ridge combiner on OOF subscores")
    print("-" * 60)

    combiner, combiner_cv, combiner_std = train_combiner(
        oof_subscores, labels, sample_weights, folds,
    )

    print(f"  Ridge combiner: cv_corr={combiner_cv:.4f} +/- {combiner_std:.3f}")
    print(f"  Coefficients: {dict(zip(MODULE_NAMES, combiner.coef_.round(4)))}")
    print(f"  Intercept: {combiner.intercept_:.4f}")

    # OOF-based per-source correlations
    oof_combined = combiner.predict(oof_subscores)
    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC
    oof_human_corr = pearson_correlation(oof_combined[human_mask], labels[human_mask]) if human_mask.sum() > 2 else 0.0
    oof_synth_corr = pearson_correlation(oof_combined[synth_mask], labels[synth_mask]) if synth_mask.sum() > 2 else 0.0

    print(f"  OOF human_corr: {oof_human_corr:.4f}  (n={int(human_mask.sum())})")
    print(f"  OOF synth_corr: {oof_synth_corr:.4f}  (n={int(synth_mask.sum())})")

    # End-to-end validation with trained models (train=test, optimistic)
    print()
    print("Trained model evaluation (all data, optimistic)")
    print("-" * 60)

    e2e = evaluate_trained_models(
        module_models, combiner, module_matrices,
        labels, sample_weights, sources,
    )

    print(f"  val_corr:   {e2e['val_corr']:.4f}")
    print(f"  MES mean:   {e2e['mes_mean']:.1f}  std: {e2e['mes_std']:.1f}")
    if "human_corr" in e2e:
        print(f"  human_corr: {e2e['human_corr']:.4f}  (n={e2e['human_n']})")
    if "synth_corr" in e2e:
        print(f"  synth_corr: {e2e['synth_corr']:.4f}  (n={e2e['synth_n']})")

    # Compare with linear baseline
    from prepare import evaluate_weights
    from train import WEIGHTS as LINEAR_WEIGHTS

    linear_results = evaluate_weights(LINEAR_WEIGHTS, features, labels, sources=sources)

    print()
    print("Comparison with linear baseline")
    print("-" * 60)
    print(f"  Linear (13 feat):  cv_corr={linear_results['cv_corr']:.4f}")
    print(f"  Subscore (OOF+Ridge): cv_corr={combiner_cv:.4f}")
    gap = combiner_cv - linear_results["cv_corr"]
    pct = gap / linear_results["cv_corr"] * 100
    print(f"  Improvement: {gap:+.4f} ({pct:+.1f}%)")
    if "human_corr" in linear_results:
        human_gap = oof_human_corr - linear_results["human_corr"]
        print(f"  Human corr:  {linear_results['human_corr']:.4f} -> {oof_human_corr:.4f} ({human_gap:+.4f})")

    # Save artifacts
    print()
    print("Saving artifacts")
    print("-" * 60)

    metrics = {
        "combiner_cv_corr": combiner_cv,
        "combiner_cv_std": combiner_std,
        "oof_human_corr": oof_human_corr,
        "oof_synth_corr": oof_synth_corr,
        "linear_cv_corr": linear_results["cv_corr"],
        "improvement": gap,
    }

    save_artifacts(
        module_models, combiner, metrics,
        n_entries=n, n_human=n_human, n_synthetic=n_synth,
    )

    elapsed = time.time() - start
    print()
    print("=" * 60)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Artifacts saved to {ARTIFACTS_DIR}")
    print("=" * 60)

    # Print in autoresearch-compatible format
    print()
    print("---")
    print(f"combiner_cv_corr: {combiner_cv:.6f}")
    print(f"combiner_cv_std:  {combiner_std:.6f}")
    print(f"oof_human_corr:     {oof_human_corr:.6f}")
    print(f"oof_synth_corr:   {oof_synth_corr:.6f}")
    print(f"linear_cv_corr:   {linear_results['cv_corr']:.6f}")
    print(f"improvement:      {gap:+.6f}")


if __name__ == "__main__":
    main()
