"""Model registry for subscore and hybrid embedding models.

Supports five architectures (auto-detected from combiner.json):
- **v7 (FEP subscores)**: Ridge regression on embeddings for 5 FEP-aligned
  dimensions + Ridge combiner on subscores + features. Loaded when
  combiner.json has architecture "subscore_embedding_v7".
  Does not require xgboost.
- **v6 (hybrid)**: Ridge regression on 1024-dim document embedding + top-k
  hand-crafted features. Loaded when combiner.json has architecture "hybrid_v6".
  Does not require xgboost for MES prediction.
- **v5 (ensemble)**: Blends linear baseline with v3 XGBoost predictions.
  Loaded when combiner.json has architecture "ensemble_v5".
- **v3 (dimension-aligned)**: 5 XGBoost models trained per rubric dimension,
  each using all 82 features.
- **v1 (module-aligned, legacy)**: 5 XGBoost models trained per feature module.

All models are optional — if artifacts are missing or dependencies are not
installed, functions raise ImportError or FileNotFoundError so callers can
fall back to the linear scoring method.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mental_entropy.features import (
    BC_FEATURE_KEYS,
    CE_FEATURE_KEYS,
    CLE_FEATURE_KEYS,
    NE_FEATURE_KEYS,
    SE_FEATURE_KEYS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ARTIFACTS_DIR = Path(__file__).parent / "_artifacts"

# ---------------------------------------------------------------------------
# Feature-group mapping (module-aligned, kept for backward compat)
# ---------------------------------------------------------------------------

#: Module names in canonical order (legacy architecture).
MODULE_NAMES: tuple[str, ...] = ("ce", "se", "ne", "cle", "bc")

#: Sorted feature lists per module (order must match training).
MODULE_FEATURE_KEYS: dict[str, list[str]] = {
    "ce": sorted(CE_FEATURE_KEYS),
    "se": sorted(SE_FEATURE_KEYS),
    "ne": sorted(NE_FEATURE_KEYS),
    "cle": sorted(CLE_FEATURE_KEYS),
    "bc": sorted(BC_FEATURE_KEYS),
}

# ---------------------------------------------------------------------------
# Dimension names (v7 FEP-aligned)
# ---------------------------------------------------------------------------

#: FEP-aligned dimension names in canonical order.
DIMENSION_NAMES: tuple[str, ...] = (
    "prediction_coherence",
    "model_complexity",
    "compression_progress",
    "belief_integration",
    "precision_weighting",
)

#: Legacy dimension names (v3/v5/v6 era).
_LEGACY_DIMENSION_NAMES: tuple[str, ...] = (
    "continuity",
    "topic_focus",
    "contradiction_integration",
    "cognitive_clarity",
    "narrative_closure",
)

#: All 77 feature names in sorted order (used by dimension-aligned models).
ALL_FEATURE_KEYS: list[str] = sorted(
    list(CE_FEATURE_KEYS) + list(SE_FEATURE_KEYS) + list(NE_FEATURE_KEYS)
    + list(CLE_FEATURE_KEYS) + list(BC_FEATURE_KEYS)
)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_subscore_models: dict[str, Any] | None = None
_combiner: dict[str, Any] | None = None
_architecture: str | None = None
_hybrid_model: dict[str, Any] | None = None
_dim_ridge_models: dict[str, dict[str, Any]] | None = None


def _check_xgboost() -> None:
    """Raise ImportError if xgboost is not available."""
    try:
        import xgboost  # noqa: F401
    except ImportError:
        raise ImportError(
            "xgboost is required for subscore models. "
            "Install it with: pip install mental-entropy[models]"
        )


def _detect_architecture() -> str:
    """Detect model architecture from combiner.json."""
    path = _ARTIFACTS_DIR / "combiner.json"
    if not path.exists():
        raise FileNotFoundError(f"Combiner artifact not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data.get("architecture", "module_aligned_v1")


def _get_architecture() -> str:
    """Get the current architecture (loads combiner if needed)."""
    global _architecture
    if _architecture is not None:
        return _architecture
    _architecture = _detect_architecture()
    return _architecture


def _load_dim_ridge_models() -> dict[str, dict[str, Any]]:
    """Load per-dimension Ridge models for v7 architecture."""
    global _dim_ridge_models
    if _dim_ridge_models is not None:
        return _dim_ridge_models

    combiner = _load_combiner()
    dim_names = combiner.get("dimension_names", list(DIMENSION_NAMES))

    models: dict[str, dict[str, Any]] = {}
    for dim_name in dim_names:
        path = _ARTIFACTS_DIR / f"dim_{dim_name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Dimension Ridge model not found: {path}. "
                "Run autoresearch-macos/train_v7_fep.py to train."
            )
        with open(path) as f:
            models[dim_name] = json.load(f)

    _dim_ridge_models = models
    return _dim_ridge_models


def _load_subscore_models() -> dict[str, Any]:
    """Load all subscore XGBoost models from artifacts (legacy v3/v5)."""
    global _subscore_models, _architecture
    if _subscore_models is not None:
        return _subscore_models

    _check_xgboost()
    from xgboost import XGBRegressor

    arch = _detect_architecture()
    _architecture = arch

    # Determine which dimension names to use
    dim_names = _LEGACY_DIMENSION_NAMES
    if arch in ("dimension_aligned_v3", "ensemble_v5", "hybrid_v6"):
        dim_names = _LEGACY_DIMENSION_NAMES
    else:
        dim_names = None  # module-aligned

    models: dict[str, Any] = {}

    if dim_names is not None:
        for dim in dim_names:
            path = _ARTIFACTS_DIR / f"{dim}_model.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Dimension model artifact not found: {path}. "
                    "Run autoresearch-macos/train_subscores_v5.py to train."
                )
            model = XGBRegressor()
            model.load_model(str(path))
            models[dim] = model
    else:
        for module in MODULE_NAMES:
            path = _ARTIFACTS_DIR / f"{module}_model.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Subscore model artifact not found: {path}. "
                    "Run autoresearch-macos/train_subscores.py to train."
                )
            model = XGBRegressor()
            model.load_model(str(path))
            models[module] = model

    _subscore_models = models
    return _subscore_models


def _load_combiner() -> dict[str, Any]:
    """Load Ridge combiner coefficients from artifacts."""
    global _combiner
    if _combiner is not None:
        return _combiner

    path = _ARTIFACTS_DIR / "combiner.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Combiner artifact not found: {path}. "
            "Run autoresearch-macos/train_v7_fep.py to train models."
        )

    with open(path) as f:
        _combiner = json.load(f)

    return _combiner


def _load_hybrid_model() -> dict[str, Any]:
    """Load hybrid embedding model coefficients from artifacts."""
    global _hybrid_model
    if _hybrid_model is not None:
        return _hybrid_model

    path = _ARTIFACTS_DIR / "embedding_model.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Hybrid embedding model not found: {path}. "
            "Run autoresearch-macos/train_embedding_regression.py to train."
        )

    with open(path) as f:
        _hybrid_model = json.load(f)

    return _hybrid_model


def reset_cache() -> None:
    """Clear cached models (useful for testing or reloading after retrain)."""
    global _subscore_models, _combiner, _architecture, _hybrid_model, _dim_ridge_models
    _subscore_models = None
    _combiner = None
    _architecture = None
    _hybrid_model = None
    _dim_ridge_models = None


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _features_to_array(features: dict[str, float], module: str) -> np.ndarray:
    """Extract a module's features into a 1D array in canonical order."""
    keys = MODULE_FEATURE_KEYS[module]
    return np.array([features.get(k, 0.0) for k in keys], dtype=np.float64)


def _all_features_to_array(features: dict[str, float]) -> np.ndarray:
    """Extract all 77 features into a 1D array in canonical sorted order."""
    return np.array([features.get(k, 0.0) for k in ALL_FEATURE_KEYS], dtype=np.float64)


# ---------------------------------------------------------------------------
# v7 FEP subscore inference
# ---------------------------------------------------------------------------

def v7_available() -> bool:
    """Check if v7 FEP subscore artifacts exist on disk."""
    try:
        arch = _detect_architecture()
    except Exception:
        return False
    if arch != "subscore_embedding_v7":
        return False
    combiner_path = _ARTIFACTS_DIR / "combiner.json"
    if not combiner_path.exists():
        return False
    with open(combiner_path) as f:
        combiner = json.load(f)
    for dim in combiner.get("dimension_names", DIMENSION_NAMES):
        if not (_ARTIFACTS_DIR / f"dim_{dim}.json").exists():
            return False
    return True


def predict_subscores_from_embedding(
    doc_embedding: list[float],
    features: dict[str, float],
) -> dict[str, float]:
    """Predict FEP-aligned subscores from document embedding.

    Uses per-dimension Ridge models (embedding → dimension score 1-5).

    Args:
        doc_embedding: 1024-dim document embedding.
        features: Full feature dict (used for combiner features).

    Returns:
        Dict with subscore values keyed by dimension name + "_subscore".
    """
    dim_models = _load_dim_ridge_models()
    emb = np.array(doc_embedding[:1024], dtype=np.float64)

    subscores: dict[str, float] = {}
    for dim_name, model_data in dim_models.items():
        coef = np.array(model_data["coef"], dtype=np.float64)
        intercept = float(model_data["intercept"])
        raw = float(np.dot(coef, emb)) + intercept
        # Clamp to 1-5 scale
        subscores[f"{dim_name}_subscore"] = min(5.0, max(1.0, raw))

    return subscores


def predict_mes_v7(
    doc_embedding: list[float],
    features: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Predict MES score using v7 FEP subscore architecture.

    Stage 1: Per-dimension Ridge (embedding → 5 subscores)
    Stage 2: Combiner Ridge (subscores + features → overall_entropy)
    Overall_entropy (1-10) is converted to MES (0-100).

    Args:
        doc_embedding: 1024-dim document embedding.
        features: Full feature dict.

    Returns:
        Tuple of (mes_score, subscores_dict).
    """
    subscores = predict_subscores_from_embedding(doc_embedding, features)
    combiner = _load_combiner()

    coef = np.array(combiner["coef"], dtype=np.float64)
    intercept = float(combiner["intercept"])
    dim_names = combiner.get("dimension_names", list(DIMENSION_NAMES))
    combiner_features = combiner.get("feature_names", [])

    # Build input: [subscores | optional features]
    subscore_vec = np.array(
        [subscores[f"{name}_subscore"] for name in dim_names],
        dtype=np.float64,
    )

    if combiner_features:
        feat_vec = np.array(
            [features.get(f, 0.0) for f in combiner_features],
            dtype=np.float64,
        )
        x = np.concatenate([subscore_vec, feat_vec])
    else:
        x = subscore_vec

    # Predict on 1-10 scale, convert to 0-100
    raw = float(np.dot(coef, x)) + intercept
    score_100 = (raw - 1.0) / 9.0 * 100.0
    score_100 = _apply_mes_calibration(score_100, combiner)
    mes_score = min(100.0, max(0.0, score_100))

    return mes_score, subscores


def _apply_mes_calibration(score_100: float, combiner: dict[str, Any]) -> float:
    """Apply optional MES-scale calibration stored in the combiner artifact."""
    calibration = combiner.get("calibration")
    if not calibration:
        return score_100

    kind = calibration.get("kind")
    if kind == "linear_mes":
        slope = float(calibration.get("slope", 1.0))
        intercept = float(calibration.get("intercept", 0.0))
        return slope * score_100 + intercept
    if kind == "isotonic_mes":
        x = np.array(calibration.get("x_thresholds", []), dtype=np.float64)
        y = np.array(calibration.get("y_thresholds", []), dtype=np.float64)
        if len(x) < 2 or len(x) != len(y):
            return score_100
        return float(np.interp(score_100, x, y, left=y[0], right=y[-1]))

    return score_100


def get_dimension_display() -> dict[str, dict[str, str]]:
    """Get display names and descriptions for dimensions.

    Returns:
        Dict mapping dimension name to {"name": ..., "description": ...}.
    """
    combiner = _load_combiner()
    return combiner.get("dimension_display", {})


# ---------------------------------------------------------------------------
# Legacy inference (v3/v5/v6)
# ---------------------------------------------------------------------------

def subscore_models_available() -> bool:
    """Check if trained subscore model artifacts exist on disk (legacy v3/v5)."""
    combiner_path = _ARTIFACTS_DIR / "combiner.json"
    if not combiner_path.exists():
        return False

    try:
        arch = _detect_architecture()
    except Exception:
        return False

    # v7 uses its own path
    if arch == "subscore_embedding_v7":
        return False

    if arch in ("dimension_aligned_v3", "ensemble_v5", "hybrid_v6"):
        for dim in _LEGACY_DIMENSION_NAMES:
            if not (_ARTIFACTS_DIR / f"{dim}_model.json").exists():
                return False
    else:
        for module in MODULE_NAMES:
            if not (_ARTIFACTS_DIR / f"{module}_model.json").exists():
                return False

    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def predict_subscores(features: dict[str, float]) -> dict[str, float]:
    """Predict subscores from features (legacy v3/v5 — uses XGBoost)."""
    models = _load_subscore_models()
    arch = _get_architecture()
    subscores: dict[str, float] = {}

    if arch in ("dimension_aligned_v3", "ensemble_v5", "hybrid_v6"):
        X = _all_features_to_array(features).reshape(1, -1)
        for dim in _LEGACY_DIMENSION_NAMES:
            pred = float(models[dim].predict(X)[0])
            subscores[f"{dim}_subscore"] = pred
    else:
        for module in MODULE_NAMES:
            X = _features_to_array(features, module).reshape(1, -1)
            pred = float(models[module].predict(X)[0])
            subscores[f"{module}_subscore"] = pred

    return subscores


def _compute_linear_baseline(features: dict[str, float]) -> float:
    """Compute the linear baseline score (0-100) from hand-tuned weights."""
    _WEIGHTS = [
        ("ce_adj_p75",                 0.17, "inv"),
        ("ne_fragment_sentence_rate",  0.14, "dir"),
        ("ce_inter_block_break_rate",  0.11, "dir"),
        ("bc_belief_sentence_count",   0.11, "dir_norm", 18.0),
        ("ce_n_blocks",                0.10, "dir_norm", 20.0),
        ("cle_length_cv",              0.08, "dir"),
        ("ce_skip_mean",               0.07, "inv"),
        ("ne_start_to_centroid",       0.06, "inv"),
        ("se_dominant_cluster_frac",   0.05, "inv"),
        ("ne_arc_linearity",           0.03, "inv"),
        ("ne_end_to_centroid",         0.03, "inv"),
        ("cle_hedge_rate",             0.03, "dir"),
        ("ne_start_end_sim",           0.02, "inv"),
    ]
    raw = 0.0
    for entry in _WEIGHTS:
        name, weight, transform = entry[0], entry[1], entry[2]
        param = entry[3] if len(entry) > 3 else None
        default = 1.0 if transform == "inv" else 0.0
        value = features.get(name, default)
        if transform == "inv":
            raw += weight * (1.0 - value)
        elif transform == "dir":
            raw += weight * value
        elif transform == "dir_norm" and param:
            raw += weight * min(value / param, 1.0)
    return min(100.0, max(0.0, raw * 100.0))


def predict_mes(features: dict[str, float]) -> float:
    """Predict MES score using legacy subscore models + Ridge combiner."""
    subscores = predict_subscores(features)
    combiner = _load_combiner()

    coef = np.array(combiner["coef"], dtype=np.float64)
    intercept = float(combiner["intercept"])

    arch = _get_architecture()
    if arch in ("dimension_aligned_v3", "ensemble_v5", "hybrid_v6"):
        names = list(_LEGACY_DIMENSION_NAMES)
    else:
        names = combiner.get("module_names", list(MODULE_NAMES))

    subscore_vec = np.array(
        [subscores[f"{name}_subscore"] for name in names],
        dtype=np.float64,
    )

    subscore_raw = intercept + float(np.dot(coef, subscore_vec))

    if arch == "ensemble_v5":
        blend_alpha = float(combiner.get("blend_alpha", 0.4))
        linear_score = _compute_linear_baseline(features)
        subscore_100 = min(100.0, max(0.0, (subscore_raw - 1.0) / 9.0 * 100.0))
        blended = blend_alpha * linear_score + (1.0 - blend_alpha) * subscore_100
        return min(100.0, max(0.0, blended))

    return min(100.0, max(0.0, subscore_raw))


# ---------------------------------------------------------------------------
# Hybrid embedding model (v6 fallback)
# ---------------------------------------------------------------------------


def hybrid_model_available() -> bool:
    """Check if hybrid embedding model artifact exists on disk."""
    emb_path = _ARTIFACTS_DIR / "embedding_model.json"
    if not emb_path.exists():
        return False
    try:
        arch = _detect_architecture()
        return arch in ("hybrid_v6", "subscore_embedding_v7")
    except Exception:
        return False


def predict_mes_from_embedding(
    doc_embedding: list[float],
    features: dict[str, float],
) -> float:
    """Predict MES score using the hybrid embedding + features model.

    Uses Ridge regression on [1024-dim embedding | top-k features].
    The model predicts on the 1-10 overall_entropy scale, which is
    then converted to 0-100.

    Args:
        doc_embedding: 1024-dim document embedding from mxbai model.
        features: Dict containing all hand-crafted features.

    Returns:
        MES score clamped to [0, 100].
    """
    model = _load_hybrid_model()
    coef = np.array(model["coef"], dtype=np.float64)
    intercept = float(model["intercept"])
    feature_names: list[str] = model["feature_names"]
    embedding_dim = int(model["embedding_dim"])

    emb = np.array(doc_embedding[:embedding_dim], dtype=np.float64)
    feat_vals = np.array(
        [features.get(f, 0.0) for f in feature_names],
        dtype=np.float64,
    )
    x = np.concatenate([emb, feat_vals])

    raw = float(np.dot(coef, x)) + intercept
    score_100 = (raw - 1.0) / 9.0 * 100.0
    return min(100.0, max(0.0, score_100))
