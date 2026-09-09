"""Mental Entropy Score (MES) weighted aggregator.

This module combines all feature modules into a single MES score (0-100)
where higher values indicate more entropy (fragmented, disorganized thinking).

Supports four scoring methods:
- "embedding": v7 FEP subscore architecture — per-dimension Ridge on embeddings
  + combiner on subscores + features. Returns 5 FEP-aligned subscores.
  Falls back to v6 hybrid Ridge on embedding + features.
- "subscore": Legacy v3/v5 — XGBoost dimension models + Ridge combiner.
- "linear": Hand-tuned weighted sum of 13 features.
- "auto": Uses v7 if available, else embedding, else subscore, else linear.

v7 trained on 4240 entries (2995 human + 1245 synthetic) with multi-rater
FEP-aligned consensus labels. human_corr = 0.617, cv_corr = 0.747.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mental_entropy import (
    embed_journal_entry,
    ce_features_from_result,
    se_features_from_result,
    ne_features_from_result,
    cle_features_from_result,
    bc_features_from_result,
    gd_features_from_result,
)

if TYPE_CHECKING:
    from mental_entropy.embedding.types import EmbeddingResult


# ---------------------------------------------------------------------------
# Weight configuration (autoresearch/mar12 expanded-data)
# ---------------------------------------------------------------------------
# Each entry: (feature_name, weight, transform[, param])
#   "inv"      -> contribution = weight * (1 - x)
#   "dir"      -> contribution = weight * x
#   "dir_norm" -> contribution = weight * min(x / param, 1.0)
#
# Weights sum to 1.0; final score = sum(contributions) * 100, clamped [0,100].
# Calibrated on 2209 entries (964 LLM-labeled + 1245 synthetic).

_WEIGHTS: list[tuple[str, float, str] | tuple[str, float, str, float]] = [
    ("ce_adj_p75",                 0.17, "inv"),              # Upper quartile coherence
    ("ne_fragment_sentence_rate",  0.14, "dir"),              # Fragment detection
    ("ce_inter_block_break_rate",  0.11, "dir"),              # Inter-block breaks
    ("bc_belief_sentence_count",   0.11, "dir_norm", 18.0),  # Belief density
    ("ce_n_blocks",                0.10, "dir_norm", 20.0),   # Block count
    ("cle_length_cv",              0.08, "dir"),              # Sentence length variability
    ("ce_skip_mean",               0.07, "inv"),              # Skip-connection quality
    ("ne_start_to_centroid",       0.06, "inv"),              # Opening alignment
    ("se_dominant_cluster_frac",   0.05, "inv"),              # Topic concentration
    ("ne_arc_linearity",           0.03, "inv"),              # Narrative linearity
    ("ne_end_to_centroid",         0.03, "inv"),              # Closing alignment
    ("cle_hedge_rate",             0.03, "dir"),              # Hedging language
    ("ne_start_end_sim",           0.02, "inv"),              # Narrative return
]

# Valid method values
_VALID_METHODS = ("auto", "linear", "subscore", "embedding")


def _apply_transform(value: float, transform: str, param: float | None = None) -> float:
    """Apply a transform to a raw feature value."""
    if transform == "inv":
        return 1.0 - value
    elif transform == "dir":
        return value
    elif transform == "dir_norm":
        return min(value / param, 1.0) if param else value
    return value


def _compute_linear_mes(features: dict[str, float]) -> float:
    """Compute MES using the hand-tuned linear weighted formula."""
    raw = 0.0
    for entry in _WEIGHTS:
        name, weight, transform = entry[0], entry[1], entry[2]
        param = entry[3] if len(entry) > 3 else None
        # Default: 1.0 for inv features (worst case), 0.0 for dir features
        default = 1.0 if transform == "inv" else 0.0
        value = features.get(name, default)
        raw += weight * _apply_transform(value, transform, param)

    return min(100.0, max(0.0, raw * 100.0))


def _compute_subscore_mes(features: dict[str, float]) -> float:
    """Compute MES using trained per-module XGBoost models + Ridge combiner.

    Raises:
        ImportError: If xgboost is not installed.
        FileNotFoundError: If model artifacts are missing.
    """
    from mental_entropy.models._registry import predict_mes
    return predict_mes(features)


def compute_mes_score(
    features: dict[str, float],
    method: str = "auto",
) -> float:
    """Compute Mental Entropy Score from all features.

    Returns 0-100 where:
    - 0-20: Very low entropy (highly organized)
    - 21-35: Low entropy (well-organized)
    - 36-50: Moderate entropy (some fragmentation)
    - 51-65: Elevated entropy (noticeable fragmentation)
    - 66-80: High entropy (significant disorganization)
    - 81-100: Very high entropy (severely fragmented)

    Args:
        features: Dict containing all CE/SE/NE/CLE/BC features.
        method: Scoring method to use:
            - "auto": Use embedding if available, else subscore, else linear.
            - "embedding": Use hybrid embedding model (raises if unavailable).
              NOTE: This method requires an embedding vector — use
              ``compute_mes_from_text`` or ``compute_mes_from_result`` instead.
            - "linear": Always use the hand-tuned linear weighted formula.
            - "subscore": Always use trained per-module models (raises if unavailable).

    Returns:
        MES score from 0 to 100.

    Raises:
        ValueError: If method is not valid, or if method="embedding" (needs embedding).
        ImportError: If method="subscore" and xgboost is not installed.
        FileNotFoundError: If method="subscore" and model artifacts are missing.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Invalid method {method!r}. Must be one of: {_VALID_METHODS}"
        )

    if method == "embedding":
        raise ValueError(
            "method='embedding' requires a document embedding. "
            "Use compute_mes_from_text() or compute_mes_from_result() instead."
        )

    if method == "linear":
        return _compute_linear_mes(features)

    if method == "subscore":
        return _compute_subscore_mes(features)

    # method == "auto": try subscore, fall back to linear
    # (embedding handled at higher level in compute_mes_from_text/_from_result)
    try:
        from mental_entropy.models._registry import subscore_models_available
        if subscore_models_available():
            return _compute_subscore_mes(features)
    except Exception:
        pass

    return _compute_linear_mes(features)


def compute_mes_from_text(
    text: str,
    method: str = "auto",
) -> dict[str, float]:
    """Compute MES score and all features from raw journal text.

    This is the main entry point for end-to-end scoring.

    Args:
        text: Raw journal entry text.
        method: Scoring method ("auto", "linear", "subscore", or "embedding").

    Returns:
        Dict containing:
        - 'mes_score': The overall MES score (0-100)
        - All individual features from each module (ce_*, se_*, ne_*, cle_*, bc_*)
        - Per-module subscores if method uses subscore models
    """
    # Embed the text
    result = embed_journal_entry(text)

    # Compute all feature sets
    features: dict[str, float] = {}
    features.update(ce_features_from_result(result))
    features.update(se_features_from_result(result))
    features.update(ne_features_from_result(result))
    features.update(cle_features_from_result(result))
    features.update(bc_features_from_result(result))
    features.update(gd_features_from_result(result))

    # Score using the best available method
    mes_score, subscores = _score_with_embedding(
        result.doc_embedding, features, method,
    )

    # Build result dict
    result_dict: dict[str, float] = {"mes_score": mes_score}
    result_dict.update(features)
    if subscores:
        result_dict.update(subscores)

    return result_dict


def compute_mes_from_result(
    result: "EmbeddingResult",
    method: str = "auto",
) -> dict[str, float]:
    """Compute MES score and all features from an EmbeddingResult.

    Use this when you've already embedded the text and want to avoid
    re-embedding.

    Args:
        result: EmbeddingResult from embed_journal_entry().
        method: Scoring method ("auto", "linear", "subscore", or "embedding").

    Returns:
        Dict containing 'mes_score' and all individual features.
    """
    features: dict[str, float] = {}
    features.update(ce_features_from_result(result))
    features.update(se_features_from_result(result))
    features.update(ne_features_from_result(result))
    features.update(cle_features_from_result(result))
    features.update(bc_features_from_result(result))
    features.update(gd_features_from_result(result))

    mes_score, subscores = _score_with_embedding(
        result.doc_embedding, features, method,
    )

    result_dict: dict[str, float] = {"mes_score": mes_score}
    result_dict.update(features)
    if subscores:
        result_dict.update(subscores)

    return result_dict


def _score_with_embedding(
    doc_embedding: list[float],
    features: dict[str, float],
    method: str,
) -> tuple[float, dict[str, float]]:
    """Score using the best available embedding-based method.

    Returns:
        Tuple of (mes_score, subscores_dict). subscores_dict may be empty.
    """
    # Try v7 FEP subscores first (for "auto" or "embedding")
    if method in ("auto", "embedding"):
        try:
            from mental_entropy.models._registry import v7_available, predict_mes_v7
            if v7_available():
                mes_score, subscores = predict_mes_v7(doc_embedding, features)
                return mes_score, subscores
        except Exception:
            if method == "embedding":
                raise

    # Try v6 hybrid embedding model
    if method in ("auto", "embedding"):
        try:
            from mental_entropy.models._registry import (
                hybrid_model_available,
                predict_mes_from_embedding,
            )
            if hybrid_model_available():
                mes_score = predict_mes_from_embedding(doc_embedding, features)
                return mes_score, {}
        except Exception:
            if method == "embedding":
                raise

    # Fall back to feature-based scoring
    mes_score = compute_mes_score(features, method=method)
    return mes_score, {}


def _subscore_available() -> bool:
    """Check if subscore models are available (cached helper)."""
    try:
        from mental_entropy.models._registry import subscore_models_available
        return subscore_models_available()
    except Exception:
        return False


def interpret_mes_score(score: float) -> str:
    """Return a human-readable interpretation of the MES score.

    Args:
        score: MES score (0-100).

    Returns:
        Description of what the score indicates.
    """
    if score <= 20:
        return "Very low entropy: Highly organized, coherent thinking with strong narrative structure."
    elif score <= 35:
        return "Low entropy: Well-organized thinking with good coherence and minimal fragmentation."
    elif score <= 50:
        return "Moderate entropy: Generally organized but with some topic shifts or fragmented passages."
    elif score <= 65:
        return "Elevated entropy: Noticeable fragmentation, topic jumping, or incomplete thoughts."
    elif score <= 80:
        return "High entropy: Significant disorganization with frequent breaks in coherence."
    else:
        return "Very high entropy: Severely fragmented thinking with minimal narrative structure."
