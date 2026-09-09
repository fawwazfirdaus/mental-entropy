"""Mental Entropy Score (MES) pipeline for journal analysis."""

from mental_entropy.embedding import embed_journal_entry
from mental_entropy.features import (
    ce_features,
    ce_features_from_result,
    se_features,
    se_features_from_result,
    ne_features,
    ne_features_from_result,
    cle_features,
    cle_features_from_result,
    bc_features,
    bc_features_from_result,
    gd_features,
    gd_features_from_result,
)
from mental_entropy.score import (
    compute_mes_score,
    compute_mes_from_text,
    compute_mes_from_result,
    interpret_mes_score,
)
from mental_entropy.temporal import (
    ScoredEntry,
    TE_FEATURE_KEYS,
    te_features,
    te_features_from_entries,
)
from mental_entropy.temporal.te import compute_user_state
from mental_entropy.insight import (
    MentalState,
    ClassificationResult,
    WeakDimension,
    InsightPrompt,
    FeedbackContext,
    FeedbackResult,
    SubscoreDelta,
    classify_mental_state,
    assemble_insight_prompt,
    compute_feedback,
)

__all__ = [
    # Embedding
    "embed_journal_entry",
    # Feature extraction
    "ce_features",
    "ce_features_from_result",
    "se_features",
    "se_features_from_result",
    "ne_features",
    "ne_features_from_result",
    "cle_features",
    "cle_features_from_result",
    "bc_features",
    "bc_features_from_result",
    "gd_features",
    "gd_features_from_result",
    # Score computation
    "compute_mes_score",
    "compute_mes_from_text",
    "compute_mes_from_result",
    "interpret_mes_score",
    # Temporal analysis
    "ScoredEntry",
    "TE_FEATURE_KEYS",
    "te_features",
    "te_features_from_entries",
    "compute_user_state",
    # Insight engine
    "MentalState",
    "ClassificationResult",
    "WeakDimension",
    "InsightPrompt",
    "FeedbackContext",
    "FeedbackResult",
    "SubscoreDelta",
    "classify_mental_state",
    "assemble_insight_prompt",
    "compute_feedback",
]


def predict_subscores(features: dict[str, float]) -> dict[str, float]:
    """Predict per-module entropy subscores using trained XGBoost models.

    Convenience re-export from mental_entropy.models. Requires the ``models``
    extra (xgboost + scikit-learn) and trained artifacts.

    Args:
        features: Dict containing all 77 features (ce_*, se_*, ne_*, cle_*, bc_*).

    Returns:
        Dict with keys ce_subscore, se_subscore, ne_subscore, cle_subscore, bc_subscore.

    Raises:
        ImportError: If xgboost is not installed.
        FileNotFoundError: If model artifacts are missing.
    """
    from mental_entropy.models import predict_subscores as _predict
    return _predict(features)
