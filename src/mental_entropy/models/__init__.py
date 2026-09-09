"""Subscore models for MES scoring.

Supports two architectures (auto-detected from artifacts):
- **v3 (dimension-aligned)**: 5 XGBoost models trained per rubric dimension,
  each using all 77 features.
- **v1 (module-aligned, legacy)**: 5 XGBoost models trained per feature module.

Requires the ``models`` extra: ``pip install mental-entropy[models]``

If xgboost is not installed or model artifacts are missing, the main
scoring API falls back to the linear weighted method automatically.
"""

from __future__ import annotations

from mental_entropy.models._registry import (
    ALL_FEATURE_KEYS,
    DIMENSION_NAMES,
    MODULE_FEATURE_KEYS,
    MODULE_NAMES,
    predict_mes,
    predict_subscores,
    reset_cache,
    subscore_models_available,
)

__all__ = [
    "ALL_FEATURE_KEYS",
    "DIMENSION_NAMES",
    "MODULE_FEATURE_KEYS",
    "MODULE_NAMES",
    "predict_mes",
    "predict_subscores",
    "reset_cache",
    "subscore_models_available",
]
