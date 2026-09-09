from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "autoresearch-macos"))


def test_coefficient_bounds_apply_dimension_and_feature_sign_contract() -> None:
    import train_v10_constrained_combiner as train_v10

    combiner = {
        "dimension_names": ["prediction_coherence"],
        "feature_names": ["ce_adj_mean", "bc_unresolved_count", "unknown_feature"],
    }

    lower, upper = train_v10.coefficient_bounds(
        ["prediction_coherence", "ce_adj_mean", "bc_unresolved_count", "unknown_feature"],
        combiner,
    )

    assert lower[0] == -np.inf
    assert upper[0] == 0.0
    assert lower[1] == -np.inf
    assert upper[1] == 0.0
    assert lower[2] == 0.0
    assert upper[2] == np.inf
    assert lower[3] == -np.inf
    assert upper[3] == np.inf
    assert lower[4] == -np.inf
    assert upper[4] == np.inf


def test_standardize_matrix_handles_constant_columns() -> None:
    import train_v10_constrained_combiner as train_v10

    X = np.array([[1.0, 5.0], [3.0, 5.0]], dtype=np.float64)

    X_std, means, scales = train_v10.standardize_matrix(X)

    assert means.tolist() == [2.0, 5.0]
    assert scales.tolist() == [1.0, 1.0]
    assert X_std[:, 1].tolist() == [0.0, 0.0]


def test_fit_constrained_combiner_respects_expected_signs() -> None:
    import train_v10_constrained_combiner as train_v10

    X = np.array([
        [4.0, 0.1],
        [3.0, 0.4],
        [2.0, 0.8],
        [1.0, 1.1],
    ], dtype=np.float64)
    y = np.array([2.0, 4.0, 7.0, 9.0], dtype=np.float64)
    sources = np.zeros(4, dtype=np.int32)
    combiner = {
        "dimension_names": ["prediction_coherence"],
        "feature_names": ["bc_unresolved_count"],
    }

    artifact = train_v10.fit_constrained_combiner(
        X,
        y,
        sources,
        ["prediction_coherence", "bc_unresolved_count"],
        combiner,
        alpha=0.0,
    )

    assert artifact["coef_standardized"][0] <= 0.0
    assert artifact["coef_standardized"][1] >= 0.0


def test_predict_overall_entropy_uses_standardized_artifact() -> None:
    import train_v10_constrained_combiner as train_v10

    artifact = {
        "coef_standardized": [2.0],
        "intercept": 5.0,
        "feature_means": [10.0],
        "feature_scales": [5.0],
    }

    assert train_v10.predict_overall_entropy([15.0], artifact) == 7.0


def test_project_relative_accepts_relative_paths() -> None:
    import train_v10_constrained_combiner as train_v10

    assert train_v10._project_relative(Path("artifacts/example.json")) == "artifacts/example.json"
