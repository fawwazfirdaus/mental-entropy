from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def test_spearman_handles_ties() -> None:
    import audit_ground_up_scoring as audit

    assert audit.spearman([1.0, 2.0, 2.0, 4.0], [1.0, 2.0, 2.0, 4.0]) == pytest.approx(1.0)


def test_labels_to_mes_scales_overall_entropy() -> None:
    import audit_ground_up_scoring as audit

    values = audit.labels_to_mes(np.array([1.0, 5.5, 10.0], dtype=np.float64))

    assert values.tolist() == [0.0, 50.0, 100.0]


def test_make_source_stratified_folds_preserves_all_indices() -> None:
    import audit_ground_up_scoring as audit

    sources = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    folds = audit.make_source_stratified_folds(sources, n_folds=3)
    validation_indices = sorted(index for _, val_idx in folds for index in val_idx.tolist())

    assert validation_indices == [0, 1, 2, 3, 4, 5]
    assert all(len(val_idx) == 2 for _, val_idx in folds)


def test_build_feature_family_audit_groups_prefixes() -> None:
    import audit_ground_up_scoring as audit

    features = [
        {"ce_break": 0.1, "gd_marker": 0.0},
        {"ce_break": 0.5, "gd_marker": 0.2},
        {"ce_break": 0.9, "gd_marker": 0.8},
    ]
    labels = np.array([1.0, 5.0, 9.0], dtype=np.float64)
    sources = np.array([0, 0, 1], dtype=np.int32)

    result = audit.build_feature_family_audit(
        features,
        labels,
        sources,
        ["ce_break", "gd_marker"],
    )

    assert result["ce"]["feature_count"] == 1
    assert result["gd"]["feature_count"] == 1
    assert result["ce"]["top_human_signals"][0]["name"] == "ce_break"


def test_build_label_audit_reports_zero_label_stds() -> None:
    import audit_ground_up_scoring as audit

    labels = np.array([1.0, 5.0, 9.0], dtype=np.float64)
    sources = np.array([0, 0, 1], dtype=np.int32)
    dim_labels = {"prediction_coherence": np.array([5.0, 3.0, 1.0], dtype=np.float64)}
    label_stds = np.zeros(3, dtype=np.float64)

    result = audit.build_label_audit(labels, sources, dim_labels, label_stds)

    assert result["label_std_all_zero"] is True
    assert result["label_stds"]["all"]["max"] == 0.0


def test_embedding_neighbor_smoothness_reports_label_gap() -> None:
    import audit_ground_up_scoring as audit

    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ], dtype=np.float64)
    labels = np.array([1.0, 2.0, 9.0], dtype=np.float64)
    sources = np.array([0, 0, 0], dtype=np.int32)

    result = audit.embedding_neighbor_smoothness(embeddings, labels, sources, k=1)

    assert result["n"] == 3
    assert result["mean_neighbor_oe_gap"] > 0.0
