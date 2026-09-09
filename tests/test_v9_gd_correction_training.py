from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "autoresearch-macos"))


def test_apply_gd_correction_only_adds_positive_uplift() -> None:
    import train_v9_gd_correction as train_v9

    artifact = {
        "feature_names": ["gd_global_disorder_score", "gd_corruption_rate"],
        "coef": [10.0, 20.0],
        "max_uplift": 25.0,
    }

    corrected = train_v9.apply_gd_correction(
        40.0,
        {"gd_global_disorder_score": 0.5, "gd_corruption_rate": 2.0},
        artifact,
    )

    assert corrected == 65.0


def test_apply_gd_correction_clamps_to_mes_range() -> None:
    import train_v9_gd_correction as train_v9

    artifact = {
        "feature_names": ["gd_global_disorder_score"],
        "coef": [50.0],
        "max_uplift": 50.0,
    }

    corrected = train_v9.apply_gd_correction(
        80.0,
        {"gd_global_disorder_score": 1.0},
        artifact,
    )

    assert corrected == 100.0


def test_correction_feature_values_adds_density_and_marker_union() -> None:
    import train_v9_gd_correction as train_v9

    values = train_v9.correction_feature_values({
        "gd_disorder_marker_count_norm": 0.4,
        "gd_disorder_markers_per_500w": 30.0,
        "gd_global_disorder_score": 0.5,
    })

    assert values["gd_density_cap"] == 1.0
    assert values["gd_marker_or_density"] == 1.0
    assert values["gd_marker_x_global"] == 0.2


def test_train_correction_uses_selected_candidate_weights_by_default() -> None:
    import inspect

    import train_v9_gd_correction as train_v9

    signature = inspect.signature(train_v9.train_correction)

    assert signature.parameters["high_label_weight"].default == 50.0
    assert signature.parameters["false_low_weight"].default == 2.0


def test_acceptance_summary_requires_locked_improvement_without_control_regression() -> None:
    import train_v9_gd_correction as train_v9

    baseline = {
        "metrics": {"mae": 20.12},
        "per_band": {
            "spectrum_7_8": {"mae": 23.45},
            "spectrum_9_10": {"mae": 39.36},
        },
        "control_checks": {
            "emotional_coherent_negative_mean_model_mes": 30.11,
            "neutral_fragmented_positive_mean_model_mes": 52.37,
        },
    }
    candidate = {
        "metrics": {"mae": 19.5},
        "per_band": {
            "spectrum_7_8": {"mae": 18.0},
            "spectrum_9_10": {"mae": 30.0},
        },
        "control_checks": {
            "emotional_coherent_negative_mean_model_mes": 32.0,
            "neutral_fragmented_positive_mean_model_mes": 61.0,
        },
    }

    summary = train_v9.acceptance_summary(candidate, baseline)

    assert summary["promotable"] is True
    assert summary["overall_mae_improved"] is True
    assert summary["neutral_fragmented_improved"] is True
    assert summary["neutral_fragmented_gte_60"] is True
    assert summary["emotional_coherent_not_regressed"] is True
    assert summary["high_band_mae_improved"] is True


def test_acceptance_summary_rejects_worse_locked_mae() -> None:
    import train_v9_gd_correction as train_v9

    baseline = {
        "metrics": {"mae": 20.12},
        "per_band": {
            "spectrum_7_8": {"mae": 23.45},
            "spectrum_9_10": {"mae": 39.36},
        },
        "control_checks": {
            "emotional_coherent_negative_mean_model_mes": 30.11,
            "neutral_fragmented_positive_mean_model_mes": 52.37,
        },
    }
    candidate = {
        "metrics": {"mae": 22.0},
        "per_band": {
            "spectrum_7_8": {"mae": 20.0},
            "spectrum_9_10": {"mae": 30.0},
        },
        "control_checks": {
            "emotional_coherent_negative_mean_model_mes": 28.0,
            "neutral_fragmented_positive_mean_model_mes": 60.0,
        },
    }

    summary = train_v9.acceptance_summary(candidate, baseline)

    assert summary["promotable"] is False
    assert summary["overall_mae_improved"] is False


def test_acceptance_summary_rejects_neutral_fragmented_below_target() -> None:
    import train_v9_gd_correction as train_v9

    baseline = {
        "metrics": {"mae": 20.12},
        "per_band": {
            "spectrum_7_8": {"mae": 23.45},
            "spectrum_9_10": {"mae": 39.36},
        },
        "control_checks": {
            "emotional_coherent_negative_mean_model_mes": 30.11,
            "neutral_fragmented_positive_mean_model_mes": 52.37,
        },
    }
    candidate = {
        "metrics": {"mae": 18.85},
        "per_band": {
            "spectrum_7_8": {"mae": 18.0},
            "spectrum_9_10": {"mae": 30.0},
        },
        "control_checks": {
            "emotional_coherent_negative_mean_model_mes": 33.42,
            "neutral_fragmented_positive_mean_model_mes": 58.83,
        },
    }

    summary = train_v9.acceptance_summary(candidate, baseline)

    assert summary["promotable"] is False
    assert summary["neutral_fragmented_improved"] is True
    assert summary["neutral_fragmented_gte_60"] is False
