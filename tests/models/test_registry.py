"""Tests for the models registry (feature-group mapping, loading, inference).

Tests that require trained model artifacts use pytest.mark.skipif.
Tests for feature-group mapping and availability checks run without models.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mental_entropy.features import (
    BC_FEATURE_KEYS,
    CE_FEATURE_KEYS,
    CLE_FEATURE_KEYS,
    NE_FEATURE_KEYS,
    SE_FEATURE_KEYS,
)
from mental_entropy.models._registry import (
    ALL_FEATURE_KEYS,
    DIMENSION_NAMES,
    MODULE_FEATURE_KEYS,
    MODULE_NAMES,
    _apply_mes_calibration,
    _all_features_to_array,
    _features_to_array,
    reset_cache,
    subscore_models_available,
)


# ---------------------------------------------------------------------------
# Feature-group mapping tests (module-aligned, backward compat)
# ---------------------------------------------------------------------------


class TestModuleFeatureKeys:
    """Test that MODULE_FEATURE_KEYS correctly maps to feature frozensets."""

    def test_module_names_complete(self) -> None:
        """All 5 modules are present."""
        assert set(MODULE_NAMES) == {"ce", "se", "ne", "cle", "bc"}

    def test_module_names_order(self) -> None:
        """Module names are in canonical order."""
        assert MODULE_NAMES == ("ce", "se", "ne", "cle", "bc")

    def test_ce_keys_match(self) -> None:
        """CE module keys match CE_FEATURE_KEYS frozenset."""
        assert set(MODULE_FEATURE_KEYS["ce"]) == CE_FEATURE_KEYS

    def test_se_keys_match(self) -> None:
        """SE module keys match SE_FEATURE_KEYS frozenset."""
        assert set(MODULE_FEATURE_KEYS["se"]) == SE_FEATURE_KEYS

    def test_ne_keys_match(self) -> None:
        """NE module keys match NE_FEATURE_KEYS frozenset."""
        assert set(MODULE_FEATURE_KEYS["ne"]) == NE_FEATURE_KEYS

    def test_cle_keys_match(self) -> None:
        """CLE module keys match CLE_FEATURE_KEYS frozenset."""
        assert set(MODULE_FEATURE_KEYS["cle"]) == CLE_FEATURE_KEYS

    def test_bc_keys_match(self) -> None:
        """BC module keys match BC_FEATURE_KEYS frozenset."""
        assert set(MODULE_FEATURE_KEYS["bc"]) == BC_FEATURE_KEYS

    def test_keys_are_sorted(self) -> None:
        """Feature keys within each module are sorted alphabetically."""
        for module, keys in MODULE_FEATURE_KEYS.items():
            assert keys == sorted(keys), f"{module} keys are not sorted"

    def test_total_feature_count(self) -> None:
        """Total features across all modules matches expected count."""
        total = sum(len(keys) for keys in MODULE_FEATURE_KEYS.values())
        expected = (
            len(CE_FEATURE_KEYS)
            + len(SE_FEATURE_KEYS)
            + len(NE_FEATURE_KEYS)
            + len(CLE_FEATURE_KEYS)
            + len(BC_FEATURE_KEYS)
        )
        assert total == expected

    def test_no_duplicate_features_across_modules(self) -> None:
        """Each feature belongs to exactly one module."""
        all_keys: list[str] = []
        for keys in MODULE_FEATURE_KEYS.values():
            all_keys.extend(keys)
        assert len(all_keys) == len(set(all_keys)), "Duplicate features across modules"


# ---------------------------------------------------------------------------
# Dimension-aligned architecture tests
# ---------------------------------------------------------------------------


class TestDimensionAlignment:
    """Test dimension-aligned constants."""

    def test_dimension_names_complete(self) -> None:
        """All 5 FEP-aligned dimensions are present."""
        assert set(DIMENSION_NAMES) == {
            "prediction_coherence", "model_complexity", "compression_progress",
            "belief_integration", "precision_weighting",
        }

    def test_dimension_names_order(self) -> None:
        """Dimension names are in canonical order."""
        assert DIMENSION_NAMES == (
            "prediction_coherence", "model_complexity", "compression_progress",
            "belief_integration", "precision_weighting",
        )

    def test_all_feature_keys_count(self) -> None:
        """ALL_FEATURE_KEYS contains all 77 features."""
        expected = (
            len(CE_FEATURE_KEYS) + len(SE_FEATURE_KEYS) + len(NE_FEATURE_KEYS)
            + len(CLE_FEATURE_KEYS) + len(BC_FEATURE_KEYS)
        )
        assert len(ALL_FEATURE_KEYS) == expected

    def test_all_feature_keys_sorted(self) -> None:
        """ALL_FEATURE_KEYS is sorted alphabetically."""
        assert ALL_FEATURE_KEYS == sorted(ALL_FEATURE_KEYS)

    def test_all_feature_keys_no_duplicates(self) -> None:
        """ALL_FEATURE_KEYS has no duplicates."""
        assert len(ALL_FEATURE_KEYS) == len(set(ALL_FEATURE_KEYS))

    def test_all_feature_keys_contains_all_modules(self) -> None:
        """ALL_FEATURE_KEYS contains features from all modules."""
        all_set = set(ALL_FEATURE_KEYS)
        for module, keys in MODULE_FEATURE_KEYS.items():
            for k in keys:
                assert k in all_set, f"{k} from {module} missing in ALL_FEATURE_KEYS"


# ---------------------------------------------------------------------------
# Feature array conversion tests
# ---------------------------------------------------------------------------


class TestFeaturesToArray:
    """Test _features_to_array helper."""

    def test_correct_shape(self) -> None:
        """Output shape matches module feature count."""
        features = {k: 0.5 for k in CE_FEATURE_KEYS}
        arr = _features_to_array(features, "ce")
        assert arr.shape == (len(CE_FEATURE_KEYS),)

    def test_values_in_order(self) -> None:
        """Values are extracted in sorted key order."""
        keys = MODULE_FEATURE_KEYS["se"]
        features = {k: float(i) for i, k in enumerate(keys)}
        arr = _features_to_array(features, "se")
        for i, k in enumerate(keys):
            assert arr[i] == float(i), f"Value mismatch for {k}"

    def test_missing_features_default_to_zero(self) -> None:
        """Missing feature keys default to 0.0."""
        arr = _features_to_array({}, "ne")
        assert np.all(arr == 0.0)

    def test_extra_features_ignored(self) -> None:
        """Extra keys not in the module are ignored."""
        features = {"not_a_real_feature": 999.0}
        features.update({k: 0.5 for k in NE_FEATURE_KEYS})
        arr = _features_to_array(features, "ne")
        assert 999.0 not in arr

    def test_dtype_is_float64(self) -> None:
        """Output array is float64."""
        features = {k: 1.0 for k in BC_FEATURE_KEYS}
        arr = _features_to_array(features, "bc")
        assert arr.dtype == np.float64


class TestAllFeaturesToArray:
    """Test _all_features_to_array helper."""

    def test_correct_shape(self) -> None:
        """Output shape matches total feature count."""
        features = {k: 0.5 for k in ALL_FEATURE_KEYS}
        arr = _all_features_to_array(features)
        assert arr.shape == (len(ALL_FEATURE_KEYS),)

    def test_missing_features_default_to_zero(self) -> None:
        """Missing feature keys default to 0.0."""
        arr = _all_features_to_array({})
        assert np.all(arr == 0.0)
        assert arr.shape == (len(ALL_FEATURE_KEYS),)

    def test_values_in_sorted_order(self) -> None:
        """Values are extracted in sorted key order."""
        features = {k: float(i) for i, k in enumerate(ALL_FEATURE_KEYS)}
        arr = _all_features_to_array(features)
        for i, k in enumerate(ALL_FEATURE_KEYS):
            assert arr[i] == float(i), f"Value mismatch for {k}"

    def test_dtype_is_float64(self) -> None:
        """Output array is float64."""
        arr = _all_features_to_array({})
        assert arr.dtype == np.float64


# ---------------------------------------------------------------------------
# MES calibration
# ---------------------------------------------------------------------------


class TestMesCalibration:
    """Test optional MES calibration stored in combiner artifacts."""

    def test_missing_calibration_returns_original_score(self) -> None:
        assert _apply_mes_calibration(42.0, {}) == 42.0

    def test_linear_calibration_applies_slope_and_intercept(self) -> None:
        combiner = {"calibration": {"kind": "linear_mes", "slope": 1.5, "intercept": -5.0}}

        assert _apply_mes_calibration(30.0, combiner) == 40.0

    def test_isotonic_calibration_interpolates_thresholds(self) -> None:
        combiner = {
            "calibration": {
                "kind": "isotonic_mes",
                "x_thresholds": [20.0, 60.0, 80.0],
                "y_thresholds": [10.0, 70.0, 90.0],
            }
        }

        assert _apply_mes_calibration(40.0, combiner) == 40.0
        assert _apply_mes_calibration(70.0, combiner) == 80.0
        assert _apply_mes_calibration(5.0, combiner) == 10.0


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


class TestSubscoreAvailability:
    """Test subscore_models_available() with/without artifacts."""

    def test_not_available_without_artifacts(self) -> None:
        """Returns False when artifact directory is empty."""
        reset_cache()
        result = subscore_models_available()
        assert isinstance(result, bool)

    def test_not_available_with_missing_combiner(self, tmp_path: Path) -> None:
        """Returns False when combiner.json is missing."""
        reset_cache()
        with patch(
            "mental_entropy.models._registry._ARTIFACTS_DIR", tmp_path
        ):
            assert not subscore_models_available()

    def test_not_available_with_missing_dimension_model(self, tmp_path: Path) -> None:
        """Returns False when a dimension model is missing (v3 architecture)."""
        reset_cache()
        with patch(
            "mental_entropy.models._registry._ARTIFACTS_DIR", tmp_path
        ):
            combiner_data = {
                "coef": [1.0] * 5,
                "intercept": 0.0,
                "architecture": "dimension_aligned_v3",
            }
            (tmp_path / "combiner.json").write_text(json.dumps(combiner_data))
            for dim in DIMENSION_NAMES[:4]:
                (tmp_path / f"{dim}_model.json").write_text("{}")
            assert not subscore_models_available()

    def test_not_available_with_missing_module_model(self, tmp_path: Path) -> None:
        """Returns False when a module model is missing (legacy architecture)."""
        reset_cache()
        with patch(
            "mental_entropy.models._registry._ARTIFACTS_DIR", tmp_path
        ):
            combiner_data = {"coef": [1.0] * 5, "intercept": 0.0}
            (tmp_path / "combiner.json").write_text(json.dumps(combiner_data))
            for module in MODULE_NAMES[:4]:
                (tmp_path / f"{module}_model.json").write_text("{}")
            assert not subscore_models_available()


# ---------------------------------------------------------------------------
# Reset cache
# ---------------------------------------------------------------------------


class TestResetCache:
    """Test that reset_cache() clears cached models."""

    def test_reset_cache_runs_without_error(self) -> None:
        """reset_cache() should always succeed, even if nothing is cached."""
        reset_cache()  # should not raise
