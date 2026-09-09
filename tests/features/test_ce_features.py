"""Tests for the CE (coherence entropy) feature extraction module.

These tests use synthetic unit vectors (no transformer model needed).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mental_entropy.features.ce import (
    BREAK_T,
    CE_FEATURE_KEYS,
    SHARP_DROP_T,
    ce_features,
    ce_features_from_result,
)
from mental_entropy.embedding.types import EmbeddingResult, SentenceEmbedding

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 1024


def make_unit_vector(direction: int = 0) -> list[float]:
    """Create a unit vector pointing along a given axis in 1024-dim space."""
    vec = [0.0] * EMBEDDING_DIM
    vec[direction % EMBEDDING_DIM] = 1.0
    return vec


def make_unit_vector_from_2d(x: float, y: float) -> list[float]:
    """
    Create a 1024-dim unit vector from a 2D direction (x, y).
    
    The 2D vector is normalized and embedded into the first two dimensions.
    """
    norm = math.sqrt(x * x + y * y)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    vec = [0.0] * EMBEDDING_DIM
    vec[0] = x / norm
    vec[1] = y / norm
    return vec


def make_embedding_result(embeddings: list[list[float]]) -> EmbeddingResult:
    """Create an EmbeddingResult from a list of embeddings."""
    sentences = [
        SentenceEmbedding(id=i, text=f"Sentence {i}", embedding=emb)
        for i, emb in enumerate(embeddings)
    ]
    # doc_embedding doesn't matter for CE features
    return EmbeddingResult(sentences=sentences, doc_embedding=None)


# ---------------------------------------------------------------------------
# Test: exact keyset
# ---------------------------------------------------------------------------


def test_ce_features_returns_exact_keys():
    """CE features dict must have exactly the expected keys."""
    # Use a simple 2-sentence case
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    features = ce_features(embeddings)

    assert set(features.keys()) == CE_FEATURE_KEYS
    assert len(features) == 29  # 24 original + 5 block-aware features


def test_ce_features_all_values_json_serializable():
    """All returned values must be Python int or float (not numpy types)."""
    import json
    
    embeddings = [make_unit_vector(0), make_unit_vector(0), make_unit_vector(1)]
    features = ce_features(embeddings)
    
    # Should not raise
    json_str = json.dumps(features)
    assert json_str  # Non-empty
    
    # Check types explicitly
    for key, value in features.items():
        assert isinstance(value, (int, float)), f"Key {key} has type {type(value)}"
        assert not isinstance(value, np.generic), f"Key {key} is a numpy type"


# ---------------------------------------------------------------------------
# Test: edge cases (n=0, n=1)
# ---------------------------------------------------------------------------


def test_ce_features_empty_input():
    """n=0: return all keys with zero defaults."""
    features = ce_features([])
    
    assert set(features.keys()) == CE_FEATURE_KEYS
    assert features["ce_n_sentences"] == 0
    assert features["ce_n_adj"] == 0
    assert features["ce_adj_mean"] == 0.0
    assert features["ce_break_count"] == 0
    assert features["ce_longest_coherent_run"] == 0
    assert features["ce_skip_mean"] == 0.0


def test_ce_features_single_sentence():
    """n=1: return all keys with zero defaults except ce_n_sentences=1."""
    features = ce_features([make_unit_vector(0)])
    
    assert set(features.keys()) == CE_FEATURE_KEYS
    assert features["ce_n_sentences"] == 1
    assert features["ce_n_adj"] == 0
    assert features["ce_adj_mean"] == 0.0
    assert features["ce_break_count"] == 0
    assert features["ce_longest_coherent_run"] == 0


# ---------------------------------------------------------------------------
# Test: perfectly coherent text (identical embeddings)
# ---------------------------------------------------------------------------


def test_ce_features_perfectly_coherent():
    """Identical embeddings => adj_mean=1, no breaks, full coherent run."""
    n = 5
    embeddings = [make_unit_vector(0) for _ in range(n)]
    features = ce_features(embeddings)
    
    assert features["ce_n_sentences"] == n
    assert features["ce_n_adj"] == n - 1
    
    # All adjacent similarities should be 1.0 (dot product of identical unit vectors)
    assert abs(features["ce_adj_mean"] - 1.0) < 1e-5
    assert abs(features["ce_adj_min"] - 1.0) < 1e-5
    assert abs(features["ce_adj_max"] - 1.0) < 1e-5
    assert abs(features["ce_adj_std"]) < 1e-5  # No variance
    
    # No breaks (1.0 > BREAK_T)
    assert features["ce_break_count"] == 0
    assert features["ce_break_rate"] == 0.0
    assert features["ce_longest_coherent_run"] == n - 1
    
    # No drops (all similarities are equal)
    assert abs(features["ce_drop_mean"]) < 1e-5
    assert abs(features["ce_drop_std"]) < 1e-5
    assert features["ce_sharp_drop_count"] == 0
    
    # No low-coherence mass (all above threshold)
    assert features["ce_low_mass"] == 0.0
    assert features["ce_low_mass_sum"] == 0.0
    
    # Skip coherence also perfect
    assert abs(features["ce_skip_mean"] - 1.0) < 1e-5
    assert abs(features["ce_skip_min"] - 1.0) < 1e-5
    assert features["ce_skip_break_count"] == 0


# ---------------------------------------------------------------------------
# Test: fragmented text (alternating orthogonal vectors)
# ---------------------------------------------------------------------------


def test_ce_features_fragmented_alternating():
    """
    Alternating orthogonal vectors: A, B, A, B, A
    Adjacent sims should be 0, skip sims should be 1.
    """
    # A = axis 0, B = axis 1 (orthogonal)
    a = make_unit_vector(0)
    b = make_unit_vector(1)
    embeddings = [a, b, a, b, a]  # n=5, n_adj=4
    
    features = ce_features(embeddings)
    
    assert features["ce_n_sentences"] == 5
    assert features["ce_n_adj"] == 4
    
    # Adjacent sims: dot(A,B) = 0 for all
    assert abs(features["ce_adj_mean"]) < 1e-5
    assert abs(features["ce_adj_min"]) < 1e-5
    assert abs(features["ce_adj_max"]) < 1e-5
    
    # All adjacencies are breaks (0 < BREAK_T)
    assert features["ce_break_count"] == 4
    assert abs(features["ce_break_rate"] - 1.0) < 1e-5
    assert features["ce_longest_coherent_run"] == 0
    
    # Low-coherence mass: all at 0, so low[i] = max(0, BREAK_T - 0) = BREAK_T
    expected_low_mass = BREAK_T
    assert abs(features["ce_low_mass"] - expected_low_mass) < 1e-5
    assert abs(features["ce_low_mass_sum"] - expected_low_mass * 4) < 1e-5
    
    # Skip coherence: dot(A,A) = 1, dot(B,B) = 1 => skip sims all 1.0
    # s2 has length n-2 = 3: (A,A), (B,B), (A,A)
    assert abs(features["ce_skip_mean"] - 1.0) < 1e-5
    assert abs(features["ce_skip_min"] - 1.0) < 1e-5
    assert features["ce_skip_break_count"] == 0


# ---------------------------------------------------------------------------
# Test: sharp drop scenario
# ---------------------------------------------------------------------------


def test_ce_features_sharp_drop():
    """
    Construct s=[0.9, 0.6] using 2D-plane unit vectors.
    drop[0] = 0.6 - 0.9 = -0.3 < SHARP_DROP_T (-0.15) => 1 sharp drop.
    """
    # We need 3 embeddings e0, e1, e2 such that:
    # dot(e0, e1) = 0.9
    # dot(e1, e2) = 0.6
    
    # e0 = (1, 0)
    e0 = make_unit_vector_from_2d(1.0, 0.0)
    
    # e1 such that dot(e0, e1) = 0.9
    # e1 = (0.9, sqrt(1 - 0.81)) = (0.9, ~0.436)
    e1 = make_unit_vector_from_2d(0.9, math.sqrt(1 - 0.81))
    
    # e2 such that dot(e1, e2) = 0.6
    # e1 · e2 = 0.9 * e2[0] + 0.436 * e2[1] = 0.6
    # Let's pick e2[0] = x, e2[1] = y with x^2 + y^2 = 1
    # 0.9x + 0.436y = 0.6
    # y = (0.6 - 0.9x) / 0.436
    # We need x^2 + ((0.6 - 0.9x)/0.436)^2 = 1
    # Let's solve: pick x = 0.5 => y = (0.6 - 0.45)/0.436 = 0.344
    # Check: 0.5^2 + 0.344^2 = 0.25 + 0.118 = 0.368 != 1
    # Need to normalize: norm = sqrt(0.368) = 0.607
    # e2_raw = (0.5, 0.344), normalize => e2 = (0.824, 0.567)
    # Verify: dot(e1, e2) = 0.9*0.824 + 0.436*0.567 = 0.742 + 0.247 = 0.989 (not 0.6)
    
    # Different approach: use angle-based construction
    # cos(theta) = similarity
    # e0 at angle 0
    # e1 at angle such that cos(theta1) = 0.9 => theta1 = acos(0.9) ≈ 25.8°
    # e2 at angle theta2 such that cos(theta2 - theta1) = 0.6 => theta2 - theta1 = acos(0.6) ≈ 53.1°
    # So theta2 ≈ 78.9°
    
    theta1 = math.acos(0.9)  # angle between e0 and e1
    # e1 = (cos(theta1), sin(theta1))
    e1 = make_unit_vector_from_2d(math.cos(theta1), math.sin(theta1))
    
    # e2 at angle theta2 = theta1 + acos(0.6) from e1's perspective
    # But we want dot(e1, e2) = 0.6
    # If e1 is at angle theta1, and e2 is at angle theta2,
    # dot(e1, e2) = cos(theta2 - theta1) = 0.6
    # So theta2 = theta1 + acos(0.6)
    theta2 = theta1 + math.acos(0.6)
    e2 = make_unit_vector_from_2d(math.cos(theta2), math.sin(theta2))
    
    # Verify our construction
    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))
    
    s0 = dot(e0, e1)
    s1 = dot(e1, e2)
    assert abs(s0 - 0.9) < 0.01, f"Expected s0=0.9, got {s0}"
    assert abs(s1 - 0.6) < 0.01, f"Expected s1=0.6, got {s1}"
    
    embeddings = [e0, e1, e2]
    features = ce_features(embeddings)
    
    assert features["ce_n_sentences"] == 3
    assert features["ce_n_adj"] == 2
    
    # Verify adjacency stats
    assert abs(features["ce_adj_mean"] - 0.75) < 0.01  # (0.9 + 0.6) / 2
    assert abs(features["ce_adj_min"] - 0.6) < 0.01
    assert abs(features["ce_adj_max"] - 0.9) < 0.01
    
    # Drop: s1 - s0 = 0.6 - 0.9 = -0.3
    assert abs(features["ce_drop_min"] - (-0.3)) < 0.01
    
    # Sharp drop: -0.3 < SHARP_DROP_T (-0.15) => 1 sharp drop
    assert features["ce_sharp_drop_count"] == 1
    assert abs(features["ce_sharp_drop_rate"] - 1.0) < 0.01  # 1/1


# ---------------------------------------------------------------------------
# Test: NaN policy
# ---------------------------------------------------------------------------


def test_ce_features_raises_on_nan():
    """A single NaN in any embedding should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    embeddings[0][0] = float("nan")
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        ce_features(embeddings)


def test_ce_features_raises_on_inf():
    """Inf in any embedding should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    embeddings[1][5] = float("inf")
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        ce_features(embeddings)


# ---------------------------------------------------------------------------
# Test: wrapper for EmbeddingResult
# ---------------------------------------------------------------------------


def test_ce_features_from_result():
    """Test the EmbeddingResult wrapper."""
    embeddings = [make_unit_vector(0), make_unit_vector(0), make_unit_vector(1)]
    result = make_embedding_result(embeddings)
    
    features = ce_features_from_result(result)
    
    # Should match direct call
    expected = ce_features(embeddings)
    assert features == expected


def test_ce_features_from_result_empty():
    """Test wrapper with empty result."""
    result = EmbeddingResult(sentences=[], doc_embedding=None)
    features = ce_features_from_result(result)
    
    assert features["ce_n_sentences"] == 0
    assert set(features.keys()) == CE_FEATURE_KEYS


# ---------------------------------------------------------------------------
# Test: two sentences (boundary for drop calculation)
# ---------------------------------------------------------------------------


def test_ce_features_two_sentences():
    """n=2: has one adjacency but no drops (drop array empty)."""
    embeddings = [make_unit_vector(0), make_unit_vector(1)]
    features = ce_features(embeddings)
    
    assert features["ce_n_sentences"] == 2
    assert features["ce_n_adj"] == 1
    
    # Adjacent sim = 0 (orthogonal)
    assert abs(features["ce_adj_mean"]) < 1e-5
    
    # No drops possible (need 2+ adjacencies)
    assert features["ce_drop_min"] == 0.0
    assert features["ce_drop_mean"] == 0.0
    assert features["ce_sharp_drop_count"] == 0
    
    # No skip connections possible (need n >= 3)
    assert features["ce_skip_mean"] == 0.0
    assert features["ce_skip_min"] == 0.0
    assert features["ce_skip_break_count"] == 0


# ---------------------------------------------------------------------------
# Test: percentile edge cases
# ---------------------------------------------------------------------------


def test_ce_features_percentiles_with_few_values():
    """Percentiles should work correctly even with few values."""
    # 2 sentences = 1 adjacency => percentiles all equal to that one value
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    features = ce_features(embeddings)
    
    # All percentiles should equal the single value (1.0)
    assert abs(features["ce_adj_p10"] - 1.0) < 1e-5
    assert abs(features["ce_adj_p25"] - 1.0) < 1e-5
    assert abs(features["ce_adj_median"] - 1.0) < 1e-5
    assert abs(features["ce_adj_p75"] - 1.0) < 1e-5
    assert abs(features["ce_adj_p90"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test: longest coherent run
# ---------------------------------------------------------------------------


def test_ce_features_longest_coherent_run_mixed():
    """Test longest coherent run with mixed coherent/incoherent."""
    # Create pattern: coherent, coherent, break, coherent
    # Similarities: 1.0, 1.0, 0.0, 1.0
    # coherent mask: T, T, F, T
    # longest run: 2
    a = make_unit_vector(0)
    b = make_unit_vector(1)
    embeddings = [a, a, a, b, a]  # sims: 1, 1, 0, 0
    
    features = ce_features(embeddings)
    
    # Coherent mask: [T, T, F, F] (1.0 >= BREAK_T, 1.0 >= BREAK_T, 0 < BREAK_T, 0 < BREAK_T)
    assert features["ce_longest_coherent_run"] == 2


def test_ce_features_longest_coherent_run_all_breaks():
    """All adjacencies below threshold => longest run = 0."""
    # All orthogonal => all sims = 0
    embeddings = [make_unit_vector(i) for i in range(5)]
    features = ce_features(embeddings)
    
    assert features["ce_longest_coherent_run"] == 0
    assert features["ce_break_count"] == 4


# ---------------------------------------------------------------------------
# Test: low-coherence mass
# ---------------------------------------------------------------------------


def test_ce_features_low_mass_partial():
    """Test low mass when some values are below threshold."""
    # Create s = [0.5, 0.2]
    # With BREAK_T=0.45:
    # low = [max(0, 0.45-0.5), max(0, 0.45-0.2)] = [0, 0.25]
    # low_mass = mean = 0.1, low_mass_sum = 0.2
    
    # Need 3 embeddings with dot(e0,e1)=0.5, dot(e1,e2)=0.2
    theta1 = math.acos(0.5)
    theta2 = theta1 + math.acos(0.2)
    
    e0 = make_unit_vector_from_2d(1.0, 0.0)
    e1 = make_unit_vector_from_2d(math.cos(theta1), math.sin(theta1))
    e2 = make_unit_vector_from_2d(math.cos(theta2), math.sin(theta2))
    
    embeddings = [e0, e1, e2]
    features = ce_features(embeddings)
    
    # low = [0, 0.2], mean = 0.1
    # Use BREAK_T from module to stay in sync
    expected_low = max(0, BREAK_T - 0.2)  # BREAK_T - 0.2
    expected_mass = expected_low / 2  # mean of [0, 0.2]
    assert abs(features["ce_low_mass"] - expected_mass) < 0.01
    assert abs(features["ce_low_mass_sum"] - expected_low) < 0.01


# ---------------------------------------------------------------------------
# Test: skip connections
# ---------------------------------------------------------------------------


def test_ce_features_skip_connections():
    """Test skip connection computation with n=4."""
    # 4 sentences => s2 has length 2: dot(e0,e2), dot(e1,e3)
    a = make_unit_vector(0)
    b = make_unit_vector(1)
    embeddings = [a, b, a, b]
    
    features = ce_features(embeddings)
    
    # s2 = [dot(a,a), dot(b,b)] = [1.0, 1.0]
    assert abs(features["ce_skip_mean"] - 1.0) < 1e-5
    assert abs(features["ce_skip_min"] - 1.0) < 1e-5
    assert features["ce_skip_break_count"] == 0


def test_ce_features_skip_connections_with_breaks():
    """Test skip connection breaks."""
    # All different orthogonal vectors => all skip sims = 0
    embeddings = [make_unit_vector(i) for i in range(4)]
    features = ce_features(embeddings)
    
    # s2 = [0, 0] (all orthogonal)
    assert abs(features["ce_skip_mean"]) < 1e-5
    assert abs(features["ce_skip_min"]) < 1e-5
    # 0 < BREAK_T => all are breaks
    assert features["ce_skip_break_count"] == 2
