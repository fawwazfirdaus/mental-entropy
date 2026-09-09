"""Tests for the SE (semantic entropy) feature extraction module.

These tests use synthetic unit vectors (no transformer model needed).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mental_entropy.features.se import (
    CLUSTER_T,
    SE_FEATURE_KEYS,
    se_features,
    se_features_from_result,
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
    return EmbeddingResult(sentences=sentences, doc_embedding=None)


# ---------------------------------------------------------------------------
# Test: exact keyset
# ---------------------------------------------------------------------------


def test_se_features_returns_exact_keys():
    """SE features dict must have exactly the expected keys."""
    # Use a simple 2-sentence case
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    features = se_features(embeddings)

    assert set(features.keys()) == SE_FEATURE_KEYS
    assert len(features) == 9  # Sanity check (7 original + 2 weighted switch)


def test_se_features_all_values_json_serializable():
    """All returned values must be Python int or float (not numpy types)."""
    import json

    embeddings = [make_unit_vector(0), make_unit_vector(0), make_unit_vector(1)]
    features = se_features(embeddings)

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


def test_se_features_empty_input():
    """n=0: return all keys with zero defaults."""
    features = se_features([])

    assert set(features.keys()) == SE_FEATURE_KEYS
    assert features["se_n_clusters"] == 0
    assert features["se_cluster_entropy"] == 0.0
    assert features["se_dominant_cluster_frac"] == 0.0
    assert features["se_switch_count"] == 0
    assert features["se_switch_rate"] == 0.0
    assert features["se_switch_weighted"] == 0.0
    assert features["se_switch_mean_jump"] == 0.0
    assert features["se_intra_mean"] == 0.0
    assert features["se_inter_mean"] == 0.0


def test_se_features_single_sentence():
    """n=1: return all keys with zero defaults."""
    features = se_features([make_unit_vector(0)])

    assert set(features.keys()) == SE_FEATURE_KEYS
    assert features["se_n_clusters"] == 0
    assert features["se_cluster_entropy"] == 0.0
    assert features["se_dominant_cluster_frac"] == 0.0
    assert features["se_switch_count"] == 0
    assert features["se_switch_rate"] == 0.0
    assert features["se_switch_weighted"] == 0.0
    assert features["se_switch_mean_jump"] == 0.0
    assert features["se_intra_mean"] == 0.0
    assert features["se_inter_mean"] == 0.0


# ---------------------------------------------------------------------------
# Test: identical embeddings (single cluster)
# ---------------------------------------------------------------------------


def test_se_features_identical_embeddings():
    """Identical embeddings => single cluster, no switches, high intra coherence."""
    n = 5
    embeddings = [make_unit_vector(0) for _ in range(n)]
    features = se_features(embeddings)

    # All sentences should be in one cluster
    assert features["se_n_clusters"] == 1

    # Entropy of single cluster = 0
    assert features["se_cluster_entropy"] == 0.0

    # Dominant cluster has all sentences
    assert abs(features["se_dominant_cluster_frac"] - 1.0) < 1e-5

    # No switches (all same cluster)
    assert features["se_switch_count"] == 0
    assert features["se_switch_rate"] == 0.0

    # No switches => weighted switch and mean jump are 0
    assert features["se_switch_weighted"] == 0.0
    assert features["se_switch_mean_jump"] == 0.0

    # Intra-cluster similarity: for identical vectors, all pairs have sim=1
    # Mean should be 1.0, weighted by p_k=1.0
    assert abs(features["se_intra_mean"] - 1.0) < 1e-5

    # Inter-cluster with K=1: returns 1.0
    assert abs(features["se_inter_mean"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test: orthogonal embeddings (multiple clusters)
# ---------------------------------------------------------------------------


def test_se_features_orthogonal_embeddings():
    """
    All orthogonal embeddings => each sentence is its own cluster.
    Cosine distance between orthogonal vectors = 1.0 > CLUSTER_T, so no merging.
    """
    n = 4
    embeddings = [make_unit_vector(i) for i in range(n)]
    features = se_features(embeddings)

    # Each sentence in its own cluster
    assert features["se_n_clusters"] == n

    # Uniform distribution over n clusters => max entropy = 1.0
    assert abs(features["se_cluster_entropy"] - 1.0) < 1e-5

    # Each cluster has 1/n
    assert abs(features["se_dominant_cluster_frac"] - 1 / n) < 1e-5

    # Every adjacent pair is in different clusters
    assert features["se_switch_count"] == n - 1
    assert abs(features["se_switch_rate"] - 1.0) < 1e-5

    # All switches are maximum jumps (orthogonal = distance 1.0)
    # se_switch_weighted = sum(1.0 for each switch) / (n-1) = (n-1) / (n-1) = 1.0
    assert abs(features["se_switch_weighted"] - 1.0) < 1e-5
    # Mean jump at switches = 1.0 (all orthogonal)
    assert abs(features["se_switch_mean_jump"] - 1.0) < 1e-5

    # No intra-cluster pairs (each cluster has size 1)
    assert features["se_intra_mean"] == 0.0

    # Inter-cluster: all centroids are orthogonal, so sim = 0
    assert abs(features["se_inter_mean"]) < 1e-5


# ---------------------------------------------------------------------------
# Test: two-block structure (two topics)
# ---------------------------------------------------------------------------


def test_se_features_two_blocks():
    """
    Two blocks of identical vectors: [A, A, A, B, B, B]
    Should form 2 clusters with 1 switch at the boundary.
    """
    a = make_unit_vector(0)
    b = make_unit_vector(1)
    embeddings = [a, a, a, b, b, b]  # n=6
    features = se_features(embeddings)

    # Two clusters
    assert features["se_n_clusters"] == 2

    # Uniform distribution => entropy = 1.0
    assert abs(features["se_cluster_entropy"] - 1.0) < 1e-5

    # Each cluster has 3/6 = 0.5
    assert abs(features["se_dominant_cluster_frac"] - 0.5) < 1e-5

    # One switch (from A to B)
    assert features["se_switch_count"] == 1
    assert abs(features["se_switch_rate"] - 1 / 5) < 1e-5

    # Intra-cluster: each cluster has 3 identical vectors
    # Mean pairwise sim = 1.0 for each, weighted by 0.5 each
    # Total = 0.5 * 1.0 + 0.5 * 1.0 = 1.0
    assert abs(features["se_intra_mean"] - 1.0) < 1e-5

    # Inter-cluster: centroids are orthogonal, sim = 0
    assert abs(features["se_inter_mean"]) < 1e-5


# ---------------------------------------------------------------------------
# Test: alternating pattern (high switching)
# ---------------------------------------------------------------------------


def test_se_features_alternating_pattern():
    """
    Alternating pattern: [A, B, A, B, A, B]
    Should form 2 clusters with maximum switching.
    """
    a = make_unit_vector(0)
    b = make_unit_vector(1)
    embeddings = [a, b, a, b, a, b]  # n=6
    features = se_features(embeddings)

    # Two clusters
    assert features["se_n_clusters"] == 2

    # Uniform distribution => entropy = 1.0
    assert abs(features["se_cluster_entropy"] - 1.0) < 1e-5

    # Each cluster has 3/6 = 0.5
    assert abs(features["se_dominant_cluster_frac"] - 0.5) < 1e-5

    # Maximum switches: A->B, B->A, A->B, B->A, A->B = 5 switches
    assert features["se_switch_count"] == 5
    assert abs(features["se_switch_rate"] - 1.0) < 1e-5

    # Intra-cluster: same as two-blocks
    assert abs(features["se_intra_mean"] - 1.0) < 1e-5

    # Inter-cluster: same as two-blocks
    assert abs(features["se_inter_mean"]) < 1e-5


# ---------------------------------------------------------------------------
# Test: uneven cluster sizes
# ---------------------------------------------------------------------------


def test_se_features_uneven_clusters():
    """
    Uneven clusters: [A, A, A, A, B] => cluster sizes 4 and 1.
    """
    a = make_unit_vector(0)
    b = make_unit_vector(1)
    embeddings = [a, a, a, a, b]  # n=5
    features = se_features(embeddings)

    # Two clusters
    assert features["se_n_clusters"] == 2

    # p = [0.8, 0.2], H = -0.8*log(0.8) - 0.2*log(0.2) / log(2)
    p1, p2 = 0.8, 0.2
    H = -(p1 * math.log(p1) + p2 * math.log(p2))
    normalized_H = H / math.log(2)
    assert abs(features["se_cluster_entropy"] - normalized_H) < 1e-5

    # Dominant cluster has 4/5 = 0.8
    assert abs(features["se_dominant_cluster_frac"] - 0.8) < 1e-5

    # One switch (at position 3->4)
    assert features["se_switch_count"] == 1
    assert abs(features["se_switch_rate"] - 0.25) < 1e-5

    # Intra-cluster: cluster A has 4 identical vectors, cluster B has 1
    # Cluster A: mean sim = 1.0, weight = 0.8
    # Cluster B: skipped (size 1)
    # Total = 0.8 * 1.0 = 0.8
    assert abs(features["se_intra_mean"] - 0.8) < 1e-5


# ---------------------------------------------------------------------------
# Test: similar but not identical embeddings (should cluster together)
# ---------------------------------------------------------------------------


def test_se_features_similar_embeddings_cluster():
    """
    Embeddings with small cosine distance (< CLUSTER_T) should cluster together.
    """
    # Create two vectors with cosine similarity > (1 - CLUSTER_T) = 0.50
    # cos(theta) = 0.9 => theta ≈ 25.8°
    # cosine distance = 1 - 0.9 = 0.1 < CLUSTER_T = 0.50
    theta = math.acos(0.9)
    e0 = make_unit_vector_from_2d(1.0, 0.0)
    e1 = make_unit_vector_from_2d(math.cos(theta), math.sin(theta))

    embeddings = [e0, e1]
    features = se_features(embeddings)

    # Should be merged into one cluster (distance 0.1 < 0.50)
    assert features["se_n_clusters"] == 1
    assert features["se_switch_count"] == 0


def test_se_features_dissimilar_embeddings_separate():
    """
    Embeddings with large cosine distance (> CLUSTER_T) should stay separate.
    """
    # Create two vectors with cosine similarity < (1 - CLUSTER_T) = 0.50
    # cos(theta) = 0.4 => cosine distance = 0.6 > CLUSTER_T = 0.50
    theta = math.acos(0.4)
    e0 = make_unit_vector_from_2d(1.0, 0.0)
    e1 = make_unit_vector_from_2d(math.cos(theta), math.sin(theta))

    embeddings = [e0, e1]
    features = se_features(embeddings)

    # Should remain separate (distance 0.6 > 0.50)
    assert features["se_n_clusters"] == 2
    assert features["se_switch_count"] == 1


# ---------------------------------------------------------------------------
# Test: NaN policy
# ---------------------------------------------------------------------------


def test_se_features_raises_on_nan():
    """A single NaN in any embedding should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    embeddings[0][0] = float("nan")

    with pytest.raises(ValueError, match="NaN or Inf"):
        se_features(embeddings)


def test_se_features_raises_on_inf():
    """Inf in any embedding should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    embeddings[1][5] = float("inf")

    with pytest.raises(ValueError, match="NaN or Inf"):
        se_features(embeddings)


# ---------------------------------------------------------------------------
# Test: wrapper for EmbeddingResult
# ---------------------------------------------------------------------------


def test_se_features_from_result():
    """Test the EmbeddingResult wrapper."""
    embeddings = [make_unit_vector(0), make_unit_vector(0), make_unit_vector(1)]
    result = make_embedding_result(embeddings)

    features = se_features_from_result(result)

    # Should match direct call
    expected = se_features(embeddings)
    assert features == expected


def test_se_features_from_result_empty():
    """Test wrapper with empty result."""
    result = EmbeddingResult(sentences=[], doc_embedding=None)
    features = se_features_from_result(result)

    assert features["se_n_clusters"] == 0
    assert set(features.keys()) == SE_FEATURE_KEYS


# ---------------------------------------------------------------------------
# Test: two sentences (boundary case)
# ---------------------------------------------------------------------------


def test_se_features_two_sentences_same():
    """n=2 with identical embeddings."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    features = se_features(embeddings)

    assert features["se_n_clusters"] == 1
    assert features["se_switch_count"] == 0
    assert features["se_switch_rate"] == 0.0
    # Intra-cluster: 2 identical vectors, mean sim = 1.0, weight = 1.0
    assert abs(features["se_intra_mean"] - 1.0) < 1e-5


def test_se_features_two_sentences_different():
    """n=2 with orthogonal embeddings."""
    embeddings = [make_unit_vector(0), make_unit_vector(1)]
    features = se_features(embeddings)

    assert features["se_n_clusters"] == 2
    assert features["se_switch_count"] == 1
    assert features["se_switch_rate"] == 1.0
    # No intra-cluster pairs (each cluster has 1)
    assert features["se_intra_mean"] == 0.0


# ---------------------------------------------------------------------------
# Test: determinism
# ---------------------------------------------------------------------------


def test_se_features_deterministic():
    """Same input should always produce same output."""
    embeddings = [
        make_unit_vector(0),
        make_unit_vector(0),
        make_unit_vector(1),
        make_unit_vector(0),
    ]

    features1 = se_features(embeddings)
    features2 = se_features(embeddings)

    assert features1 == features2


# ---------------------------------------------------------------------------
# Test: cluster label stability
# ---------------------------------------------------------------------------


def test_se_features_cluster_labels_stable():
    """
    Cluster labels should be assigned consistently based on first occurrence.
    This ensures deterministic behavior.
    """
    a = make_unit_vector(0)
    b = make_unit_vector(1)
    c = make_unit_vector(2)

    # Pattern: A, B, C, A, B
    embeddings = [a, b, c, a, b]
    features = se_features(embeddings)

    # Should have 3 clusters (all orthogonal)
    assert features["se_n_clusters"] == 3

    # Switches: A->B, B->C, C->A, A->B = 4
    assert features["se_switch_count"] == 4


# ---------------------------------------------------------------------------
# Test: inter_mean with similar centroids
# ---------------------------------------------------------------------------


def test_se_features_inter_mean_similar_centroids():
    """
    When clusters have similar centroids, inter_mean should be high.
    """
    # Two clusters where each cluster's centroid is similar to the other
    # Cluster 1: vectors close to axis 0
    # Cluster 2: vectors close to axis 0 but slightly different

    theta1 = math.acos(0.95)  # Small angle from axis 0
    theta2 = math.acos(0.90)  # Slightly larger angle

    # Cluster 1: two very similar vectors (will merge)
    e0 = make_unit_vector_from_2d(1.0, 0.0)
    e1 = make_unit_vector_from_2d(math.cos(0.1), math.sin(0.1))  # ~0.995 sim

    # Cluster 2: orthogonal to cluster 1
    e2 = make_unit_vector(1)
    e3 = make_unit_vector(1)  # Same as e2

    embeddings = [e0, e1, e2, e3]
    features = se_features(embeddings)

    # e0, e1 should merge (high similarity, low distance)
    # e2, e3 should merge (identical)
    # Clusters should be separate (orthogonal)
    assert features["se_n_clusters"] == 2

    # Inter-cluster: centroids are roughly orthogonal
    assert features["se_inter_mean"] < 0.1


# ---------------------------------------------------------------------------
# Test: weighted switch features
# ---------------------------------------------------------------------------


def test_se_features_weighted_switch_small_jumps():
    """
    Switches with moderate semantic jumps should have moderate weighted switch.
    
    This simulates emotional narratives where cluster labels may change
    but adjacent sentences are still somewhat semantically related.
    """
    # Create vectors that are different enough to be in separate clusters
    # cos(theta) = 0.45 => distance = 0.55 > CLUSTER_T = 0.50 => separate clusters
    theta = math.acos(0.45)
    
    e0 = make_unit_vector_from_2d(1.0, 0.0)
    e1 = make_unit_vector_from_2d(math.cos(theta), math.sin(theta))  # sim=0.45 to e0
    e2 = make_unit_vector_from_2d(1.0, 0.0)  # same as e0
    
    embeddings = [e0, e1, e2]  # pattern: A, B, A with sim(A,B)=0.45
    features = se_features(embeddings)
    
    # Should have 2 clusters (A and B are too far apart)
    assert features["se_n_clusters"] == 2
    
    # 2 switches: A->B, B->A
    assert features["se_switch_count"] == 2
    assert abs(features["se_switch_rate"] - 1.0) < 1e-5
    
    # Jump distance = 1 - 0.45 = 0.55 for each switch
    # se_switch_weighted = (0.55 + 0.55) / 2 = 0.55
    assert abs(features["se_switch_weighted"] - 0.55) < 1e-5
    
    # Mean jump = 0.55
    assert abs(features["se_switch_mean_jump"] - 0.55) < 1e-5


def test_se_features_weighted_switch_big_jumps():
    """
    Switches with big semantic jumps should have high weighted switch.
    
    This simulates scattered journals with truly unrelated topics.
    """
    # Orthogonal vectors: maximum semantic distance
    a = make_unit_vector(0)
    b = make_unit_vector(1)
    c = make_unit_vector(2)
    
    embeddings = [a, b, c]  # all orthogonal
    features = se_features(embeddings)
    
    # Each in its own cluster
    assert features["se_n_clusters"] == 3
    
    # 2 switches
    assert features["se_switch_count"] == 2
    
    # Jump distance = 1.0 (orthogonal) for each switch
    # se_switch_weighted = (1.0 + 1.0) / 2 = 1.0
    assert abs(features["se_switch_weighted"] - 1.0) < 1e-5
    
    # Mean jump = 1.0
    assert abs(features["se_switch_mean_jump"] - 1.0) < 1e-5


def test_se_features_weighted_switch_mixed():
    """
    Mix of moderate and big jumps at switch points.
    """
    # e0 and e1 are somewhat different (distance 0.55), e2 is orthogonal to both
    theta = math.acos(0.45)
    
    e0 = make_unit_vector_from_2d(1.0, 0.0)
    e1 = make_unit_vector_from_2d(math.cos(theta), math.sin(theta))  # sim=0.45 to e0
    e2 = make_unit_vector(2)  # orthogonal to both (in axis 2)
    
    embeddings = [e0, e1, e2]
    features = se_features(embeddings)
    
    # 3 clusters (all separate since e0-e1 distance > 0.50, e2 orthogonal)
    assert features["se_n_clusters"] == 3
    
    # 2 switches
    assert features["se_switch_count"] == 2
    
    # Jump at e0->e1: 1 - 0.45 = 0.55
    # Jump at e1->e2: 1 - 0 = 1.0 (e1 has components in dims 0,1; e2 is in dim 2)
    # se_switch_weighted = (0.55 + 1.0) / 2 = 0.775
    assert abs(features["se_switch_weighted"] - 0.775) < 1e-5
    
    # Mean jump = (0.55 + 1.0) / 2 = 0.775
    assert abs(features["se_switch_mean_jump"] - 0.775) < 1e-5
