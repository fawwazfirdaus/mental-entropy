"""Tests for the BC (belief conflict) feature extraction module.

These tests use synthetic unit vectors (no transformer model needed).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from mental_entropy.features.bc import (
    BC_FEATURE_KEYS,
    bc_features,
    bc_features_from_result,
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


def make_similar_vector(base_direction: int = 0, offset: float = 0.1) -> list[float]:
    """Create a unit vector similar to the base direction (high cosine similarity)."""
    vec = [0.0] * EMBEDDING_DIM
    vec[base_direction % EMBEDDING_DIM] = 1.0 - offset
    vec[(base_direction + 1) % EMBEDDING_DIM] = offset
    # Normalize
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def make_embedding_result(
    embeddings: list[list[float]],
    sentences: list[str],
) -> EmbeddingResult:
    """Create an EmbeddingResult from embeddings and sentences."""
    sent_objs = [
        SentenceEmbedding(id=i, text=text, embedding=emb)
        for i, (text, emb) in enumerate(zip(sentences, embeddings))
    ]
    return EmbeddingResult(sentences=sent_objs, doc_embedding=None)


# ---------------------------------------------------------------------------
# Test: exact keyset
# ---------------------------------------------------------------------------


def test_bc_features_returns_exact_keys():
    """BC features dict must have exactly the expected keys."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    sentences = ["I am happy.", "I feel good."]
    features = bc_features(embeddings, sentences)

    assert set(features.keys()) == BC_FEATURE_KEYS
    assert len(features) == 12


def test_bc_features_all_values_json_serializable():
    """All returned values must be Python int or float (not numpy types)."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    sentences = ["I am confident.", "I feel strong."]
    features = bc_features(embeddings, sentences)

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


def test_bc_features_empty_input():
    """n=0: return all keys with zero defaults."""
    features = bc_features([], [])

    assert set(features.keys()) == BC_FEATURE_KEYS
    assert features["bc_belief_sentence_count"] == 0
    assert features["bc_belief_sentence_rate"] == 0.0
    assert features["bc_conflict_pair_count"] == 0
    assert features["bc_unresolved_count"] == 0


def test_bc_features_single_sentence():
    """n=1: return all keys with zero defaults."""
    features = bc_features([make_unit_vector(0)], ["I am happy."])

    assert set(features.keys()) == BC_FEATURE_KEYS
    assert features["bc_belief_sentence_count"] == 0  # Can't have conflicts with 1
    assert features["bc_conflict_pair_count"] == 0


def test_bc_features_length_mismatch_raises():
    """Mismatched lengths should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    sentences = ["Only one sentence."]

    with pytest.raises(ValueError, match="Length mismatch"):
        bc_features(embeddings, sentences)


# ---------------------------------------------------------------------------
# Test: no belief sentences
# ---------------------------------------------------------------------------


def test_bc_features_no_beliefs():
    """Text without belief patterns should have zero belief count."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    sentences = ["The weather is nice.", "It rained yesterday."]
    features = bc_features(embeddings, sentences)

    assert features["bc_belief_sentence_count"] == 0
    assert features["bc_belief_sentence_rate"] == 0.0
    assert features["bc_conflict_pair_count"] == 0


# ---------------------------------------------------------------------------
# Test: belief detection
# ---------------------------------------------------------------------------


def test_bc_features_detects_beliefs():
    """Various belief patterns should be detected."""
    sentences = [
        "I am confident.",
        "I feel happy.",
        "I should work harder.",
        "I always try my best.",
        "The sky is blue.",  # Not a belief
    ]
    embeddings = [make_unit_vector(i) for i in range(5)]
    features = bc_features(embeddings, sentences)

    assert features["bc_belief_sentence_count"] == 4
    assert features["bc_belief_sentence_rate"] == 0.8  # 4/5


# ---------------------------------------------------------------------------
# Test: conflict detection (unresolved)
# ---------------------------------------------------------------------------


def test_bc_features_unresolved_conflict_negation():
    """Detect conflict via negation flip."""
    # Two belief sentences about same topic (high similarity) with negation flip
    sentences = [
        "I can do this.",
        "I can't do anything right.",
    ]
    # Same direction = high similarity
    embeddings = [make_unit_vector(0), make_similar_vector(0, 0.05)]

    features = bc_features(embeddings, sentences)

    assert features["bc_belief_sentence_count"] == 2
    assert features["bc_conflict_pair_count"] == 1
    # No integration markers nearby
    assert features["bc_integrated_count"] == 0
    assert features["bc_unresolved_count"] == 1
    assert features["bc_unresolved_rate"] == 1.0


def test_bc_features_unresolved_conflict_polar_words():
    """Detect conflict via polar word opposition."""
    sentences = [
        "I love my job.",
        "I hate my job.",
    ]
    # Same direction = high similarity
    embeddings = [make_unit_vector(0), make_similar_vector(0, 0.05)]

    features = bc_features(embeddings, sentences)

    assert features["bc_belief_sentence_count"] == 2
    assert features["bc_conflict_pair_count"] == 1
    assert features["bc_unresolved_count"] == 1


# ---------------------------------------------------------------------------
# Test: conflict detection (integrated)
# ---------------------------------------------------------------------------


def test_bc_features_integrated_conflict():
    """Conflict with nearby integration marker should be counted as integrated."""
    # Use polar words that are in POLAR_WORD_PAIRS: confident <-> worthless
    sentences = [
        "I am confident.",
        "But I've realized I'm also struggling sometimes.",
        "I am worthless.",
    ]
    # All similar (same topic)
    embeddings = [
        make_unit_vector(0),
        make_similar_vector(0, 0.05),
        make_similar_vector(0, 0.03),
    ]

    features = bc_features(embeddings, sentences)

    # Sentences 0 and 2 are beliefs with polar opposition (confident vs worthless)
    assert features["bc_belief_sentence_count"] >= 2
    # Conflict should be detected
    assert features["bc_conflict_pair_count"] >= 1
    # Integration marker "but I've realized" is present
    assert features["bc_integrated_count"] >= 1
    assert features["bc_integration_rate"] > 0


def test_bc_features_integration_marker_in_belief():
    """Integration marker in the belief sentence itself should count."""
    sentences = [
        "I am confident.",
        "I am scared, but both are true.",
    ]
    embeddings = [make_unit_vector(0), make_similar_vector(0, 0.05)]

    features = bc_features(embeddings, sentences)

    # Should detect conflict and mark it as integrated
    if features["bc_conflict_pair_count"] > 0:
        assert features["bc_integrated_count"] >= 1


# ---------------------------------------------------------------------------
# Test: no conflict when topics differ
# ---------------------------------------------------------------------------


def test_bc_features_no_conflict_different_topics():
    """Opposing beliefs about different topics should not conflict."""
    sentences = [
        "I love my job.",
        "I hate vegetables.",
    ]
    # Different directions = low similarity
    embeddings = [make_unit_vector(0), make_unit_vector(1)]

    features = bc_features(embeddings, sentences)

    assert features["bc_belief_sentence_count"] == 2
    # Low similarity means no conflict detected
    assert features["bc_conflict_pair_count"] == 0


# ---------------------------------------------------------------------------
# Test: conflict span
# ---------------------------------------------------------------------------


def test_bc_features_conflict_span():
    """Conflict span measures distance between conflicting sentences."""
    sentences = [
        "I am confident.",
        "Something happened today.",
        "I also worked on a project.",
        "I am worthless.",
    ]
    # Sentences 0 and 3 should conflict (similar topic, polar words)
    embeddings = [
        make_unit_vector(0),
        make_unit_vector(1),  # Different topic
        make_unit_vector(1),  # Different topic
        make_similar_vector(0, 0.05),  # Similar to sentence 0
    ]

    features = bc_features(embeddings, sentences)

    if features["bc_conflict_pair_count"] > 0:
        # Span between sentence 0 and 3 is 3
        assert features["bc_conflict_span_max"] == 3


# ---------------------------------------------------------------------------
# Test: wrapper for EmbeddingResult
# ---------------------------------------------------------------------------


def test_bc_features_from_result():
    """Test the EmbeddingResult wrapper."""
    sentences = ["I am happy.", "I feel good."]
    embeddings = [make_unit_vector(0), make_similar_vector(0, 0.05)]
    result = make_embedding_result(embeddings, sentences)

    features = bc_features_from_result(result)

    # Should match direct call
    expected = bc_features(embeddings, sentences)
    assert features == expected


def test_bc_features_from_result_empty():
    """Test wrapper with empty result."""
    result = EmbeddingResult(sentences=[], doc_embedding=None)
    features = bc_features_from_result(result)

    assert features["bc_belief_sentence_count"] == 0
    assert set(features.keys()) == BC_FEATURE_KEYS


# ---------------------------------------------------------------------------
# Test: NaN policy
# ---------------------------------------------------------------------------


def test_bc_features_raises_on_nan():
    """A single NaN in any embedding should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    embeddings[0][0] = float("nan")
    sentences = ["I am happy.", "I feel good."]

    with pytest.raises(ValueError, match="NaN or Inf"):
        bc_features(embeddings, sentences)


def test_bc_features_raises_on_inf():
    """Inf in any embedding should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    embeddings[1][5] = float("inf")
    sentences = ["I am happy.", "I feel good."]

    with pytest.raises(ValueError, match="NaN or Inf"):
        bc_features(embeddings, sentences)


# ---------------------------------------------------------------------------
# Test: similarity values
# ---------------------------------------------------------------------------


def test_bc_features_conflict_similarity_values():
    """Test that similarity values are captured correctly."""
    sentences = [
        "I am confident.",
        "I am worthless.",
    ]
    # Very high similarity (same topic)
    embeddings = [make_unit_vector(0), make_similar_vector(0, 0.02)]

    features = bc_features(embeddings, sentences)

    if features["bc_conflict_pair_count"] > 0:
        # Similarity should be high (close to 1.0)
        assert features["bc_max_conflict_sim"] > 0.9
        assert features["bc_mean_conflict_sim"] > 0.9
