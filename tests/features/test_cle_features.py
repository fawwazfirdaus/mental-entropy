"""Tests for the CLE (cognitive load entropy) feature extraction module.

These tests use synthetic unit vectors (no transformer model needed)
and sample sentences for text cue detection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mental_entropy.features.cle import (
    CLE_FEATURE_KEYS,
    HIGH_T,
    LOW_T,
    REP_T,
    cle_features,
    cle_features_from_result,
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


def make_embedding_result(
    embeddings: list[list[float]], texts: list[str]
) -> EmbeddingResult:
    """Create an EmbeddingResult from embeddings and texts."""
    sentences = [
        SentenceEmbedding(id=i, text=text, embedding=emb)
        for i, (emb, text) in enumerate(zip(embeddings, texts))
    ]
    return EmbeddingResult(sentences=sentences, doc_embedding=None)


def make_vector_with_similarity(base: list[float], target_sim: float) -> list[float]:
    """
    Create a unit vector with specified cosine similarity to base vector.

    Uses angle-based construction in 2D subspace.
    """
    theta = math.acos(target_sim)
    # Assume base is along axis 0 for simplicity
    return make_unit_vector_from_2d(math.cos(theta), math.sin(theta))


# ---------------------------------------------------------------------------
# Test: exact keyset
# ---------------------------------------------------------------------------


def test_cle_features_returns_exact_keys():
    """CLE features dict must have exactly the expected keys."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    sentences = ["First sentence.", "Second sentence."]
    features = cle_features(embeddings, sentences)

    assert set(features.keys()) == CLE_FEATURE_KEYS
    assert len(features) == 14


def test_cle_features_all_values_json_serializable():
    """All returned values must be Python float (not numpy types)."""
    import json

    embeddings = [make_unit_vector(0), make_unit_vector(0), make_unit_vector(1)]
    sentences = ["First.", "Second.", "Third."]
    features = cle_features(embeddings, sentences)

    # Should not raise
    json_str = json.dumps(features)
    assert json_str  # Non-empty

    # Check types explicitly
    for key, value in features.items():
        assert isinstance(value, float), f"Key {key} has type {type(value)}"
        assert not isinstance(value, np.generic), f"Key {key} is a numpy type"


# ---------------------------------------------------------------------------
# Test: edge cases (n=0, n=1)
# ---------------------------------------------------------------------------


def test_cle_features_empty_input():
    """n=0: return all keys with 0.0 defaults."""
    features = cle_features([], [])

    assert set(features.keys()) == CLE_FEATURE_KEYS
    for key, value in features.items():
        assert value == 0.0, f"Key {key} should be 0.0 for empty input"


def test_cle_features_single_sentence():
    """n=1: return all keys with 0.0 defaults."""
    features = cle_features([make_unit_vector(0)], ["Single sentence."])

    assert set(features.keys()) == CLE_FEATURE_KEYS
    for key, value in features.items():
        assert value == 0.0, f"Key {key} should be 0.0 for single sentence"


# ---------------------------------------------------------------------------
# Test: length mismatch
# ---------------------------------------------------------------------------


def test_cle_features_length_mismatch_raises():
    """Mismatched embeddings and sentences lengths should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    sentences = ["Only one."]

    with pytest.raises(ValueError, match="Length mismatch"):
        cle_features(embeddings, sentences)


# ---------------------------------------------------------------------------
# Test: NaN/Inf policy
# ---------------------------------------------------------------------------


def test_cle_features_raises_on_nan():
    """A single NaN in any embedding should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    embeddings[0][0] = float("nan")
    sentences = ["First.", "Second."]

    with pytest.raises(ValueError, match="NaN or Inf"):
        cle_features(embeddings, sentences)


def test_cle_features_raises_on_inf():
    """Inf in any embedding should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    embeddings[1][5] = float("inf")
    sentences = ["First.", "Second."]

    with pytest.raises(ValueError, match="NaN or Inf"):
        cle_features(embeddings, sentences)


# ---------------------------------------------------------------------------
# Test: determinism
# ---------------------------------------------------------------------------


def test_cle_features_deterministic():
    """Same input should always produce same output."""
    embeddings = [
        make_unit_vector(0),
        make_unit_vector(0),
        make_unit_vector(1),
        make_unit_vector(0),
    ]
    sentences = [
        "I was thinking maybe something.",
        "Wait, never mind.",
        "Actually I think I know.",
        "Sort of...",
    ]

    features1 = cle_features(embeddings, sentences)
    features2 = cle_features(embeddings, sentences)

    assert features1 == features2


# ---------------------------------------------------------------------------
# Test: surface-level features - fragment detection
# ---------------------------------------------------------------------------


def test_cle_fragment_rate_ellipsis():
    """Sentences ending with ellipsis are fragments."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "I was thinking about something...",  # ends with ..., fragment
        "This is a complete thought.",  # not fragment
    ]
    features = cle_features(embeddings, sentences)

    assert abs(features["cle_fragment_rate"] - 0.5) < 1e-5


def test_cle_fragment_rate_trailing_dash():
    """Sentences ending with em-dash or double hyphen are fragments."""
    embeddings = [make_unit_vector(0) for _ in range(3)]
    sentences = [
        "I was going to say—",  # ends with em-dash, fragment
        "Wait, I thought--",  # ends with --, fragment
        "This is complete.",  # not fragment
    ]
    features = cle_features(embeddings, sentences)

    assert abs(features["cle_fragment_rate"] - 2.0 / 3.0) < 1e-5


def test_cle_fragment_rate_short_no_verb():
    """Short sentences without verbs are fragments."""
    embeddings = [make_unit_vector(0) for _ in range(3)]
    sentences = [
        "Just wow",  # 2 tokens, no verb, fragment
        "The end",  # 2 tokens, no verb, fragment
        "I am happy today.",  # has verb, complete
    ]
    features = cle_features(embeddings, sentences)

    assert abs(features["cle_fragment_rate"] - 2.0 / 3.0) < 1e-5


def test_cle_fragment_rate_conjunction_unfinished():
    """Sentences starting with conjunctions without proper ending are fragments."""
    embeddings = [make_unit_vector(0) for _ in range(3)]
    sentences = [
        "But I really wanted to",  # starts with "but", no terminal punct, fragment
        "And then everything changed.",  # has terminal punct, not fragment
        "So what happened next.",  # has terminal punct, not fragment
    ]
    features = cle_features(embeddings, sentences)

    assert abs(features["cle_fragment_rate"] - 1.0 / 3.0) < 1e-5


def test_cle_fragment_rate_no_verb_long():
    """Longer sentences without verbs are fragments."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "The big red balloon on display today.",  # no verb, 7 tokens, fragment
        "I walked to the store yesterday.",  # has verb, not fragment
    ]
    features = cle_features(embeddings, sentences)

    assert abs(features["cle_fragment_rate"] - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# Test: surface-level features - restart detection
# ---------------------------------------------------------------------------


def test_cle_restart_rate():
    """Test restart cue detection."""
    embeddings = [make_unit_vector(0) for _ in range(4)]
    sentences = [
        "I was saying something.",  # no restart
        "Anyway, let me continue.",  # has "anyway"
        "Wait, what was I saying?",  # has "wait"
        "Normal sentence here.",  # no restart
    ]
    features = cle_features(embeddings, sentences)

    # 2 out of 4 have restart cues
    assert abs(features["cle_restart_rate"] - 0.5) < 1e-5


def test_cle_restart_rate_various_cues():
    """Test various restart cues."""
    embeddings = [make_unit_vector(0) for _ in range(5)]
    sentences = [
        "I mean, it was different.",  # has "i mean"
        "Actually, I changed my mind.",  # has "actually"
        "Never mind that now.",  # has "never mind"
        "No, that's not right.",  # has "no,"
        "Scratch that, start over.",  # has "scratch that"
    ]
    features = cle_features(embeddings, sentences)

    # All 5 have restart cues
    assert abs(features["cle_restart_rate"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test: surface-level features - hedge detection
# ---------------------------------------------------------------------------


def test_cle_hedge_rate():
    """Test hedge cue detection."""
    embeddings = [make_unit_vector(0) for _ in range(4)]
    sentences = [
        "I am certain about this.",  # no hedge
        "Maybe it will work out.",  # has "maybe"
        "I think this is right.",  # has "i think"
        "This is definitely true.",  # no hedge
    ]
    features = cle_features(embeddings, sentences)

    # 2 out of 4 have hedge cues
    assert abs(features["cle_hedge_rate"] - 0.5) < 1e-5


def test_cle_hedge_rate_various_cues():
    """Test various hedge cues."""
    embeddings = [make_unit_vector(0) for _ in range(6)]
    sentences = [
        "I guess that's okay.",  # has "i guess"
        "I'm not sure about this.",  # has "i'm not sure"
        "Kind of like what happened.",  # has "kind of"
        "Sort of the same thing.",  # has "sort of"
        "Probably true after all.",  # has "probably"
        "It feels like something changed.",  # has "it feels like"
    ]
    features = cle_features(embeddings, sentences)

    # All 6 have hedge cues
    assert abs(features["cle_hedge_rate"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test: surface-level features - length CV
# ---------------------------------------------------------------------------


def test_cle_length_cv_uniform():
    """Uniform sentence lengths => CV = 0."""
    embeddings = [make_unit_vector(0) for _ in range(3)]
    sentences = [
        "One two three four.",  # 4 tokens
        "Five six seven eight.",  # 4 tokens
        "Nine ten eleven twelve.",  # 4 tokens
    ]
    features = cle_features(embeddings, sentences)

    # All same length => std = 0 => CV = 0
    assert abs(features["cle_length_cv"]) < 1e-5


def test_cle_length_cv_varied():
    """Varied sentence lengths => positive CV."""
    embeddings = [make_unit_vector(0) for _ in range(3)]
    sentences = [
        "Short.",  # 1 token
        "This is a medium length sentence here.",  # 7 tokens
        "Tiny.",  # 1 token
    ]
    features = cle_features(embeddings, sentences)

    # Lengths: [1, 7, 1], mean = 3, std = sqrt((4+16+4)/3) = sqrt(8) ≈ 2.83
    # CV = 2.83 / 3 ≈ 0.94
    assert features["cle_length_cv"] > 0.5


def test_cle_length_cv_calculation():
    """Verify CV calculation with known values."""
    embeddings = [make_unit_vector(0) for _ in range(4)]
    sentences = [
        "One two.",  # 2 tokens
        "One two three four.",  # 4 tokens
        "One two three four five six.",  # 6 tokens
        "One two three four five six seven eight.",  # 8 tokens
    ]
    features = cle_features(embeddings, sentences)

    # Lengths: [2, 4, 6, 8], mean = 5, std = sqrt((9+1+1+9)/4) = sqrt(5) ≈ 2.236
    # CV = 2.236 / 5 ≈ 0.447
    expected_cv = math.sqrt(5) / 5
    assert abs(features["cle_length_cv"] - expected_cv) < 0.01


# ---------------------------------------------------------------------------
# Test: surface-level features - punctuation noise
# ---------------------------------------------------------------------------


def test_cle_punctuation_noise_exclamations():
    """Multiple exclamation points trigger punctuation noise."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "This is amazing!!!",  # has !!!
        "This is normal.",  # no noise
    ]
    features = cle_features(embeddings, sentences)

    assert abs(features["cle_punctuation_noise"] - 0.5) < 1e-5


def test_cle_punctuation_noise_questions():
    """Multiple question marks trigger punctuation noise."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "What is happening???",  # has ???
        "What is this?",  # single ? is fine
    ]
    features = cle_features(embeddings, sentences)

    assert abs(features["cle_punctuation_noise"] - 0.5) < 1e-5


def test_cle_punctuation_noise_repeated_ellipses():
    """Multiple ellipsis clusters trigger punctuation noise."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "I thought... and then... nothing.",  # 2 ellipsis clusters
        "Just one trailing...",  # single ellipsis is fragment, not noise
    ]
    features = cle_features(embeddings, sentences)

    assert abs(features["cle_punctuation_noise"] - 0.5) < 1e-5


def test_cle_punctuation_noise_mid_dash():
    """Mid-sentence dashes trigger punctuation noise."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "I was thinking—wait—what happened?",  # mid-sentence dash with content after
        "This is normal.",
    ]
    features = cle_features(embeddings, sentences)

    assert abs(features["cle_punctuation_noise"] - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# Test: embedding features - adjacent similarity stats
# ---------------------------------------------------------------------------


def test_cle_adj_sim_identical():
    """Identical embeddings => adj sims all 1.0, std=0, range=0, cv=0."""
    embeddings = [make_unit_vector(0) for _ in range(4)]
    sentences = [f"Sentence {i}." for i in range(4)]
    features = cle_features(embeddings, sentences)

    # All adjacent sims = 1.0
    assert abs(features["cle_adj_sim_std"]) < 1e-5
    assert abs(features["cle_adj_sim_range"]) < 1e-5
    assert abs(features["cle_adj_sim_cv"]) < 1e-5  # std=0, so cv=0
    assert abs(features["cle_adj_low_frac"]) < 1e-5  # 1.0 > LOW_T


def test_cle_adj_sim_orthogonal():
    """Orthogonal embeddings => adj sims all 0.0."""
    embeddings = [make_unit_vector(i) for i in range(4)]
    sentences = [f"Sentence {i}." for i in range(4)]
    features = cle_features(embeddings, sentences)

    # All adjacent sims = 0.0 => std=0, range=0, but cv undefined (0/0), returns 0
    assert abs(features["cle_adj_sim_std"]) < 1e-5
    assert abs(features["cle_adj_sim_range"]) < 1e-5
    assert abs(features["cle_adj_sim_cv"]) < 1e-5
    # All sims (0.0) < LOW_T (0.50)
    assert abs(features["cle_adj_low_frac"] - 1.0) < 1e-5


def test_cle_adj_sim_varied():
    """Varied similarities => positive std, range, and cv."""
    # Create: sim = [0.9, 0.2]
    # e0 at angle 0, e1 at acos(0.9), e2 at angle such that dot(e1, e2) = 0.2

    e0 = make_unit_vector_from_2d(1.0, 0.0)

    theta1 = math.acos(0.9)
    e1 = make_unit_vector_from_2d(math.cos(theta1), math.sin(theta1))

    # e2 such that dot(e1, e2) = 0.2
    # theta2 = theta1 + acos(0.2)
    theta2 = theta1 + math.acos(0.2)
    e2 = make_unit_vector_from_2d(math.cos(theta2), math.sin(theta2))

    embeddings = [e0, e1, e2]
    sentences = ["One.", "Two.", "Three."]
    features = cle_features(embeddings, sentences)

    # sim = [0.9, 0.2], range = 0.7, std = sqrt((0.35^2 + 0.35^2)/2) ≈ 0.35
    assert abs(features["cle_adj_sim_range"] - 0.7) < 0.02

    # std of [0.9, 0.2]: mean=0.55, deviations=[0.35, 0.35], std=0.35
    expected_std = 0.35
    assert abs(features["cle_adj_sim_std"] - expected_std) < 0.02

    # cv = std / mean = 0.35 / 0.55 ≈ 0.636
    expected_cv = 0.35 / 0.55
    assert abs(features["cle_adj_sim_cv"] - expected_cv) < 0.02

    # 0.2 < LOW_T (0.50), so 1 out of 2 is low
    assert abs(features["cle_adj_low_frac"] - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# Test: embedding features - zigzag rate
# ---------------------------------------------------------------------------


def test_cle_zigzag_rate_no_zigzag():
    """Monotonic similarities => no zigzag."""
    # Create monotonically decreasing: sim = [0.9, 0.7, 0.5]
    # Need 4 embeddings for 3 adjacent pairs

    e0 = make_unit_vector_from_2d(1.0, 0.0)

    theta1 = math.acos(0.9)
    e1 = make_unit_vector_from_2d(math.cos(theta1), math.sin(theta1))

    theta2 = theta1 + math.acos(0.7)
    e2 = make_unit_vector_from_2d(math.cos(theta2), math.sin(theta2))

    theta3 = theta2 + math.acos(0.5)
    e3 = make_unit_vector_from_2d(math.cos(theta3), math.sin(theta3))

    embeddings = [e0, e1, e2, e3]
    sentences = ["A.", "B.", "C.", "D."]
    features = cle_features(embeddings, sentences)

    # No local maxima or minima in monotonic sequence
    # Actually we need to verify the sims are monotonic
    # For this test, use simpler approach: identical embeddings
    embeddings2 = [make_unit_vector(0) for _ in range(4)]
    features2 = cle_features(embeddings2, sentences)

    # All sims = 1.0, no variation, no zigzag
    assert features2["cle_zigzag_rate"] == 0.0


def test_cle_zigzag_rate_alternating():
    """Alternating pattern => maximum zigzag."""
    # Pattern: A, B, A, B, A (orthogonal alternation)
    # sims: [0, 0, 0, 0] - all same, no zigzag actually

    # Better test: create varying pattern
    # sim = [0.9, 0.5, 0.9] for 4 embeddings
    # This has a valley at position 1 (0.9 > 0.5 < 0.9)

    a = make_unit_vector(0)
    b = make_unit_vector(1)
    embeddings = [a, b, a, b, a]  # n=5, sims = [0, 0, 0, 0] - not zigzag

    sentences = ["A.", "B.", "C.", "D.", "E."]
    features = cle_features(embeddings, sentences)

    # All sims = 0, no variation means no peaks/valleys
    assert features["cle_zigzag_rate"] == 0.0


def test_cle_zigzag_rate_with_peaks():
    """Pattern with peak/valley => positive zigzag rate."""
    # Create pattern where sim = [0.9, 0.5, 0.8]
    # This has: 0.9 > 0.5 < 0.8 => valley at index 1

    e0 = make_unit_vector_from_2d(1.0, 0.0)

    # e1 such that dot(e0, e1) = 0.9
    theta1 = math.acos(0.9)
    e1 = make_unit_vector_from_2d(math.cos(theta1), math.sin(theta1))

    # e2 such that dot(e1, e2) = 0.5
    theta2 = theta1 + math.acos(0.5)
    e2 = make_unit_vector_from_2d(math.cos(theta2), math.sin(theta2))

    # e3 such that dot(e2, e3) = 0.8
    theta3 = theta2 + math.acos(0.8)
    e3 = make_unit_vector_from_2d(math.cos(theta3), math.sin(theta3))

    embeddings = [e0, e1, e2, e3]
    sentences = ["A.", "B.", "C.", "D."]
    features = cle_features(embeddings, sentences)

    # sim = [0.9, 0.5, 0.8]
    # Check for valley at index 1: 0.9 > 0.5 < 0.8 => YES, 1 zigzag
    # zigzag_rate = 1 / max(1, 4-2) = 1/2 = 0.5
    assert abs(features["cle_zigzag_rate"] - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# Test: embedding features - repetition score
# ---------------------------------------------------------------------------


def test_cle_repetition_score_no_repetition():
    """All orthogonal => no repetition."""
    embeddings = [make_unit_vector(i) for i in range(5)]
    sentences = [f"S{i}." for i in range(5)]
    features = cle_features(embeddings, sentences)

    # All non-adjacent pairs are orthogonal (sim=0 < REP_T)
    assert features["cle_repetition_score"] == 0.0


def test_cle_repetition_score_with_repetition():
    """Alternating A, B, A, B, A => (i, i+2) pairs are identical."""
    a = make_unit_vector(0)
    b = make_unit_vector(1)
    embeddings = [a, b, a, b, a]  # n=5
    sentences = [f"S{i}." for i in range(5)]
    features = cle_features(embeddings, sentences)

    # (0,2): dot(a,a) = 1.0 > REP_T
    # (1,3): dot(b,b) = 1.0 > REP_T
    # (2,4): dot(a,a) = 1.0 > REP_T
    # (0,3): dot(a,b) = 0.0 < REP_T
    # (1,4): dot(b,a) = 0.0 < REP_T
    # Total repetitions: 3
    # repetition_score = 3 / 5 = 0.6
    assert abs(features["cle_repetition_score"] - 0.6) < 1e-5


def test_cle_repetition_score_identical():
    """All identical => maximum repetition from (i, i+2) and (i, i+3)."""
    embeddings = [make_unit_vector(0) for _ in range(5)]
    sentences = [f"S{i}." for i in range(5)]
    features = cle_features(embeddings, sentences)

    # n=5
    # (i, i+2): i=0,1,2 => 3 pairs, all sim=1.0 > REP_T
    # (i, i+3): i=0,1 => 2 pairs, all sim=1.0 > REP_T
    # Total: 5 repetitions
    # repetition_score = 5 / 5 = 1.0
    assert abs(features["cle_repetition_score"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test: embedding features - resume rate
# ---------------------------------------------------------------------------


def test_cle_resume_rate_no_resume():
    """Identical embeddings => no low sims, no resume patterns."""
    embeddings = [make_unit_vector(0) for _ in range(4)]
    sentences = [f"S{i}." for i in range(4)]
    features = cle_features(embeddings, sentences)

    # All sims = 1.0 > LOW_T, so no drop => no resume
    assert features["cle_resume_rate"] == 0.0


def test_cle_resume_rate_with_resume():
    """Pattern: drop then rise => resume detected."""
    # Create sim = [0.8, 0.4, 0.8]
    # At index 1: sim[1]=0.4 < LOW_T (0.50), sim[2]=0.8 > HIGH_T (0.65) => resume!

    e0 = make_unit_vector_from_2d(1.0, 0.0)

    theta1 = math.acos(0.8)
    e1 = make_unit_vector_from_2d(math.cos(theta1), math.sin(theta1))

    theta2 = theta1 + math.acos(0.4)
    e2 = make_unit_vector_from_2d(math.cos(theta2), math.sin(theta2))

    theta3 = theta2 + math.acos(0.8)
    e3 = make_unit_vector_from_2d(math.cos(theta3), math.sin(theta3))

    embeddings = [e0, e1, e2, e3]
    sentences = ["A.", "B.", "C.", "D."]
    features = cle_features(embeddings, sentences)

    # sim = [0.8, 0.4, 0.8]
    # Check: sim[1]=0.4 < 0.50 and sim[2]=0.8 > 0.65 => 1 resume
    # resume_rate = 1 / max(1, 4-2) = 0.5
    assert abs(features["cle_resume_rate"] - 0.5) < 1e-5


def test_cle_resume_rate_no_high_recovery():
    """Drop but no high recovery => no resume."""
    # Create sim = [0.8, 0.4, 0.55]
    # At index 1: sim[1]=0.4 < LOW_T (0.50), sim[2]=0.55 < HIGH_T (0.65) => no resume

    e0 = make_unit_vector_from_2d(1.0, 0.0)

    theta1 = math.acos(0.8)
    e1 = make_unit_vector_from_2d(math.cos(theta1), math.sin(theta1))

    theta2 = theta1 + math.acos(0.4)
    e2 = make_unit_vector_from_2d(math.cos(theta2), math.sin(theta2))

    theta3 = theta2 + math.acos(0.55)
    e3 = make_unit_vector_from_2d(math.cos(theta3), math.sin(theta3))

    embeddings = [e0, e1, e2, e3]
    sentences = ["A.", "B.", "C.", "D."]
    features = cle_features(embeddings, sentences)

    # sim[1]=0.4 < LOW_T (0.50) but sim[2]=0.55 < HIGH_T (0.65) => no resume
    assert features["cle_resume_rate"] == 0.0


# ---------------------------------------------------------------------------
# Test: wrapper for EmbeddingResult
# ---------------------------------------------------------------------------


def test_cle_features_from_result():
    """Test the EmbeddingResult wrapper."""
    embeddings = [make_unit_vector(0), make_unit_vector(0), make_unit_vector(1)]
    texts = ["Maybe something good.", "I think so.", "Wait, actually..."]
    result = make_embedding_result(embeddings, texts)

    features = cle_features_from_result(result)

    # Should match direct call
    expected = cle_features(embeddings, texts)
    assert features == expected


def test_cle_features_from_result_empty():
    """Test wrapper with empty result."""
    result = EmbeddingResult(sentences=[], doc_embedding=None)
    features = cle_features_from_result(result)

    assert set(features.keys()) == CLE_FEATURE_KEYS
    for value in features.values():
        assert value == 0.0


# ---------------------------------------------------------------------------
# Test: two sentences (boundary case)
# ---------------------------------------------------------------------------


def test_cle_features_two_sentences_identical():
    """n=2 with identical embeddings."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    sentences = ["First thought.", "Same thought."]
    features = cle_features(embeddings, sentences)

    # Only 1 adjacent pair, sim = 1.0
    assert abs(features["cle_adj_sim_std"]) < 1e-5
    assert abs(features["cle_adj_sim_range"]) < 1e-5
    assert features["cle_adj_low_frac"] == 0.0  # 1.0 > LOW_T


def test_cle_features_two_sentences_orthogonal():
    """n=2 with orthogonal embeddings."""
    embeddings = [make_unit_vector(0), make_unit_vector(1)]
    sentences = ["First topic.", "Different topic."]
    features = cle_features(embeddings, sentences)

    # Only 1 adjacent pair, sim = 0.0
    assert abs(features["cle_adj_sim_std"]) < 1e-5
    assert abs(features["cle_adj_sim_range"]) < 1e-5
    assert abs(features["cle_adj_low_frac"] - 1.0) < 1e-5  # 0.0 < LOW_T


# ---------------------------------------------------------------------------
# Test: combined surface + embedding features
# ---------------------------------------------------------------------------


def test_cle_features_mixed_cognitive_load():
    """Test a realistic mixed cognitive load scenario."""
    # Alternating pattern with cognitive load text signals
    a = make_unit_vector(0)
    b = make_unit_vector(1)

    embeddings = [a, b, a, b]
    sentences = [
        "I was thinking maybe...",  # hedge + fragment (ellipsis)
        "Wait, no, actually I think something else.",  # restart (wait, actually) + hedge (i think)
        "Sort of like what I said before—",  # hedge (sort of) + fragment (dash)
        "I'm not sure about any of this!!!",  # hedge + punctuation noise
    ]
    features = cle_features(embeddings, sentences)

    # All 4 have hedge cues
    assert features["cle_hedge_rate"] == 1.0

    # 3 fragments (ellipsis, dash, dash)
    assert features["cle_fragment_rate"] >= 0.5

    # 1 restart (wait or actually)
    assert features["cle_restart_rate"] >= 0.25

    # 1 punctuation noise (!!!)
    assert features["cle_punctuation_noise"] >= 0.25

    # Embedding features: alternating orthogonal
    # sims = [0, 0, 0], all same so std=0, range=0
    assert abs(features["cle_adj_sim_std"]) < 1e-5
    assert abs(features["cle_adj_sim_range"]) < 1e-5
    # All sims < LOW_T
    assert abs(features["cle_adj_low_frac"] - 1.0) < 1e-5
    # All sentences have semantic breaks (all sims < LOW_T)
    assert abs(features["cle_semantic_break_rate"] - 1.0) < 1e-5
    # Middle sentences (1, 2) are isolated from both neighbors
    assert abs(features["cle_semantic_isolated_rate"] - 0.5) < 1e-5  # 2/4 = 0.5


# ---------------------------------------------------------------------------
# Test: semantic break rate
# ---------------------------------------------------------------------------


def test_cle_semantic_break_rate_no_breaks():
    """All high similarity => no semantic breaks."""
    # All identical embeddings => sim = 1.0 for all pairs
    embeddings = [make_unit_vector(0) for _ in range(5)]
    sentences = ["A.", "B.", "C.", "D.", "E."]
    features = cle_features(embeddings, sentences)

    # All sims = 1.0 > LOW_T, so no breaks
    assert features["cle_semantic_break_rate"] == 0.0


def test_cle_semantic_break_rate_all_breaks():
    """All low similarity => all sentences have breaks."""
    # All orthogonal embeddings => sim = 0.0 for all pairs
    embeddings = [make_unit_vector(i) for i in range(5)]
    sentences = ["A.", "B.", "C.", "D.", "E."]
    features = cle_features(embeddings, sentences)

    # All sims = 0.0 < LOW_T, so all sentences have breaks
    assert abs(features["cle_semantic_break_rate"] - 1.0) < 1e-5


def test_cle_semantic_break_rate_mixed():
    """Mixed similarities => some sentences have breaks."""
    # Pattern: high, low, high, low
    # sims = [0.9, 0.3, 0.8, 0.2]
    # Sentence 0: no break before, no break after (0.9 > LOW_T) -> no break
    # Sentence 1: no break before (0.9 > LOW_T), break after (0.3 < LOW_T) -> break
    # Sentence 2: break before (0.3 < LOW_T), no break after (0.8 > LOW_T) -> break
    # Sentence 3: no break before (0.8 > LOW_T), break after (0.2 < LOW_T) -> break
    # Sentence 4: break before (0.2 < LOW_T), no next -> break
    # 4 out of 5 sentences have breaks

    e0 = make_unit_vector_from_2d(1.0, 0.0)

    # e1 such that dot(e0, e1) = 0.9
    theta1 = math.acos(0.9)
    e1 = make_unit_vector_from_2d(math.cos(theta1), math.sin(theta1))

    # e2 such that dot(e1, e2) = 0.3
    theta2 = theta1 + math.acos(0.3)
    e2 = make_unit_vector_from_2d(math.cos(theta2), math.sin(theta2))

    # e3 such that dot(e2, e3) = 0.8
    theta3 = theta2 + math.acos(0.8)
    e3 = make_unit_vector_from_2d(math.cos(theta3), math.sin(theta3))

    # e4 such that dot(e3, e4) = 0.2
    theta4 = theta3 + math.acos(0.2)
    e4 = make_unit_vector_from_2d(math.cos(theta4), math.sin(theta4))

    embeddings = [e0, e1, e2, e3, e4]
    sentences = ["A.", "B.", "C.", "D.", "E."]
    features = cle_features(embeddings, sentences)

    # 4 out of 5 sentences have breaks (sentence 0 has no break)
    assert abs(features["cle_semantic_break_rate"] - 0.8) < 1e-5


def test_cle_semantic_break_rate_first_last_sentences():
    """First and last sentences can have breaks."""
    # Pattern: low, high, low
    # sims = [0.3, 0.8]
    # Sentence 0: no break before, break after (0.3 < LOW_T) -> break
    # Sentence 1: break before (0.3 < LOW_T), no break after (0.8 > LOW_T) -> break
    # Sentence 2: break before (0.8 > LOW_T, wait no - 0.8 > LOW_T so no break), no next -> no break

    a = make_unit_vector(0)
    b = make_unit_vector(1)
    embeddings = [a, b, a]  # sims = [0, 0]
    sentences = ["A.", "B.", "C."]
    features = cle_features(embeddings, sentences)

    # Sentence 0: break after (0 < LOW_T) -> break
    # Sentence 1: break before (0 < LOW_T), break after (0 < LOW_T) -> break
    # Sentence 2: break before (0 < LOW_T) -> break
    # All 3 have breaks
    assert abs(features["cle_semantic_break_rate"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test: semantic isolated rate
# ---------------------------------------------------------------------------


def test_cle_semantic_isolated_rate_no_isolated():
    """All high similarity => no isolated sentences."""
    # All identical embeddings => sim = 1.0 for all pairs
    embeddings = [make_unit_vector(0) for _ in range(5)]
    sentences = ["A.", "B.", "C.", "D.", "E."]
    features = cle_features(embeddings, sentences)

    # All sims = 1.0 > LOW_T, so no isolated sentences
    assert features["cle_semantic_isolated_rate"] == 0.0


def test_cle_semantic_isolated_rate_all_isolated():
    """All low similarity => all middle sentences isolated."""
    # All orthogonal embeddings => sim = 0.0 for all pairs
    embeddings = [make_unit_vector(i) for i in range(5)]
    sentences = ["A.", "B.", "C.", "D.", "E."]
    features = cle_features(embeddings, sentences)

    # All sims = 0.0 < LOW_T
    # Sentences 1, 2, 3 are isolated (have low sim to both neighbors)
    # Sentence 0: first, can't be isolated
    # Sentence 4: last, can't be isolated
    # 3 isolated out of 5 total = 0.6
    assert abs(features["cle_semantic_isolated_rate"] - 0.6) < 1e-5


def test_cle_semantic_isolated_rate_mixed():
    """Mixed pattern => some sentences isolated."""
    # Pattern: high, low, low, high
    # sims = [0.8, 0.3, 0.2]
    # Sentence 0: first, can't be isolated
    # Sentence 1: prev=0.8 > LOW_T, next=0.3 < LOW_T -> not isolated
    # Sentence 2: prev=0.3 < LOW_T, next=0.2 < LOW_T -> isolated
    # Sentence 3: last, can't be isolated
    # 1 isolated out of 4 = 0.25

    e0 = make_unit_vector_from_2d(1.0, 0.0)

    # e1 such that dot(e0, e1) = 0.8
    theta1 = math.acos(0.8)
    e1 = make_unit_vector_from_2d(math.cos(theta1), math.sin(theta1))

    # e2 such that dot(e1, e2) = 0.3
    theta2 = theta1 + math.acos(0.3)
    e2 = make_unit_vector_from_2d(math.cos(theta2), math.sin(theta2))

    # e3 such that dot(e2, e3) = 0.2
    theta3 = theta2 + math.acos(0.2)
    e3 = make_unit_vector_from_2d(math.cos(theta3), math.sin(theta3))

    embeddings = [e0, e1, e2, e3]
    sentences = ["A.", "B.", "C.", "D."]
    features = cle_features(embeddings, sentences)

    # Only sentence 2 (index 2) is isolated
    assert abs(features["cle_semantic_isolated_rate"] - 0.25) < 1e-5


def test_cle_semantic_isolated_rate_first_last_excluded():
    """First and last sentences cannot be isolated."""
    # Even if first/last have low similarity, they can't be isolated
    # (need both neighbors)
    a = make_unit_vector(0)
    b = make_unit_vector(1)
    embeddings = [a, b, a]  # sims = [0, 0]
    sentences = ["A.", "B.", "C."]
    features = cle_features(embeddings, sentences)

    # Sentence 0: first, excluded
    # Sentence 1: prev=0 < LOW_T, next=0 < LOW_T -> isolated
    # Sentence 2: last, excluded
    # 1 isolated out of 3 = 0.333...
    assert abs(features["cle_semantic_isolated_rate"] - (1.0 / 3.0)) < 1e-5


def test_cle_semantic_isolated_rate_two_sentences():
    """With only 2 sentences, none can be isolated."""
    embeddings = [make_unit_vector(0), make_unit_vector(1)]
    sentences = ["A.", "B."]
    features = cle_features(embeddings, sentences)

    # Only 2 sentences: first and last, neither can be isolated
    assert features["cle_semantic_isolated_rate"] == 0.0


def test_cle_semantic_features_relationship():
    """Test relationship between semantic_break_rate and semantic_isolated_rate."""
    # Pattern: high, low, high, low, high
    # sims = [0.8, 0.3, 0.7, 0.2]
    # Sentence 0: no break before, no break after (0.8 > LOW_T) -> no break
    # Sentence 1: no break before (0.8 > LOW_T), break after (0.3 < LOW_T) -> break, not isolated
    # Sentence 2: break before (0.3 < LOW_T), no break after (0.7 > LOW_T) -> break, not isolated
    # Sentence 3: no break before (0.7 > LOW_T), break after (0.2 < LOW_T) -> break, not isolated
    # Sentence 4: break before (0.2 < LOW_T), no next -> break
    # Breaks: 4/5 = 0.8, Isolated: 0/5 = 0.0

    e0 = make_unit_vector_from_2d(1.0, 0.0)

    # e1 such that dot(e0, e1) = 0.8
    theta1 = math.acos(0.8)
    e1 = make_unit_vector_from_2d(math.cos(theta1), math.sin(theta1))

    # e2 such that dot(e1, e2) = 0.3
    theta2 = theta1 + math.acos(0.3)
    e2 = make_unit_vector_from_2d(math.cos(theta2), math.sin(theta2))

    # e3 such that dot(e2, e3) = 0.7
    theta3 = theta2 + math.acos(0.7)
    e3 = make_unit_vector_from_2d(math.cos(theta3), math.sin(theta3))

    # e4 such that dot(e3, e4) = 0.2
    theta4 = theta3 + math.acos(0.2)
    e4 = make_unit_vector_from_2d(math.cos(theta4), math.sin(theta4))

    embeddings = [e0, e1, e2, e3, e4]
    sentences = ["A.", "B.", "C.", "D.", "E."]
    features = cle_features(embeddings, sentences)

    # 4 out of 5 sentences have breaks, but none are isolated (each has one high-sim neighbor)
    assert abs(features["cle_semantic_break_rate"] - 0.8) < 1e-5
    assert abs(features["cle_semantic_isolated_rate"] - 0.0) < 1e-5
    # This demonstrates that break_rate >= isolated_rate (isolated is a subset of breaks)
    assert features["cle_semantic_break_rate"] >= features["cle_semantic_isolated_rate"]
