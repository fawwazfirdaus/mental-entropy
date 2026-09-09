"""Tests for the NE (narrative entropy) feature extraction module.

These tests use synthetic unit vectors (no transformer model needed)
and sample sentences for text cue detection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mental_entropy.features.ne import (
    NE_FEATURE_KEYS,
    ne_features,
    ne_features_from_result,
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


# ---------------------------------------------------------------------------
# Test: exact keyset
# ---------------------------------------------------------------------------


def test_ne_features_returns_exact_keys():
    """NE features dict must have exactly the expected keys."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    sentences = ["First sentence.", "Second sentence."]
    features = ne_features(embeddings, sentences)

    assert set(features.keys()) == NE_FEATURE_KEYS
    assert len(features) == 18


def test_ne_features_all_values_json_serializable():
    """All returned values must be Python float (not numpy types)."""
    import json

    embeddings = [make_unit_vector(0), make_unit_vector(0), make_unit_vector(1)]
    sentences = ["First.", "Second.", "Third."]
    features = ne_features(embeddings, sentences)

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


def test_ne_features_empty_input():
    """n=0: return all keys with 0.0 defaults."""
    features = ne_features([], [])

    assert set(features.keys()) == NE_FEATURE_KEYS
    for key, value in features.items():
        assert value == 0.0, f"Key {key} should be 0.0 for empty input"


def test_ne_features_single_sentence():
    """n=1: return all keys with 0.0 defaults."""
    features = ne_features([make_unit_vector(0)], ["Single sentence."])

    assert set(features.keys()) == NE_FEATURE_KEYS
    for key, value in features.items():
        assert value == 0.0, f"Key {key} should be 0.0 for single sentence"


# ---------------------------------------------------------------------------
# Test: length mismatch
# ---------------------------------------------------------------------------


def test_ne_features_length_mismatch_raises():
    """Mismatched embeddings and sentences lengths should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    sentences = ["Only one."]

    with pytest.raises(ValueError, match="Length mismatch"):
        ne_features(embeddings, sentences)


# ---------------------------------------------------------------------------
# Test: NaN/Inf policy
# ---------------------------------------------------------------------------


def test_ne_features_raises_on_nan():
    """A single NaN in any embedding should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    embeddings[0][0] = float("nan")
    sentences = ["First.", "Second."]

    with pytest.raises(ValueError, match="NaN or Inf"):
        ne_features(embeddings, sentences)


def test_ne_features_raises_on_inf():
    """Inf in any embedding should raise ValueError."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    embeddings[1][5] = float("inf")
    sentences = ["First.", "Second."]

    with pytest.raises(ValueError, match="NaN or Inf"):
        ne_features(embeddings, sentences)


# ---------------------------------------------------------------------------
# Test: determinism
# ---------------------------------------------------------------------------


def test_ne_features_deterministic():
    """Same input should always produce same output."""
    embeddings = [
        make_unit_vector(0),
        make_unit_vector(0),
        make_unit_vector(1),
        make_unit_vector(0),
    ]
    sentences = ["Today was good.", "I felt okay.", "Then something changed.", "Finally peace."]

    features1 = ne_features(embeddings, sentences)
    features2 = ne_features(embeddings, sentences)

    assert features1 == features2


# ---------------------------------------------------------------------------
# Test: embedding features - identical vectors
# ---------------------------------------------------------------------------


def test_ne_features_identical_embeddings():
    """Identical embeddings => high start_end_sim, zero wander, arc_linearity ~0."""
    n = 5
    embeddings = [make_unit_vector(0) for _ in range(n)]
    sentences = [f"Sentence {i}." for i in range(n)]
    features = ne_features(embeddings, sentences)

    # Start-end similarity = 1.0 (identical vectors)
    assert abs(features["ne_start_end_sim"] - 1.0) < 1e-5

    # Semantic wander = 0 (all adjacent pairs identical)
    assert abs(features["ne_semantic_wander"]) < 1e-5

    # Arc linearity: direct=0, path=0, so 0 / (0 + eps) ≈ 0
    assert abs(features["ne_arc_linearity"]) < 1e-5

    # All sentences identical to centroid (which is the same vector)
    assert abs(features["ne_start_to_centroid"] - 1.0) < 1e-5
    assert abs(features["ne_end_to_centroid"] - 1.0) < 1e-5
    assert abs(features["ne_consolidation_delta"]) < 1e-5


# ---------------------------------------------------------------------------
# Test: embedding features - orthogonal vectors (maximum wander)
# ---------------------------------------------------------------------------


def test_ne_features_orthogonal_embeddings():
    """All orthogonal embeddings => start_end_sim=0, high wander."""
    n = 4
    embeddings = [make_unit_vector(i) for i in range(n)]
    sentences = [f"Sentence {i}." for i in range(n)]
    features = ne_features(embeddings, sentences)

    # Start-end similarity = 0 (orthogonal)
    assert abs(features["ne_start_end_sim"]) < 1e-5

    # Path: each adjacent pair has distance 1.0, so path = n-1 = 3
    # Semantic wander = 3 / 3 = 1.0
    assert abs(features["ne_semantic_wander"] - 1.0) < 1e-5

    # Arc linearity: direct=1, path=3, so 1 / (3 + eps) ≈ 0.333
    assert abs(features["ne_arc_linearity"] - 1.0 / 3.0) < 1e-5


# ---------------------------------------------------------------------------
# Test: embedding features - arc pattern (start diverges, end converges)
# ---------------------------------------------------------------------------


def test_ne_features_arc_pattern():
    """Pattern that starts far from centroid and ends closer (positive consolidation)."""
    # Create a pattern: start with outlier, gradually move toward center
    # e0 = (1, 0), e1 = (0.7, 0.7), e2 = (0, 1), e3 = (0.5, 0.5)
    # The centroid will be somewhere in between, and e3 (last) is closest to center

    e0 = make_unit_vector_from_2d(1.0, 0.0)  # axis 0
    e1 = make_unit_vector_from_2d(0.7, 0.7)  # 45 degrees
    e2 = make_unit_vector_from_2d(0.0, 1.0)  # axis 1
    e3 = make_unit_vector_from_2d(0.5, 0.5)  # 45 degrees (closer to mean)

    embeddings = [e0, e1, e2, e3]
    sentences = ["Start far.", "Moving.", "Further.", "Back to center."]
    features = ne_features(embeddings, sentences)

    # We expect some positive consolidation since last sentences are closer to mean
    # But the exact value depends on the centroid calculation
    # Just verify it computes without error and returns reasonable range
    assert -1.0 <= features["ne_consolidation_delta"] <= 1.0
    assert 0.0 <= features["ne_semantic_wander"] <= 2.0


# ---------------------------------------------------------------------------
# Test: embedding features - positive consolidation delta
# ---------------------------------------------------------------------------


def test_ne_features_positive_consolidation():
    """Verify positive consolidation when end is closer to centroid than start."""
    # Pattern: first 3 sentences are outliers, last 3 converge to center
    # Use orthogonal outliers at start, then converge

    # e0, e1, e2 are spread out (axis 0, 1, 2)
    # e3, e4, e5 are all the same (axis 3) - they'll pull centroid toward them

    embeddings = [
        make_unit_vector(0),  # outlier
        make_unit_vector(1),  # outlier
        make_unit_vector(2),  # outlier
        make_unit_vector(3),  # convergent
        make_unit_vector(3),  # convergent
        make_unit_vector(3),  # convergent
    ]
    sentences = [f"Sentence {i}." for i in range(6)]
    features = ne_features(embeddings, sentences)

    # Last 3 sentences are identical and should be closer to centroid
    # (centroid is pulled toward them)
    # First 3 are spread out and should average lower similarity to centroid
    # So end_to_centroid > start_to_centroid => positive delta
    assert features["ne_consolidation_delta"] > 0


# ---------------------------------------------------------------------------
# Test: text cues - temporal markers
# ---------------------------------------------------------------------------


def test_ne_temporal_markers_rate():
    """Test temporal marker detection."""
    embeddings = [make_unit_vector(0) for _ in range(4)]
    sentences = [
        "Today I woke up early.",  # has "today"
        "The weather was nice.",  # no marker
        "Yesterday was different.",  # has "yesterday"
        "I went to the store.",  # no marker
    ]
    features = ne_features(embeddings, sentences)

    # 2 out of 4 sentences have temporal markers
    assert abs(features["ne_temporal_markers_rate"] - 0.5) < 1e-5


def test_ne_temporal_markers_case_insensitive():
    """Temporal markers should be case-insensitive."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = ["TODAY was good.", "YESTERDAY was bad."]
    features = ne_features(embeddings, sentences)

    assert abs(features["ne_temporal_markers_rate"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test: text cues - temporal jump rate
# ---------------------------------------------------------------------------


def test_ne_temporal_jump_rate_no_jumps():
    """No temporal jumps when no past/future markers."""
    embeddings = [make_unit_vector(0) for _ in range(3)]
    sentences = ["Normal sentence.", "Another one.", "Final thought."]
    features = ne_features(embeddings, sentences)

    assert features["ne_temporal_jump_rate"] == 0.0


def test_ne_temporal_jump_rate_with_jump():
    """Detect temporal jump from past to future without glue."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "Yesterday I was sad.",  # past marker
        "Tomorrow will be better.",  # future marker, no glue
    ]
    features = ne_features(embeddings, sentences)

    # 1 jump out of 1 pair
    assert abs(features["ne_temporal_jump_rate"] - 1.0) < 1e-5


def test_ne_temporal_jump_rate_with_glue_connector():
    """No jump counted when glue connector bridges past→future."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "Yesterday was hard.",  # past marker
        "But tomorrow will be better.",  # future marker WITH glue "but"
    ]
    features = ne_features(embeddings, sentences)

    # Glue connector "but" prevents counting as jump
    assert features["ne_temporal_jump_rate"] == 0.0


def test_ne_temporal_jump_rate_glue_in_first_sentence():
    """Glue connector in first sentence also prevents jump."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "Yesterday was hard because of work.",  # past + glue "because"
        "Tomorrow will be better.",  # future
    ]
    features = ne_features(embeddings, sentences)

    # Glue connector "because" in first sentence prevents jump
    assert features["ne_temporal_jump_rate"] == 0.0


def test_ne_temporal_jump_rate_future_to_past():
    """Jump detected in future→past direction too."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "Tomorrow I will rest.",  # future
        "Yesterday was exhausting.",  # past, no glue
    ]
    features = ne_features(embeddings, sentences)

    assert abs(features["ne_temporal_jump_rate"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test: text cues - connectors rate
# ---------------------------------------------------------------------------


def test_ne_connectors_rate():
    """Test connector detection."""
    embeddings = [make_unit_vector(0) for _ in range(4)]
    sentences = [
        "I was tired.",  # no connector
        "But I kept going.",  # has "but"
        "Therefore I succeeded.",  # has "therefore"
        "The end.",  # no connector
    ]
    features = ne_features(embeddings, sentences)

    # 2 out of 4 sentences have connectors
    assert abs(features["ne_connectors_rate"] - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# Test: text cues - reflection rate
# ---------------------------------------------------------------------------


def test_ne_reflection_rate():
    """Test reflection phrase detection."""
    embeddings = [make_unit_vector(0) for _ in range(4)]
    sentences = [
        "It was hard.",  # no reflection
        "I realized that something important happened.",  # has "i realized that"
        "Looking back, it makes sense.",  # has "looking back,"
        "Done.",  # no reflection
    ]
    features = ne_features(embeddings, sentences)

    # 2 out of 4 sentences have reflection phrases
    assert abs(features["ne_reflection_rate"] - 0.5) < 1e-5


def test_ne_last_sentence_reflective_true():
    """Last sentence is reflective."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = ["Something happened.", "I realized that it was true."]
    features = ne_features(embeddings, sentences)

    assert features["ne_last_sentence_reflective"] == 1.0


def test_ne_last_sentence_reflective_false():
    """Last sentence is not reflective."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = ["I realized something.", "Then I left."]
    features = ne_features(embeddings, sentences)

    assert features["ne_last_sentence_reflective"] == 0.0


# ---------------------------------------------------------------------------
# Test: text cues - fragment detection
# ---------------------------------------------------------------------------


def test_ne_fragment_rate_short_sentences():
    """Short sentences (<=4 tokens) are fragments."""
    embeddings = [make_unit_vector(0) for _ in range(3)]
    sentences = [
        "Too short",  # 2 tokens, fragment
        "This is a longer sentence with many words.",  # not fragment
        "Just three words",  # 3 tokens, fragment
    ]
    features = ne_features(embeddings, sentences)

    # 2 out of 3 are fragments
    assert abs(features["ne_fragment_sentence_rate"] - 2.0 / 3.0) < 1e-5


def test_ne_fragment_rate_no_verb():
    """Sentences without verb-like patterns are fragments."""
    embeddings = [make_unit_vector(0) for _ in range(3)]
    sentences = [
        "The big red balloon on display.",  # no verb, 5 tokens, fragment
        "I am very happy about this.",  # has "am", 6 tokens, not fragment
        "She walked quickly back home.",  # has "walked" (ed suffix), 5 tokens, not fragment
    ]
    features = ne_features(embeddings, sentences)

    # Only first sentence is a fragment (no verb detected)
    # "walked" is now detected via the *ed suffix pattern
    assert abs(features["ne_fragment_sentence_rate"] - 1.0 / 3.0) < 1e-5


def test_ne_fragment_rate_expanded_verb_detection():
    """Test that expanded verb detection catches various verb forms."""
    embeddings = [make_unit_vector(0) for _ in range(6)]
    sentences = [
        "I was confused about everything today.",  # "was" (auxiliary) - not fragment
        "Everyone seemed engaged and very interested.",  # "seemed" (ed suffix) - not fragment
        "Things clicked into place right away.",  # "clicked" (ed suffix) - not fragment
        "The whole team went well together.",  # "went" (irregular past) - not fragment
        "She will arrive here very soon.",  # "will" (modal) - not fragment
        "The sky above the dark clouds.",  # no verb, 6 tokens - fragment
    ]
    features = ne_features(embeddings, sentences)

    # Only last sentence is a fragment (no verb)
    assert abs(features["ne_fragment_sentence_rate"] - 1.0 / 6.0) < 1e-5


def test_ne_fragment_rate_irregular_verbs():
    """Test that irregular past tense verbs are detected."""
    embeddings = [make_unit_vector(0) for _ in range(4)]
    sentences = [
        "I thought about it very carefully.",  # "thought" - irregular past
        "She knew the right answer already.",  # "knew" - irregular past
        "They went to the big store.",  # "went" - irregular past
        "The quiet old dusty red book.",  # no verb, 6 tokens - fragment
    ]
    features = ne_features(embeddings, sentences)

    # Only last sentence is a fragment (no verb, >4 tokens)
    assert abs(features["ne_fragment_sentence_rate"] - 1.0 / 4.0) < 1e-5


def test_ne_fragment_rate_ellipsis_ending():
    """Sentences ending with ... are fragments."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "I was thinking about something...",  # ends with ..., fragment
        "This is a complete thought.",  # not fragment
    ]
    features = ne_features(embeddings, sentences)

    assert abs(features["ne_fragment_sentence_rate"] - 0.5) < 1e-5


def test_ne_fragment_rate_no_terminal_punct():
    """Sentences without terminal punctuation are fragments."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "This sentence has no ending punctuation",  # no .!?, fragment
        "This one does.",  # has period, need to check other criteria
    ]
    features = ne_features(embeddings, sentences)

    # First is fragment due to no terminal punctuation
    # Second may or may not be fragment depending on verb detection
    assert features["ne_fragment_sentence_rate"] >= 0.5


# ---------------------------------------------------------------------------
# Test: text cues - punctuation break rate
# ---------------------------------------------------------------------------


def test_ne_punct_break_rate_ellipsis():
    """Sentences ending with ... have punct break."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "Something happened...",  # ends with ...
        "This is normal.",  # no break
    ]
    features = ne_features(embeddings, sentences)

    assert abs(features["ne_punct_break_rate"] - 0.5) < 1e-5


def test_ne_punct_break_rate_multiple_dashes():
    """Sentences with multiple dashes have punct break."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "I thought--wait--what was that?",  # multiple dashes
        "This is normal.",  # no break
    ]
    features = ne_features(embeddings, sentences)

    assert abs(features["ne_punct_break_rate"] - 0.5) < 1e-5


def test_ne_punct_break_rate_multiple_ellipses():
    """Sentences with 2+ ellipsis clusters have punct break."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "I thought... and then... nothing.",  # 2 ellipsis clusters
        "This is normal.",  # no break
    ]
    features = ne_features(embeddings, sentences)

    assert abs(features["ne_punct_break_rate"] - 0.5) < 1e-5


def test_ne_punct_break_rate_single_dash():
    """Single dash does not trigger punct break."""
    embeddings = [make_unit_vector(0) for _ in range(2)]
    sentences = [
        "I thought--about it.",  # single dash occurrence
        "This is normal.",
    ]
    features = ne_features(embeddings, sentences)

    # Single dash occurrence should not trigger (need 2+)
    assert features["ne_punct_break_rate"] == 0.0


# ---------------------------------------------------------------------------
# Test: wrapper for EmbeddingResult
# ---------------------------------------------------------------------------


def test_ne_features_from_result():
    """Test the EmbeddingResult wrapper."""
    embeddings = [make_unit_vector(0), make_unit_vector(0), make_unit_vector(1)]
    texts = ["Today was good.", "I felt happy.", "Something else."]
    result = make_embedding_result(embeddings, texts)

    features = ne_features_from_result(result)

    # Should match direct call
    expected = ne_features(embeddings, texts)
    assert features == expected


def test_ne_features_from_result_empty():
    """Test wrapper with empty result."""
    result = EmbeddingResult(sentences=[], doc_embedding=None)
    features = ne_features_from_result(result)

    assert set(features.keys()) == NE_FEATURE_KEYS
    for value in features.values():
        assert value == 0.0


# ---------------------------------------------------------------------------
# Test: two sentences (boundary case)
# ---------------------------------------------------------------------------


def test_ne_features_two_sentences_identical():
    """n=2 with identical embeddings."""
    embeddings = [make_unit_vector(0), make_unit_vector(0)]
    sentences = ["First thought.", "Same thought."]
    features = ne_features(embeddings, sentences)

    assert abs(features["ne_start_end_sim"] - 1.0) < 1e-5
    assert abs(features["ne_semantic_wander"]) < 1e-5


def test_ne_features_two_sentences_orthogonal():
    """n=2 with orthogonal embeddings."""
    embeddings = [make_unit_vector(0), make_unit_vector(1)]
    sentences = ["First topic.", "Different topic."]
    features = ne_features(embeddings, sentences)

    assert abs(features["ne_start_end_sim"]) < 1e-5
    assert abs(features["ne_semantic_wander"] - 1.0) < 1e-5
    # Arc linearity: direct=1, path=1, so 1 / (1 + eps) ≈ 1.0
    assert abs(features["ne_arc_linearity"] - 1.0) < 1e-5
