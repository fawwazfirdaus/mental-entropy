"""Narrative Entropy (NE) feature extraction for MES v1.

This module computes deterministic numeric features from ordered sentence embeddings
and raw sentence text to measure narrative/reflective structure vs scattered fragments.

Low NE = clear arc or integration (beginning→middle→end, description→insight).
High NE = fragments, no arc, unresolved starts, erratic temporal movement.

Features use:
- Embedding-derived structure (topic arc and integration)
- Surface-level narrative cues from text (NOT sentiment/emotion)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from mental_entropy.utils.linguistic import (
    ABANDONMENT_MARKERS,
    CLOSURE_CUES,
    CONNECTORS,
    FUTURE_MARKERS,
    GLUE_CONNECTORS,
    PAST_MARKERS,
    REFLECTION_PHRASES,
    TEMPORAL_MARKERS,
    ENDS_WITH_ELLIPSIS,
    ENDS_WITH_SENTENCE_PUNCT,
    ELLIPSIS_CLUSTER,
    MULTIPLE_DASHES,
    contains_any,
    has_verb,
    tokenize_simple,
)

if TYPE_CHECKING:
    from mental_entropy.embedding.types import EmbeddingResult

# ---------------------------------------------------------------------------
# Expected feature keys (for validation / documentation)
# ---------------------------------------------------------------------------
NE_FEATURE_KEYS: frozenset[str] = frozenset(
    [
        "ne_start_end_sim",
        "ne_arc_linearity",
        "ne_semantic_wander",
        "ne_end_to_centroid",
        "ne_start_to_centroid",
        "ne_consolidation_delta",
        "ne_temporal_markers_rate",
        "ne_temporal_jump_rate",
        "ne_connectors_rate",
        "ne_reflection_rate",
        "ne_last_sentence_reflective",
        "ne_fragment_sentence_rate",
        "ne_punct_break_rate",
        # Narrative closure features (v3)
        "ne_end_closure_cue",
        "ne_end_abandonment_cue",
        "ne_end_open_question",
        "ne_end_fragment",
        "ne_end_similarity",
    ]
)


# ---------------------------------------------------------------------------
# Internal: Text cue detection helpers
# ---------------------------------------------------------------------------
def _has_temporal_marker(text: str) -> bool:
    """Check if sentence contains any temporal marker."""
    return contains_any(text, TEMPORAL_MARKERS)


def _has_past_marker(text: str) -> bool:
    """Check if sentence contains a strict past marker for jump detection."""
    return contains_any(text, PAST_MARKERS)


def _has_future_marker(text: str) -> bool:
    """Check if sentence contains a strict future marker for jump detection."""
    return contains_any(text, FUTURE_MARKERS)


def _has_glue_connector(text: str) -> bool:
    """Check if sentence contains a glue connector."""
    return contains_any(text, GLUE_CONNECTORS)


def _has_connector(text: str) -> bool:
    """Check if sentence contains any discourse connector."""
    return contains_any(text, CONNECTORS)


def _has_reflection_phrase(text: str) -> bool:
    """Check if sentence contains any reflection phrase."""
    return contains_any(text, REFLECTION_PHRASES)


def _is_fragment(text: str) -> bool:
    """
    Check if sentence is fragment-like. Fragment if ANY:
    - token count <= 4 and lacks a clear clause ending
    - no verb-like pattern (expanded detection)
    - ends with "..."
    - does NOT end with sentence punctuation [.!?]
    """
    tokens = tokenize_simple(text)
    if len(tokens) <= 4:
        if has_verb(text) and ENDS_WITH_SENTENCE_PUNCT.search(text):
            return False
        return True

    if not has_verb(text):
        return True

    if ENDS_WITH_ELLIPSIS.search(text):
        return True

    if not ENDS_WITH_SENTENCE_PUNCT.search(text):
        return True

    return False


def _has_closure_cue(text: str) -> bool:
    """Check if sentence contains closure/resolution language."""
    return contains_any(text, CLOSURE_CUES)


def _has_abandonment_marker(text: str) -> bool:
    """Check if sentence contains abandonment/trailing-off language."""
    return contains_any(text, ABANDONMENT_MARKERS)


def _ends_with_question(text: str) -> bool:
    """Check if sentence ends with a question mark."""
    return bool(re.search(r"\?\s*$", text))


def _has_punct_break(text: str) -> bool:
    """
    Check if sentence has punctuation break markers:
    - ends with "..."
    - contains multiple dash markers (-- or em-dash)
    - contains 2+ ellipsis clusters
    """
    if ENDS_WITH_ELLIPSIS.search(text):
        return True

    # Check for multiple dash markers (count occurrences)
    dash_matches = MULTIPLE_DASHES.findall(text)
    if len(dash_matches) >= 2:
        return True

    # Check for 2+ ellipsis clusters
    ellipsis_matches = ELLIPSIS_CLUSTER.findall(text)
    if len(ellipsis_matches) >= 2:
        return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ne_features(
    sentence_embeddings: list[list[float]],
    sentences: list[str],
) -> dict[str, float]:
    """
    Compute NE (narrative entropy) features from ordered sentence embeddings and text.

    Args:
        sentence_embeddings: List of embedding vectors (each a list of floats).
            Expected to be L2-normalized (unit length). Order is preserved.
        sentences: List of raw sentence strings in the same order.

    Returns:
        A dict with exactly the keys in NE_FEATURE_KEYS.
        All values are Python float (JSON-serializable).

    Raises:
        ValueError: If embeddings contain NaN or Inf values, or if lengths mismatch.
    """
    n = len(sentence_embeddings)

    # Validate lengths match
    if len(sentences) != n:
        raise ValueError(
            f"Length mismatch: {n} embeddings vs {len(sentences)} sentences"
        )

    # Handle edge cases: n == 0 or n == 1
    if n <= 1:
        return _empty_features()

    # Convert to numpy array
    X = np.asarray(sentence_embeddings, dtype=np.float32)

    # Validate: check for NaN/Inf
    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain NaN or Inf values")

    # ---------------------------------------------------------------------------
    # Embedding-only features
    # ---------------------------------------------------------------------------

    # ne_start_end_sim: cosine similarity between first and last sentence
    start_end_sim = float(np.dot(X[0], X[-1]))

    # Path length: sum of (1 - cos_sim) for adjacent pairs
    adj_sims = (X[:-1] * X[1:]).sum(axis=1)  # cosine similarities
    adj_dists = 1.0 - adj_sims  # cosine distances
    path = float(np.sum(adj_dists))

    # Direct distance from start to end
    direct = 1.0 - start_end_sim

    # ne_arc_linearity: direct / (path + epsilon)
    arc_linearity = float(direct / (path + 1e-8))

    # ne_semantic_wander: path / (n-1)
    semantic_wander = float(path / (n - 1))

    # Compute document centroid (L2-normalized)
    centroid = X.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm

    # Similarities to centroid
    sims_to_centroid = X @ centroid  # (n,)

    # ne_start_to_centroid: mean of first k sentences' similarity to centroid
    k = min(3, n)
    start_to_centroid = float(np.mean(sims_to_centroid[:k]))

    # ne_end_to_centroid: mean of last k sentences' similarity to centroid
    end_to_centroid = float(np.mean(sims_to_centroid[-k:]))

    # ne_consolidation_delta: end - start (positive = converging toward meaning)
    consolidation_delta = end_to_centroid - start_to_centroid

    # ---------------------------------------------------------------------------
    # Text-based narrative cues
    # ---------------------------------------------------------------------------

    # Count sentences with temporal markers
    temporal_count = sum(1 for s in sentences if _has_temporal_marker(s))
    temporal_markers_rate = float(temporal_count / n)

    # Count sentences with discourse connectors
    connector_count = sum(1 for s in sentences if _has_connector(s))
    connectors_rate = float(connector_count / n)

    # Count sentences with reflection phrases
    reflection_count = sum(1 for s in sentences if _has_reflection_phrase(s))
    reflection_rate = float(reflection_count / n)

    # Last sentence reflective (boolean as 0.0 or 1.0)
    last_sentence_reflective = 1.0 if _has_reflection_phrase(sentences[-1]) else 0.0

    # Temporal jump rate: count adjacent pairs that flip past↔future without glue
    jump_count = 0
    for i in range(1, n):
        prev_sent = sentences[i - 1]
        curr_sent = sentences[i]

        prev_past = _has_past_marker(prev_sent)
        prev_future = _has_future_marker(prev_sent)
        curr_past = _has_past_marker(curr_sent)
        curr_future = _has_future_marker(curr_sent)

        # Check for flip: (prev_past AND curr_future) OR (prev_future AND curr_past)
        is_flip = (prev_past and curr_future) or (prev_future and curr_past)

        if is_flip:
            # Check if either sentence has glue connector
            has_glue = _has_glue_connector(prev_sent) or _has_glue_connector(curr_sent)
            if not has_glue:
                jump_count += 1

    temporal_jump_rate = float(jump_count / (n - 1))

    # Fragment sentence rate
    fragment_count = sum(1 for s in sentences if _is_fragment(s))
    fragment_sentence_rate = float(fragment_count / n)

    # Punctuation break rate
    punct_break_count = sum(1 for s in sentences if _has_punct_break(s))
    punct_break_rate = float(punct_break_count / n)

    # ---------------------------------------------------------------------------
    # Narrative closure features
    # ---------------------------------------------------------------------------

    # Check last 2 sentences for closure/abandonment cues
    end_k = min(2, n)
    end_sentences = sentences[-end_k:]

    # ne_end_closure_cue: any of last 2 sentences contain closure language
    end_closure_cue = 1.0 if any(_has_closure_cue(s) for s in end_sentences) else 0.0

    # ne_end_abandonment_cue: any of last 2 sentences contain abandonment language
    end_abandonment_cue = 1.0 if any(_has_abandonment_marker(s) for s in end_sentences) else 0.0

    # ne_end_open_question: last sentence ends with question mark
    end_open_question = 1.0 if _ends_with_question(sentences[-1]) else 0.0

    # ne_end_fragment: last sentence is a fragment
    end_fragment = 1.0 if _is_fragment(sentences[-1]) else 0.0

    # ne_end_similarity: cosine similarity of last sentence to mean of all preceding
    if n >= 2:
        preceding_mean = X[:-1].mean(axis=0)
        preceding_norm = np.linalg.norm(preceding_mean)
        if preceding_norm > 0:
            preceding_mean = preceding_mean / preceding_norm
        end_similarity = float(np.dot(X[-1], preceding_mean))
    else:
        end_similarity = 0.0

    # ---------------------------------------------------------------------------
    # Assemble feature dict
    # ---------------------------------------------------------------------------
    features: dict[str, float] = {
        "ne_start_end_sim": start_end_sim,
        "ne_arc_linearity": arc_linearity,
        "ne_semantic_wander": semantic_wander,
        "ne_end_to_centroid": end_to_centroid,
        "ne_start_to_centroid": start_to_centroid,
        "ne_consolidation_delta": consolidation_delta,
        "ne_temporal_markers_rate": temporal_markers_rate,
        "ne_temporal_jump_rate": temporal_jump_rate,
        "ne_connectors_rate": connectors_rate,
        "ne_reflection_rate": reflection_rate,
        "ne_last_sentence_reflective": last_sentence_reflective,
        "ne_fragment_sentence_rate": fragment_sentence_rate,
        "ne_punct_break_rate": punct_break_rate,
        "ne_end_closure_cue": end_closure_cue,
        "ne_end_abandonment_cue": end_abandonment_cue,
        "ne_end_open_question": end_open_question,
        "ne_end_fragment": end_fragment,
        "ne_end_similarity": end_similarity,
    }

    return features


def _empty_features() -> dict[str, float]:
    """Return a feature dict with all zeros for n <= 1 sentences."""
    return {
        "ne_start_end_sim": 0.0,
        "ne_arc_linearity": 0.0,
        "ne_semantic_wander": 0.0,
        "ne_end_to_centroid": 0.0,
        "ne_start_to_centroid": 0.0,
        "ne_consolidation_delta": 0.0,
        "ne_temporal_markers_rate": 0.0,
        "ne_temporal_jump_rate": 0.0,
        "ne_connectors_rate": 0.0,
        "ne_reflection_rate": 0.0,
        "ne_last_sentence_reflective": 0.0,
        "ne_fragment_sentence_rate": 0.0,
        "ne_punct_break_rate": 0.0,
        "ne_end_closure_cue": 0.0,
        "ne_end_abandonment_cue": 0.0,
        "ne_end_open_question": 0.0,
        "ne_end_fragment": 0.0,
        "ne_end_similarity": 0.0,
    }


def ne_features_from_result(result: "EmbeddingResult") -> dict[str, float]:
    """
    Compute NE features from an EmbeddingResult.

    Convenience wrapper that extracts embeddings and text from result.sentences.

    Args:
        result: An EmbeddingResult with ordered SentenceEmbedding objects.

    Returns:
        NE feature dict (same as ne_features).
    """
    embeddings = [sent.embedding for sent in result.sentences]
    sentences = [sent.text for sent in result.sentences]
    return ne_features(embeddings, sentences)
