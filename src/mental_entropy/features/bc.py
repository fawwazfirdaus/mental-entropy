"""Belief Conflict (BC) feature extraction for MES v1.

This module computes deterministic numeric features from ordered sentence embeddings
and raw sentence text to measure unresolved contradictions and belief conflicts
in journal writing.

BC measures structural coherence of self-beliefs and values. Contradictions that
are acknowledged and integrated (e.g., "I love him but I also hate him - both are true")
indicate organized processing. Unresolved contradictions (e.g., "I am confident"
followed by "I am worthless" with no integration) signal high entropy.

Key distinction:
- INTEGRATED contradiction: Both sides acknowledged, held together with awareness
- UNRESOLVED contradiction: Opposing beliefs stated without integration markers
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from mental_entropy.utils.linguistic import (
    BELIEF_PATTERNS,
    INTEGRATION_MARKERS,
    NEGATION_WORDS,
    POLAR_WORD_PAIRS,
    contains_any,
)
from mental_entropy.utils.thresholds import CONFLICT_SIM_T

if TYPE_CHECKING:
    from mental_entropy.embedding.types import EmbeddingResult

# ---------------------------------------------------------------------------
# Expected feature keys (for validation / documentation)
# ---------------------------------------------------------------------------

BC_FEATURE_KEYS: frozenset[str] = frozenset(
    [
        # Core belief detection
        "bc_belief_sentence_count",
        "bc_belief_sentence_rate",
        # Conflict detection
        "bc_conflict_pair_count",
        "bc_conflict_rate",
        # Conflict characteristics
        "bc_max_conflict_sim",
        "bc_mean_conflict_sim",
        "bc_conflict_span_mean",
        "bc_conflict_span_max",
        # Resolution/Integration
        "bc_integrated_count",
        "bc_integration_rate",
        "bc_unresolved_count",
        "bc_unresolved_rate",
    ]
)


# ---------------------------------------------------------------------------
# Internal: Belief and conflict detection helpers
# ---------------------------------------------------------------------------


def _is_belief_sentence(text: str) -> bool:
    """Check if sentence contains a belief statement pattern.

    Belief patterns include:
    - Identity statements: "I am", "I'm", "I feel"
    - Modal commitments: "I will", "I can", "I should"
    - Habitual self-perception: "I always", "I never"
    - Value statements: "I love", "I hate", "I want", "I need"
    """
    text_lower = text.lower()
    for pattern in BELIEF_PATTERNS:
        if pattern in text_lower:
            return True
    return False


def _has_negation(text: str) -> bool:
    """Check if text contains negation words."""
    return contains_any(text, NEGATION_WORDS)


def _find_polar_words(text: str) -> set[str]:
    """Find polar words present in text."""
    text_lower = text.lower()
    found = set()
    for word in POLAR_WORD_PAIRS:
        # Use word boundary matching
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            found.add(word)
    return found


def _has_polarity_opposition(text1: str, text2: str) -> bool:
    """Determine if two texts have opposing polarity.

    Opposition is detected via:
    1. Negation flip: one has negation, the other doesn't
       (e.g., "I can do this" vs "I can't do anything")
    2. Polar word opposition: one has a word, other has its opposite
       (e.g., "I love him" vs "I hate him")
    """
    neg1 = _has_negation(text1)
    neg2 = _has_negation(text2)

    # Negation flip (one has negation, other doesn't)
    if neg1 != neg2:
        return True

    # Polar word opposition
    polar1 = _find_polar_words(text1)
    polar2 = _find_polar_words(text2)

    for word in polar1:
        opposite = POLAR_WORD_PAIRS.get(word)
        if opposite and opposite in polar2:
            return True

    return False


def _find_conflict_pairs(
    embeddings: np.ndarray,
    sentences: list[str],
    belief_indices: list[int],
) -> list[tuple[int, int, float]]:
    """Find pairs of belief sentences that may be in conflict.

    A conflict pair is two belief sentences where:
    1. Embedding similarity >= CONFLICT_SIM_T (same topic)
    2. Polarity is opposite (negation flip or polar word opposition)

    Args:
        embeddings: Numpy array of shape (n, dim) with L2-normalized embeddings.
        sentences: List of raw sentence strings.
        belief_indices: Indices of sentences containing belief statements.

    Returns:
        List of (idx_i, idx_j, similarity) tuples for conflict pairs.
    """
    conflicts = []

    for i, idx_i in enumerate(belief_indices):
        for idx_j in belief_indices[i + 1 :]:
            # Check topic similarity (cosine = dot product for L2-normalized)
            sim = float(np.dot(embeddings[idx_i], embeddings[idx_j]))
            if sim < CONFLICT_SIM_T:
                continue

            # Check polarity opposition
            if _has_polarity_opposition(sentences[idx_i], sentences[idx_j]):
                conflicts.append((idx_i, idx_j, sim))

    return conflicts


def _has_integration_marker(text: str) -> bool:
    """Check if text contains an integration/resolution marker."""
    return contains_any(text, INTEGRATION_MARKERS)


def _is_conflict_integrated(
    conflict_pair: tuple[int, int, float],
    sentences: list[str],
    window: int = 2,
) -> bool:
    """Check if a conflict pair has nearby integration markers.

    Integration markers within `window` sentences of either
    conflict sentence count as resolution.

    Args:
        conflict_pair: Tuple of (idx_i, idx_j, similarity).
        sentences: List of raw sentence strings.
        window: Number of sentences before/after to check for integration.

    Returns:
        True if integration marker found within window of conflict.
    """
    idx_i, idx_j, _ = conflict_pair

    # Define the window around the conflict
    start = max(0, min(idx_i, idx_j) - window)
    end = min(len(sentences), max(idx_i, idx_j) + window + 1)

    # Check sentences in window for integration markers
    for idx in range(start, end):
        if _has_integration_marker(sentences[idx]):
            return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bc_features(
    sentence_embeddings: list[list[float]],
    sentences: list[str],
) -> dict[str, int | float]:
    """Compute BC (belief conflict) features from embeddings and text.

    Detects contradictions between belief statements and measures whether
    they are integrated (resolved) or left unresolved (high entropy signal).

    Args:
        sentence_embeddings: List of embedding vectors (each a list of floats).
            Expected to be L2-normalized (unit length). Order is preserved.
        sentences: List of raw sentence strings in the same order.

    Returns:
        A dict with exactly the keys in BC_FEATURE_KEYS.
        All values are Python int or float (JSON-serializable).

    Raises:
        ValueError: If embeddings contain NaN or Inf values, or if lengths mismatch.
    """
    n = len(sentence_embeddings)

    # Validate lengths match
    if len(sentences) != n:
        raise ValueError(
            f"Length mismatch: {n} embeddings vs {len(sentences)} sentences"
        )

    # Handle edge cases
    if n <= 1:
        return _empty_features()

    # Convert to numpy array
    X = np.asarray(sentence_embeddings, dtype=np.float32)

    # Validate: check for NaN/Inf
    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain NaN or Inf values")

    # Step 1: Identify belief sentences
    belief_indices = [i for i in range(n) if _is_belief_sentence(sentences[i])]
    bc_belief_sentence_count = len(belief_indices)
    bc_belief_sentence_rate = float(bc_belief_sentence_count / n)

    # Handle no belief sentences or single belief
    if bc_belief_sentence_count < 2:
        return _features_no_conflicts(bc_belief_sentence_count, bc_belief_sentence_rate)

    # Step 2: Find conflict pairs
    conflict_pairs = _find_conflict_pairs(X, sentences, belief_indices)
    bc_conflict_pair_count = len(conflict_pairs)
    bc_conflict_rate = float(bc_conflict_pair_count / bc_belief_sentence_count)

    # Handle no conflicts
    if bc_conflict_pair_count == 0:
        return _features_no_conflicts(bc_belief_sentence_count, bc_belief_sentence_rate)

    # Step 3: Compute conflict characteristics
    similarities = [sim for _, _, sim in conflict_pairs]
    bc_max_conflict_sim = float(max(similarities))
    bc_mean_conflict_sim = float(np.mean(similarities))

    spans = [abs(j - i) for i, j, _ in conflict_pairs]
    bc_conflict_span_mean = float(np.mean(spans))
    bc_conflict_span_max = int(max(spans))

    # Step 4: Check integration for each conflict
    integrated = [_is_conflict_integrated(cp, sentences) for cp in conflict_pairs]
    bc_integrated_count = int(sum(integrated))
    bc_integration_rate = float(bc_integrated_count / bc_conflict_pair_count)
    bc_unresolved_count = int(bc_conflict_pair_count - bc_integrated_count)
    bc_unresolved_rate = float(bc_unresolved_count / bc_conflict_pair_count)

    return {
        "bc_belief_sentence_count": bc_belief_sentence_count,
        "bc_belief_sentence_rate": bc_belief_sentence_rate,
        "bc_conflict_pair_count": bc_conflict_pair_count,
        "bc_conflict_rate": bc_conflict_rate,
        "bc_max_conflict_sim": bc_max_conflict_sim,
        "bc_mean_conflict_sim": bc_mean_conflict_sim,
        "bc_conflict_span_mean": bc_conflict_span_mean,
        "bc_conflict_span_max": bc_conflict_span_max,
        "bc_integrated_count": bc_integrated_count,
        "bc_integration_rate": bc_integration_rate,
        "bc_unresolved_count": bc_unresolved_count,
        "bc_unresolved_rate": bc_unresolved_rate,
    }


def _empty_features() -> dict[str, int | float]:
    """Return a feature dict with all zeros for n <= 1 sentences."""
    return {
        "bc_belief_sentence_count": 0,
        "bc_belief_sentence_rate": 0.0,
        "bc_conflict_pair_count": 0,
        "bc_conflict_rate": 0.0,
        "bc_max_conflict_sim": 0.0,
        "bc_mean_conflict_sim": 0.0,
        "bc_conflict_span_mean": 0.0,
        "bc_conflict_span_max": 0,
        "bc_integrated_count": 0,
        "bc_integration_rate": 0.0,
        "bc_unresolved_count": 0,
        "bc_unresolved_rate": 0.0,
    }


def _features_no_conflicts(
    belief_count: int, belief_rate: float
) -> dict[str, int | float]:
    """Return features when beliefs exist but no conflicts detected."""
    return {
        "bc_belief_sentence_count": belief_count,
        "bc_belief_sentence_rate": belief_rate,
        "bc_conflict_pair_count": 0,
        "bc_conflict_rate": 0.0,
        "bc_max_conflict_sim": 0.0,
        "bc_mean_conflict_sim": 0.0,
        "bc_conflict_span_mean": 0.0,
        "bc_conflict_span_max": 0,
        "bc_integrated_count": 0,
        "bc_integration_rate": 0.0,
        "bc_unresolved_count": 0,
        "bc_unresolved_rate": 0.0,
    }


def bc_features_from_result(result: "EmbeddingResult") -> dict[str, int | float]:
    """Compute BC features from an EmbeddingResult.

    Convenience wrapper that extracts embeddings and text from result.sentences.

    Args:
        result: An EmbeddingResult with ordered SentenceEmbedding objects.

    Returns:
        BC feature dict (same as bc_features).
    """
    embeddings = [sent.embedding for sent in result.sentences]
    sentences = [sent.text for sent in result.sentences]
    return bc_features(embeddings, sentences)
