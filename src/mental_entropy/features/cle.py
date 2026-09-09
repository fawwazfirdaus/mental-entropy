"""Cognitive Load Entropy (CLE) feature extraction for MES v1.

This module computes deterministic numeric features from ordered sentence embeddings
and raw sentence text to measure micro-level cognitive overload, thought crowding,
and low compressibility in writing.

CLE reflects instability in the thinking process itself, not topic structure or
narrative arc. It does NOT measure emotion, sentiment, topic changes, or narrative
coherence.

Features come from two sources:
- Surface-level linguistic signals (primary): fragments, restarts, hedging, etc.
- Local embedding instability signals (secondary): jitter, zigzag, repetition
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from mental_entropy.utils.linguistic import (
    HEDGE_CUES,
    RESTART_CUES,
    ELLIPSIS_CLUSTER,
    ENDS_WITH_DASH,
    ENDS_WITH_ELLIPSIS,
    ENDS_WITH_SENTENCE_PUNCT,
    MID_SENTENCE_DASH,
    MULTIPLE_EXCLAMATIONS,
    MULTIPLE_QUESTIONS,
    STARTS_WITH_CONJUNCTION,
    contains_any,
    has_verb,
    tokenize_simple,
)

from mental_entropy.utils.thresholds import HIGH_T, LOW_T, REP_T

if TYPE_CHECKING:
    from mental_entropy.embedding.types import EmbeddingResult

# ---------------------------------------------------------------------------
# Expected feature keys (for validation / documentation)
# ---------------------------------------------------------------------------
CLE_FEATURE_KEYS: frozenset[str] = frozenset(
    [
        # Surface-level features
        "cle_fragment_rate",
        "cle_restart_rate",
        "cle_hedge_rate",
        "cle_length_cv",
        "cle_punctuation_noise",
        # Embedding-based features (local only)
        "cle_adj_sim_std",
        "cle_adj_sim_range",
        "cle_adj_sim_cv",
        "cle_adj_low_frac",
        "cle_zigzag_rate",
        "cle_repetition_score",
        "cle_resume_rate",
        "cle_semantic_break_rate",
        "cle_semantic_isolated_rate",
    ]
)

# ---------------------------------------------------------------------------
# Internal: Text cue detection helpers
# ---------------------------------------------------------------------------


def _is_fragment(text: str) -> bool:
    """
    Check if sentence is a fragment indicating cognitive load.

    A sentence is considered a fragment if:
    - Ends with ellipsis (...) or em dash (— or --)
    - Ends abruptly without a main verb
    - Starts with conjunctions (but, and, so) and appears unfinished
    - Is extremely short (< 4 tokens) and not a complete clause
    """
    # Check for trailing ellipsis
    if ENDS_WITH_ELLIPSIS.search(text):
        return True

    # Check for trailing dash
    if ENDS_WITH_DASH.search(text):
        return True

    tokens = tokenize_simple(text)
    token_count = len(tokens)

    # Extremely short and not a complete clause
    if token_count < 4:
        # Allow very short sentences with verbs as complete clauses
        if not has_verb(text):
            return True
        # Short sentences starting with conjunctions that lack proper ending
        if STARTS_WITH_CONJUNCTION.search(text):
            if not ENDS_WITH_SENTENCE_PUNCT.search(text):
                return True

    # Starts with conjunction and appears unfinished
    if STARTS_WITH_CONJUNCTION.search(text):
        if not ENDS_WITH_SENTENCE_PUNCT.search(text):
            return True

    # No verb and more than trivial length
    if token_count >= 4 and not has_verb(text):
        return True

    return False


def _has_restart_cue(text: str) -> bool:
    """Check if sentence contains a restart/reset cue."""
    return contains_any(text, RESTART_CUES)


def _has_hedge_cue(text: str) -> bool:
    """Check if sentence contains a hedging cue."""
    return contains_any(text, HEDGE_CUES)


def _has_punctuation_noise(text: str) -> bool:
    """
    Check if sentence has excessive punctuation patterns.

    Detects:
    - Multiple exclamation points (!!!)
    - Multiple question marks (???)
    - Repeated ellipses (2+ occurrences of ...)
    - Abrupt dashes mid-sentence
    """
    # Multiple exclamations
    if MULTIPLE_EXCLAMATIONS.search(text):
        return True

    # Multiple questions
    if MULTIPLE_QUESTIONS.search(text):
        return True

    # Repeated ellipsis clusters (2+)
    ellipsis_matches = ELLIPSIS_CLUSTER.findall(text)
    if len(ellipsis_matches) >= 2:
        return True

    # Mid-sentence dash (not just trailing)
    # Check if dash appears and there's content after it
    dash_match = MID_SENTENCE_DASH.search(text)
    if dash_match:
        # Check if there's substantial content after the dash
        after_dash = text[dash_match.end() :].strip()
        if after_dash and len(after_dash) > 1:
            return True

    return False


def _has_semantic_break(
    sentence_idx: int,
    adj_sims: np.ndarray,
    threshold: float = LOW_T,
) -> bool:
    """
    Detect if sentence has a semantic break (cognitive jump).

    A sentence has a semantic break if:
    - The transition BEFORE it is low (< threshold), OR
    - The transition AFTER it is low (< threshold)

    This captures cognitive jumps even in syntactically complete sentences.
    Unlike syntactic fragments, this detects semantic disruption patterns.

    Args:
        sentence_idx: Index of the sentence (0-based).
        adj_sims: Array of adjacent similarities, shape (n-1,).
            adj_sims[i] = similarity between sentence i and i+1.
        threshold: Similarity threshold (default LOW_T = 0.50).

    Returns:
        True if sentence has at least one low-similarity transition.
    """
    # Check transition before this sentence
    if sentence_idx > 0 and adj_sims[sentence_idx - 1] < threshold:
        return True
    # Check transition after this sentence
    if sentence_idx < len(adj_sims) and adj_sims[sentence_idx] < threshold:
        return True
    return False


def _is_semantically_isolated(
    sentence_idx: int,
    adj_sims: np.ndarray,
    threshold: float = LOW_T,
) -> bool:
    """
    Detect if sentence is isolated from both neighbors.

    A sentence is isolated if:
    - The transition BEFORE it is low (< threshold), AND
    - The transition AFTER it is low (< threshold)

    This is a stronger signal of cognitive disruption than a single break.
    Only applies to middle sentences (not first or last).

    Args:
        sentence_idx: Index of the sentence (0-based).
        adj_sims: Array of adjacent similarities, shape (n-1,).
            adj_sims[i] = similarity between sentence i and i+1.
        threshold: Similarity threshold (default LOW_T = 0.50).

    Returns:
        True if sentence is isolated from both neighbors, False otherwise.
    """
    # First or last sentence cannot be isolated from both neighbors
    if sentence_idx == 0 or sentence_idx >= len(adj_sims):
        return False

    prev_low = adj_sims[sentence_idx - 1] < threshold
    next_low = adj_sims[sentence_idx] < threshold
    return prev_low and next_low


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def cle_features(
    sentence_embeddings: list[list[float]],
    sentences: list[str],
) -> dict[str, float]:
    """
    Compute CLE (cognitive load entropy) features from ordered sentence embeddings and text.

    Args:
        sentence_embeddings: List of embedding vectors (each a list of floats).
            Expected to be L2-normalized (unit length). Order is preserved.
        sentences: List of raw sentence strings in the same order.

    Returns:
        A dict with exactly the keys in CLE_FEATURE_KEYS.
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
    # Surface-level CLE features (primary)
    # ---------------------------------------------------------------------------

    # cle_fragment_rate
    fragment_count = sum(1 for s in sentences if _is_fragment(s))
    cle_fragment_rate = float(fragment_count / n)

    # cle_restart_rate
    restart_count = sum(1 for s in sentences if _has_restart_cue(s))
    cle_restart_rate = float(restart_count / n)

    # cle_hedge_rate
    hedge_count = sum(1 for s in sentences if _has_hedge_cue(s))
    cle_hedge_rate = float(hedge_count / n)

    # cle_length_cv (coefficient of variation of sentence lengths)
    lengths = np.array([len(tokenize_simple(s)) for s in sentences], dtype=np.float32)
    mean_len = float(np.mean(lengths))
    if mean_len > 0:
        cle_length_cv = float(np.std(lengths) / mean_len)
    else:
        cle_length_cv = 0.0

    # cle_punctuation_noise
    punct_noise_count = sum(1 for s in sentences if _has_punctuation_noise(s))
    cle_punctuation_noise = float(punct_noise_count / n)

    # ---------------------------------------------------------------------------
    # Embedding-based CLE features (local only)
    # ---------------------------------------------------------------------------

    # Compute adjacent similarities: sim[i] = dot(e[i], e[i+1])
    sim = (X[:-1] * X[1:]).sum(axis=1)  # shape: (n-1,)
    n_adj = len(sim)

    # cle_adj_sim_std: standard deviation of adjacent similarities
    cle_adj_sim_std = float(np.std(sim))

    # cle_adj_sim_range: max - min
    cle_adj_sim_range = float(np.max(sim) - np.min(sim))

    # cle_adj_sim_cv: coefficient of variation (std/mean) of adjacent similarities
    # Better captures cognitive instability relative to baseline coherence
    adj_mean = float(np.mean(sim))
    cle_adj_sim_cv = float(cle_adj_sim_std / adj_mean) if adj_mean > 0 else 0.0

    # cle_adj_low_frac: fraction of adjacent sims below LOW_T
    low_count = int(np.sum(sim < LOW_T))
    cle_adj_low_frac = float(low_count / n_adj)

    # cle_zigzag_rate: count zigzag patterns
    zigzag_count = 0
    if len(sim) >= 3:
        for i in range(1, len(sim) - 1):
            # Local maximum or local minimum
            is_peak = sim[i - 1] < sim[i] > sim[i + 1]
            is_valley = sim[i - 1] > sim[i] < sim[i + 1]
            if is_peak or is_valley:
                zigzag_count += 1
    cle_zigzag_rate = float(zigzag_count / max(1, n - 2))

    # cle_repetition_score: non-adjacent repetition (i, i+2) and (i, i+3)
    repetition_count = 0
    for i in range(n):
        # Check (i, i+2)
        if i + 2 < n:
            sim_i2 = float(np.dot(X[i], X[i + 2]))
            if sim_i2 > REP_T:
                repetition_count += 1
        # Check (i, i+3)
        if i + 3 < n:
            sim_i3 = float(np.dot(X[i], X[i + 3]))
            if sim_i3 > REP_T:
                repetition_count += 1
    cle_repetition_score = float(repetition_count / n)

    # cle_resume_rate: interruption-resume patterns
    # sim[i] drops sharply (below LOW_T) then sim[i+1] rises (above HIGH_T)
    resume_count = 0
    if len(sim) >= 2:
        for i in range(len(sim) - 1):
            if sim[i] < LOW_T and sim[i + 1] > HIGH_T:
                resume_count += 1
    cle_resume_rate = float(resume_count / max(1, n - 2))

    # cle_semantic_break_rate: fraction of sentences with semantic breaks
    # Detects cognitive jumps even in syntactically complete sentences
    semantic_break_count = sum(
        1 for i in range(n) if _has_semantic_break(i, sim, LOW_T)
    )
    cle_semantic_break_rate = float(semantic_break_count / n)

    # cle_semantic_isolated_rate: fraction of sentences isolated from both neighbors
    # Stronger signal of cognitive disruption (complete disconnection)
    semantic_isolated_count = sum(
        1 for i in range(n) if _is_semantically_isolated(i, sim, LOW_T)
    )
    cle_semantic_isolated_rate = float(semantic_isolated_count / n)

    # ---------------------------------------------------------------------------
    # Assemble feature dict
    # ---------------------------------------------------------------------------
    features: dict[str, float] = {
        # Surface-level features
        "cle_fragment_rate": cle_fragment_rate,
        "cle_restart_rate": cle_restart_rate,
        "cle_hedge_rate": cle_hedge_rate,
        "cle_length_cv": cle_length_cv,
        "cle_punctuation_noise": cle_punctuation_noise,
        # Embedding-based features
        "cle_adj_sim_std": cle_adj_sim_std,
        "cle_adj_sim_range": cle_adj_sim_range,
        "cle_adj_sim_cv": cle_adj_sim_cv,
        "cle_adj_low_frac": cle_adj_low_frac,
        "cle_zigzag_rate": cle_zigzag_rate,
        "cle_repetition_score": cle_repetition_score,
        "cle_resume_rate": cle_resume_rate,
        "cle_semantic_break_rate": cle_semantic_break_rate,
        "cle_semantic_isolated_rate": cle_semantic_isolated_rate,
    }

    return features


def _empty_features() -> dict[str, float]:
    """Return a feature dict with all zeros for n <= 1 sentences."""
    return {
        "cle_fragment_rate": 0.0,
        "cle_restart_rate": 0.0,
        "cle_hedge_rate": 0.0,
        "cle_length_cv": 0.0,
        "cle_punctuation_noise": 0.0,
        "cle_adj_sim_std": 0.0,
        "cle_adj_sim_range": 0.0,
        "cle_adj_sim_cv": 0.0,
        "cle_adj_low_frac": 0.0,
        "cle_zigzag_rate": 0.0,
        "cle_repetition_score": 0.0,
        "cle_resume_rate": 0.0,
        "cle_semantic_break_rate": 0.0,
        "cle_semantic_isolated_rate": 0.0,
    }


def cle_features_from_result(result: "EmbeddingResult") -> dict[str, float]:
    """
    Compute CLE features from an EmbeddingResult.

    Convenience wrapper that extracts embeddings and text from result.sentences.

    Args:
        result: An EmbeddingResult with ordered SentenceEmbedding objects.

    Returns:
        CLE feature dict (same as cle_features).
    """
    embeddings = [sent.embedding for sent in result.sentences]
    sentences = [sent.text for sent in result.sentences]
    return cle_features(embeddings, sentences)
