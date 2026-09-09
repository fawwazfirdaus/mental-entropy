"""Coherence Entropy (CE) feature extraction for MES v1.

This module computes deterministic numeric features from ordered sentence embeddings
to measure semantic continuity / coherence. It does NOT compute a CE score—only features.

All features are computed from embeddings only (not raw text).
Since embeddings are L2-normalized unit vectors, cosine similarity = dot product.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mental_entropy.utils.thresholds import BREAK_T, SHARP_DROP_T

if TYPE_CHECKING:
    from mental_entropy.embedding.types import EmbeddingResult

# ---------------------------------------------------------------------------
# Expected feature keys (for validation / documentation)
# ---------------------------------------------------------------------------
CE_FEATURE_KEYS: frozenset[str] = frozenset(
    [
        # Basic counts
        "ce_n_sentences",
        "ce_n_adj",
        # Adjacent similarity statistics
        "ce_adj_mean",
        "ce_adj_std",
        "ce_adj_min",
        "ce_adj_max",
        "ce_adj_median",
        "ce_adj_p10",
        "ce_adj_p25",
        "ce_adj_p75",
        "ce_adj_p90",
        # Absolute coherence breaks
        "ce_break_count",
        "ce_break_rate",
        "ce_longest_coherent_run",
        # Relative drops
        "ce_drop_min",
        "ce_drop_mean",
        "ce_drop_std",
        "ce_sharp_drop_count",
        "ce_sharp_drop_rate",
        # Low-coherence mass
        "ce_low_mass",
        "ce_low_mass_sum",
        # Skip-connection coherence
        "ce_skip_mean",
        "ce_skip_min",
        "ce_skip_break_count",
        # Block-aware coherence (distinguishes intentional topic switches from fragmentation)
        "ce_n_blocks",
        "ce_intra_block_break_count",
        "ce_intra_block_break_rate",
        "ce_inter_block_break_count",
        "ce_inter_block_break_rate",
    ]
)


def _percentile(arr: np.ndarray, q: float) -> float:
    """Compute percentile with consistent method across numpy versions."""
    # numpy >= 1.22 uses 'method', older versions use 'interpolation'
    try:
        return float(np.percentile(arr, q, method="linear"))
    except TypeError:
        return float(np.percentile(arr, q, interpolation="linear"))  # type: ignore[call-overload]


def _longest_run(mask: np.ndarray) -> int:
    """
    Compute the longest consecutive run of True values in a boolean array.

    Simple deterministic loop—no fancy libraries needed.
    """
    if len(mask) == 0:
        return 0
    max_run = 0
    current_run = 0
    for val in mask:
        if val:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return max_run


def ce_features(
    sentence_embeddings: list[list[float]],
    block_ids: list[int] | None = None,
) -> dict[str, int | float]:
    """
    Compute CE (coherence entropy) features from ordered sentence embeddings.

    Args:
        sentence_embeddings: List of embedding vectors (each a list of floats).
            Expected to be L2-normalized (unit length). Order is preserved.
        block_ids: Optional list of block IDs for each sentence (0-indexed).
            If provided, computes block-aware features that distinguish
            intra-block breaks (fragmentation) from inter-block breaks (expected).

    Returns:
        A dict with exactly the keys in CE_FEATURE_KEYS.
        All values are Python int or float (JSON-serializable).

    Raises:
        ValueError: If any embedding contains NaN or Inf values, or if
            block_ids length doesn't match embeddings length.
    """
    n = len(sentence_embeddings)

    # Handle empty input
    if n == 0:
        return _empty_features(n_sentences=0)

    # Validate block_ids if provided
    if block_ids is not None and len(block_ids) != n:
        raise ValueError(
            f"block_ids length ({len(block_ids)}) doesn't match "
            f"embeddings length ({n})"
        )

    # Convert to numpy array
    X = np.asarray(sentence_embeddings, dtype=np.float32)

    # Validate: check for NaN/Inf
    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain NaN or Inf values")

    # Handle single sentence
    if n == 1:
        return _empty_features(n_sentences=1)

    # ---------------------------------------------------------------------------
    # Compute adjacent similarities: s[i] = dot(e[i], e[i+1])
    # Since embeddings are unit vectors, dot product = cosine similarity
    # ---------------------------------------------------------------------------
    # Vectorized: element-wise multiply adjacent rows, sum across embedding dim
    s = (X[:-1] * X[1:]).sum(axis=1)  # shape: (n-1,)
    n_adj = len(s)

    # ---------------------------------------------------------------------------
    # Adjacent similarity statistics
    # ---------------------------------------------------------------------------
    adj_mean = float(np.mean(s))
    adj_std = float(np.std(s))
    adj_min = float(np.min(s))
    adj_max = float(np.max(s))
    adj_median = float(np.median(s))
    adj_p10 = _percentile(s, 10)
    adj_p25 = _percentile(s, 25)
    adj_p75 = _percentile(s, 75)
    adj_p90 = _percentile(s, 90)

    # ---------------------------------------------------------------------------
    # Absolute coherence breaks
    # ---------------------------------------------------------------------------
    breaks = s < BREAK_T
    break_count = int(np.sum(breaks))
    break_rate = break_count / n_adj

    # Longest coherent run
    coherent = s >= BREAK_T
    longest_coherent_run = _longest_run(coherent)

    # ---------------------------------------------------------------------------
    # Relative drops between adjacent similarities
    # ---------------------------------------------------------------------------
    if len(s) >= 2:
        drop = np.diff(s)  # drop[j] = s[j+1] - s[j], length = n_adj - 1
        drop_min = float(np.min(drop))
        drop_mean = float(np.mean(drop))
        drop_std = float(np.std(drop))

        sharp = drop < SHARP_DROP_T
        sharp_drop_count = int(np.sum(sharp))
        sharp_drop_rate = sharp_drop_count / len(drop)
    else:
        # Only one adjacency, no drops
        drop_min = 0.0
        drop_mean = 0.0
        drop_std = 0.0
        sharp_drop_count = 0
        sharp_drop_rate = 0.0

    # ---------------------------------------------------------------------------
    # Low-coherence mass (area under threshold)
    # ---------------------------------------------------------------------------
    low = np.maximum(0, BREAK_T - s)
    low_mass = float(np.mean(low))
    low_mass_sum = float(np.sum(low))

    # ---------------------------------------------------------------------------
    # Skip-connection coherence (two-step continuity)
    # ---------------------------------------------------------------------------
    if n >= 3:
        # s2[k] = dot(e[k], e[k+2]) for k in [0..n-3]
        s2 = (X[:-2] * X[2:]).sum(axis=1)  # shape: (n-2,)
        skip_mean = float(np.mean(s2))
        skip_min = float(np.min(s2))
        skip_break_count = int(np.sum(s2 < BREAK_T))
    else:
        skip_mean = 0.0
        skip_min = 0.0
        skip_break_count = 0

    # ---------------------------------------------------------------------------
    # Block-aware coherence features
    # ---------------------------------------------------------------------------
    if block_ids is not None:
        n_blocks = len(set(block_ids))

        # Identify which adjacent pairs are intra-block vs inter-block
        # For adjacent pair (i, i+1), check if block_ids[i] == block_ids[i+1]
        intra_block_mask = np.array(
            [block_ids[i] == block_ids[i + 1] for i in range(n - 1)]
        )
        inter_block_mask = ~intra_block_mask

        # Count breaks in each category
        intra_block_breaks = breaks & intra_block_mask
        inter_block_breaks = breaks & inter_block_mask

        intra_block_break_count = int(np.sum(intra_block_breaks))
        inter_block_break_count = int(np.sum(inter_block_breaks))

        # Compute rates (avoid division by zero)
        n_intra = int(np.sum(intra_block_mask))
        n_inter = int(np.sum(inter_block_mask))

        intra_block_break_rate = (
            float(intra_block_break_count / n_intra) if n_intra > 0 else 0.0
        )
        inter_block_break_rate = (
            float(inter_block_break_count / n_inter) if n_inter > 0 else 0.0
        )
    else:
        # No block information - use defaults
        n_blocks = 1
        intra_block_break_count = break_count
        intra_block_break_rate = break_rate
        inter_block_break_count = 0
        inter_block_break_rate = 0.0

    # ---------------------------------------------------------------------------
    # Assemble feature dict
    # ---------------------------------------------------------------------------
    features: dict[str, int | float] = {
        # Basic counts
        "ce_n_sentences": n,
        "ce_n_adj": n_adj,
        # Adjacent similarity statistics
        "ce_adj_mean": adj_mean,
        "ce_adj_std": adj_std,
        "ce_adj_min": adj_min,
        "ce_adj_max": adj_max,
        "ce_adj_median": adj_median,
        "ce_adj_p10": adj_p10,
        "ce_adj_p25": adj_p25,
        "ce_adj_p75": adj_p75,
        "ce_adj_p90": adj_p90,
        # Absolute coherence breaks
        "ce_break_count": break_count,
        "ce_break_rate": break_rate,
        "ce_longest_coherent_run": longest_coherent_run,
        # Relative drops
        "ce_drop_min": drop_min,
        "ce_drop_mean": drop_mean,
        "ce_drop_std": drop_std,
        "ce_sharp_drop_count": sharp_drop_count,
        "ce_sharp_drop_rate": sharp_drop_rate,
        # Low-coherence mass
        "ce_low_mass": low_mass,
        "ce_low_mass_sum": low_mass_sum,
        # Skip-connection coherence
        "ce_skip_mean": skip_mean,
        "ce_skip_min": skip_min,
        "ce_skip_break_count": skip_break_count,
        # Block-aware coherence
        "ce_n_blocks": n_blocks,
        "ce_intra_block_break_count": intra_block_break_count,
        "ce_intra_block_break_rate": intra_block_break_rate,
        "ce_inter_block_break_count": inter_block_break_count,
        "ce_inter_block_break_rate": inter_block_break_rate,
    }

    return features


def _empty_features(n_sentences: int) -> dict[str, int | float]:
    """Return a feature dict with safe defaults for n < 2 sentences."""
    return {
        # Basic counts
        "ce_n_sentences": n_sentences,
        "ce_n_adj": max(n_sentences - 1, 0),
        # Adjacent similarity statistics (all 0.0 when no adjacencies)
        "ce_adj_mean": 0.0,
        "ce_adj_std": 0.0,
        "ce_adj_min": 0.0,
        "ce_adj_max": 0.0,
        "ce_adj_median": 0.0,
        "ce_adj_p10": 0.0,
        "ce_adj_p25": 0.0,
        "ce_adj_p75": 0.0,
        "ce_adj_p90": 0.0,
        # Absolute coherence breaks
        "ce_break_count": 0,
        "ce_break_rate": 0.0,
        "ce_longest_coherent_run": 0,
        # Relative drops
        "ce_drop_min": 0.0,
        "ce_drop_mean": 0.0,
        "ce_drop_std": 0.0,
        "ce_sharp_drop_count": 0,
        "ce_sharp_drop_rate": 0.0,
        # Low-coherence mass
        "ce_low_mass": 0.0,
        "ce_low_mass_sum": 0.0,
        # Skip-connection coherence
        "ce_skip_mean": 0.0,
        "ce_skip_min": 0.0,
        "ce_skip_break_count": 0,
        # Block-aware coherence
        "ce_n_blocks": 1 if n_sentences > 0 else 0,
        "ce_intra_block_break_count": 0,
        "ce_intra_block_break_rate": 0.0,
        "ce_inter_block_break_count": 0,
        "ce_inter_block_break_rate": 0.0,
    }


def ce_features_from_result(result: "EmbeddingResult") -> dict[str, int | float]:
    """
    Compute CE features from an EmbeddingResult.

    Convenience wrapper that extracts embeddings and block_ids from result.sentences.
    Block-aware features distinguish intra-block breaks (fragmentation) from
    inter-block breaks (expected topic transitions).

    Args:
        result: An EmbeddingResult with ordered SentenceEmbedding objects.

    Returns:
        CE feature dict (same as ce_features).
    """
    embeddings = [sent.embedding for sent in result.sentences]
    block_ids = [sent.block_id for sent in result.sentences]
    return ce_features(embeddings, block_ids=block_ids)
