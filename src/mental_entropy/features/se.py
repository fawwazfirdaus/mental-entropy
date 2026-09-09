"""Semantic Entropy (SE) feature extraction for MES v1.

This module computes deterministic numeric features from ordered sentence embeddings
to measure semantic dispersion / topic fragmentation. It does NOT compute an SE score—only features.

All features are computed from embeddings only (not raw text).
Since embeddings are L2-normalized unit vectors, cosine similarity = dot product.

Clustering uses agglomerative (average linkage) with a cosine-distance threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mental_entropy.utils.thresholds import CLUSTER_T

if TYPE_CHECKING:
    from mental_entropy.embedding.types import EmbeddingResult

# ---------------------------------------------------------------------------
# Expected feature keys (for validation / documentation)
# ---------------------------------------------------------------------------
SE_FEATURE_KEYS: frozenset[str] = frozenset(
    [
        "se_n_clusters",
        "se_cluster_entropy",
        "se_dominant_cluster_frac",
        "se_switch_count",
        "se_switch_rate",
        "se_switch_weighted",
        "se_switch_mean_jump",
        "se_intra_mean",
        "se_inter_mean",
    ]
)


# ---------------------------------------------------------------------------
# Internal: Agglomerative clustering with threshold (NumPy-only)
# ---------------------------------------------------------------------------
def _agglomerative_threshold_labels(X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Perform agglomerative clustering with average linkage and a distance threshold.

    Args:
        X: (n, d) array of L2-normalized embeddings.
        threshold: Stop merging when minimum inter-cluster distance exceeds this.

    Returns:
        (n,) array of integer cluster labels (0-indexed, sorted by first occurrence).
    """
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32)
    if n == 1:
        return np.array([0], dtype=np.int32)

    # Cosine distance matrix: D[i,j] = 1 - dot(X[i], X[j])
    # For L2-normalized vectors, dot = cosine similarity
    sim = X @ X.T  # (n, n) similarity matrix
    dist = 1.0 - sim  # cosine distance

    # Initialize: each point is its own cluster
    # clusters[i] = set of original indices in cluster i
    clusters: dict[int, set[int]] = {i: {i} for i in range(n)}
    # Track which cluster each point belongs to
    point_to_cluster = list(range(n))

    # Average linkage distance between clusters i and j:
    # mean of dist[a, b] for a in clusters[i], b in clusters[j]
    # We'll compute this on-the-fly during merges

    while len(clusters) > 1:
        # Find the pair of clusters with minimum average linkage distance
        min_dist = float("inf")
        merge_pair = (-1, -1)

        cluster_ids = sorted(clusters.keys())
        for idx_i, ci in enumerate(cluster_ids):
            for cj in cluster_ids[idx_i + 1 :]:
                # Compute average distance between clusters ci and cj
                members_i = clusters[ci]
                members_j = clusters[cj]
                total_dist = 0.0
                for a in members_i:
                    for b in members_j:
                        total_dist += dist[a, b]
                avg_dist = total_dist / (len(members_i) * len(members_j))

                if avg_dist < min_dist:
                    min_dist = avg_dist
                    merge_pair = (ci, cj)

        # Stop if minimum distance exceeds threshold
        if min_dist > threshold:
            break

        # Merge clusters
        ci, cj = merge_pair
        clusters[ci] = clusters[ci] | clusters[cj]
        for pt in clusters[cj]:
            point_to_cluster[pt] = ci
        del clusters[cj]

    # Assign final labels sorted by minimum original index in each cluster
    final_clusters = sorted(clusters.keys(), key=lambda c: min(clusters[c]))
    label_map = {c: i for i, c in enumerate(final_clusters)}

    labels = np.array([label_map[point_to_cluster[i]] for i in range(n)], dtype=np.int32)
    return labels


# ---------------------------------------------------------------------------
# Internal: Compute intra-cluster mean similarity
# ---------------------------------------------------------------------------
def _intra_cluster_mean(X: np.ndarray, labels: np.ndarray, n: int, K: int) -> float:
    """
    Compute weighted mean intra-cluster cosine similarity.

    For each cluster k with m_k >= 2 members:
      - Compute mean pairwise cosine similarity within the cluster
      - Weight by p_k = m_k / n
      - Sum across all clusters

    Clusters with m_k == 1 are skipped (no pairs).

    Uses efficient formula: for unit vectors, sum of all pairs dot products is
    (||sum(E)||^2 - m) / 2, and mean = (||sum(E)||^2 - m) / (m * (m-1)).
    """
    if K == 0 or n == 0:
        return 0.0

    weighted_sum = 0.0
    for k in range(K):
        mask = labels == k
        m_k = int(np.sum(mask))
        if m_k < 2:
            continue

        # Members of cluster k
        E_k = X[mask]  # (m_k, d)

        # Sum of all member embeddings
        s = E_k.sum(axis=0)  # (d,)

        # ||s||^2 = s · s
        s_norm_sq = float(np.dot(s, s))

        # Since embeddings are L2-normalized, ||e_i||^2 = 1 for all i
        # So sum of diag = m_k
        diag_sum = m_k

        # Mean pairwise similarity = (||s||^2 - diag_sum) / (m_k * (m_k - 1))
        mean_sim = (s_norm_sq - diag_sum) / (m_k * (m_k - 1))

        # Weight by p_k
        p_k = m_k / n
        weighted_sum += p_k * mean_sim

    return weighted_sum


# ---------------------------------------------------------------------------
# Internal: Compute inter-cluster mean similarity
# ---------------------------------------------------------------------------
def _inter_cluster_mean(X: np.ndarray, labels: np.ndarray, K: int) -> float:
    """
    Compute mean cosine similarity between cluster centroids.

    Centroids are computed as L2-normalized mean of member embeddings.
    Returns mean similarity across all pairs of centroids.

    If K == 1, returns 1.0 (single cluster = maximum coherence).
    """
    if K <= 1:
        return 1.0

    # Compute L2-normalized centroids
    centroids = []
    for k in range(K):
        mask = labels == k
        E_k = X[mask]
        centroid = E_k.mean(axis=0)
        # L2 normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)

    centroids = np.array(centroids)  # (K, d)

    # Compute pairwise similarities between centroids
    sim_matrix = centroids @ centroids.T  # (K, K)

    # Mean of upper triangle (excluding diagonal)
    total_sim = 0.0
    count = 0
    for i in range(K):
        for j in range(i + 1, K):
            total_sim += sim_matrix[i, j]
            count += 1

    return float(total_sim / count) if count > 0 else 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def se_features(sentence_embeddings: list[list[float]]) -> dict[str, int | float]:
    """
    Compute SE (semantic entropy) features from ordered sentence embeddings.

    Args:
        sentence_embeddings: List of embedding vectors (each a list of floats).
            Expected to be L2-normalized (unit length). Order is preserved.

    Returns:
        A dict with exactly the keys in SE_FEATURE_KEYS.
        All values are Python int or float (JSON-serializable).

    Raises:
        ValueError: If any embedding contains NaN or Inf values.
    """
    n = len(sentence_embeddings)

    # Handle edge cases: n == 0 or n == 1
    if n <= 1:
        return _empty_features()

    # Convert to numpy array
    X = np.asarray(sentence_embeddings, dtype=np.float32)

    # Validate: check for NaN/Inf
    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain NaN or Inf values")

    # ---------------------------------------------------------------------------
    # Cluster sentences using agglomerative clustering with threshold
    # ---------------------------------------------------------------------------
    labels = _agglomerative_threshold_labels(X, CLUSTER_T)
    K = int(labels.max() + 1) if len(labels) > 0 else 0

    # ---------------------------------------------------------------------------
    # Cluster statistics
    # ---------------------------------------------------------------------------
    # Count sentences per cluster
    cluster_counts = np.bincount(labels, minlength=K)
    p_k = cluster_counts / n  # proportions

    # ---------------------------------------------------------------------------
    # se_n_clusters
    # ---------------------------------------------------------------------------
    se_n_clusters = K

    # ---------------------------------------------------------------------------
    # se_cluster_entropy: normalized entropy H / log(K)
    # ---------------------------------------------------------------------------
    if K == 1:
        se_cluster_entropy = 0.0
    else:
        # H = -sum(p_k * log(p_k)) for p_k > 0
        # Avoid log(0) by filtering
        p_nonzero = p_k[p_k > 0]
        H = -float(np.sum(p_nonzero * np.log(p_nonzero)))
        se_cluster_entropy = float(H / np.log(K))

    # ---------------------------------------------------------------------------
    # se_dominant_cluster_frac: max(p_k)
    # ---------------------------------------------------------------------------
    se_dominant_cluster_frac = float(np.max(p_k))

    # ---------------------------------------------------------------------------
    # se_switch_count: number of cluster label changes between adjacent sentences
    # ---------------------------------------------------------------------------
    switches = labels[:-1] != labels[1:]
    se_switch_count = int(np.sum(switches))

    # ---------------------------------------------------------------------------
    # se_switch_rate: switch_count / (n - 1)
    # ---------------------------------------------------------------------------
    se_switch_rate = se_switch_count / (n - 1)

    # ---------------------------------------------------------------------------
    # Adjacent jumps: cosine distance between consecutive sentences
    # jump[i] = 1 - cosine_sim(e[i], e[i+1])
    # ---------------------------------------------------------------------------
    adj_sims = (X[:-1] * X[1:]).sum(axis=1)  # cosine similarities
    adj_jumps = 1.0 - adj_sims  # cosine distances (0 = identical, larger = bigger jump)

    # ---------------------------------------------------------------------------
    # se_switch_weighted: sum of jump distances at switch points / (n-1)
    # Measures total semantic distance covered by topic switches
    # ---------------------------------------------------------------------------
    switched_jumps = adj_jumps[switches]  # jumps only where cluster changed
    se_switch_weighted = float(np.sum(switched_jumps)) / (n - 1)

    # ---------------------------------------------------------------------------
    # se_switch_mean_jump: mean jump distance when switches occur
    # Distinguishes small phase shifts from big topic jumps
    # ---------------------------------------------------------------------------
    if se_switch_count > 0:
        se_switch_mean_jump = float(np.mean(switched_jumps))
    else:
        se_switch_mean_jump = 0.0

    # ---------------------------------------------------------------------------
    # se_intra_mean: weighted mean intra-cluster similarity
    # ---------------------------------------------------------------------------
    se_intra_mean = _intra_cluster_mean(X, labels, n, K)

    # ---------------------------------------------------------------------------
    # se_inter_mean: mean inter-cluster centroid similarity
    # ---------------------------------------------------------------------------
    se_inter_mean = _inter_cluster_mean(X, labels, K)

    # ---------------------------------------------------------------------------
    # Assemble feature dict
    # ---------------------------------------------------------------------------
    features: dict[str, int | float] = {
        "se_n_clusters": se_n_clusters,
        "se_cluster_entropy": se_cluster_entropy,
        "se_dominant_cluster_frac": se_dominant_cluster_frac,
        "se_switch_count": se_switch_count,
        "se_switch_rate": se_switch_rate,
        "se_switch_weighted": se_switch_weighted,
        "se_switch_mean_jump": se_switch_mean_jump,
        "se_intra_mean": se_intra_mean,
        "se_inter_mean": se_inter_mean,
    }

    return features


def _empty_features() -> dict[str, int | float]:
    """Return a feature dict with safe defaults for n <= 1 sentences."""
    return {
        "se_n_clusters": 0,
        "se_cluster_entropy": 0.0,
        "se_dominant_cluster_frac": 0.0,
        "se_switch_count": 0,
        "se_switch_rate": 0.0,
        "se_switch_weighted": 0.0,
        "se_switch_mean_jump": 0.0,
        "se_intra_mean": 0.0,
        "se_inter_mean": 0.0,
    }


def se_features_from_result(result: "EmbeddingResult") -> dict[str, int | float]:
    """
    Compute SE features from an EmbeddingResult.

    Convenience wrapper that extracts embeddings from result.sentences in order.

    Args:
        result: An EmbeddingResult with ordered SentenceEmbedding objects.

    Returns:
        SE feature dict (same as se_features).
    """
    embeddings = [sent.embedding for sent in result.sentences]
    return se_features(embeddings)
