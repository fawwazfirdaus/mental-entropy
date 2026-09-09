"""Centralized threshold constants for Mental Entropy Score (MES) feature extraction.

All thresholds are calibrated for mxbai embeddings (mixedbread-ai/mxbai-embed-large-v1)
which produce a wider similarity distribution than models like E5.

These thresholds are used across feature extraction modules:
- CE (Coherence Entropy): BREAK_T, SHARP_DROP_T
- SE (Semantic Entropy): CLUSTER_T
- CLE (Cognitive Load Entropy): LOW_T, HIGH_T, REP_T
"""

# ---------------------------------------------------------------------------
# CE (Coherence Entropy) Thresholds
# ---------------------------------------------------------------------------

BREAK_T: float = 0.45
"""Absolute coherence break threshold. Adjacent similarity below this is a 'break'.

Calibrated for mxbai embeddings (mixedbread-ai/mxbai-embed-large-v1) which produce
a wider similarity distribution than E5. Threshold chosen empirically to separate
coherent and fragmented journal text.
"""

SHARP_DROP_T: float = -0.15
"""Relative sharp drop threshold. A drop in adjacent similarity below this is 'sharp'.

Calibrated for mxbai's similarity distribution. A drop of -0.15 represents
a meaningful decrease in coherence for typical journal text.
"""

# ---------------------------------------------------------------------------
# SE (Semantic Entropy) Thresholds
# ---------------------------------------------------------------------------

CLUSTER_T: float = 0.52
"""Cosine distance threshold for agglomerative clustering.

Merging stops when the minimum average-linkage distance exceeds this value.
Calibrated for mxbai embeddings with mean pooling. A threshold of 0.52 achieves
100% accuracy on coherent vs fragmented classification, providing optimal
separation between distinct topics while keeping related sentences clustered.
"""

# ---------------------------------------------------------------------------
# CLE (Cognitive Load Entropy) Thresholds
# ---------------------------------------------------------------------------

LOW_T: float = 0.50
"""Low similarity threshold for adjacent pairs.

Adjacent similarity below this indicates a cognitive disruption or jitter.
Calibrated for mxbai embeddings which produce similarities typically in 0.40-0.80 range
with mean ~0.55. This threshold flags the bottom ~25% of transitions as "low".
"""

HIGH_T: float = 0.65
"""High similarity threshold for resume detection.

Used to detect when similarity rises again after a drop (interruption-resume pattern).
Set at ~p70 of typical adjacent similarities to require genuine recovery.
"""

REP_T: float = 0.70
"""Repetition threshold for non-adjacent pairs.

Similarity above this between (i, i+2) or (i, i+3) indicates looping/rumination.
Raised from 0.65 to 0.70 to reduce false positives on coherent journals where
thematically related sentences naturally have high non-adjacent similarity.
"""

# ---------------------------------------------------------------------------
# BC (Belief Conflict) Thresholds
# ---------------------------------------------------------------------------

CONFLICT_SIM_T: float = 0.55
"""Topic similarity threshold for belief conflict detection.

Two belief sentences are considered potentially in conflict only if their
embedding similarity exceeds this threshold. This ensures conflicts are
about the SAME topic (not unrelated beliefs).

Set at ~p50 of adjacent similarities in coherent text to capture
topically related but not necessarily adjacent statements.
Calibrated for mxbai embeddings.
"""
