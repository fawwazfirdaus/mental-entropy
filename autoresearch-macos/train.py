"""
MES Weight Optimization - Scoring Configuration.

This is the file the agent modifies. Edit the WEIGHTS list below to
experiment with different feature weights and transforms.

Usage: uv run train.py
"""

from prepare import load_cached_data, evaluate_weights

# ---------------------------------------------------------------------------
# WEIGHT CONFIGURATION (edit this section)
# ---------------------------------------------------------------------------
#
# Each entry: (feature_name, weight, transform, [param])
#
# Transforms:
#   "inv"      → score += w * (1 - x)        Higher x = lower entropy
#   "dir"      → score += w * x              Higher x = higher entropy
#   "dir_norm" → score += w * min(x/P, 1)    Normalize then direct
#   "inv_clip" → score += w * (1 - min(x,P)) Clip then invert
#   "inv_nz"   → score += w * (1-x) if x>0   Invert only when non-zero
#
# Constraints:
#   - Weights should sum to ~1.0
#   - Final score = sum(w * transform(x)) * 100, clamped to [0, 100]
#
# Correlation reference (from 245 LLM-labeled entries):
#   Negative r → higher feature = lower entropy → use "inv"
#   Positive r → higher feature = higher entropy → use "dir"
#
# Available features (77 total, grouped by correlation strength):
#
# |r| > 0.30 (strongest):
#   se_dominant_cluster_frac  r=-0.336  inv   Topic concentration
#   ne_start_end_sim          r=-0.335  inv   Narrative return
#   ce_adj_min                r=-0.324  inv   Worst adjacent similarity
#   ne_start_to_centroid      r=-0.315  inv   Opening alignment
#   ne_end_to_centroid        r=-0.312  inv   Closing alignment
#   se_intra_mean             r=-0.311  inv   Within-topic coherence
#   ce_adj_p25                r=-0.306  inv   Bottom quartile coherence
#   se_inter_mean             r=-0.304  inv   Topic separation
#   ce_adj_p10                r=-0.302  inv   Bottom 10% coherence
#
# |r| 0.20-0.30 (moderate):
#   ce_adj_mean               r=-0.291  inv   Average transition quality
#   ce_adj_median             r=-0.271  inv   Median transition quality
#   ce_skip_min               r=-0.252  inv   Skip-connection minimum
#   ne_arc_linearity          r=-0.236  inv   Narrative linearity
#   ce_adj_p75                r=-0.225  inv   Upper quartile coherence
#   cle_fragment_rate         r=+0.221  dir   Surface fragmentation
#   ne_fragment_sentence_rate r=+0.207  dir   Fragment detection
#
# |r| 0.10-0.20 (supporting):
#   ne_temporal_markers_rate  r=-0.173  inv   Temporal structure
#   se_cluster_entropy        r=+0.165  dir   Topic dispersion
#   ne_consolidation_delta    r=-0.155  inv   Narrative convergence
#   ce_intra_block_break_rate r=+0.156  dir   Within-paragraph breaks
#   ne_arc_linearity          r=-0.149  inv   Arc structure
#   cle_length_cv             r=+0.118  dir   Sentence length variability
#
# |r| < 0.10 (weak):
#   bc_unresolved_rate        r=+0.041  dir   Unresolved conflicts
#   bc_belief_sentence_rate   r=-0.045  inv   Belief examination
#   ... and 50+ more features with weak individual correlation

WEIGHTS = [
    # exp_2209: grid-optimized 13 features on 2209 entries (cv_corr=0.627)
    # Dataset: 964 LLM-labeled real journals + 1245 synthetic
    # (includes 250 high-entropy synthetic journals labeled by Claude)
    ("ce_adj_p75",                0.17, "inv"),     # Upper quartile coherence
    ("ne_fragment_sentence_rate", 0.14, "dir"),     # Fragment detection
    ("ce_inter_block_break_rate", 0.11, "dir"),     # Inter-block breaks
    ("bc_belief_sentence_count",  0.11, "dir_norm", 18.0),  # Belief density
    ("ce_n_blocks",               0.10, "dir_norm", 20.0),  # Block count
    ("cle_length_cv",             0.08, "dir"),     # Sentence length variability
    ("ce_skip_mean",              0.07, "inv"),     # Skip-connection quality
    ("ne_start_to_centroid",      0.06, "inv"),     # Opening alignment
    ("se_dominant_cluster_frac",  0.05, "inv"),     # Topic concentration
    ("ne_arc_linearity",          0.03, "inv"),     # Narrative linearity
    ("ne_end_to_centroid",        0.03, "inv"),     # Closing alignment
    ("cle_hedge_rate",            0.03, "dir"),     # Hedging language
    ("ne_start_end_sim",          0.02, "inv"),     # Narrative return
]

# ---------------------------------------------------------------------------
# EVALUATION (do not edit below this line)
# ---------------------------------------------------------------------------

features, labels, feature_names, sources = load_cached_data()
results = evaluate_weights(WEIGHTS, features, labels, sources=sources)

# Print in autoresearch format
print("---")
print(f"val_corr:         {results['val_corr']:.6f}")
print(f"cv_corr:          {results['cv_corr']:.6f}")
print(f"cv_std:           {results['cv_std']:.6f}")
print(f"mse:              {results['mse']:.4f}")
print(f"score_mean:       {results['score_mean']:.1f}")
print(f"score_std:        {results['score_std']:.1f}")
print(f"n_features:       {results['n_features']}")
print(f"weight_sum:       {results['weight_sum']:.3f}")
if "human_corr" in results:
    print(f"human_corr:       {results['human_corr']:.6f}  (n={results['human_n']})")
if "synth_corr" in results:
    print(f"synth_corr:       {results['synth_corr']:.6f}  (n={results['synth_n']})")
