"""
Per-Dimension Rubric Analysis for MES Features.

Correlates all 77 features against each of the 5 rubric dimensions
(continuity, topic_focus, contradiction_integration, cognitive_clarity,
narrative_closure) separately for human-written vs synthetic journals.

Key questions answered:
  1. Which dimensions have the narrowest label range for human journals?
  2. Which features best predict each dimension for human journals?
  3. Does the module-dimension alignment match our architecture?
  4. Where are the coverage gaps?

Usage:
    cd autoresearch-macos && uv run python diagnose_dimensions.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

from prepare import (
    DIMENSION_NAMES,
    SOURCE_HUMAN,
    SOURCE_SYNTHETIC,
    load_cached_data,
    load_dimension_labels,
    pearson_correlation,
)


def _module_for_feature(feat: str) -> str:
    """Extract module prefix from feature name."""
    prefix = feat.split("_")[0]
    return prefix if prefix in ("ce", "se", "ne", "cle", "bc") else "?"


def _text_histogram(values: np.ndarray, bins: list[float], width: int = 40) -> list[str]:
    """Return text-based histogram lines."""
    counts, edges = np.histogram(values, bins=bins)
    max_count = max(counts) if len(counts) > 0 else 1
    lines = []
    for i, c in enumerate(counts):
        bar_len = int(c / max_count * width) if max_count > 0 else 0
        label = f"  {edges[i]:.0f}-{edges[i+1]:.0f}"
        lines.append(f"{label:>8s} | {'█' * bar_len} {c}")
    return lines


def main() -> None:
    # Load data
    features, labels, feature_names, sources = load_cached_data()
    dim_labels = load_dimension_labels()

    n = len(features)
    human_mask = sources == SOURCE_HUMAN
    synth_mask = sources == SOURCE_SYNTHETIC
    n_human = int(human_mask.sum())
    n_synth = int(synth_mask.sum())

    print("=" * 72)
    print("PER-DIMENSION RUBRIC ANALYSIS")
    print(f"Total entries: {n} (human={n_human}, synthetic={n_synth})")
    print(f"Features: {len(feature_names)}")
    print(f"Dimensions: {', '.join(DIMENSION_NAMES)}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Section 1: Dimension label distributions
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 1: DIMENSION LABEL DISTRIBUTIONS")
    print("=" * 72)

    bins_dim = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    bins_overall = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]

    for dim_name in DIMENSION_NAMES + ["overall_entropy"]:
        if dim_name == "overall_entropy":
            vals = labels
            bins = bins_overall
        else:
            vals = dim_labels[dim_name]
            bins = bins_dim

        h_vals = vals[human_mask]
        s_vals = vals[synth_mask]

        print(f"\n--- {dim_name} ---")
        print(f"  Human  (n={n_human}): mean={h_vals.mean():.2f}  std={h_vals.std():.2f}  "
              f"min={h_vals.min():.0f}  max={h_vals.max():.0f}")
        for line in _text_histogram(h_vals, bins):
            print(f"    {line}")

        print(f"  Synth  (n={n_synth}): mean={s_vals.mean():.2f}  std={s_vals.std():.2f}  "
              f"min={s_vals.min():.0f}  max={s_vals.max():.0f}")
        for line in _text_histogram(s_vals, bins):
            print(f"    {line}")

    # ------------------------------------------------------------------
    # Section 2: Feature × Dimension correlation matrix
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 2: FEATURE × DIMENSION CORRELATIONS (HUMAN-WRITTEN)")
    print("=" * 72)

    # Build feature matrix
    feat_matrix = np.zeros((n, len(feature_names)), dtype=np.float64)
    for i, fd in enumerate(features):
        for j, fname in enumerate(feature_names):
            feat_matrix[i, j] = fd.get(fname, 0.0)

    # Compute correlations: features × (5 dimensions + overall_entropy)
    all_dims = DIMENSION_NAMES + ["overall_entropy"]
    corr_human = np.zeros((len(feature_names), len(all_dims)), dtype=np.float64)
    corr_synth = np.zeros((len(feature_names), len(all_dims)), dtype=np.float64)

    for d_idx, dim_name in enumerate(all_dims):
        if dim_name == "overall_entropy":
            target = labels
        else:
            target = dim_labels[dim_name]

        for f_idx, fname in enumerate(feature_names):
            f_vals = feat_matrix[:, f_idx]
            corr_human[f_idx, d_idx] = pearson_correlation(
                f_vals[human_mask], target[human_mask]
            )
            corr_synth[f_idx, d_idx] = pearson_correlation(
                f_vals[synth_mask], target[synth_mask]
            )

    # Print top features per dimension (human only)
    for d_idx, dim_name in enumerate(all_dims):
        print(f"\n--- {dim_name} (human, top 10 by |r|) ---")
        human_corrs = corr_human[:, d_idx]
        top_idx = np.argsort(np.abs(human_corrs))[::-1][:10]
        print(f"  {'Feature':<40s} {'r_human':>8s} {'r_synth':>8s} {'module':>6s}")
        for idx in top_idx:
            fname = feature_names[idx]
            r_h = human_corrs[idx]
            r_s = corr_synth[idx, d_idx]
            mod = _module_for_feature(fname)
            print(f"  {fname:<40s} {r_h:>+8.3f} {r_s:>+8.3f} {mod:>6s}")

    # ------------------------------------------------------------------
    # Section 3: Module × Dimension alignment
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 3: MODULE × DIMENSION ALIGNMENT (HUMAN-WRITTEN)")
    print("=" * 72)
    print("\nMean |r| per module per dimension (human journals only):")

    modules = ["ce", "se", "ne", "cle", "bc"]
    # Header
    header = f"  {'Module':<8s}"
    for dim in all_dims:
        short = dim[:12]
        header += f" {short:>14s}"
    print(header)
    print("  " + "-" * (8 + 15 * len(all_dims)))

    module_dim_scores = {}
    for mod in modules:
        mod_feat_idx = [j for j, fn in enumerate(feature_names) if fn.startswith(mod + "_")]
        row = f"  {mod:<8s}"
        mod_scores = {}
        for d_idx, dim in enumerate(all_dims):
            if mod_feat_idx:
                mean_abs_r = float(np.mean(np.abs(corr_human[mod_feat_idx, d_idx])))
            else:
                mean_abs_r = 0.0
            mod_scores[dim] = mean_abs_r
            row += f" {mean_abs_r:>14.3f}"
        module_dim_scores[mod] = mod_scores
        print(row)

    # Best module per dimension
    print("\n  Best module per dimension:")
    for d_idx, dim in enumerate(all_dims):
        scores = {m: module_dim_scores[m][dim] for m in modules}
        best = max(scores, key=scores.get)
        print(f"    {dim:<35s} → {best} (mean |r| = {scores[best]:.3f})")

    # ------------------------------------------------------------------
    # Section 4: Human vs Synthetic divergence by dimension
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 4: HUMAN vs SYNTHETIC DIVERGENCE BY DIMENSION")
    print("=" * 72)
    print("\nMean |r_human - r_synth| per dimension (feature correlation gap):")

    for d_idx, dim in enumerate(all_dims):
        gaps = np.abs(corr_human[:, d_idx] - corr_synth[:, d_idx])
        mean_gap = float(np.mean(gaps))
        max_gap = float(np.max(gaps))
        max_feat = feature_names[int(np.argmax(gaps))]
        print(f"  {dim:<35s} mean_gap={mean_gap:.3f}  max_gap={max_gap:.3f} ({max_feat})")

    # Show which dimension is MOST consistent between human and synthetic
    dim_gaps = []
    for d_idx, dim in enumerate(all_dims):
        mean_gap = float(np.mean(np.abs(corr_human[:, d_idx] - corr_synth[:, d_idx])))
        dim_gaps.append((dim, mean_gap))
    dim_gaps.sort(key=lambda x: x[1])
    print("\n  Dimensions ranked by consistency (lowest gap = most transferable):")
    for dim, gap in dim_gaps:
        print(f"    {dim:<35s} {gap:.3f}")

    # ------------------------------------------------------------------
    # Section 5: Coverage gaps
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 5: COVERAGE GAPS (FEATURES WITH |r_human| > 0.15 PER DIMENSION)")
    print("=" * 72)

    for d_idx, dim in enumerate(all_dims):
        human_corrs = corr_human[:, d_idx]
        strong = [(feature_names[j], human_corrs[j])
                  for j in range(len(feature_names))
                  if abs(human_corrs[j]) > 0.15]
        strong.sort(key=lambda x: abs(x[1]), reverse=True)

        if strong:
            print(f"\n  {dim}: {len(strong)} features with |r| > 0.15")
            for fname, r in strong[:8]:
                direction = "↑ entropy" if r > 0 else "↓ entropy"
                print(f"    {fname:<40s} r={r:+.3f}  ({direction})")
        else:
            print(f"\n  {dim}: *** NO features with |r| > 0.15 — MAJOR GAP ***")

    # ------------------------------------------------------------------
    # Section 6: Dimension inter-correlation (human)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 6: DIMENSION INTER-CORRELATION (HUMAN-WRITTEN)")
    print("=" * 72)
    print("\nHow do the 5 sub-dimensions correlate with each other and overall_entropy?")
    print("(Lower continuity/focus/clarity = higher entropy, so expect negative r)")

    header = f"  {'':>14s}"
    for dim in all_dims:
        header += f" {dim[:12]:>12s}"
    print(header)

    for d1_idx, d1 in enumerate(all_dims):
        if d1 == "overall_entropy":
            vals1 = labels[human_mask]
        else:
            vals1 = dim_labels[d1][human_mask]

        row = f"  {d1[:14]:>14s}"
        for d2_idx, d2 in enumerate(all_dims):
            if d2 == "overall_entropy":
                vals2 = labels[human_mask]
            else:
                vals2 = dim_labels[d2][human_mask]
            r = pearson_correlation(vals1, vals2)
            row += f" {r:>+12.3f}"
        print(row)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    # Which dimension has the weakest feature coverage for human journals?
    dim_coverage = {}
    for d_idx, dim in enumerate(all_dims):
        n_strong = sum(1 for j in range(len(feature_names))
                       if abs(corr_human[j, d_idx]) > 0.15)
        max_r = float(np.max(np.abs(corr_human[:, d_idx])))
        dim_coverage[dim] = (n_strong, max_r)

    print("\nDimension coverage summary (human journals):")
    print(f"  {'Dimension':<35s} {'n_feat |r|>0.15':>15s} {'max |r|':>8s}")
    for dim in all_dims:
        n_strong, max_r = dim_coverage[dim]
        flag = " ← GAP" if n_strong < 3 else ""
        print(f"  {dim:<35s} {n_strong:>15d} {max_r:>8.3f}{flag}")

    # Architecture alignment check
    print("\nArchitecture alignment check:")
    expected = {
        "ce": "continuity",
        "se": "topic_focus",
        "ne": "narrative_closure",
        "cle": "cognitive_clarity",
        "bc": "contradiction_integration",
    }
    for mod, expected_dim in expected.items():
        actual_best = max(DIMENSION_NAMES, key=lambda d: module_dim_scores[mod][d])
        aligned = "✓" if actual_best == expected_dim else "✗"
        print(f"  {mod:>4s} → expected: {expected_dim:<28s} actual best: {actual_best:<28s} {aligned}")


if __name__ == "__main__":
    main()
