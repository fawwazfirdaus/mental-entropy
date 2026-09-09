#!/usr/bin/env python3
"""
Correlate computed MES features with LLM-labeled entropy scores.

This script:
1. Loads labeled journal entries
2. Computes CE/SE/NE/CLE features for each
3. Calculates correlations with overall_entropy
4. Reports the most predictive features

Usage:
    python scripts/correlate_features.py --input journals_labeled.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mental_entropy import (
    embed_journal_entry,
    ce_features_from_result,
    se_features_from_result,
    ne_features_from_result,
    cle_features_from_result,
    bc_features_from_result,
)


def load_labeled_journals(path: Path) -> list[dict]:
    """Load labeled journal entries from CSV."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                'journal': row['journal'],
                'overall_entropy': int(row['overall_entropy']),
                'continuity': int(row['continuity']),
                'topic_focus': int(row['topic_focus']),
                'contradiction_integration': int(row['contradiction_integration']),
                'cognitive_clarity': int(row['cognitive_clarity']),
                'narrative_closure': int(row['narrative_closure']),
            })
    return entries


def compute_all_features(text: str) -> dict[str, float]:
    """Compute all MES features for a journal entry."""
    # Embed the text
    result = embed_journal_entry(text)

    # Compute all feature sets (sentence text is inside result.sentences[i].text)
    features = {}
    features.update(ce_features_from_result(result))
    features.update(se_features_from_result(result))
    features.update(ne_features_from_result(result))
    features.update(cle_features_from_result(result))
    features.update(bc_features_from_result(result))

    return features


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n == 0:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = (var_x * var_y) ** 0.5

    if denom == 0:
        return 0.0

    return num / denom


def main():
    parser = argparse.ArgumentParser(
        description="Correlate MES features with LLM entropy labels"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("journals_labeled.csv"),
        help="Input CSV with labeled journals"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON file for feature data (optional)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of entries to process"
    )

    args = parser.parse_args()

    # Load labeled entries
    print(f"Loading labeled journals from {args.input}...")
    entries = load_labeled_journals(args.input)
    print(f"Loaded {len(entries)} entries")

    if args.limit:
        entries = entries[:args.limit]
        print(f"Limited to {len(entries)} entries")

    # Compute features for each entry
    print("\nComputing features (this may take a while for first run - model loading)...")
    all_features = []
    entropy_scores = []

    for i, entry in enumerate(entries):
        pct = (i + 1) / len(entries) * 100
        print(f"\r[{i+1}/{len(entries)}] ({pct:.1f}%) Processing...", end="", flush=True)

        try:
            features = compute_all_features(entry['journal'])
            all_features.append(features)
            entropy_scores.append(entry['overall_entropy'])
        except Exception as e:
            print(f"\nWarning: Failed to process entry {i+1}: {e}")
            continue

    print(f"\n\nSuccessfully processed {len(all_features)} entries")

    if not all_features:
        print("No features computed, exiting.")
        return

    # Get all feature names
    feature_names = sorted(all_features[0].keys())

    # Calculate correlations
    print("\nCalculating correlations with overall_entropy...")
    correlations = {}

    for feature_name in feature_names:
        feature_values = [f[feature_name] for f in all_features]
        corr = pearson_correlation(feature_values, entropy_scores)
        correlations[feature_name] = corr

    # Sort by absolute correlation (strongest first)
    sorted_correlations = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Print results
    print("\n" + "=" * 70)
    print("FEATURE CORRELATIONS WITH OVERALL_ENTROPY")
    print("(Positive = higher feature value → higher entropy)")
    print("(Negative = higher feature value → lower entropy)")
    print("=" * 70)

    # Group by module
    ce_corrs = [(k, v) for k, v in sorted_correlations if k.startswith('ce_')]
    se_corrs = [(k, v) for k, v in sorted_correlations if k.startswith('se_')]
    ne_corrs = [(k, v) for k, v in sorted_correlations if k.startswith('ne_')]
    cle_corrs = [(k, v) for k, v in sorted_correlations if k.startswith('cle_')]
    bc_corrs = [(k, v) for k, v in sorted_correlations if k.startswith('bc_')]

    print("\n📊 TOP 20 MOST PREDICTIVE FEATURES (by |correlation|):\n")
    for i, (name, corr) in enumerate(sorted_correlations[:20], 1):
        direction = "↑" if corr > 0 else "↓"
        bar_len = int(abs(corr) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"{i:2}. {name:35} {corr:+.3f} {direction} |{bar}|")

    print("\n" + "-" * 70)
    print("\n📈 BY MODULE (sorted by |correlation|):\n")

    for module_name, module_corrs in [("CE (Coherence)", ce_corrs),
                                        ("SE (Semantic)", se_corrs),
                                        ("NE (Narrative)", ne_corrs),
                                        ("CLE (Cognitive Load)", cle_corrs),
                                        ("BC (Belief Conflict)", bc_corrs)]:
        print(f"\n{module_name}:")
        module_sorted = sorted(module_corrs, key=lambda x: abs(x[1]), reverse=True)
        for name, corr in module_sorted[:10]:
            direction = "↑" if corr > 0 else "↓"
            print(f"  {name:35} {corr:+.3f} {direction}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    strong_positive = [k for k, v in correlations.items() if v > 0.3]
    strong_negative = [k for k, v in correlations.items() if v < -0.3]
    moderate = [k for k, v in correlations.items() if 0.15 < abs(v) <= 0.3]

    print(f"\nStrong positive correlations (>0.3): {len(strong_positive)}")
    for f in strong_positive[:5]:
        print(f"  - {f}: {correlations[f]:+.3f}")

    print(f"\nStrong negative correlations (<-0.3): {len(strong_negative)}")
    for f in strong_negative[:5]:
        print(f"  - {f}: {correlations[f]:+.3f}")

    print(f"\nModerate correlations (0.15-0.3): {len(moderate)}")

    # Also correlate with LLM sub-dimensions
    print("\n" + "=" * 70)
    print("CORRELATION WITH LLM SUB-DIMENSIONS")
    print("=" * 70)

    llm_dims = ['continuity', 'topic_focus', 'contradiction_integration', 'cognitive_clarity', 'narrative_closure']

    for dim in llm_dims:
        dim_scores = [entries[i][dim] for i in range(len(all_features))]

        # Find top 5 correlated features for this dimension
        dim_corrs = {}
        for feature_name in feature_names:
            feature_values = [f[feature_name] for f in all_features]
            corr = pearson_correlation(feature_values, dim_scores)
            dim_corrs[feature_name] = corr

        sorted_dim = sorted(dim_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

        print(f"\n{dim.upper()} (LLM-rated, 1-5):")
        print("  Top correlates:")
        for name, corr in sorted_dim[:5]:
            direction = "↑" if corr > 0 else "↓"
            print(f"    {name:35} {corr:+.3f} {direction}")

    # Save full data if requested
    if args.output:
        output_data = {
            'correlations': correlations,
            'sorted_by_abs': [(k, v) for k, v in sorted_correlations],
            'n_samples': len(all_features),
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nFull data saved to {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
