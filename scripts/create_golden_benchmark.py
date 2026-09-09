#!/usr/bin/env python3
"""Create golden benchmark set of anchor journals for MES evaluation.

Selects 5-8 journal entries that represent clear entropy levels (0th, 25th,
50th, 75th, 100th percentiles) based on high inter-rater agreement.

These serve as an evaluation benchmark — not for training.
The model should correctly order these anchors after any training change.

Usage:
    cd mental-entropy
    python scripts/create_golden_benchmark.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONSENSUS_CSV = PROJECT_ROOT / "data" / "multirater" / "consensus_labels.csv"
RAW_RATINGS_CSV = PROJECT_ROOT / "data" / "multirater" / "raw_ratings.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "golden_benchmark.json"

# Target percentiles and score ranges
TARGETS = [
    {"name": "very_low", "percentile": 0, "score_range": (1.0, 2.0), "description": "Calm, organized, single-topic, reflective"},
    {"name": "low", "percentile": 25, "score_range": (2.5, 3.5), "description": "Mostly coherent, minor tangents"},
    {"name": "medium", "percentile": 50, "score_range": (4.0, 5.0), "description": "Mixed coherence and fragmentation"},
    {"name": "high", "percentile": 75, "score_range": (6.0, 7.5), "description": "Noticeable topic jumping, hedging"},
    {"name": "very_high", "percentile": 100, "score_range": (8.0, 10.0), "description": "Severely fragmented, contradictory, scattered"},
]


def load_consensus_entries() -> list[dict]:
    """Load consensus labels with raw per-rater scores."""
    entries = []
    with open(CONSENSUS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            entries.append({
                "idx": i,
                "journal": row["journal"],
                "source": row["source"],
                "overall_entropy": float(row["overall_entropy"]),
                "label_std": float(row.get("label_std", "0")),
                "n_raters": int(row.get("n_raters", "3")),
                "text_length": len(row["journal"]),
            })
    return entries


def select_anchors(entries: list[dict]) -> list[dict]:
    """Select best anchor journal for each target percentile."""
    anchors = []

    for target in TARGETS:
        lo, hi = target["score_range"]

        # Filter: score in range, human source preferred, low label_std
        candidates = [
            e for e in entries
            if lo <= e["overall_entropy"] <= hi
            and e["text_length"] >= 200  # Skip very short entries
        ]

        if not candidates:
            print(f"  WARNING: No candidates for {target['name']} ({lo}-{hi})")
            # Widen range
            candidates = [
                e for e in entries
                if lo - 0.5 <= e["overall_entropy"] <= hi + 0.5
                and e["text_length"] >= 100
            ]

        if not candidates:
            print(f"  SKIPPING: {target['name']} — no entries in range")
            continue

        # Sort by: human source first, then lowest label_std, then moderate text length
        candidates.sort(key=lambda e: (
            0 if e["source"] == "human" else 1,  # Prefer human
            e["label_std"],  # Low disagreement
            abs(e["text_length"] - 800),  # Prefer ~800 chars (representative length)
        ))

        best = candidates[0]
        anchors.append({
            "target": target["name"],
            "percentile": target["percentile"],
            "description": target["description"],
            "idx": best["idx"],
            "overall_entropy": best["overall_entropy"],
            "label_std": best["label_std"],
            "source": best["source"],
            "text_length": best["text_length"],
            "journal_preview": best["journal"][:200] + "..." if len(best["journal"]) > 200 else best["journal"],
            "journal_full": best["journal"],
            "n_candidates": len(candidates),
        })

    return anchors


def main() -> None:
    print("=" * 72)
    print("GOLDEN BENCHMARK CREATION")
    print("=" * 72)

    entries = load_consensus_entries()
    n_human = sum(1 for e in entries if e["source"] == "human")
    print(f"Loaded {len(entries)} entries ({n_human} human)")

    # Stats
    scores = [e["overall_entropy"] for e in entries]
    stds = [e["label_std"] for e in entries]
    print(f"Score range: {min(scores):.1f} - {max(scores):.1f}")
    print(f"Label std: mean={sum(stds)/len(stds):.2f}, min={min(stds):.2f}, max={max(stds):.2f}")

    print(f"\n--- Selecting anchors ---")
    anchors = select_anchors(entries)

    print(f"\n--- Golden Benchmark ({len(anchors)} anchors) ---")
    for a in anchors:
        print(f"\n  [{a['target']}] (percentile {a['percentile']})")
        print(f"    Score: {a['overall_entropy']:.2f}, Std: {a['label_std']:.2f}, Source: {a['source']}")
        print(f"    Length: {a['text_length']} chars, Candidates: {a['n_candidates']}")
        print(f"    Preview: {a['journal_preview'][:120]}...")

    # Verify ordering
    scores_ordered = [a["overall_entropy"] for a in anchors]
    is_ordered = all(scores_ordered[i] <= scores_ordered[i+1] for i in range(len(scores_ordered)-1))
    print(f"\n  Correctly ordered: {is_ordered}")
    if not is_ordered:
        print("  WARNING: Anchors not in ascending order — review manually!")

    # Save
    benchmark = {
        "version": "1.0",
        "n_anchors": len(anchors),
        "description": "Golden benchmark set for MES evaluation. Anchors represent clear entropy levels with high inter-rater agreement.",
        "usage": "After any model change, verify model.predict(anchor[i]) < model.predict(anchor[j]) for all i < j.",
        "anchors": [{k: v for k, v in a.items() if k != "journal_full"} for a in anchors],
        "anchor_texts": {a["target"]: a["journal_full"] for a in anchors},
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
