#!/usr/bin/env python3
"""Refresh cached labels/stds from consensus CSV without recomputing features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "autoresearch-macos" / "features_cache.json"
DEFAULT_CONSENSUS_PATH = PROJECT_ROOT / "data" / "multirater" / "consensus_labels.csv"
DEFAULT_LOCKED_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "human_locked_eval_v1.json"

DIMENSION_NAMES = [
    "prediction_coherence",
    "model_complexity",
    "compression_progress",
    "belief_integration",
    "precision_weighting",
]


def load_locked_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(entry["source_index"]) for entry in payload.get("entries", [])}


def load_filtered_consensus(path: Path, locked_indices: set[int]) -> list[tuple[int, dict[str, str]]]:
    rows: list[tuple[int, dict[str, str]]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for source_index, row in enumerate(reader):
            if source_index not in locked_indices:
                rows.append((source_index, row))
    return rows


def refresh_cache_payload(
    cache: dict[str, Any],
    consensus_rows: list[tuple[int, dict[str, str]]],
) -> dict[str, Any]:
    n_cache = len(cache.get("labels", []))
    if n_cache != len(consensus_rows):
        raise ValueError(
            f"Cache row count {n_cache} does not match filtered consensus "
            f"row count {len(consensus_rows)}"
        )

    labels = [float(row["overall_entropy"]) for _, row in consensus_rows]
    old_labels = [float(value) for value in cache.get("labels", [])]
    mismatches = [
        i for i, (old, new) in enumerate(zip(old_labels, labels, strict=True))
        if abs(old - new) > 1e-9
    ]
    if mismatches:
        first = mismatches[0]
        raise ValueError(
            f"Cache labels do not align with consensus row order at cache row {first}: "
            f"cache={old_labels[first]} consensus={labels[first]}"
        )

    dimension_labels: dict[str, list[float]] = {}
    for name in DIMENSION_NAMES:
        dimension_labels[name] = [
            float(row.get(name, "5") or 5.0)
            for _, row in consensus_rows
        ]

    updated = dict(cache)
    updated["labels"] = labels
    updated["label_stds"] = [
        float(row.get("label_std", 0.0) or 0.0)
        for _, row in consensus_rows
    ]
    updated["dimension_labels"] = dimension_labels
    updated["source_indices"] = [source_index for source_index, _ in consensus_rows]
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh features_cache label metadata from consensus_labels.csv."
    )
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--consensus-path", type=Path, default=DEFAULT_CONSENSUS_PATH)
    parser.add_argument("--locked-eval-path", type=Path, default=DEFAULT_LOCKED_EVAL_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache = json.loads(args.cache_path.read_text(encoding="utf-8"))
    locked_indices = load_locked_indices(args.locked_eval_path)
    consensus_rows = load_filtered_consensus(args.consensus_path, locked_indices)
    updated = refresh_cache_payload(cache, consensus_rows)
    args.cache_path.write_text(
        json.dumps(updated, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )

    label_stds = updated["label_stds"]
    print("Refreshed feature cache label metadata")
    print(f"  rows: {len(label_stds)}")
    print(f"  locked rows excluded: {len(locked_indices)}")
    print(f"  label_std mean: {sum(label_stds) / len(label_stds):.3f}")
    print(f"  label_std max: {max(label_stds):.3f}")
    print(f"  cache: {args.cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
