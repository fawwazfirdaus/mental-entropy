#!/usr/bin/env python3
"""Refresh locked eval labels from current consensus rows.

Preserves locked row membership and control types, but updates label values,
expected MES, reasoning, and disagreement metadata from consensus_labels.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSENSUS_PATH = PROJECT_ROOT / "data" / "multirater" / "consensus_labels.csv"
DEFAULT_LOCKED_PATH = PROJECT_ROOT / "data" / "eval" / "human_locked_eval_v1.json"


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def load_consensus_by_index(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return {
            index: row
            for index, row in enumerate(csv.DictReader(f))
        }


def refresh_payload(
    payload: dict[str, Any],
    consensus_by_index: dict[int, dict[str, str]],
) -> dict[str, Any]:
    entries = []
    for entry in payload["entries"]:
        source_index = int(entry["source_index"])
        consensus = consensus_by_index[source_index]
        label_oe = float(consensus["overall_entropy"])
        refreshed = {
            **entry,
            "text": consensus["journal"],
            "label_overall_entropy": label_oe,
            "expected_mes": _round((label_oe - 1.0) / 9.0 * 100.0),
            "label_std": float(consensus.get("label_std", 0.0)),
            "reasoning": consensus.get("reasoning", entry.get("reasoning", "")),
        }
        if "overall_entropy_raw" in consensus:
            refreshed["overall_entropy_raw"] = consensus["overall_entropy_raw"]
        if "overall_entropy_spread" in consensus:
            refreshed["overall_entropy_spread"] = float(consensus["overall_entropy_spread"])
        entries.append(refreshed)

    return {
        **payload,
        "label_source": (
            "three-pass consensus labels with raw rater disagreement from "
            "data/multirater/consensus_labels.csv"
        ),
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh locked eval labels.")
    parser.add_argument("--consensus", type=Path, default=DEFAULT_CONSENSUS_PATH)
    parser.add_argument("--locked", type=Path, default=DEFAULT_LOCKED_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.locked.read_text(encoding="utf-8"))
    consensus_by_index = load_consensus_by_index(args.consensus)
    refreshed = refresh_payload(payload, consensus_by_index)
    args.locked.write_text(
        json.dumps(refreshed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Refreshed locked eval labels")
    print(f"  entries: {len(refreshed['entries'])}")
    print(f"  path: {args.locked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
