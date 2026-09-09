#!/usr/bin/env python3
"""Extract journal_text from scored synthetic JSONL into CSV for label_journals.py.

Usage:
    python scripts/extract_synthetic_for_labeling.py
    python scripts/extract_synthetic_for_labeling.py --input data/synthetic_journals_800_scored.jsonl --output data/synthetic_for_labeling.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract synthetic journal texts to CSV")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/synthetic_journals_800_scored.jsonl"),
        help="Input scored JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_for_labeling.csv"),
        help="Output CSV with journal column",
    )
    args = parser.parse_args()

    entries = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            text = entry.get("journal_text", "").strip()
            if text:
                entries.append(text)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["journal"])
        writer.writeheader()
        for text in entries:
            writer.writerow({"journal": text})

    print(f"Extracted {len(entries)} journal texts to {args.output}")


if __name__ == "__main__":
    main()
