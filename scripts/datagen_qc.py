"""Quality control checks for synthetic journal JSONL.

Reports phrase/opening repetition and can optionally write a filtered JSONL.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


OVERUSED_PHRASES: tuple[str, ...] = (
    "i keep",
    "the hum of",
    "just closed my laptop",
    "i'm sitting",
    "sitting in",
)


def _tokenize_opening(text: str, n: int = 5) -> str:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    return " ".join(tokens[:n])


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {idx}") from exc
        if not isinstance(payload, dict):
            continue
        rows.append(payload)
    return rows


def _phrase_counts(texts: list[str]) -> dict[str, int]:
    return {phrase: sum(1 for t in texts if phrase in t.lower()) for phrase in OVERUSED_PHRASES}


def run_qc(
    rows: list[dict[str, object]],
    max_opening_count: int,
    max_phrase_ratio: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    texts = [str(row.get("journal_text", "")) for row in rows]
    openings = [_tokenize_opening(t) for t in texts]
    opening_counter = Counter(openings)
    phrase_counter = _phrase_counts(texts)
    n_rows = max(1, len(rows))

    flagged_ids: set[int] = set()
    for row, text, opening in zip(rows, texts, openings):
        entry_id = row.get("id")
        if not isinstance(entry_id, int):
            continue
        if opening_counter[opening] > max_opening_count:
            flagged_ids.add(entry_id)
            continue
        lower_text = text.lower()
        for phrase, count in phrase_counter.items():
            if phrase in lower_text and (count / n_rows) > max_phrase_ratio and lower_text.count(phrase) >= 2:
                flagged_ids.add(entry_id)
                break

    kept = [row for row in rows if isinstance(row.get("id"), int) and row["id"] not in flagged_ids]
    summary = {
        "total_rows": len(rows),
        "kept_rows": len(kept),
        "flagged_rows": len(rows) - len(kept),
        "top_openings": opening_counter.most_common(10),
        "phrase_counts": phrase_counter,
    }
    return kept, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="QC report/filter for synthetic journal JSONL.")
    parser.add_argument("--input", required=True, help="Input JSONL file path.")
    parser.add_argument("--output", help="Optional output JSONL path for filtered rows.")
    parser.add_argument("--max-opening-count", type=int, default=4)
    parser.add_argument("--max-phrase-ratio", type=float, default=0.75)
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = _load_rows(input_path)
    kept, summary = run_qc(
        rows=rows,
        max_opening_count=max(1, args.max_opening_count),
        max_phrase_ratio=max(0.0, min(1.0, args.max_phrase_ratio)),
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
