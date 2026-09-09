"""Extract journal_text entries from synthetic journal JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def iter_journal_records(path: str, *, skip_invalid: bool) -> Iterator[dict[str, object]]:
    """Yield JSON objects from a JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                if skip_invalid:
                    continue
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            if not isinstance(data, dict):
                if skip_invalid:
                    continue
                raise ValueError(f"Expected JSON object on line {line_number}")
            yield data


def iter_journal_texts(path: str, *, skip_invalid: bool, strip: bool) -> Iterator[str]:
    """Yield journal_text values from a JSONL file."""
    for record in iter_journal_records(path, skip_invalid=skip_invalid):
        journal_text = record.get("journal_text")
        if not isinstance(journal_text, str):
            if skip_invalid:
                continue
            raise ValueError("Missing or invalid journal_text field.")
        yield journal_text.strip() if strip else journal_text


def write_text_output(texts: Iterable[str], output_path: str | None) -> None:
    """Write journal texts to a plain text file or stdout."""
    handle = sys.stdout if output_path is None else open(output_path, "w", encoding="utf-8")
    try:
        first = True
        for text in texts:
            if not first:
                handle.write("\n\n")
            handle.write(text)
            first = False
    finally:
        if handle is not sys.stdout:
            handle.close()


def write_jsonl_output(texts: Iterable[str], output_path: str | None) -> None:
    """Write journal texts to JSONL (one object per line)."""
    handle = sys.stdout if output_path is None else open(output_path, "w", encoding="utf-8")
    try:
        for text in texts:
            handle.write(json.dumps({"journal_text": text}, ensure_ascii=False) + "\n")
    finally:
        if handle is not sys.stdout:
            handle.close()


def write_embeddings(
    records: Iterable[dict[str, object]],
    output_path: str,
    *,
    skip_invalid: bool,
) -> None:
    """Embed journal_text entries and write embeddings to JSONL."""
    from mental_entropy.embedding.embed import embed_journal_entry

    with open(output_path, "w", encoding="utf-8") as handle:
        for idx, record in enumerate(records, start=1):
            journal_text = record.get("journal_text")
            if not isinstance(journal_text, str):
                if skip_invalid:
                    continue
                raise ValueError("Missing or invalid journal_text field for embedding.")
            entry_id = record.get("id")
            if not isinstance(entry_id, int):
                entry_id = idx

            result = embed_journal_entry(journal_text)
            entry = {
                "id": entry_id,
                "sentence_embeddings": [
                    {
                        "id": sentence.id,
                        "text": sentence.text,
                        "embedding": sentence.embedding,
                    }
                    for sentence in result.sentences
                ],
                "doc_embedding": result.doc_embedding,
            }
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract journal_text from JSONL.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", help="Output path (defaults to stdout).")
    parser.add_argument("--format", choices=["txt", "jsonl"], default="txt")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid lines.")
    parser.add_argument("--no-strip", action="store_true", help="Preserve leading/trailing whitespace.")
    parser.add_argument(
        "--embed-output",
        help="Optional JSONL path to write embeddings (requires model dependencies).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    strip = not args.no_strip

    if args.embed_output:
        records_iter = iter_journal_records(args.input, skip_invalid=args.skip_invalid)
        write_embeddings(records_iter, args.embed_output, skip_invalid=args.skip_invalid)
        if args.output is None:
            return

    texts_iter = iter_journal_texts(args.input, skip_invalid=args.skip_invalid, strip=strip)

    if args.format == "jsonl":
        write_jsonl_output(texts_iter, args.output)
    else:
        write_text_output(texts_iter, args.output)


if __name__ == "__main__":
    main()
