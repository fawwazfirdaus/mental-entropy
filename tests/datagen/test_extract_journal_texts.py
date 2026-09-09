"""Tests for JSONL extraction utilities."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mental_entropy.datagen.extract_journal_texts import iter_journal_texts, write_embeddings


def test_iter_journal_texts_basic(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    entries = [
        {"journal_text": "First entry."},
        {"journal_text": "Second entry."},
    ]
    path.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    texts = list(iter_journal_texts(str(path), skip_invalid=False, strip=True))
    assert texts == ["First entry.", "Second entry."]


def test_iter_journal_texts_skip_invalid(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    lines = [
        json.dumps({"journal_text": "Valid entry."}),
        "not-json",
        json.dumps({"other": "missing text"}),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    texts = list(iter_journal_texts(str(path), skip_invalid=True, strip=True))
    assert texts == ["Valid entry."]


def test_iter_journal_texts_invalid_raises(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(ValueError):
        list(iter_journal_texts(str(path), skip_invalid=False, strip=True))


def test_write_embeddings_skip_invalid_records(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "embeddings.jsonl"
    records = [
        {"id": 1, "journal_text": "Valid entry."},
        {"id": 2, "other": "missing"},
    ]

    def _fake_embed_journal_entry(text: str) -> SimpleNamespace:
        sentence = SimpleNamespace(id=1, text=text, embedding=[0.1, 0.2])
        return SimpleNamespace(sentences=[sentence], doc_embedding=[0.3, 0.4])

    monkeypatch.setattr(
        "mental_entropy.embedding.embed.embed_journal_entry",
        _fake_embed_journal_entry,
    )

    write_embeddings(records, str(output_path), skip_invalid=True)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["id"] == 1
