"""Type definitions for the embedding layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SentenceEmbedding:
    """A single sentence embedding with metadata."""

    id: int
    """Sentence index (0-based, preserves original order)."""

    text: str
    """Original sentence text."""

    embedding: list[float]
    """1024-dimensional L2-normalized embedding."""

    block_id: int = 0
    """Paragraph/block index (0-based). Increments at blank line boundaries."""


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Result of embedding a journal entry."""

    sentences: list[SentenceEmbedding]
    """Ordered list of sentence embeddings."""

    doc_embedding: list[float] | None
    """1024-dimensional L2-normalized document embedding, or None if no sentences."""

