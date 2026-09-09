"""Embedding layer for MES pipeline."""

from mental_entropy.embedding.embed import embed_journal_entry
from mental_entropy.embedding.types import EmbeddingResult, SentenceEmbedding

__all__ = ["embed_journal_entry", "EmbeddingResult", "SentenceEmbedding"]

