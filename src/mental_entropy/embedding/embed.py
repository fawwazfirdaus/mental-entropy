"""Main embedding API for the MES pipeline."""

from __future__ import annotations

from mental_entropy.embedding.model import compute_doc_embedding, embed_sentences
from mental_entropy.embedding.text import normalize_text, split_sentences_with_blocks
from mental_entropy.embedding.types import EmbeddingResult, SentenceEmbedding


def embed_journal_entry(
    raw_text: str,
    *,
    batch_size: int = 32,
) -> EmbeddingResult:
    """
    Embed a raw journal entry into sentence-level and document-level embeddings.

    This function performs:
    1. Text normalization (strip, normalize line endings, collapse spaces/tabs)
    2. Sentence splitting (blank-line blocks + spaCy sentencizer + regex fallback)
    3. mxbai embedding with mean pooling and L2 normalization
    4. Document embedding as mean of sentence embeddings (L2 normalized)

    Args:
        raw_text: The raw journal entry text.
        batch_size: Number of sentences to embed at once (default 32).

    Returns:
        EmbeddingResult containing:
        - sentences: Ordered list of SentenceEmbedding objects
        - doc_embedding: 1024-dim L2-normalized vector, or None if no sentences

    Guarantees:
    - Deterministic: same input always produces same output
    - All embeddings are L2-normalized (unit length)
    - Sentence order is preserved
    - Model is frozen (no training, no gradients)
    """
    # Step 1: Normalize text
    normalized = normalize_text(raw_text)

    # Step 2: Split into sentences with block IDs
    sentences_with_blocks = split_sentences_with_blocks(normalized)

    # Handle empty input
    if not sentences_with_blocks:
        return EmbeddingResult(sentences=[], doc_embedding=None)

    # Extract just the sentence texts for embedding
    sentence_texts = [text for text, _ in sentences_with_blocks]

    # Steps 3-5: Embed sentences (tokenization, CLS pooling, L2 norm)
    embeddings_tensor = embed_sentences(sentence_texts, batch_size=batch_size)

    # Step 6: Build sentence embedding objects with block IDs
    sentence_embeddings: list[SentenceEmbedding] = []
    for idx, ((text, block_id), embedding) in enumerate(
        zip(sentences_with_blocks, embeddings_tensor)
    ):
        sentence_embeddings.append(
            SentenceEmbedding(
                id=idx,
                text=text,
                embedding=embedding.tolist(),
                block_id=block_id,
            )
        )

    # Step 7: Compute document embedding
    doc_emb_tensor = compute_doc_embedding(embeddings_tensor)
    doc_embedding = doc_emb_tensor.tolist() if doc_emb_tensor is not None else None

    # Step 8: Return structured result
    return EmbeddingResult(
        sentences=sentence_embeddings,
        doc_embedding=doc_embedding,
    )
