"""Smoke tests for the embedding layer.

These tests verify:
1. Basic functionality works end-to-end
2. Embeddings are L2-normalized (unit length)
3. Results are deterministic (same input -> same output)
4. Sentence order is preserved
5. Trained spaCy model is loaded for sentence splitting
"""

from __future__ import annotations

import math


def test_using_trained_spacy_model():
    """Verify en_core_web_sm is loaded for sentence splitting."""
    from mental_entropy.embedding.text import is_using_trained_model

    assert is_using_trained_model() is True


def test_embed_journal_entry_basic():
    """Test basic embedding functionality."""
    from mental_entropy import embed_journal_entry

    text = "Today was good. I felt happy."
    result = embed_journal_entry(text)

    assert len(result.sentences) == 2
    assert result.sentences[0].id == 0
    assert result.sentences[1].id == 1
    assert result.sentences[0].text == "Today was good."
    assert result.sentences[1].text == "I felt happy."
    assert len(result.sentences[0].embedding) == 1024
    assert len(result.sentences[1].embedding) == 1024
    assert result.doc_embedding is not None
    assert len(result.doc_embedding) == 1024


def test_embeddings_are_l2_normalized():
    """Test that all embeddings have unit length."""
    from mental_entropy import embed_journal_entry

    text = "First sentence. Second sentence. Third one here."
    result = embed_journal_entry(text)

    def l2_norm(vec: list[float]) -> float:
        return math.sqrt(sum(x * x for x in vec))

    # Check sentence embeddings
    for sent in result.sentences:
        norm = l2_norm(sent.embedding)
        assert abs(norm - 1.0) < 1e-5, f"Sentence {sent.id} norm={norm}, expected ~1.0"

    # Check doc embedding
    if result.doc_embedding:
        norm = l2_norm(result.doc_embedding)
        assert abs(norm - 1.0) < 1e-5, f"Doc embedding norm={norm}, expected ~1.0"


def test_determinism():
    """Test that same input produces identical output."""
    from mental_entropy import embed_journal_entry

    text = "Testing determinism. Same input, same output."

    result1 = embed_journal_entry(text)
    result2 = embed_journal_entry(text)

    # Same number of sentences
    assert len(result1.sentences) == len(result2.sentences)

    # Identical embeddings (exact float equality on CPU)
    for s1, s2 in zip(result1.sentences, result2.sentences):
        assert s1.id == s2.id
        assert s1.text == s2.text
        assert s1.embedding == s2.embedding, f"Sentence {s1.id} embeddings differ"

    # Identical doc embeddings
    assert result1.doc_embedding == result2.doc_embedding


def test_empty_input():
    """Test handling of empty input."""
    from mental_entropy import embed_journal_entry

    result = embed_journal_entry("")
    assert result.sentences == []
    assert result.doc_embedding is None

    result = embed_journal_entry("   \n\n   ")
    assert result.sentences == []
    assert result.doc_embedding is None


def test_sentence_order_preserved():
    """Test that sentence order matches input order."""
    from mental_entropy import embed_journal_entry

    text = """First. Second. Third.

Fourth after blank line. Fifth."""

    result = embed_journal_entry(text)

    # Verify order by checking text content
    texts = [s.text for s in result.sentences]
    assert "First" in texts[0]
    assert "Second" in texts[1]
    assert "Third" in texts[2]
    assert "Fourth" in texts[3]
    assert "Fifth" in texts[4]

    # Verify IDs are sequential
    for i, sent in enumerate(result.sentences):
        assert sent.id == i


def test_multiline_journal():
    """Test realistic multi-paragraph journal entry."""
    from mental_entropy import embed_journal_entry

    journal = """
Today I woke up early. The sun was just rising.

I went for a run in the park. It felt refreshing!
My mind was clearer afterward.

Work was intense but productive. I finished the report.
"""

    result = embed_journal_entry(journal)

    # Should have multiple sentences
    assert len(result.sentences) >= 5

    # All should have valid embeddings
    for sent in result.sentences:
        assert len(sent.embedding) == 1024
        assert sent.text.strip() != ""

    # Doc embedding should exist
    assert result.doc_embedding is not None
    assert len(result.doc_embedding) == 1024


def test_punctuation_only_filtered():
    """Test that punctuation-only 'sentences' are filtered out."""
    from mental_entropy import embed_journal_entry

    # Text with punctuation-only segments
    text = "Real sentence. ... Another real one!"

    result = embed_journal_entry(text)

    # Should not include "..." as a sentence
    texts = [s.text for s in result.sentences]
    assert "..." not in texts
    assert "Real sentence." in texts or "Real sentence" in texts[0]


def test_output_text_is_unmodified():
    """Test that the output text field contains the original sentence unchanged."""
    from mental_entropy import embed_journal_entry

    text = "This is a test sentence."
    result = embed_journal_entry(text)

    # The text field should contain the original sentence, not any model-specific formatting
    for sent in result.sentences:
        # Should not have any common embedding model prefixes
        assert not sent.text.startswith("passage: "), (
            f"Text field should not contain 'passage: ' prefix, got: {sent.text}"
        )
        assert not sent.text.startswith("query: "), (
            f"Text field should not contain 'query: ' prefix, got: {sent.text}"
        )
        assert not sent.text.startswith("search_document: "), (
            f"Text field should not contain 'search_document: ' prefix, got: {sent.text}"
        )


def test_embedding_list_length_matches_sentences():
    """Test that embedding count always equals sentence count."""
    from mental_entropy import embed_journal_entry

    texts = [
        "One sentence.",
        "Two sentences here. And another one.",
        "Three. Four. Five.",
    ]

    for text in texts:
        result = embed_journal_entry(text)
        # Each sentence should have exactly one embedding
        assert len(result.sentences) > 0
        for sent in result.sentences:
            assert len(sent.embedding) == 1024


def test_long_input_truncation():
    """Test that very long sentences are truncated without crashing."""
    from mental_entropy import embed_journal_entry

    # Create a very long sentence (well over 512 tokens)
    long_sentence = "This is a word. " * 500  # ~1500 tokens

    result = embed_journal_entry(long_sentence)

    # Should not crash, should return valid embeddings
    assert len(result.sentences) >= 1
    assert len(result.sentences[0].embedding) == 1024

    # Embedding should still be L2-normalized
    import math

    norm = math.sqrt(sum(x * x for x in result.sentences[0].embedding))
    assert abs(norm - 1.0) < 1e-5


def test_mean_pooling_consistent():
    """
    Test that mean pooling produces consistent embeddings regardless of batch context.

    This is verified indirectly: the same sentence should produce identical embeddings
    whether it's embedded alone or with other sentences in a batch.
    """
    from mental_entropy import embed_journal_entry

    # Single sentence - no padding needed
    result1 = embed_journal_entry("Short sentence.")

    # Same sentence but with another sentence that's much longer
    # This forces padding on the short sentence
    result2 = embed_journal_entry(
        "Short sentence. This is a much longer sentence that will require more tokens and cause padding on the first sentence when batched together."
    )

    # The first sentence's embedding should be very similar in both cases
    # (not identical due to different batch contexts, but cosine similarity should be very high)
    emb1 = result1.sentences[0].embedding
    emb2 = result2.sentences[0].embedding

    # Cosine similarity (both are L2-normalized, so dot product = cosine sim)
    cosine_sim = sum(a * b for a, b in zip(emb1, emb2))

    # Should be extremely similar (>0.99) - mean pooling with attention mask is deterministic
    assert cosine_sim > 0.99, (
        f"Cosine similarity {cosine_sim} too low, mean pooling may be broken"
    )
