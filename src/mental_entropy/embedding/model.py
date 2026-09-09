"""Embedding model singletons, batching, pooling, and normalization.

Currently uses mixedbread-ai/mxbai-embed-large-v1 with mean pooling and L2 normalization.

Mean pooling was chosen over CLS pooling because it:
- Achieves 100% accuracy on coherent vs fragmented classification (vs 83% for CLS)
- Better handles short emotional sentences by preserving contextual connections
- Produces higher similarities for thematically related content
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Model configuration
MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
# Pin to a specific revision for determinism (can be overridden via env var)
# Default to the latest revision of the model as of December 2025
MODEL_REVISION = os.environ.get("MES_MODEL_REVISION", "db9d1fe0f31addb4978201b2bf3e577f3f8900d2")
MAX_LENGTH = 512
EMBEDDING_DIM = 1024

# Singletons (loaded once at first use)
_tokenizer: "PreTrainedTokenizerBase | None" = None
_model: "PreTrainedModel | None" = None
_device: torch.device | None = None


def _get_device() -> torch.device:
    """Get the device to use (CPU by default for determinism)."""
    global _device
    if _device is None:
        # Default to CPU for maximum determinism
        # Can be overridden via env var if needed
        device_str = os.environ.get("MES_DEVICE", "cpu")
        _device = torch.device(device_str)
    return _device


def _get_tokenizer() -> "PreTrainedTokenizerBase":
    """Lazily load the tokenizer (singleton)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
        )
    return _tokenizer


def _get_model() -> "PreTrainedModel":
    """Lazily load the model (singleton, frozen, eval mode)."""
    global _model
    if _model is None:
        device = _get_device()
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
        )
        _model.to(device)
        _model.eval()
        _model.requires_grad_(False)
    return _model


def mean_pooling(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply mean pooling over token embeddings (excluding padding tokens).

    Mean pooling averages all non-padding token embeddings, which better captures
    shared vocabulary and contextual connections compared to CLS pooling.

    Args:
        last_hidden_state: Shape (batch_size, seq_len, hidden_dim)
        attention_mask: Shape (batch_size, seq_len), 1 for real tokens, 0 for padding

    Returns:
        Pooled embeddings of shape (batch_size, hidden_dim)
    """
    # Expand attention mask to match hidden state dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    # Sum embeddings for non-padding tokens
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)

    # Sum of mask (number of real tokens per sequence)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    # Mean = sum / count
    return sum_embeddings / sum_mask


def l2_normalize(embeddings: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize embeddings to unit length.

    Args:
        embeddings: Shape (batch_size, hidden_dim)

    Returns:
        Normalized embeddings of shape (batch_size, hidden_dim)
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def embed_sentences(
    sentences: list[str],
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Embed sentences using the frozen embedding model.

    Args:
        sentences: List of raw sentence strings.
        batch_size: Number of sentences to process at once.

    Returns:
        L2-normalized embeddings of shape (num_sentences, EMBEDDING_DIM).
    """
    if not sentences:
        return torch.empty(0, EMBEDDING_DIM)

    tokenizer = _get_tokenizer()
    model = _get_model()
    device = _get_device()

    all_embeddings: list[torch.Tensor] = []

    # Process in batches
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]

        # Tokenize
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Forward pass (no gradients)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

            # Mean pooling (average all non-padding tokens)
            pooled = mean_pooling(last_hidden_state, attention_mask)

            # L2 normalize
            normalized = l2_normalize(pooled)

            all_embeddings.append(normalized.cpu())

    # Concatenate all batches
    return torch.cat(all_embeddings, dim=0)


def compute_doc_embedding(sentence_embeddings: torch.Tensor) -> torch.Tensor | None:
    """
    Compute document embedding as mean of sentence embeddings.

    Args:
        sentence_embeddings: Shape (num_sentences, EMBEDDING_DIM), already L2-normalized.

    Returns:
        L2-normalized document embedding of shape (EMBEDDING_DIM,), or None if no sentences.
    """
    if sentence_embeddings.numel() == 0:
        return None

    # Mean across sentences
    doc_emb = sentence_embeddings.mean(dim=0, keepdim=True)

    # L2 normalize
    doc_emb = l2_normalize(doc_emb)

    return doc_emb.squeeze(0)
