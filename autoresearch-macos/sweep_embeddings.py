"""
Embedding Model Sweep — Test whether a different embedding model improves MES prediction.

Tests 5 embedding models through the same Ridge pipeline:
1. mxbai-embed-large-v1 (baseline, 1024-dim)
2. bge-large-en-v1.5 (1024-dim, CLS pooling)
3. e5-large-v2 (1024-dim, needs "passage: " prefix)
4. gte-large (1024-dim, mean pooling)
5. nomic-embed-text-v1.5 (768-dim, needs "search_document: " prefix)

Each model is evaluated with:
- Pure Ridge: embedding → overall_entropy (5 alphas × 3 weight strategies)
- Hybrid Ridge: [embedding | top-10 features] → overall_entropy (3 alphas × 3 weights)

Usage:
    cd autoresearch-macos && PYTHONPATH=../src ../.venv/bin/python -u sweep_embeddings.py
    cd autoresearch-macos && PYTHONPATH=../src ../.venv/bin/python -u sweep_embeddings.py --force  # re-embed all
    cd autoresearch-macos && PYTHONPATH=../src ../.venv/bin/python -u sweep_embeddings.py --models mxbai bge  # subset
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoTokenizer

# Add src to path for mental_entropy imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mental_entropy.embedding.model import mean_pooling, l2_normalize
from mental_entropy.embedding.text import normalize_text, split_sentences

from prepare import (
    SOURCE_HUMAN,
    SOURCE_SYNTHETIC,
    load_cached_data,
    pearson_correlation,
)
from train_subscores_v4 import get_stratified_folds, compute_v4_objective
from train_embedding_regression import (
    oof_evaluate,
    make_sample_weights_fn,
    compute_feature_human_correlations,
)
from train_subscores_v3 import ALL_FEATURE_KEYS

W = 72
CONSENSUS_CSV = Path(__file__).parent.parent / "data" / "multirater" / "consensus_labels.csv"
CACHE_DIR = Path(__file__).parent  # store embeddings_*.npy here
RESULTS_PATH = Path(__file__).parent / "embedding_sweep_results.json"

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "mxbai": {
        "hf_name": "mixedbread-ai/mxbai-embed-large-v1",
        "dim": 1024,
        "pooling": "mean",
        "prefix": None,
        "trust_remote_code": False,
    },
    "bge": {
        "hf_name": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "pooling": "cls",
        "prefix": None,
        "trust_remote_code": False,
    },
    "e5": {
        "hf_name": "intfloat/e5-large-v2",
        "dim": 1024,
        "pooling": "mean",
        "prefix": "passage: ",
        "trust_remote_code": False,
    },
    "gte": {
        "hf_name": "thenlper/gte-large",
        "dim": 1024,
        "pooling": "mean",
        "prefix": None,
        "trust_remote_code": False,
    },
    "nomic": {
        "hf_name": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
        "pooling": "mean",
        "prefix": "search_document: ",
        "trust_remote_code": True,
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_journal_texts() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load journal texts, labels, and sources from consensus CSV.

    Returns:
        (texts, labels, sources) aligned with existing cache order.
    """
    if not CONSENSUS_CSV.exists():
        print(f"ERROR: Consensus CSV not found at {CONSENSUS_CSV}")
        sys.exit(1)

    texts: list[str] = []
    labels: list[float] = []
    sources: list[int] = []

    with open(CONSENSUS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["journal"])
            labels.append(float(row["overall_entropy"]))
            src = row.get("source", "human")
            sources.append(SOURCE_SYNTHETIC if src == "synthetic" else SOURCE_HUMAN)

    return texts, np.array(labels, dtype=np.float64), np.array(sources, dtype=np.int32)


# ---------------------------------------------------------------------------
# Generic embedding
# ---------------------------------------------------------------------------


def cls_pooling(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor,
) -> torch.Tensor:
    """CLS token pooling — take the first token's representation."""
    return last_hidden_state[:, 0, :]


def embed_with_model(
    texts: list[str],
    config: dict,
    batch_size: int = 16,
) -> np.ndarray:
    """Embed all journal entries using a specific model.

    For each text: normalize → split sentences → embed sentences → average → L2 norm.
    This matches the production pipeline in embed.py.

    Args:
        texts: Raw journal texts.
        config: Model configuration dict.
        batch_size: Sentence batch size for inference.

    Returns:
        np.ndarray of shape (n_entries, config["dim"]).
    """
    hf_name = config["hf_name"]
    pooling_fn = mean_pooling if config["pooling"] == "mean" else cls_pooling
    prefix = config["prefix"]

    print(f"  Loading model: {hf_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_name, trust_remote_code=config["trust_remote_code"],
    )
    model = AutoModel.from_pretrained(
        hf_name, trust_remote_code=config["trust_remote_code"],
    )
    model.eval()
    model.requires_grad_(False)

    doc_embeddings: list[np.ndarray] = []
    t0 = time.time()

    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(texts) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(texts)}] {rate:.1f} entries/s, ETA {eta/60:.1f}m")

        # Normalize and split sentences (same as production pipeline)
        clean = normalize_text(text)
        sentences = split_sentences(clean)

        if not sentences:
            doc_embeddings.append(np.zeros(config["dim"], dtype=np.float32))
            continue

        # Apply prefix if needed
        if prefix:
            sentences = [prefix + s for s in sentences]

        # Embed sentences in batches
        all_sent_embs: list[torch.Tensor] = []
        for j in range(0, len(sentences), batch_size):
            batch = sentences[j : j + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.inference_mode():
                outputs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                pooled = pooling_fn(outputs.last_hidden_state, encoded["attention_mask"])
                normalized = l2_normalize(pooled)
                all_sent_embs.append(normalized.cpu())

        # Average sentence embeddings → doc embedding
        sent_tensor = torch.cat(all_sent_embs, dim=0)
        doc_emb = sent_tensor.mean(dim=0, keepdim=True)
        doc_emb = l2_normalize(doc_emb).squeeze(0).numpy()
        doc_embeddings.append(doc_emb)

    elapsed = time.time() - t0
    print(f"  Done: {len(texts)} entries in {elapsed:.0f}s ({len(texts)/elapsed:.1f}/s)")

    # Cleanup model from memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return np.array(doc_embeddings, dtype=np.float64)


def get_or_compute_embeddings(
    name: str,
    config: dict,
    texts: list[str],
    force: bool = False,
) -> np.ndarray:
    """Load cached embeddings or compute and cache them.

    Args:
        name: Short model name (e.g., "mxbai", "bge").
        config: Model configuration dict.
        texts: Journal texts.
        force: If True, recompute even if cache exists.

    Returns:
        np.ndarray of shape (n_entries, config["dim"]).
    """
    cache_path = CACHE_DIR / f"embeddings_{name}.npy"

    if cache_path.exists() and not force:
        print(f"  Loading cached embeddings from {cache_path.name}")
        E = np.load(cache_path)
        if E.shape[0] == len(texts) and E.shape[1] == config["dim"]:
            return E
        print(f"  Cache shape mismatch ({E.shape}), recomputing...")

    E = embed_with_model(texts, config)
    np.save(cache_path, E)
    print(f"  Saved embeddings to {cache_path.name}")
    return E


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(
    models_to_test: list[str],
    force: bool = False,
) -> dict:
    """Run the full embedding model sweep.

    Args:
        models_to_test: List of model names from MODEL_CONFIGS.
        force: If True, recompute all embeddings.

    Returns:
        Full results dict.
    """
    print("=" * W)
    print("EMBEDDING MODEL SWEEP")
    print("=" * W)

    # Load data
    print("\n--- Loading data ---")
    texts, labels, sources = load_journal_texts()
    features, cached_labels, feature_names, cached_sources = load_cached_data()
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())
    print(f"Entries: {len(texts)} ({n_human} human, {n_synth} synthetic)")

    # Verify alignment
    assert len(texts) == len(cached_labels), (
        f"Text count {len(texts)} != cache count {len(cached_labels)}"
    )

    # Get folds (same across all models for fair comparison)
    folds = get_stratified_folds(sources)
    print(f"Folds: {len(folds)}")

    # Compute feature correlations for top-k selection (fixed across models)
    feat_corrs = compute_feature_human_correlations(
        features, labels, sources, sorted(ALL_FEATURE_KEYS),
    )
    top_10_names = [name for name, _ in feat_corrs[:10]]
    print(f"Top-10 features (fixed): {top_10_names}")

    # Build feature matrix for hybrid
    top_10_matrix = np.array(
        [[f.get(name, 0.0) for name in top_10_names] for f in features],
        dtype=np.float64,
    )

    # Weight strategies
    weight_strategies = {
        "equal": None,
        "balanced": make_sample_weights_fn(1.0),
        "human_heavy": make_sample_weights_fn(0.3),
    }

    # Ridge alphas
    pure_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    hybrid_alphas = [0.1, 1.0, 10.0]

    all_results: list[dict] = []

    for model_name in models_to_test:
        config = MODEL_CONFIGS[model_name]
        print(f"\n{'=' * W}")
        print(f"MODEL: {model_name} ({config['hf_name']}, {config['dim']}-dim, {config['pooling']} pooling)")
        print(f"{'=' * W}")

        E = get_or_compute_embeddings(model_name, config, texts, force=force)
        print(f"  Embeddings shape: {E.shape}")

        # --- Pure Ridge ---
        print(f"\n  --- Pure Ridge ({model_name}) ---")
        for alpha in pure_alphas:
            for wname, wfn in weight_strategies.items():
                result = oof_evaluate(
                    Ridge, {"alpha": alpha}, E, labels, folds, sources, wfn,
                )
                v4 = compute_v4_objective(result)
                row = {
                    "model": model_name,
                    "dim": config["dim"],
                    "type": "pure_ridge",
                    "alpha": alpha,
                    "weights": wname,
                    "cv_corr": float(result["cv_corr"]),
                    "human_corr": float(result["oof_human_corr"]),
                    "synth_corr": float(result["oof_synth_corr"]),
                    "fold_std": float(result["fold_human_corr_std"]),
                    "v4_score": float(v4),
                }
                all_results.append(row)
                print(f"    α={alpha:7.3f} w={wname:12s}  human={row['human_corr']:.3f}  cv={row['cv_corr']:.3f}  v4={row['v4_score']:.3f}")

        # --- Hybrid Ridge (embedding + top-10 features) ---
        print(f"\n  --- Hybrid Ridge ({model_name} + top-10 features) ---")
        X_hybrid = np.hstack([E, top_10_matrix])
        for alpha in hybrid_alphas:
            for wname, wfn in weight_strategies.items():
                result = oof_evaluate(
                    Ridge, {"alpha": alpha}, X_hybrid, labels, folds, sources, wfn,
                )
                v4 = compute_v4_objective(result)
                row = {
                    "model": model_name,
                    "dim": config["dim"],
                    "type": "hybrid_ridge",
                    "alpha": alpha,
                    "weights": wname,
                    "cv_corr": float(result["cv_corr"]),
                    "human_corr": float(result["oof_human_corr"]),
                    "synth_corr": float(result["oof_synth_corr"]),
                    "fold_std": float(result["fold_human_corr_std"]),
                    "v4_score": float(v4),
                }
                all_results.append(row)
                print(f"    α={alpha:7.3f} w={wname:12s}  human={row['human_corr']:.3f}  cv={row['cv_corr']:.3f}  v4={row['v4_score']:.3f}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * W}")
    print("FINAL COMPARISON — Best config per model")
    print(f"{'=' * W}")

    # Group by model, find best by v4_score
    best_per_model: dict[str, dict] = {}
    for r in all_results:
        m = r["model"]
        if m not in best_per_model or r["v4_score"] > best_per_model[m]["v4_score"]:
            best_per_model[m] = r

    print(f"\n{'Model':<10} {'Type':<14} {'α':>6} {'Weights':<14} {'human_corr':>11} {'cv_corr':>8} {'v4_score':>9}")
    print("-" * W)
    for m in models_to_test:
        if m in best_per_model:
            r = best_per_model[m]
            print(f"{m:<10} {r['type']:<14} {r['alpha']:>6.2f} {r['weights']:<14} {r['human_corr']:>11.4f} {r['cv_corr']:>8.4f} {r['v4_score']:>9.4f}")

    # Also show best by human_corr
    print(f"\n{'Model':<10} {'Type':<14} {'α':>6} {'Weights':<14} {'human_corr':>11} {'cv_corr':>8}")
    print("-" * W)
    best_by_human: dict[str, dict] = {}
    for r in all_results:
        m = r["model"]
        if m not in best_by_human or r["human_corr"] > best_by_human[m]["human_corr"]:
            best_by_human[m] = r
    for m in models_to_test:
        if m in best_by_human:
            r = best_by_human[m]
            marker = " ← WINNER" if r["human_corr"] == max(best_by_human[x]["human_corr"] for x in best_by_human) else ""
            print(f"{m:<10} {r['type']:<14} {r['alpha']:>6.2f} {r['weights']:<14} {r['human_corr']:>11.4f} {r['cv_corr']:>8.4f}{marker}")

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_entries": len(texts),
        "n_human": n_human,
        "n_synth": n_synth,
        "n_configs": len(all_results),
        "top_10_features": top_10_names,
        "best_per_model_v4": best_per_model,
        "best_per_model_human": best_by_human,
        "all_results": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Embedding model sweep for MES")
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to test (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-embedding even if cached .npy files exist",
    )
    args = parser.parse_args()

    run_sweep(args.models, force=args.force)


if __name__ == "__main__":
    main()
