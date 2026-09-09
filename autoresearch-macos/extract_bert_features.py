"""
BERT Internal Signals — Extract pseudo-perplexity and attention entropy features.

Tests whether BERT MLM internal signals (pseudo-perplexity, attention entropy)
improve the Ridge scoring model on top of embeddings + hand-crafted features.

For each model (mxbai, bge), extracts 10 BERT features per journal entry:
- 5 pseudo-perplexity features (masked LM loss statistics)
- 5 attention entropy features (Shannon entropy of attention distributions)

Then evaluates via Ridge pipeline:
- Baseline: [embedding | 10 hand-crafted features]
- +All BERT: [embedding | 10 features | 10 BERT features]
- +PPL only: [embedding | 10 features | 5 PPL features]
- +Attn only: [embedding | 10 features | 5 attention features]
- +Best-k BERT: [embedding | 10 features | top-k BERT by |corr|], k=2,3,5

Usage:
    cd autoresearch-macos && PYTHONPATH=../src ../.venv/bin/python -u extract_bert_features.py
    cd autoresearch-macos && PYTHONPATH=../src ../.venv/bin/python -u extract_bert_features.py --models mxbai
    cd autoresearch-macos && PYTHONPATH=../src ../.venv/bin/python -u extract_bert_features.py --extract-only
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Add paths for sibling and src imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
CACHE_DIR = Path(__file__).parent
RESULTS_PATH = Path(__file__).parent / "bert_features_results.json"

# ---------------------------------------------------------------------------
# Model configurations (MLM variants)
# ---------------------------------------------------------------------------

MLM_CONFIGS = {
    "mxbai": {
        "hf_name": "mixedbread-ai/mxbai-embed-large-v1",
        "embedding_cache": "embeddings_mxbai.npy",
    },
    "bge": {
        "hf_name": "BAAI/bge-large-en-v1.5",
        "embedding_cache": "embeddings_bge.npy",
    },
}

# BERT feature names
PPL_FEATURE_KEYS = [
    "bert_ppl_mean",
    "bert_ppl_std",
    "bert_ppl_max",
    "bert_ppl_range",
    "bert_ppl_cv",
]

ATTN_FEATURE_KEYS = [
    "bert_attn_entropy_mean",
    "bert_attn_entropy_std",
    "bert_attn_entropy_last_layer",
    "bert_attn_entropy_first_layer",
    "bert_attn_entropy_layer_diff",
]

ALL_BERT_FEATURE_KEYS = PPL_FEATURE_KEYS + ATTN_FEATURE_KEYS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_journal_texts() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load journal texts, labels, and sources from consensus CSV."""
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
# Pseudo-perplexity extraction
# ---------------------------------------------------------------------------


def _compute_pseudo_perplexity_sentence(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    sentence: str,
    mask_frac: float = 0.15,
    n_trials: int = 3,
    seeds: tuple[int, ...] = (42, 123, 7),
) -> float:
    """Compute pseudo-perplexity for a single sentence.

    Masks `mask_frac` of tokens, computes cross-entropy on masked positions.
    Repeats with different seeds and averages for stability.

    Returns:
        Average cross-entropy loss across trials. Returns 0.0 if sentence
        has too few tokens to mask.
    """
    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    input_ids = encoded["input_ids"].squeeze(0)  # (seq_len,)
    n_tokens = input_ids.shape[0]

    # Skip [CLS] and [SEP] tokens for masking
    maskable = list(range(1, n_tokens - 1))
    if len(maskable) < 2:
        return 0.0

    n_mask = max(1, int(len(maskable) * mask_frac))
    trial_losses: list[float] = []

    for seed in seeds[:n_trials]:
        torch.manual_seed(seed)
        mask_indices = torch.tensor(
            sorted(torch.randperm(len(maskable))[:n_mask].tolist()),
            dtype=torch.long,
        )
        mask_positions = torch.tensor([maskable[i] for i in mask_indices], dtype=torch.long)

        # Create masked input
        masked_ids = input_ids.clone().unsqueeze(0)  # (1, seq_len)
        masked_ids[0, mask_positions] = tokenizer.mask_token_id

        # Create labels: -100 everywhere except masked positions
        labels = torch.full_like(masked_ids, -100)
        labels[0, mask_positions] = input_ids[mask_positions]

        # Move to model device (MPS/CUDA/CPU)
        device = next(model.parameters()).device
        masked_ids = masked_ids.to(device)
        attn_mask = encoded["attention_mask"].to(device)
        labels = labels.to(device)

        with torch.inference_mode():
            outputs = model(
                input_ids=masked_ids,
                attention_mask=attn_mask,
                labels=labels,
            )
            trial_losses.append(float(outputs.loss))

    return float(np.mean(trial_losses))


def extract_ppl_features_entry(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    sentences: list[str],
) -> dict[str, float]:
    """Extract 5 pseudo-perplexity features from an entry's sentences.

    Returns:
        Dict with keys: bert_ppl_mean, bert_ppl_std, bert_ppl_max,
        bert_ppl_range, bert_ppl_cv.
    """
    if len(sentences) == 0:
        return {k: 0.0 for k in PPL_FEATURE_KEYS}

    ppl_values: list[float] = []
    for sent in sentences:
        ppl = _compute_pseudo_perplexity_sentence(model, tokenizer, sent)
        ppl_values.append(ppl)

    arr = np.array(ppl_values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std()) if len(arr) > 1 else 0.0
    mx = float(arr.max())
    rng = float(arr.max() - arr.min()) if len(arr) > 1 else 0.0
    cv = float(std / mean) if mean > 1e-10 else 0.0

    return {
        "bert_ppl_mean": mean,
        "bert_ppl_std": std,
        "bert_ppl_max": mx,
        "bert_ppl_range": rng,
        "bert_ppl_cv": cv,
    }


# ---------------------------------------------------------------------------
# Attention entropy extraction
# ---------------------------------------------------------------------------


def extract_attn_features_entry(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    sentences: list[str],
    batch_size: int = 32,
) -> dict[str, float]:
    """Extract 5 attention entropy features from an entry's sentences.

    For each sentence, runs a forward pass with output_attentions=True,
    computes Shannon entropy of each attention head, then aggregates
    across sentences.

    Returns:
        Dict with keys: bert_attn_entropy_mean, bert_attn_entropy_std,
        bert_attn_entropy_last_layer, bert_attn_entropy_first_layer,
        bert_attn_entropy_layer_diff.
    """
    if len(sentences) == 0:
        return {k: 0.0 for k in ATTN_FEATURE_KEYS}

    # Collect per-layer entropy means across all sentences
    # Each element: (n_layers,) array of mean entropy per layer for one sentence
    all_layer_entropies: list[np.ndarray] = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move to model device (MPS/CUDA/CPU)
        device = next(model.parameters()).device
        input_ids_dev = encoded["input_ids"].to(device)
        attn_mask_dev = encoded["attention_mask"].to(device)

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids_dev,
                attention_mask=attn_mask_dev,
                output_attentions=True,
            )

        # outputs.attentions is a tuple of (batch, n_heads, seq_len, seq_len)
        # one per layer
        n_layers = len(outputs.attentions)
        attn_mask = encoded["attention_mask"]  # (batch, seq_len)

        for b_idx in range(len(batch)):
            layer_means: list[float] = []
            seq_mask = attn_mask[b_idx].bool()  # (seq_len,)
            n_valid = int(seq_mask.sum())

            for layer_idx in range(n_layers):
                attn = outputs.attentions[layer_idx][b_idx]  # (n_heads, seq_len, seq_len)
                # Only look at attention FROM valid tokens
                attn_valid = attn[:, :n_valid, :]  # (n_heads, n_valid, seq_len)

                # Shannon entropy per head: H = -sum(a * log(a + eps))
                eps = 1e-10
                log_attn = torch.log(attn_valid + eps)
                entropy_per_head = -(attn_valid * log_attn).sum(dim=-1)  # (n_heads, n_valid)
                # Mean entropy across heads and valid positions
                layer_mean = float(entropy_per_head.mean())
                layer_means.append(layer_mean)

            all_layer_entropies.append(np.array(layer_means, dtype=np.float64))

    if len(all_layer_entropies) == 0:
        return {k: 0.0 for k in ATTN_FEATURE_KEYS}

    # Stack: (n_sentences, n_layers)
    layer_matrix = np.stack(all_layer_entropies, axis=0)

    # Overall mean and std of entropy across all layers and sentences
    overall_mean = float(layer_matrix.mean())
    overall_std = float(layer_matrix.std()) if layer_matrix.size > 1 else 0.0

    # First and last layer averages (averaged across sentences)
    first_layer_mean = float(layer_matrix[:, 0].mean())
    last_layer_mean = float(layer_matrix[:, -1].mean())
    layer_diff = float(last_layer_mean - first_layer_mean)

    return {
        "bert_attn_entropy_mean": overall_mean,
        "bert_attn_entropy_std": overall_std,
        "bert_attn_entropy_last_layer": last_layer_mean,
        "bert_attn_entropy_first_layer": first_layer_mean,
        "bert_attn_entropy_layer_diff": layer_diff,
    }


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------


def extract_bert_features_for_model(
    model_name: str,
    texts: list[str],
    force: bool = False,
) -> list[dict[str, float]]:
    """Extract all 10 BERT features for a model, with caching.

    Args:
        model_name: "mxbai" or "bge".
        texts: Raw journal texts.
        force: If True, recompute even if cache exists.

    Returns:
        List of feature dicts, one per entry.
    """
    config = MLM_CONFIGS[model_name]
    cache_path = CACHE_DIR / f"bert_features_{model_name}.json"

    # Check cache
    if cache_path.exists() and not force:
        print(f"  Loading cached BERT features from {cache_path.name}")
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) == len(texts):
            return cached
        print(f"  Cache size mismatch ({len(cached)} vs {len(texts)}), recomputing...")

    hf_name = config["hf_name"]
    # Use MPS (Apple Silicon GPU) if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n  Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"\n  Using CPU")
    print(f"  Loading MLM model: {hf_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForMaskedLM.from_pretrained(hf_name, attn_implementation="eager")
    model.eval()
    model.requires_grad_(False)
    model = model.to(device)

    results: list[dict[str, float]] = []
    t0 = time.time()

    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(texts) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(texts)}] {rate:.1f} entries/s, ETA {eta/60:.1f}m")

        # Preprocess text
        clean = normalize_text(text)
        sentences = split_sentences(clean)

        # Extract PPL features
        ppl_feats = extract_ppl_features_entry(model, tokenizer, sentences)

        # Extract attention entropy features
        attn_feats = extract_attn_features_entry(model, tokenizer, sentences)

        # Merge
        entry_feats = {**ppl_feats, **attn_feats}
        results.append(entry_feats)

    elapsed = time.time() - t0
    print(f"  Done: {len(texts)} entries in {elapsed:.0f}s ({len(texts)/elapsed:.1f}/s)")

    # Save cache
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"  Saved BERT features to {cache_path.name}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Ridge evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    models_to_test: list[str],
    force: bool = False,
) -> dict:
    """Run full BERT feature evaluation.

    For each model:
    1. Extract BERT features (or load from cache)
    2. Build feature matrices for each config
    3. Evaluate via Ridge with human_heavy weighting

    Returns:
        Full results dict.
    """
    print("=" * W)
    print("BERT INTERNAL SIGNALS EXPERIMENT")
    print("=" * W)

    # --- Load data ---
    print("\n--- Loading data ---")
    texts, labels, sources = load_journal_texts()
    features, cached_labels, feature_names, cached_sources = load_cached_data()
    n_human = int((sources == SOURCE_HUMAN).sum())
    n_synth = int((sources == SOURCE_SYNTHETIC).sum())
    print(f"Entries: {len(texts)} ({n_human} human, {n_synth} synthetic)")

    assert len(texts) == len(cached_labels), (
        f"Text count {len(texts)} != cache count {len(cached_labels)}"
    )

    # --- Folds (same for all) ---
    folds = get_stratified_folds(sources)
    print(f"Folds: {len(folds)}")

    # --- Top-10 hand-crafted features (same as v6 baseline) ---
    feat_corrs = compute_feature_human_correlations(
        features, labels, sources, sorted(ALL_FEATURE_KEYS),
    )
    top_10_names = [name for name, _ in feat_corrs[:10]]
    print(f"Top-10 features (fixed): {top_10_names}")

    top_10_matrix = np.array(
        [[f.get(name, 0.0) for name in top_10_names] for f in features],
        dtype=np.float64,
    )

    # --- Weight strategy: human_heavy only ---
    weight_fn = make_sample_weights_fn(0.3)

    # --- Ridge alphas ---
    alphas = [0.01, 0.1, 1.0, 10.0]

    all_results: list[dict] = []

    for model_name in models_to_test:
        config = MLM_CONFIGS[model_name]
        print(f"\n{'=' * W}")
        print(f"MODEL: {model_name} ({config['hf_name']})")
        print(f"{'=' * W}")

        # Load embeddings
        emb_path = CACHE_DIR / config["embedding_cache"]
        if not emb_path.exists():
            print(f"  ERROR: Embedding cache not found at {emb_path}")
            continue
        E = np.load(emb_path).astype(np.float64)
        print(f"  Embeddings: {E.shape}")

        # Extract BERT features
        print(f"\n  --- Extracting BERT features ({model_name}) ---")
        bert_feats = extract_bert_features_for_model(model_name, texts, force=force)

        # Build BERT feature matrices
        ppl_matrix = np.array(
            [[bf[k] for k in PPL_FEATURE_KEYS] for bf in bert_feats],
            dtype=np.float64,
        )
        attn_matrix = np.array(
            [[bf[k] for k in ATTN_FEATURE_KEYS] for bf in bert_feats],
            dtype=np.float64,
        )
        all_bert_matrix = np.hstack([ppl_matrix, attn_matrix])

        # Compute BERT feature correlations with labels (human only)
        bert_corrs: list[tuple[str, float]] = []
        human_mask = sources == SOURCE_HUMAN
        human_labels = labels[human_mask]
        for j, fname in enumerate(ALL_BERT_FEATURE_KEYS):
            vals = all_bert_matrix[:, j]
            r = abs(pearson_correlation(vals[human_mask], human_labels))
            bert_corrs.append((fname, r))
        bert_corrs.sort(key=lambda x: -x[1])

        print(f"\n  BERT feature correlations with labels (human only):")
        for fname, r in bert_corrs:
            print(f"    {fname:<40s}  |r| = {r:.4f}")

        # --- Configs to evaluate ---
        configs: dict[str, np.ndarray] = {}

        # Baseline: [embedding | 10 features]
        X_baseline = np.hstack([E, top_10_matrix])
        configs["baseline"] = X_baseline

        # +All BERT: [embedding | 10 features | 10 BERT features]
        configs["+all_bert"] = np.hstack([X_baseline, all_bert_matrix])

        # +PPL only: [embedding | 10 features | 5 PPL features]
        configs["+ppl_only"] = np.hstack([X_baseline, ppl_matrix])

        # +Attn only: [embedding | 10 features | 5 attention features]
        configs["+attn_only"] = np.hstack([X_baseline, attn_matrix])

        # +Best-k BERT: top-k by |correlation|
        for k in [2, 3, 5]:
            top_k_names = [name for name, _ in bert_corrs[:k]]
            top_k_indices = [ALL_BERT_FEATURE_KEYS.index(n) for n in top_k_names]
            top_k_matrix = all_bert_matrix[:, top_k_indices]
            configs[f"+best_{k}_bert"] = np.hstack([X_baseline, top_k_matrix])

        # --- Evaluate each config ---
        print(f"\n  --- Ridge evaluation ({model_name}) ---")
        print(f"  {'Config':<20s} {'alpha':>6} {'human_corr':>11} {'cv_corr':>8} {'fold_std':>9} {'v4':>8}")
        print(f"  {'-' * 64}")

        for cfg_name, X in configs.items():
            for alpha in alphas:
                result = oof_evaluate(
                    Ridge, {"alpha": alpha}, X, labels, folds, sources, weight_fn,
                )
                v4 = compute_v4_objective(result)

                row = {
                    "model": model_name,
                    "config": cfg_name,
                    "alpha": alpha,
                    "n_features": int(X.shape[1]),
                    "human_corr": float(result["oof_human_corr"]),
                    "cv_corr": float(result["cv_corr"]),
                    "synth_corr": float(result["oof_synth_corr"]),
                    "fold_std": float(result["fold_human_corr_std"]),
                    "v4_score": float(v4),
                }
                all_results.append(row)

                print(f"  {cfg_name:<20s} {alpha:>6.2f} {row['human_corr']:>11.4f} {row['cv_corr']:>8.4f} {row['fold_std']:>9.4f} {row['v4_score']:>8.4f}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * W}")
    print("SUMMARY — Best alpha per config per model")
    print(f"{'=' * W}")

    # Group by (model, config), find best by human_corr
    best_per_cfg: dict[tuple[str, str], dict] = {}
    for r in all_results:
        key = (r["model"], r["config"])
        if key not in best_per_cfg or r["human_corr"] > best_per_cfg[key]["human_corr"]:
            best_per_cfg[key] = r

    print(f"\n  {'Model':<8} {'Config':<20s} {'n_feat':>6} {'alpha':>6} {'human_corr':>11} {'cv_corr':>8} {'v4':>8}")
    print(f"  {'-' * 70}")

    for model_name in models_to_test:
        for cfg_name in ["baseline", "+all_bert", "+ppl_only", "+attn_only", "+best_2_bert", "+best_3_bert", "+best_5_bert"]:
            key = (model_name, cfg_name)
            if key in best_per_cfg:
                r = best_per_cfg[key]
                baseline_key = (model_name, "baseline")
                delta = r["human_corr"] - best_per_cfg[baseline_key]["human_corr"]
                marker = f" ({delta:+.4f})" if cfg_name != "baseline" else " (ref)"
                print(
                    f"  {model_name:<8} {cfg_name:<20s} {r['n_features']:>6} "
                    f"{r['alpha']:>6.2f} {r['human_corr']:>11.4f} "
                    f"{r['cv_corr']:>8.4f} {r['v4_score']:>8.4f}{marker}"
                )
        print()

    # --- Save results ---
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_entries": len(texts),
        "n_human": n_human,
        "n_synth": n_synth,
        "models_tested": models_to_test,
        "top_10_features": top_10_names,
        "bert_feature_keys": ALL_BERT_FEATURE_KEYS,
        "best_per_config": {
            f"{k[0]}_{k[1]}": v for k, v in best_per_cfg.items()
        },
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
    parser = argparse.ArgumentParser(
        description="Extract BERT internal signals and evaluate their value for MES prediction",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MLM_CONFIGS.keys()),
        choices=list(MLM_CONFIGS.keys()),
        help="Models to test (default: both mxbai and bge)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if cached .json files exist",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract and cache features, skip Ridge evaluation",
    )
    args = parser.parse_args()

    if args.extract_only:
        print("=" * W)
        print("BERT FEATURE EXTRACTION ONLY")
        print("=" * W)
        texts, _, _ = load_journal_texts()
        for model_name in args.models:
            print(f"\n--- {model_name} ---")
            extract_bert_features_for_model(model_name, texts, force=args.force)
        print("\nDone. Use without --extract-only to run Ridge evaluation.")
    else:
        run_evaluation(args.models, force=args.force)


if __name__ == "__main__":
    main()
