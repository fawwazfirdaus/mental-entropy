"""
MES Weight Optimization - Data Preparation & Evaluation Harness.

This file has two modes:

1. Script mode (run once to pre-compute features):
   cd /Users/karanpatil/Desktop/mental-entropy
   PYTHONPATH=src .venv/bin/python autoresearch-macos/prepare.py

2. Import mode (used by train.py for evaluation):
   from prepare import load_cached_data, evaluate_weights
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_PATH = Path(__file__).parent / "features_cache.json"
CONSENSUS_CSV = Path(__file__).parent.parent / "data" / "multirater" / "consensus_labels.csv"
LOCKED_EVAL_PATH = (
    Path(__file__).parent.parent / "data" / "eval" / "human_locked_eval_v1.json"
)
# Legacy single-pass CSVs (fallback if consensus not available)
JOURNALS_CSV = Path(__file__).parent.parent / "journals_labeled.csv"
ADDITIONAL_LABELED_CSV = Path(__file__).parent.parent / "data" / "additional_journals_labeled_gpt54.csv"
SYNTHETIC_LABELED_CSV = Path(__file__).parent.parent / "data" / "synthetic_journals_labeled_gpt54.csv"
HIGH_ENTROPY_LABELED_CSV = Path(__file__).parent.parent / "data" / "high_entropy_labeled_gpt54.csv"
HIGH_ENTROPY_SYNTHETIC_CLAUDE_CSV = Path(__file__).parent.parent / "data" / "high_entropy_synthetic_labeled_claude.csv"
NEW_HUMAN_JOURNALS_CSV = Path(__file__).parent.parent / "data" / "new_human_journals_labeled.csv"

# Transform types for feature -> entropy contribution mapping
TRANSFORM_INV = "inv"           # score += w * (1 - x)
TRANSFORM_DIR = "dir"           # score += w * x
TRANSFORM_DIR_NORM = "dir_norm" # score += w * min(x / param, 1.0)
TRANSFORM_INV_CLIP = "inv_clip" # score += w * (1 - min(x, param))
TRANSFORM_INV_NZ = "inv_nz"     # score += w * (1 - x) if x > 0 else 0

N_FOLDS = 5

DIMENSION_NAMES = [
    "prediction_coherence",
    "model_complexity",
    "compression_progress",
    "belief_integration",
    "precision_weighting",
]


# ---------------------------------------------------------------------------
# Cache loading (always available, no heavy deps)
# ---------------------------------------------------------------------------

def load_locked_eval_source_indices(path: Path = LOCKED_EVAL_PATH) -> set[int]:
    """Load consensus row indices reserved for locked evaluation."""
    if not path.exists():
        return set()

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return {
        int(entry["source_index"])
        for entry in payload.get("entries", [])
        if "source_index" in entry
    }


def filter_locked_eval_entries(
    entries: list[dict[str, str]],
    locked_indices: set[int],
) -> list[tuple[int, dict[str, str]]]:
    """Return consensus rows excluding locked eval source indices."""
    return [
        (source_index, entry)
        for source_index, entry in enumerate(entries)
        if source_index not in locked_indices
    ]

def load_cached_data() -> tuple[list[dict[str, float]], np.ndarray, list[str], np.ndarray]:
    """
    Load pre-computed features and labels from cache.

    Returns:
        (features, labels, feature_names, sources) where:
        - features: list of dicts mapping feature_name -> float
        - labels: np.ndarray of overall_entropy values (1-10 scale)
        - feature_names: sorted list of all feature names
        - sources: np.ndarray of ints (0=human_written, 1=synthetic)
    """
    if not CACHE_PATH.exists():
        print(f"ERROR: Cache not found at {CACHE_PATH}")
        print("Run prepare.py first:")
        print(f"  cd {CACHE_PATH.parent.parent}")
        print(f"  PYTHONPATH=src .venv/bin/python autoresearch-macos/prepare.py")
        sys.exit(1)

    with open(CACHE_PATH, "r") as f:
        data = json.load(f)

    features = data["features"]
    labels = np.array(data["labels"], dtype=np.float64)
    feature_names = sorted(data["feature_names"])
    sources = np.array(data.get("sources", [0] * len(features)), dtype=np.int32)

    return features, labels, feature_names, sources


def load_dimension_labels() -> dict[str, np.ndarray]:
    """
    Load per-dimension rubric labels from cache.

    Returns:
        dict mapping dimension name -> np.ndarray of scores (1-5 scale).
        Keys: continuity, topic_focus, contradiction_integration,
              cognitive_clarity, narrative_closure.
    """
    if not CACHE_PATH.exists():
        print(f"ERROR: Cache not found at {CACHE_PATH}")
        sys.exit(1)

    with open(CACHE_PATH, "r") as f:
        data = json.load(f)

    dim_labels = data.get("dimension_labels")
    if dim_labels is None:
        print("ERROR: dimension_labels not in cache. Re-run prepare.py to regenerate.")
        sys.exit(1)

    return {name: np.array(vals, dtype=np.float64) for name, vals in dim_labels.items()}


def load_cached_embeddings() -> np.ndarray:
    """
    Load pre-computed 1024-dim document embeddings from cache.

    Returns:
        np.ndarray of shape (n_entries, 1024) with float64 values.
    """
    if not CACHE_PATH.exists():
        print(f"ERROR: Cache not found at {CACHE_PATH}")
        print("Run prepare.py first.")
        sys.exit(1)

    with open(CACHE_PATH, "r") as f:
        data = json.load(f)

    embeddings = data.get("embeddings")
    if embeddings is None:
        print("ERROR: embeddings not in cache. Re-run prepare.py to regenerate.")
        sys.exit(1)

    return np.array(embeddings, dtype=np.float64)


# Source constants
SOURCE_HUMAN = 0
SOURCE_SYNTHETIC = 1


# ---------------------------------------------------------------------------
# Evaluation functions (pure numpy, no heavy deps)
# ---------------------------------------------------------------------------

def pearson_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Compute (optionally weighted) Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0

    if weights is None:
        mx, my = x.mean(), y.mean()
        dx, dy = x - mx, y - my
        num = (dx * dy).sum()
        denom = math.sqrt(float((dx * dx).sum() * (dy * dy).sum()))
    else:
        w = weights / weights.sum()  # normalize to sum=1
        mx = float((w * x).sum())
        my = float((w * y).sum())
        dx, dy = x - mx, y - my
        num = float((w * dx * dy).sum())
        denom = math.sqrt(float((w * dx * dx).sum()) * float((w * dy * dy).sum()))

    if denom == 0:
        return 0.0
    return float(num / denom)


def apply_transform(value: float, transform: str, param: float | None) -> float:
    """Apply a transform to a raw feature value."""
    if transform == TRANSFORM_INV:
        return 1.0 - value
    elif transform == TRANSFORM_DIR:
        return value
    elif transform == TRANSFORM_DIR_NORM:
        return min(value / param, 1.0)
    elif transform == TRANSFORM_INV_CLIP:
        return 1.0 - min(value, param)
    elif transform == TRANSFORM_INV_NZ:
        return (1.0 - value) if value > 0 else 0.0
    else:
        raise ValueError(f"Unknown transform: {transform}")


def compute_mes_scores(
    weights_config: list[tuple],
    features: list[dict[str, float]],
) -> np.ndarray:
    """
    Compute MES scores for all entries using the given weight configuration.

    Args:
        weights_config: List of (feature_name, weight, transform, [param]) tuples.
        features: List of feature dicts, one per entry.

    Returns:
        np.ndarray of MES scores (0-100).
    """
    n = len(features)
    scores = np.zeros(n, dtype=np.float64)

    for i, feat_dict in enumerate(features):
        raw = 0.0
        for entry in weights_config:
            name, weight, transform = entry[0], entry[1], entry[2]
            param = entry[3] if len(entry) > 3 else None
            value = feat_dict.get(name, 0.0)
            raw += weight * apply_transform(value, transform, param)
        scores[i] = min(100.0, max(0.0, raw * 100.0))

    return scores


def evaluate_weights(
    weights_config: list[tuple],
    features: list[dict[str, float]],
    labels: np.ndarray,
    n_folds: int = N_FOLDS,
    sources: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Evaluate a weight configuration against labeled data.

    Returns dict with:
        val_corr:       Pearson correlation on full dataset (higher is better)
        cv_corr:        Mean 5-fold cross-validated correlation (PRIMARY METRIC)
        cv_std:         Std of fold correlations
        mse:            Mean squared error (MES 0-100 vs label scaled to 0-100)
        score_mean:     Mean MES score produced
        score_std:      Std of MES scores
        n_features:     Number of features with non-zero weight
        weight_sum:     Sum of all weights
        human_corr:     Correlation on human-written entries only (if sources provided)
        synth_corr:     Correlation on synthetic entries only (if sources provided)
    """
    scores = compute_mes_scores(weights_config, features)

    # Build per-sample weights so each source contributes equally
    # Human entries get upweighted proportionally: weight = n_synth / n_human
    if sources is not None:
        n_human = int((sources == SOURCE_HUMAN).sum())
        n_synth = int((sources == SOURCE_SYNTHETIC).sum())
        sample_weights = np.ones(len(features), dtype=np.float64)
        if n_human > 0 and n_synth > 0:
            # Each human entry gets weight = n_synth/n_human so total human weight = total synth weight
            sample_weights[sources == SOURCE_HUMAN] = n_synth / n_human
    else:
        sample_weights = None

    # Scale labels from 1-10 to 0-100 for MSE comparison
    labels_scaled = (labels - 1) / 9 * 100

    val_corr = pearson_correlation(scores, labels, weights=sample_weights)
    mse = float(((scores - labels_scaled) ** 2).mean())

    # Cross-validation on full dataset
    n = len(features)
    indices = np.arange(n)
    rng = np.random.RandomState(42)  # deterministic
    rng.shuffle(indices)

    fold_size = n // n_folds
    fold_corrs = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n

        val_idx = indices[val_start:val_end]
        val_features = [features[i] for i in val_idx]
        val_labels = labels[val_idx]
        fold_weights = sample_weights[val_idx] if sample_weights is not None else None

        fold_scores = compute_mes_scores(weights_config, val_features)
        fold_corr = pearson_correlation(fold_scores, val_labels, weights=fold_weights)
        fold_corrs.append(fold_corr)

    cv_corr = float(np.mean(fold_corrs))
    cv_std = float(np.std(fold_corrs))

    n_features = sum(1 for e in weights_config if e[1] != 0)
    weight_sum = sum(e[1] for e in weights_config)

    result = {
        "val_corr": val_corr,
        "cv_corr": cv_corr,
        "cv_std": cv_std,
        "mse": mse,
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "n_features": n_features,
        "weight_sum": weight_sum,
    }

    # Per-source correlations
    if sources is not None:
        human_mask = sources == SOURCE_HUMAN
        synth_mask = sources == SOURCE_SYNTHETIC

        if human_mask.sum() > 2:
            result["human_corr"] = pearson_correlation(scores[human_mask], labels[human_mask])
            result["human_n"] = int(human_mask.sum())
        if synth_mask.sum() > 2:
            result["synth_corr"] = pearson_correlation(scores[synth_mask], labels[synth_mask])
            result["synth_n"] = int(synth_mask.sum())

    return result


# ---------------------------------------------------------------------------
# Feature pre-computation (script mode only, needs mental_entropy)
# ---------------------------------------------------------------------------

def _save_cache(
    all_features: list[dict[str, float]],
    all_embeddings: list[list[float]],
    labels: list[float],
    sources: list[int],
    dimension_labels: dict[str, list[float]],
    label_stds: list[float],
    failed: int,
) -> None:
    """Save computed features, embeddings, and labels to JSON cache."""
    if not all_features:
        print("No features computed. Exiting.")
        sys.exit(1)

    feature_names = sorted(all_features[0].keys())
    emb_dim = len(all_embeddings[0]) if all_embeddings else 0
    print(f"  Embeddings: {len(all_embeddings)} x {emb_dim}-dim")

    cache_data = {
        "features": all_features,
        "embeddings": all_embeddings,
        "labels": labels,
        "sources": sources,
        "dimension_labels": dimension_labels,
        "feature_names": feature_names,
        "n_entries": len(all_features),
        "n_features": len(feature_names),
        "embedding_dim": emb_dim,
        "n_human": sum(1 for s in sources if s == SOURCE_HUMAN),
        "n_synthetic": sum(1 for s in sources if s == SOURCE_SYNTHETIC),
    }
    if label_stds:
        cache_data["label_stds"] = label_stds

    with open(CACHE_PATH, "w") as f:
        json.dump(cache_data, f, indent=2)

    print(f"\nCached {len(all_features)} entries x {len(feature_names)} features")
    print(f"  Human-written: {cache_data['n_human']}")
    print(f"  Synthetic:   {cache_data['n_synthetic']}")
    if label_stds:
        avg_std = sum(label_stds) / len(label_stds)
        print(f"  Label stds:  mean={avg_std:.2f} (consensus mode)")
    print(f"Saved to {CACHE_PATH}")


def load_label_stds() -> np.ndarray | None:
    """Load per-entry inter-rater standard deviations from cache.

    Returns None if not available (single-pass labels).
    """
    if not CACHE_PATH.exists():
        return None
    with open(CACHE_PATH, "r") as f:
        data = json.load(f)
    stds = data.get("label_stds")
    if stds is None:
        return None
    return np.array(stds, dtype=np.float64)


def _extract_features(text, embed_fn, ce_fn, se_fn, ne_fn, cle_fn, bc_fn, gd_fn):
    """Compute all features for a single journal entry.

    Returns:
        (features_dict, doc_embedding_list) where doc_embedding_list is the
        1024-dim document embedding as a Python list of floats.
    """
    result = embed_fn(text)
    features: dict[str, float] = {}
    features.update(ce_fn(result))
    features.update(se_fn(result))
    features.update(ne_fn(result))
    features.update(cle_fn(result))
    features.update(bc_fn(result))
    features.update(gd_fn(result))
    # Capture the raw 1024-dim document embedding
    doc_emb = result.doc_embedding.tolist() if hasattr(result.doc_embedding, 'tolist') else list(result.doc_embedding)
    return features, doc_emb


def precompute_features() -> None:
    """Run feature extraction on all labeled + synthetic entries and cache.

    If consensus labels (multi-rater) exist, uses those for all entries.
    Otherwise falls back to individual single-pass CSVs.
    """
    from mental_entropy import (
        embed_journal_entry,
        ce_features_from_result,
        se_features_from_result,
        ne_features_from_result,
        cle_features_from_result,
        bc_features_from_result,
        gd_features_from_result,
    )
    import csv

    fns = (embed_journal_entry, ce_features_from_result,
           se_features_from_result, ne_features_from_result,
           cle_features_from_result, bc_features_from_result,
           gd_features_from_result)

    all_features: list[dict[str, float]] = []
    all_embeddings: list[list[float]] = []  # 1024-dim doc embeddings
    labels: list[float] = []
    sources: list[int] = []  # 0=human_written, 1=synthetic
    dimension_labels: dict[str, list[float]] = {d: [] for d in DIMENSION_NAMES}
    label_stds: list[float] = []  # inter-rater std per entry (consensus mode)
    failed = 0

    # --- Check for consensus (multi-rater) labels ---
    if CONSENSUS_CSV.exists():
        print(f"Using CONSENSUS (multi-rater) labels from {CONSENSUS_CSV}")
        entries = []
        with open(CONSENSUS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
        print(f"  {len(entries)} entries total")
        locked_indices = load_locked_eval_source_indices()
        filtered_entries = filter_locked_eval_entries(entries, locked_indices)
        if locked_indices:
            skipped = len(entries) - len(filtered_entries)
            print(f"  Excluding {skipped} locked eval entries from training cache")

        for i, (source_index, entry) in enumerate(filtered_entries):
            pct = (i + 1) / len(filtered_entries) * 100
            print(f"\r  [{i+1}/{len(filtered_entries)}] ({pct:.1f}%)...", end="", flush=True)
            try:
                features, doc_emb = _extract_features(entry["journal"], *fns)
                all_features.append(features)
                all_embeddings.append(doc_emb)
                labels.append(float(entry["overall_entropy"]))
                sources.append(SOURCE_HUMAN if entry["source"] == "human" else SOURCE_SYNTHETIC)
                label_stds.append(float(entry.get("label_std", 0.0)))
                for d in DIMENSION_NAMES:
                    val = entry.get(d, "5")
                    dimension_labels[d].append(float(val) if val not in ("", "N/A") else 5.0)
            except Exception as e:
                print(f"\n  Warning: Failed source row {source_index}: {e}")
                failed += 1

        n_human = sum(1 for s in sources if s == SOURCE_HUMAN)
        n_synth = sum(1 for s in sources if s == SOURCE_SYNTHETIC)
        print(f"\n  Processed {len(all_features)} entries ({n_human} human, {n_synth} synthetic, {failed} failed)")
        print(f"  Label type: consensus (mean of 3 raters)")

        # Skip legacy loading — consensus has everything
        _save_cache(all_features, all_embeddings, labels, sources,
                    dimension_labels, label_stds, failed)
        return

    # --- Legacy path: individual single-pass CSVs ---
    print("No consensus labels found, using legacy single-pass CSVs...")

    # --- Part 1: Human-written journals (245 entries, LLM-labeled 1-10 scale) ---
    print(f"Loading human-written journals from {JOURNALS_CSV}...")
    human_entries = []
    with open(JOURNALS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            human_entries.append(row)
    print(f"  {len(human_entries)} entries")

    print("Computing features for human-written entries...")
    for i, entry in enumerate(human_entries):
        pct = (i + 1) / len(human_entries) * 100
        print(f"\r  [{i+1}/{len(human_entries)}] ({pct:.1f}%)...", end="", flush=True)
        try:
            features, doc_emb = _extract_features(entry["journal"], *fns)
            all_features.append(features)
            all_embeddings.append(doc_emb)
            labels.append(float(entry["overall_entropy"]))
            sources.append(SOURCE_HUMAN)
            for d in DIMENSION_NAMES:
                val = entry.get(d, "5")
                dimension_labels[d].append(float(val) if val not in ("", "N/A") else 5.0)
        except Exception as e:
            print(f"\n  Warning: Failed human entry {i+1}: {e}")
            failed += 1

    n_human = sum(1 for s in sources if s == SOURCE_HUMAN)
    print(f"\n  Processed {n_human} human-written entries ({failed} failed)")

    # --- Part 1.5: Additional human-written journals (GPT-5.4 labeled) ---
    if ADDITIONAL_LABELED_CSV.exists():
        print(f"\nLoading additional real journals from {ADDITIONAL_LABELED_CSV}...")
        add_entries = []
        with open(ADDITIONAL_LABELED_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                add_entries.append(row)
        print(f"  {len(add_entries)} entries")

        add_failed = 0
        print("Computing features for additional real journal entries...")
        for i, entry in enumerate(add_entries):
            pct = (i + 1) / len(add_entries) * 100
            print(f"\r  [{i+1}/{len(add_entries)}] ({pct:.1f}%)...", end="", flush=True)
            try:
                features, doc_emb = _extract_features(entry["journal"], *fns)
                all_features.append(features)
                all_embeddings.append(doc_emb)
                labels.append(float(entry["overall_entropy"]))
                sources.append(SOURCE_HUMAN)  # Real human journals
                for d in DIMENSION_NAMES:
                    val = entry.get(d, "5")
                    dimension_labels[d].append(float(val) if val not in ("", "N/A") else 5.0)
            except Exception as e:
                print(f"\n  Warning: Failed additional entry {i+1}: {e}")
                add_failed += 1

        n_add = len(add_entries) - add_failed
        print(f"\n  Processed {n_add} additional real entries ({add_failed} failed)")
        failed += add_failed
    else:
        print(f"\nNo additional real journals found at {ADDITIONAL_LABELED_CSV}, skipping.")

    # --- Part 1.6: New human journals (400 entries, GPT-5.4 labeled) ---
    if NEW_HUMAN_JOURNALS_CSV.exists():
        print(f"\nLoading new human journals from {NEW_HUMAN_JOURNALS_CSV}...")
        new_entries = []
        with open(NEW_HUMAN_JOURNALS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                new_entries.append(row)
        print(f"  {len(new_entries)} entries")

        new_failed = 0
        print("Computing features for new human journal entries...")
        for i, entry in enumerate(new_entries):
            pct = (i + 1) / len(new_entries) * 100
            print(f"\r  [{i+1}/{len(new_entries)}] ({pct:.1f}%)...", end="", flush=True)
            try:
                features, doc_emb = _extract_features(entry["journal"], *fns)
                all_features.append(features)
                all_embeddings.append(doc_emb)
                labels.append(float(entry["overall_entropy"]))
                sources.append(SOURCE_HUMAN)  # Real human journals
                for d in DIMENSION_NAMES:
                    val = entry.get(d, "5")
                    dimension_labels[d].append(float(val) if val not in ("", "N/A") else 5.0)
            except Exception as e:
                print(f"\n  Warning: Failed new human entry {i+1}: {e}")
                new_failed += 1

        n_new = len(new_entries) - new_failed
        print(f"\n  Processed {n_new} new human entries ({new_failed} failed)")
        failed += new_failed
    else:
        print(f"\nNo new human journals found at {NEW_HUMAN_JOURNALS_CSV}, skipping.")

    # --- Part 2: Rubric-labeled synthetic journals (from GPT-4o labeling) ---
    synth_csvs = [
        (SYNTHETIC_LABELED_CSV, "synthetic (809)"),
        (HIGH_ENTROPY_LABELED_CSV, "high-entropy (186)"),
        (HIGH_ENTROPY_SYNTHETIC_CLAUDE_CSV, "high-entropy synthetic Claude"),
    ]
    for synth_csv, label in synth_csvs:
        if synth_csv.exists():
            print(f"\nLoading {label} journals from {synth_csv}...")
            synth_entries = []
            with open(synth_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    synth_entries.append(row)
            print(f"  {len(synth_entries)} entries")

            synth_failed = 0
            print(f"Computing features for {label} entries...")
            for i, entry in enumerate(synth_entries):
                pct = (i + 1) / len(synth_entries) * 100
                print(f"\r  [{i+1}/{len(synth_entries)}] ({pct:.1f}%)...", end="", flush=True)
                try:
                    features, doc_emb = _extract_features(entry["journal"], *fns)
                    all_features.append(features)
                    all_embeddings.append(doc_emb)
                    # Labels are already on 1-10 scale from GPT-4o rubric
                    labels.append(float(entry["overall_entropy"]))
                    sources.append(SOURCE_SYNTHETIC)
                    for d in DIMENSION_NAMES:
                        val = entry.get(d, "5")
                        dimension_labels[d].append(float(val) if val not in ("", "N/A") else 5.0)
                except Exception as e:
                    print(f"\n  Warning: Failed {label} entry {i+1}: {e}")
                    synth_failed += 1

            n_synth_this = sum(1 for j, s in enumerate(sources) if s == SOURCE_SYNTHETIC) - sum(1 for j, s in enumerate(sources[:len(all_features) - len(synth_entries) + synth_failed]) if s == SOURCE_SYNTHETIC)
            print(f"\n  Processed {len(synth_entries) - synth_failed} {label} entries ({synth_failed} failed)")
            failed += synth_failed
        else:
            print(f"\nNo {label} data found at {synth_csv}, skipping.")

    print(f"\nTotal: {len(all_features)} entries ({failed} total failed)")
    _save_cache(all_features, all_embeddings, labels, sources,
                dimension_labels, [], failed)


if __name__ == "__main__":
    precompute_features()
