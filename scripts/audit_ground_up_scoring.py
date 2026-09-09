#!/usr/bin/env python3
"""Ground-up MES scoring audit.

This intentionally starts below the model stack: labels, FEP dimension labels,
feature families, and document embeddings. It does not evaluate or promote a
production scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "autoresearch-macos"))

from prepare import (  # noqa: E402
    CACHE_PATH,
    SOURCE_HUMAN,
    SOURCE_SYNTHETIC,
    load_cached_data,
    load_cached_embeddings,
    load_dimension_labels,
)


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "eval" / "ground_up_scoring_audit.json"
FEATURE_FAMILIES: dict[str, str] = {
    "ce": "Coherence Entropy",
    "se": "Semantic Entropy",
    "ne": "Narrative Entropy",
    "cle": "Cognitive Load Entropy",
    "bc": "Belief Conflict",
    "gd": "Global Disorder",
}


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        average_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = average_rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    if x_mean is None or y_mean is None:
        return None
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    x_norm = math.sqrt(sum(value * value for value in x_centered))
    y_norm = math.sqrt(sum(value * value for value in y_centered))
    if x_norm == 0.0 or y_norm == 0.0:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered)) / (x_norm * y_norm)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(_rank(xs), _rank(ys))


def labels_to_mes(labels: np.ndarray) -> np.ndarray:
    return np.clip((labels - 1.0) / 9.0 * 100.0, 0.0, 100.0)


def summarize_distribution(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(len(values)),
        "mean": _round(float(np.mean(values))),
        "std": _round(float(np.std(values))),
        "min": _round(float(np.min(values))),
        "max": _round(float(np.max(values))),
    }


def source_masks(sources: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(sources), dtype=bool),
        "human": sources == SOURCE_HUMAN,
        "synthetic": sources == SOURCE_SYNTHETIC,
    }


def build_label_audit(
    labels: np.ndarray,
    sources: np.ndarray,
    dimension_labels: dict[str, np.ndarray],
    label_stds: np.ndarray | None = None,
) -> dict[str, Any]:
    masks = source_masks(sources)
    overall: dict[str, Any] = {}
    for name, mask in masks.items():
        overall[name] = summarize_distribution(labels[mask])

    dim_rows = []
    for dim_name, dim_values in sorted(dimension_labels.items()):
        row: dict[str, Any] = {
            "name": dim_name,
            "distribution": {
                mask_name: summarize_distribution(dim_values[mask])
                for mask_name, mask in masks.items()
            },
        }
        for mask_name, mask in masks.items():
            row[f"{mask_name}_spearman_overall_entropy"] = (
                None
                if int(mask.sum()) < 2
                else _round(spearman(dim_values[mask].tolist(), labels[mask].tolist()) or 0.0)
            )
        dim_rows.append(row)

    intercorrelations = []
    names = sorted(dimension_labels)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            intercorrelations.append({
                "left": left,
                "right": right,
                "human_spearman": _round(
                    spearman(
                        dimension_labels[left][masks["human"]].tolist(),
                        dimension_labels[right][masks["human"]].tolist(),
                    ) or 0.0
                ),
            })

    result = {
        "overall_entropy": overall,
        "dimension_labels": dim_rows,
        "dimension_intercorrelations": intercorrelations,
    }
    if label_stds is not None:
        result["label_stds"] = {
            mask_name: summarize_distribution(label_stds[mask])
            for mask_name, mask in masks.items()
        }
        result["label_std_all_zero"] = bool(np.all(label_stds == 0.0))
    return result


def _feature_names_by_family(feature_names: list[str]) -> dict[str, list[str]]:
    return {
        family: sorted(name for name in feature_names if name.startswith(f"{family}_"))
        for family in FEATURE_FAMILIES
    }


def _feature_column(features: list[dict[str, float]], name: str) -> list[float]:
    return [float(row.get(name, 0.0)) for row in features]


def build_feature_family_audit(
    features: list[dict[str, float]],
    labels: np.ndarray,
    sources: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    masks = source_masks(sources)
    grouped = _feature_names_by_family(feature_names)
    output: dict[str, Any] = {}

    for family, names in grouped.items():
        rows = []
        for name in names:
            values = np.array(_feature_column(features, name), dtype=np.float64)
            row = {"name": name}
            for mask_name, mask in masks.items():
                corr = spearman(values[mask].tolist(), labels[mask].tolist())
                row[f"{mask_name}_spearman_overall_entropy"] = (
                    None if corr is None else _round(corr)
                )
            rows.append(row)

        human_abs = [
            abs(float(row["human_spearman_overall_entropy"]))
            for row in rows
            if row["human_spearman_overall_entropy"] is not None
        ]
        rows_by_abs_human = sorted(
            rows,
            key=lambda row: abs(row["human_spearman_overall_entropy"] or 0.0),
            reverse=True,
        )
        output[family] = {
            "name": FEATURE_FAMILIES[family],
            "feature_count": len(names),
            "mean_abs_human_spearman": (
                None if not human_abs else _round(_mean(human_abs) or 0.0)
            ),
            "top_human_signals": rows_by_abs_human[:10],
            "top_positive_human_signals": sorted(
                rows,
                key=lambda row: row["human_spearman_overall_entropy"] or 0.0,
                reverse=True,
            )[:5],
            "top_negative_human_signals": sorted(
                rows,
                key=lambda row: row["human_spearman_overall_entropy"] or 0.0,
            )[:5],
        }

    return output


def make_source_stratified_folds(sources: np.ndarray, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for source in sorted(set(int(value) for value in sources.tolist())):
        indices = np.where(sources == source)[0]
        for offset, index in enumerate(indices):
            folds[offset % n_folds].append(int(index))

    all_indices = np.arange(len(sources))
    result = []
    for fold in folds:
        val_idx = np.array(sorted(fold), dtype=np.int64)
        train_mask = np.ones(len(sources), dtype=bool)
        train_mask[val_idx] = False
        result.append((all_indices[train_mask], val_idx))
    return result


def ridge_oof_probe(
    X: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    alpha: float = 10.0,
    n_folds: int = 5,
) -> dict[str, Any]:
    preds = np.full(len(y), np.nan, dtype=np.float64)
    for train_idx, val_idx in make_source_stratified_folds(sources, n_folds):
        model = Ridge(alpha=alpha)
        model.fit(X[train_idx], y[train_idx])
        preds[val_idx] = model.predict(X[val_idx])

    masks = source_masks(sources)
    metrics: dict[str, Any] = {}
    for name, mask in masks.items():
        y_true = y[mask]
        y_pred = preds[mask]
        metrics[name] = {
            "pearson": _round(pearson(y_pred.tolist(), y_true.tolist()) or 0.0),
            "spearman": _round(spearman(y_pred.tolist(), y_true.tolist()) or 0.0),
            "mae": _round(float(np.mean(np.abs(y_pred - y_true)))),
        }
    return metrics


def embedding_neighbor_smoothness(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    k: int = 5,
    human_only: bool = True,
) -> dict[str, Any]:
    mask = sources == SOURCE_HUMAN if human_only else np.ones(len(sources), dtype=bool)
    E = embeddings[mask]
    y = labels[mask]
    if len(E) <= k:
        return {"n": int(len(E)), "k": k, "mean_neighbor_label_gap": None}

    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = E / norms
    sims = normalized @ normalized.T
    np.fill_diagonal(sims, -np.inf)
    neighbor_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    gaps = np.abs(y[:, None] - y[neighbor_idx])
    return {
        "n": int(len(E)),
        "k": int(k),
        "human_only": bool(human_only),
        "mean_neighbor_oe_gap": _round(float(np.mean(gaps))),
        "median_neighbor_oe_gap": _round(float(np.median(gaps))),
    }


def build_embedding_audit(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    dimension_labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    norms = np.linalg.norm(embeddings, axis=1)
    probes = {
        "overall_entropy": ridge_oof_probe(embeddings, labels, sources),
    }
    for dim_name, dim_values in sorted(dimension_labels.items()):
        probes[f"fep_{dim_name}"] = ridge_oof_probe(embeddings, dim_values, sources)

    return {
        "shape": {
            "rows": int(embeddings.shape[0]),
            "dims": int(embeddings.shape[1]),
        },
        "norms": summarize_distribution(norms),
        "ridge_oof_probes": probes,
        "nearest_neighbor_smoothness": embedding_neighbor_smoothness(
            embeddings, labels, sources,
        ),
    }


def build_ground_up_audit(
    features: list[dict[str, float]],
    labels: np.ndarray,
    feature_names: list[str],
    sources: np.ndarray,
    dimension_labels: dict[str, np.ndarray],
    embeddings: np.ndarray,
    label_stds: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "dataset": {
            "n_rows": int(len(labels)),
            "n_human": int((sources == SOURCE_HUMAN).sum()),
            "n_synthetic": int((sources == SOURCE_SYNTHETIC).sum()),
            "n_features": int(len(feature_names)),
            "embedding_dim": int(embeddings.shape[1]),
        },
        "label_audit": build_label_audit(
            labels, sources, dimension_labels, label_stds,
        ),
        "feature_family_audit": build_feature_family_audit(
            features, labels, sources, feature_names,
        ),
        "document_embedding_audit": build_embedding_audit(
            embeddings, labels, sources, dimension_labels,
        ),
        "interpretation_hints": [
            "Start here before scorer artifacts: these are raw label, feature, and embedding diagnostics.",
            "Human correlations matter most; synthetic correlations can expose generator shortcuts.",
            "Document embedding probes test whether the document vector carries OE/FEP signal before any combiner.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ground-up MES scoring audit.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_label_stds() -> np.ndarray | None:
    payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    values = payload.get("label_stds")
    if values is None:
        return None
    return np.array(values, dtype=np.float64)


def main() -> int:
    args = parse_args()
    features, labels, feature_names, sources = load_cached_data()
    dimension_labels = load_dimension_labels()
    embeddings = load_cached_embeddings()
    label_stds = load_label_stds()

    audit = build_ground_up_audit(
        features,
        labels,
        feature_names,
        sources,
        dimension_labels,
        embeddings,
        label_stds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("MES Ground-Up Scoring Audit")
    print(f"  rows: {audit['dataset']['n_rows']}")
    print("  feature families by mean abs human Spearman:")
    ranked = sorted(
        audit["feature_family_audit"].items(),
        key=lambda item: item[1]["mean_abs_human_spearman"] or 0.0,
        reverse=True,
    )
    for family, row in ranked:
        print(f"    {family}: {row['mean_abs_human_spearman']}")
    oe_probe = audit["document_embedding_audit"]["ridge_oof_probes"]["overall_entropy"]
    print(
        "  doc_embedding_overall_entropy_human_spearman: "
        f"{oe_probe['human']['spearman']}"
    )
    print(f"  report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
