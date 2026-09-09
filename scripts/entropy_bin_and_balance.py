"""Map entropy distribution from measured features and export balanced subsets.

This script:
1) loads one or more synthetic journal JSONL files
2) computes CE/SE/NE/CLE features per entry
3) derives an aggregate entropy score from feature-percentile ranks
4) bins entries into low/medium/high via score tertiles
5) writes full scored JSONL + balanced JSONL + summary report JSON
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mental_entropy import (
    ce_features_from_result,
    cle_features_from_result,
    embed_journal_entry,
    ne_features_from_result,
    se_features_from_result,
)

HIGH_FEATURES: tuple[str, ...] = (
    "ce_break_rate",
    "ce_low_mass",
    "ce_adj_std",
    "ce_sharp_drop_rate",
    "se_cluster_entropy",
    "se_switch_rate",
    "se_switch_mean_jump",
    "se_n_clusters",
    "ne_semantic_wander",
    "ne_fragment_sentence_rate",
    "ne_temporal_jump_rate",
    "ne_punct_break_rate",
    "cle_fragment_rate",
    "cle_restart_rate",
    "cle_hedge_rate",
    "cle_punctuation_noise",
    "cle_adj_low_frac",
    "cle_zigzag_rate",
    "cle_semantic_break_rate",
    "cle_semantic_isolated_rate",
)

LOW_FEATURES: tuple[str, ...] = (
    "ce_adj_mean",
    "ce_longest_coherent_run",
    "se_dominant_cluster_frac",
    "se_intra_mean",
    "ne_arc_linearity",
    "ne_start_end_sim",
    "ne_consolidation_delta",
    "ne_end_to_centroid",
)


def _to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected numeric feature value, got {type(value)}")


def _percentile_rank(values: list[float]) -> np.ndarray:
    """Return [0,1] percentile-ish ranks using stable ordering."""
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([0.5], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64) / float(n - 1)
    return ranks


def _load_rows(paths: list[str], max_entries: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        source = Path(path)
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            rows.append(
                {
                    "source_file": source.name,
                    "source_path": str(source),
                    "entry": payload,
                }
            )
            if max_entries is not None and len(rows) >= max_entries:
                return rows
    return rows


def _compute_feature_rows(rows: list[dict[str, Any]], progress_every: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        entry = row["entry"]
        text = entry.get("journal_text")
        if not isinstance(text, str) or not text.strip():
            continue
        result = embed_journal_entry(text)
        ce = ce_features_from_result(result)
        se = se_features_from_result(result)
        ne = ne_features_from_result(result)
        cle = cle_features_from_result(result)
        features = {
            **{k: _to_float(v) for k, v in ce.items()},
            **{k: _to_float(v) for k, v in se.items()},
            **{k: _to_float(v) for k, v in ne.items()},
            **{k: _to_float(v) for k, v in cle.items()},
        }
        out.append(
            {
                "source_file": row["source_file"],
                "source_path": row["source_path"],
                "entry": entry,
                "features": features,
            }
        )
        if progress_every > 0 and i % progress_every == 0:
            print(f"Computed features for {i}/{total} entries")
    return out


def _score_entropy(feature_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    feature_values: dict[str, list[float]] = {key: [] for key in HIGH_FEATURES + LOW_FEATURES}
    for row in feature_rows:
        f = row["features"]
        for key in feature_values:
            feature_values[key].append(_to_float(f[key]))

    feature_ranks: dict[str, np.ndarray] = {}
    for key, values in feature_values.items():
        ranks = _percentile_rank(values)
        if key in LOW_FEATURES:
            ranks = 1.0 - ranks
        feature_ranks[key] = ranks

    for idx, row in enumerate(feature_rows):
        high_scores = [feature_ranks[key][idx] for key in HIGH_FEATURES]
        low_scores = [feature_ranks[key][idx] for key in LOW_FEATURES]
        ce_keys = [k for k in HIGH_FEATURES + LOW_FEATURES if k.startswith("ce_")]
        se_keys = [k for k in HIGH_FEATURES + LOW_FEATURES if k.startswith("se_")]
        ne_keys = [k for k in HIGH_FEATURES + LOW_FEATURES if k.startswith("ne_")]
        cle_keys = [k for k in HIGH_FEATURES + LOW_FEATURES if k.startswith("cle_")]

        row["scores"] = {
            "entropy_overall": float(np.mean(high_scores + low_scores)),
            "entropy_ce": float(np.mean([feature_ranks[k][idx] for k in ce_keys])),
            "entropy_se": float(np.mean([feature_ranks[k][idx] for k in se_keys])),
            "entropy_ne": float(np.mean([feature_ranks[k][idx] for k in ne_keys])),
            "entropy_cle": float(np.mean([feature_ranks[k][idx] for k in cle_keys])),
        }

    overall = [row["scores"]["entropy_overall"] for row in feature_rows]
    q1, q2 = np.quantile(np.asarray(overall, dtype=np.float64), [1 / 3, 2 / 3])
    return feature_rows, {"q1": float(q1), "q2": float(q2)}


def _assign_bins(feature_rows: list[dict[str, Any]], q1: float, q2: float) -> Counter:
    counts: Counter = Counter()
    for row in feature_rows:
        s = row["scores"]["entropy_overall"]
        if s <= q1:
            label = "low"
        elif s <= q2:
            label = "medium"
        else:
            label = "high"
        row["entropy_bin"] = label
        counts[label] += 1
    return counts


def _write_scored(path: str, feature_rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in feature_rows:
            entry = row["entry"]
            payload = {
                **entry,
                "entropy_analysis": {
                    "source_file": row["source_file"],
                    "entropy_bin": row["entropy_bin"],
                    **row["scores"],
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_balanced(path: str, feature_rows: list[dict[str, Any]], seed: int) -> dict[str, int]:
    by_bin: dict[str, list[dict[str, Any]]] = {"low": [], "medium": [], "high": []}
    for row in feature_rows:
        by_bin[row["entropy_bin"]].append(row)

    min_count = min(len(by_bin["low"]), len(by_bin["medium"]), len(by_bin["high"]))
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in ("low", "medium", "high"):
        bucket = list(by_bin[key])
        rng.shuffle(bucket)
        selected.extend(bucket[:min_count])

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            entry = row["entry"]
            payload = {
                **entry,
                "entropy_analysis": {
                    "source_file": row["source_file"],
                    "entropy_bin": row["entropy_bin"],
                    **row["scores"],
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {"low": min_count, "medium": min_count, "high": min_count, "total": min_count * 3}


def main() -> None:
    parser = argparse.ArgumentParser(description="Map measured entropy distribution and export balanced subsets.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files.")
    parser.add_argument(
        "--scored-output",
        default="data/synthetic_journals_800_scored.jsonl",
        help="Output JSONL with entropy analysis attached to each entry.",
    )
    parser.add_argument(
        "--balanced-output",
        default="data/synthetic_journals_800_balanced.jsonl",
        help="Output JSONL balanced by low/medium/high entropy bins.",
    )
    parser.add_argument(
        "--report-output",
        default="data/synthetic_journals_entropy_report.json",
        help="Summary report JSON path.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-entries", type=int, default=None, help="Optional cap for fast smoke runs.")
    args = parser.parse_args()

    rows = _load_rows(args.inputs, args.max_entries)
    if not rows:
        raise ValueError("No valid input rows found.")

    print(f"Loaded {len(rows)} entries")
    feature_rows = _compute_feature_rows(rows, progress_every=max(0, args.progress_every))
    print(f"Computed features for {len(feature_rows)} entries")

    feature_rows, qs = _score_entropy(feature_rows)
    counts = _assign_bins(feature_rows, q1=qs["q1"], q2=qs["q2"])

    _write_scored(args.scored_output, feature_rows)
    balanced_counts = _write_balanced(args.balanced_output, feature_rows, seed=args.seed)

    report = {
        "input_files": args.inputs,
        "n_entries_scored": len(feature_rows),
        "quantile_cutoffs": qs,
        "bin_counts": dict(counts),
        "balanced_counts": balanced_counts,
        "outputs": {
            "scored": args.scored_output,
            "balanced": args.balanced_output,
            "report": args.report_output,
        },
    }
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
