"""Aggregate multi-rater labels into consensus scores.

Reads 3 pass CSVs from data/multirater/, computes trimmed-mean consensus,
and writes consensus_labels.csv + agreement statistics.

Usage:
    python scripts/aggregate_multirater.py
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "multirater"

PASS_FILES = {
    "a": DATA_DIR / "pass_a_labeled.csv",
    "b": DATA_DIR / "pass_b_labeled.csv",
    "c": DATA_DIR / "pass_c_labeled.csv",
}

DIMENSIONS = [
    "prediction_coherence",
    "model_complexity",
    "compression_progress",
    "belief_integration",
    "precision_weighting",
]

# Outlier thresholds: if spread > threshold, trim the outlier
DIM_OUTLIER_THRESHOLD = 2    # for 1-5 scale dimensions
OVERALL_OUTLIER_THRESHOLD = 3  # for 1-10 overall_entropy


def load_pass(pass_name: str) -> list[dict]:
    """Load a pass CSV into a list of dicts."""
    path = PASS_FILES[pass_name]
    if not path.exists():
        raise FileNotFoundError(f"Pass {pass_name} CSV not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"  Pass {pass_name.upper()}: {len(rows)} entries")
    return rows


def trimmed_mean(values: list[float], threshold: float) -> tuple[float, bool]:
    """Compute trimmed mean of 3 values.

    If spread (max - min) <= threshold, return simple mean.
    Otherwise, drop the value furthest from the median and average the rest.

    Returns:
        (consensus_value, was_trimmed)
    """
    assert len(values) == 3, f"Expected 3 values, got {len(values)}"
    s = sorted(values)

    if s[2] - s[0] <= threshold:
        return statistics.mean(s), False

    # Drop the outlier (furthest from median)
    median = s[1]
    diffs = [abs(v - median) for v in s]
    drop_idx = diffs.index(max(diffs))
    remaining = [v for i, v in enumerate(s) if i != drop_idx]
    return statistics.mean(remaining), True


def rating_stats(values: list[float]) -> dict[str, float | str | int]:
    """Return deterministic disagreement metadata for three raw ratings."""
    assert len(values) == 3, f"Expected 3 values, got {len(values)}"
    return {
        "raw": "|".join(str(round(value, 2)).rstrip("0").rstrip(".") for value in values),
        "std": round(statistics.pstdev(values), 3),
        "spread": round(max(values) - min(values), 3),
        "n_raters": len(values),
    }


def aggregate() -> None:
    """Run aggregation pipeline."""
    print("=" * 72)
    print("Aggregating Multi-Rater Labels")
    print("=" * 72)

    # Load all 3 passes
    passes: dict[str, list[dict]] = {}
    for name in ["a", "b", "c"]:
        passes[name] = load_pass(name)

    # Verify alignment
    n_a, n_b, n_c = len(passes["a"]), len(passes["b"]), len(passes["c"])
    if not (n_a == n_b == n_c):
        print(f"\n  WARNING: Pass lengths differ! A={n_a}, B={n_b}, C={n_c}")
        n = min(n_a, n_b, n_c)
        print(f"  Using first {n} entries from each pass")
    else:
        n = n_a
        print(f"\n  All passes have {n} entries")

    # Aggregate
    consensus_rows: list[dict] = []
    raw_rows: list[dict] = []
    trim_counts: dict[str, int] = {d: 0 for d in DIMENSIONS + ["overall_entropy"]}
    dim_deviations: dict[str, list[float]] = {d: [] for d in DIMENSIONS + ["overall_entropy"]}
    overall_scores: list[float] = []

    for i in range(n):
        row_a = passes["a"][i]
        row_b = passes["b"][i]
        row_c = passes["c"][i]

        consensus: dict[str, float | str] = {
            "journal": row_a["journal"],
            "source": row_a.get("source", ""),
            "source_file": row_a.get("source_file", ""),
        }

        # Aggregate each dimension
        for dim in DIMENSIONS:
            vals = [
                float(row_a.get(dim, 3)),
                float(row_b.get(dim, 3)),
                float(row_c.get(dim, 3)),
            ]
            score, trimmed = trimmed_mean(vals, DIM_OUTLIER_THRESHOLD)
            stats = rating_stats(vals)
            consensus[dim] = round(score, 2)
            consensus[f"{dim}_raw"] = stats["raw"]
            consensus[f"{dim}_std"] = stats["std"]
            consensus[f"{dim}_spread"] = stats["spread"]
            if trimmed:
                trim_counts[dim] += 1
            dim_deviations[dim].append(max(vals) - min(vals))

        # Aggregate overall_entropy
        oe_vals = [
            float(row_a.get("overall_entropy", 5)),
            float(row_b.get("overall_entropy", 5)),
            float(row_c.get("overall_entropy", 5)),
        ]
        oe_score, oe_trimmed = trimmed_mean(oe_vals, OVERALL_OUTLIER_THRESHOLD)
        oe_stats = rating_stats(oe_vals)
        consensus["overall_entropy"] = round(oe_score, 2)
        consensus["overall_entropy_raw"] = oe_stats["raw"]
        consensus["label_std"] = oe_stats["std"]
        consensus["overall_entropy_spread"] = oe_stats["spread"]
        consensus["n_raters"] = oe_stats["n_raters"]
        if oe_trimmed:
            trim_counts["overall_entropy"] += 1
        dim_deviations["overall_entropy"].append(max(oe_vals) - min(oe_vals))
        overall_scores.append(oe_score)

        # Concatenate reasoning
        reasoning_parts = []
        for pass_name, row in [("A", row_a), ("B", row_b), ("C", row_c)]:
            r = row.get("reasoning", "").strip()
            if r:
                reasoning_parts.append(f"[{pass_name}] {r}")
        consensus["reasoning"] = " | ".join(reasoning_parts)

        consensus_rows.append(consensus)

        # Raw ratings for analysis
        for pass_name, row in [("a", row_a), ("b", row_b), ("c", row_c)]:
            raw_row = {"index": i, "pass": pass_name}
            for dim in DIMENSIONS:
                raw_row[dim] = row.get(dim, "")
            raw_row["overall_entropy"] = row.get("overall_entropy", "")
            raw_row["reasoning"] = row.get("reasoning", "")
            raw_rows.append(raw_row)

    # Write consensus CSV
    consensus_path = DATA_DIR / "consensus_labels.csv"
    fieldnames = [
        "journal", "source", "source_file",
        "prediction_coherence", "prediction_coherence_raw",
        "prediction_coherence_std", "prediction_coherence_spread",
        "model_complexity", "model_complexity_raw",
        "model_complexity_std", "model_complexity_spread",
        "compression_progress", "compression_progress_raw",
        "compression_progress_std", "compression_progress_spread",
        "belief_integration", "belief_integration_raw",
        "belief_integration_std", "belief_integration_spread",
        "precision_weighting", "precision_weighting_raw",
        "precision_weighting_std", "precision_weighting_spread",
        "overall_entropy", "overall_entropy_raw", "label_std",
        "overall_entropy_spread", "n_raters", "reasoning",
    ]
    with open(consensus_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(consensus_rows)
    print(f"\n  Consensus labels: {consensus_path} ({len(consensus_rows)} entries)")

    # Write raw ratings CSV
    raw_path = DATA_DIR / "raw_ratings.csv"
    raw_fields = ["index", "pass"] + DIMENSIONS + ["overall_entropy", "reasoning"]
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"  Raw ratings: {raw_path} ({len(raw_rows)} rows)")

    # Compute agreement statistics
    stats: dict = {
        "n_entries": n,
        "dimensions": {},
        "overall_entropy": {},
    }

    for dim in DIMENSIONS + ["overall_entropy"]:
        devs = dim_deviations[dim]
        mean_dev = statistics.mean(devs) if devs else 0
        max_dev = max(devs) if devs else 0
        pct_trimmed = trim_counts[dim] / n * 100 if n > 0 else 0

        key = "overall_entropy" if dim == "overall_entropy" else "dimensions"
        target = stats[key] if dim == "overall_entropy" else stats["dimensions"].setdefault(dim, {})
        if dim != "overall_entropy":
            target = stats["dimensions"][dim] = {}
        entry = {
            "mean_abs_deviation": round(mean_dev, 3),
            "max_deviation": round(max_dev, 1),
            "pct_trimmed": round(pct_trimmed, 1),
            "trim_count": trim_counts[dim],
        }
        if dim == "overall_entropy":
            stats["overall_entropy"] = entry
        else:
            stats["dimensions"][dim] = entry

    # Overall entropy distribution
    if overall_scores:
        bins = {f"{i}-{i+1}": 0 for i in range(1, 10)}
        for s in overall_scores:
            for lo in range(1, 10):
                if lo <= s < lo + 1:
                    bins[f"{lo}-{lo+1}"] += 1
                    break
            else:
                bins["9-10"] += 1  # catch 10.0
        stats["overall_entropy_distribution"] = bins
        stats["overall_entropy_mean"] = round(statistics.mean(overall_scores), 2)
        stats["overall_entropy_std"] = round(statistics.stdev(overall_scores), 2) if len(overall_scores) > 1 else 0

    # Inter-rater correlation (Pearson between each pair of passes)
    pass_scores: dict[str, list[float]] = {"a": [], "b": [], "c": []}
    for i in range(n):
        for pn, pdata in [("a", passes["a"]), ("b", passes["b"]), ("c", passes["c"])]:
            pass_scores[pn].append(float(pdata[i].get("overall_entropy", 5)))

    def pearson(x: list[float], y: list[float]) -> float:
        n_p = len(x)
        if n_p < 3:
            return 0.0
        mx = sum(x) / n_p
        my = sum(y) / n_p
        dx = [xi - mx for xi in x]
        dy = [yi - my for yi in y]
        num = sum(dxi * dyi for dxi, dyi in zip(dx, dy))
        den_x = sum(dxi**2 for dxi in dx) ** 0.5
        den_y = sum(dyi**2 for dyi in dy) ** 0.5
        if den_x * den_y == 0:
            return 0.0
        return num / (den_x * den_y)

    stats["inter_rater_correlations"] = {
        "a_vs_b": round(pearson(pass_scores["a"], pass_scores["b"]), 4),
        "a_vs_c": round(pearson(pass_scores["a"], pass_scores["c"]), 4),
        "b_vs_c": round(pearson(pass_scores["b"], pass_scores["c"]), 4),
    }

    # Save stats
    stats_path = DATA_DIR / "agreement_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Agreement stats: {stats_path}")

    # Print summary
    print(f"\n{'='*72}")
    print("AGREEMENT SUMMARY")
    print(f"{'='*72}")
    print(f"\n  Inter-rater correlations (overall_entropy):")
    for pair, r in stats["inter_rater_correlations"].items():
        print(f"    {pair}: {r}")

    print(f"\n  Outlier trimming rates:")
    for dim in DIMENSIONS + ["overall_entropy"]:
        tc = trim_counts[dim]
        pct = tc / n * 100 if n > 0 else 0
        print(f"    {dim}: {tc}/{n} ({pct:.1f}%)")

    print(f"\n  Overall entropy consensus: mean={stats.get('overall_entropy_mean', 0)}, "
          f"std={stats.get('overall_entropy_std', 0)}")
    print(f"{'='*72}")


if __name__ == "__main__":
    aggregate()
