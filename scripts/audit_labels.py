#!/usr/bin/env python3
"""Audit MES label quality by checking sub-dimension consistency.

Analyzes journals_labeled.csv to measure internal consistency of LLM labels:
- Sub-dimension ↔ overall_entropy correlations
- Composite score vs overall agreement
- Outlier detection (sub-dimensions conflict with overall)
- Distribution analysis
- Inter-dimension correlation matrix (halo effect check)

Usage:
    python scripts/audit_labels.py
    python scripts/audit_labels.py --input path/to/journals_labeled.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

SUB_DIMS = [
    "continuity",
    "topic_focus",
    "contradiction_integration",
    "cognitive_clarity",
    "narrative_closure",
]

OUTLIER_THRESHOLD = 3  # |composite - overall| > this = outlier


def pearson_r(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    dx = [xi - mx for xi in x]
    dy = [yi - my for yi in y]
    num = sum(a * b for a, b in zip(dx, dy))
    denom = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if denom == 0:
        return 0.0
    return num / denom


def load_labels(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                entry = {
                    "overall_entropy": int(row["overall_entropy"]),
                }
                for dim in SUB_DIMS:
                    entry[dim] = int(row[dim])
                rows.append(entry)
            except (ValueError, KeyError):
                continue
    return rows


def compute_composite(entry: dict) -> float:
    """Map sub-dimension average (1-5, high=organized) to 1-10 entropy scale."""
    sub_avg = sum(entry[d] for d in SUB_DIMS) / len(SUB_DIMS)
    # sub_avg 5 (organized) → entropy 1, sub_avg 1 (fragmented) → entropy 9
    return (5 - sub_avg) * 2.0 + 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MES label quality")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("journals_labeled.csv"),
        help="Input labeled CSV (default: journals_labeled.csv)",
    )
    args = parser.parse_args()

    rows = load_labels(args.input)
    n = len(rows)
    if n == 0:
        print("No valid entries found.")
        return

    overall = [r["overall_entropy"] for r in rows]
    composites = [compute_composite(r) for r in rows]

    print(f"=== MES Label Quality Audit ({n} entries) ===\n")

    # 1. Sub-dimension → overall_entropy correlations
    print("Sub-dimension → overall_entropy correlations:")
    print("  (negative = high sub-score predicts low entropy, as expected)\n")
    for dim in SUB_DIMS:
        vals = [float(r[dim]) for r in rows]
        r = pearson_r(vals, [float(o) for o in overall])
        strength = "strong" if abs(r) > 0.6 else "moderate" if abs(r) > 0.4 else "weak"
        print(f"  {dim:<32s} r = {r:+.3f}  ({strength})")

    # 2. Composite score correlation
    comp_r = pearson_r(composites, [float(o) for o in overall])
    print(f"\nComposite score → overall_entropy: r = {comp_r:+.3f}")
    if comp_r > 0.8:
        print("  Assessment: EXCELLENT - overall is well-explained by sub-dimensions")
    elif comp_r > 0.6:
        print("  Assessment: GOOD - reasonable consistency")
    elif comp_r > 0.4:
        print("  Assessment: MODERATE - some divergence between sub-dims and overall")
    else:
        print("  Assessment: POOR - sub-dimensions and overall disagree significantly")

    # 3. Outlier detection
    outliers = []
    for i, (comp, ov) in enumerate(zip(composites, overall)):
        diff = abs(comp - ov)
        if diff > OUTLIER_THRESHOLD:
            outliers.append((i, comp, ov, diff))

    print(f"\nOutliers (|composite - overall| > {OUTLIER_THRESHOLD}): {len(outliers)} / {n} ({len(outliers)/n*100:.1f}%)")
    if outliers:
        print("  Top 5 worst:")
        outliers.sort(key=lambda x: -x[3])
        for idx, comp, ov, diff in outliers[:5]:
            r = rows[idx]
            subs = ", ".join(f"{d[:4]}={r[d]}" for d in SUB_DIMS)
            print(f"    Entry {idx}: composite={comp:.1f}, overall={ov}, diff={diff:.1f} [{subs}]")

    # 4. Distribution analysis
    print("\nDistribution of overall_entropy:")
    buckets = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
    for lo, hi in buckets:
        count = sum(1 for o in overall if lo <= o <= hi)
        bar = "#" * (count * 40 // n)
        print(f"  {lo}-{hi:>2}: {count:>4} ({count/n*100:5.1f}%)  {bar}")

    print("\nPer-dimension distributions (mean / std):")
    for dim in SUB_DIMS:
        vals = [r[dim] for r in rows]
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
        dist = {v: sum(1 for x in vals if x == v) for v in range(1, 6)}
        dist_str = "  ".join(f"{v}:{dist.get(v, 0):>3}" for v in range(1, 6))
        print(f"  {dim:<32s} mean={mean:.2f}  std={std:.2f}  [{dist_str}]")

    # 5. Inter-dimension correlation matrix (halo effect check)
    print("\nInter-dimension correlation matrix:")
    print(f"  {'':>32s}", end="")
    for dim in SUB_DIMS:
        print(f"  {dim[:6]:>6s}", end="")
    print()
    for dim_a in SUB_DIMS:
        vals_a = [float(r[dim_a]) for r in rows]
        print(f"  {dim_a:<32s}", end="")
        for dim_b in SUB_DIMS:
            vals_b = [float(r[dim_b]) for r in rows]
            r = pearson_r(vals_a, vals_b)
            print(f"  {r:+.3f}", end="")
        print()

    # Check for halo effect
    off_diag = []
    for i, dim_a in enumerate(SUB_DIMS):
        for j, dim_b in enumerate(SUB_DIMS):
            if i < j:
                vals_a = [float(r[dim_a]) for r in rows]
                vals_b = [float(r[dim_b]) for r in rows]
                off_diag.append(pearson_r(vals_a, vals_b))

    avg_inter = sum(off_diag) / len(off_diag)
    print(f"\n  Average inter-dimension correlation: {avg_inter:+.3f}")
    if avg_inter > 0.8:
        print("  WARNING: Very high inter-correlation → likely halo effect (LLM not distinguishing dimensions)")
    elif avg_inter > 0.6:
        print("  NOTE: Moderate inter-correlation → some halo effect, but dimensions partially distinct")
    else:
        print("  GOOD: Dimensions appear to capture distinct aspects")

    # Overall assessment
    print("\n" + "=" * 50)
    issues = []
    if comp_r < 0.6:
        issues.append("low composite-overall consistency")
    if len(outliers) / n > 0.10:
        issues.append(f"high outlier rate ({len(outliers)/n*100:.0f}%)")
    if avg_inter > 0.8:
        issues.append("halo effect in sub-dimensions")

    low_count = sum(1 for o in overall if o <= 4)
    if low_count / n > 0.65:
        issues.append(f"skewed distribution ({low_count/n*100:.0f}% in 1-4 range)")

    if not issues:
        print("OVERALL ASSESSMENT: GOOD — Labels appear internally consistent")
    elif len(issues) <= 1:
        print(f"OVERALL ASSESSMENT: NEEDS ATTENTION — {'; '.join(issues)}")
    else:
        print(f"OVERALL ASSESSMENT: POOR — {'; '.join(issues)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
