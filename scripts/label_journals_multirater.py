"""Multi-rater labeling for MES label quality improvement.

Labels all entries 3 times using Claude Sonnet 4.6 with different prompt
perspectives, then aggregates to consensus scores via trimmed mean.

Three prompt variants:
  A - Structure Analyst: existing MES rubric (surface markers focus)
  B - Reader Experience: phenomenological/reader-centered framing
  C - Comparative Anchor: calibrated with 3 reference examples

Usage:
    # Test with 10 entries on pass A
    ANTHROPIC_API_KEY=... python scripts/label_journals_multirater.py --pass a --limit 10

    # Run all 3 passes
    ANTHROPIC_API_KEY=... python scripts/label_journals_multirater.py --pass all --delay 0.3

    # Resume a specific pass
    ANTHROPIC_API_KEY=... python scripts/label_journals_multirater.py --pass b --delay 0.3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from label_journals import AnthropicLabeler, LABELING_RUBRIC  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
MULTIRATER_DIR = DATA_DIR / "multirater"
ALL_ENTRIES_CSV = MULTIRATER_DIR / "all_entries.csv"

# Source CSVs in the same order as prepare.py
SOURCE_CSVS: list[tuple[Path, str, str]] = [
    (PROJECT_ROOT / "journals_labeled.csv", "human", "journals_labeled.csv"),
    (DATA_DIR / "additional_journals_labeled_gpt54.csv", "human", "additional_journals_labeled_gpt54.csv"),
    (DATA_DIR / "new_human_journals_labeled.csv", "human", "new_human_journals_labeled.csv"),
    (DATA_DIR / "synthetic_journals_labeled_gpt54.csv", "synthetic", "synthetic_journals_labeled_gpt54.csv"),
    (DATA_DIR / "high_entropy_labeled_gpt54.csv", "synthetic", "high_entropy_labeled_gpt54.csv"),
    (DATA_DIR / "high_entropy_synthetic_labeled_claude.csv", "synthetic", "high_entropy_synthetic_labeled_claude.csv"),
    (DATA_DIR / "new_human_journals_2_raw.csv", "human", "new_human_journals_2_raw.csv"),
    (DATA_DIR / "new_human_journals_3_raw.csv", "human", "new_human_journals_3_raw.csv"),
]

# Journal column name detection (same as label_journals.py)
JOURNAL_COLUMNS = ["journal", "journal_text", "full_text", "text", "entry", "content"]


# ---------------------------------------------------------------------------
# Prompt Variants
# ---------------------------------------------------------------------------

RUBRIC_A = LABELING_RUBRIC  # Existing rubric unchanged

RUBRIC_B = """
You are an experienced therapist and mindfulness practitioner who understands the mind as an information-processing system. You've read thousands of personal journals. Your task is to assess how well this person's mind is functioning as a predictive model, based on your experience of reading their writing.

## What You're Measuring

You are rating how well the mind's internal model is working, as revealed through writing structure. Focus on your subjective experience as a reader: Could you follow the predictions? Did the writing compress experience into understanding? Was the expression stable?

## Important Distinctions

A journal that covers multiple topics coherently (work, then relationships, then health) = GOOD internal model.
A journal that jumps mid-sentence between fragments of different thoughts = POOR internal model.

Someone processing difficult emotions clearly and reaching insight = LOW entropy.
Someone whose writing feels scattered, hedging, and going nowhere = HIGH entropy.

## Rating Dimensions

### 1. THOUGHT FLOW (prediction_coherence, 1-5)
How easily could you follow the writer's train of thought from one sentence to the next? Could you predict where the thought was going?

1 = Constantly lost: Each sentence felt disconnected, you couldn't anticipate anything
2 = Frequently lost: Lost the thread often, many sentences came from nowhere
3 = Sometimes lost: Could follow most of it but lost the thread at several points
4 = Mostly smooth: Easy to follow with only occasional surprises
5 = Effortless: Every sentence flowed naturally, you could anticipate the direction

### 2. FOCUS (model_complexity, 1-5)
How many mental threads was the writer juggling? Did they compress experience into a few clear themes?

1 = Overwhelmed: So many topics that no structure was apparent, mind pulled everywhere
2 = Scattered: Many topics bleeding into each other, hard to identify what mattered
3 = Moderate: Several topics, some organization but the entry sprawled
4 = Focused: Two to three clear themes, well-organized
5 = Laser-focused: One or two themes everything connected to, purposeful and compressed

### 3. INSIGHT (compression_progress, 1-5)
Did the writer move from describing experience to understanding it? Did insight emerge?

1 = Pure description: Just listed events or feelings without synthesizing anything
2 = Minimal insight: Mostly narration, no real connections or conclusions drawn
3 = Partial insight: Attempted reflection but didn't fully get there
4 = Good insight: Made connections, drew some conclusions, ending felt more resolved
5 = Deep insight: Clear movement from raw experience to genuine understanding, reached new perspective

### 4. INTEGRATION (belief_integration, 1-5)
When conflicting feelings or thoughts appeared, was the writer aware of the tension?

1 = Unaware contradictions: Said opposite things without seeming to notice
2 = Stuck: Noticed conflict but went back and forth without progress
3 = Partial awareness: Acknowledged some tensions but didn't fully address them
4 = Working through it: Held contradictions with awareness and nuance
5 = Genuine integration: Transformed contradictions into deeper understanding. Or no contradictions present.

### 5. CLARITY (precision_weighting, 1-5)
Did the writing feel confident and decisive, or uncertain and searching?

1 = Very strained: Constant fragments, false starts, extreme hedging, couldn't commit to what they were saying
2 = Notably unclear: Frequent hedging, several abandoned thoughts, erratic expression
3 = Mixed: Some clear passages, some uncertain or muddled ones
4 = Mostly clear: Confident expression with only minor hesitations
5 = Crystal clear: Decisive, well-formed expression throughout — the writer knew what they thought

### 6. OVERALL ENTROPY (overall_entropy, 1-10)
Based on your overall reading experience, how well was this person's mind functioning as an information-processing system?

1-2 = Excellent: Smooth flow, compressed themes, genuine insight, stable expression
3-4 = Good: Mostly organized with minor lapses
5-6 = Mixed: Parts clear, parts hard to follow
7-8 = Poor: Significant fragmentation, hard to follow, processing noise
9-10 = Very poor: Chaotic, no compression, extreme disconnection

## Output Format

Respond with ONLY a JSON object (no markdown, no explanation before/after):

{
    "prediction_coherence": <1-5>,
    "model_complexity": <1-5>,
    "compression_progress": <1-5>,
    "belief_integration": <1-5>,
    "precision_weighting": <1-5>,
    "overall_entropy": <1-10>,
    "reasoning": "<2-3 sentences about your reading experience of this entry>"
}
"""

RUBRIC_C = """
You are an expert rater calibrating journal entries on a Mental Entropy scale. Mental Entropy measures how well the mind's internal model is functioning as an information-processing system, based on writing structure.

## Calibration Examples

Before rating, study these three reference entries carefully. Use them as anchor points.

### Reference 1: LOW entropy (overall_entropy = 2)
"Today was productive and grounding. I started the morning with a 20-minute meditation, which helped me set a clear intention for the day. At work, I had a challenging meeting with the product team about our Q3 roadmap. We disagreed on priorities, but I stayed focused on presenting the data clearly. By the end, we found a compromise that everyone could support. I feel good about how I handled the tension — staying calm while still advocating for my perspective. This evening I'm going to call my sister. We haven't talked in two weeks and I want to check in about her new job."

This scores LOW because: smooth predictive flow (each thought follows naturally), compressed around clear themes (morning → work → evening), achieves insight about handling conflict, internal tension fully integrated, confident decisive expression.

### Reference 2: MODERATE entropy (overall_entropy = 5)
"Woke up feeling off today. Not sure why exactly. Work was fine I guess — had some meetings, got through my tasks. I keep thinking about what Sarah said last week though. She didn't mean it badly but it stuck with me. Maybe I'm overthinking it. I should probably exercise more, that usually helps. Speaking of which I need to reschedule my dentist appointment. Anyway, the weather's been nice at least. I want to start journaling more consistently but I never seem to stick with it. Tomorrow should be better."

This scores MODERATE because: some predictive flow but notable jumps (Sarah → exercise → dentist → weather), several scattered topics (moderate complexity), no real insight reached (low compression progress), hedging present ("I guess", "probably", "maybe"), trails off without compression.

### Reference 3: HIGH entropy (overall_entropy = 8)
"Can't stop thinking about everything and nothing at the same time. The interview went—I don't even know. Bad? Good? They smiled but who knows what that means. Mom called again, three times, I need to—why do I always procrastinate on calling her back when I know it'll just make the guilt worse? Rent is due and I haven't checked my account. I should meditate but I can't sit still. My therapist says I need to 'sit with discomfort' but that's easy for her to say when she's not the one whose brain won't shut—just remembered I forgot to email my professor about the extension. Everything feels like it's happening at once."

This scores HIGH because: no predictive flow (constant mid-thought breaks), extremely high model complexity (interview → mom → rent → meditation → professor), zero compression progress (no insight, just listing), unresolved contradictions ("should meditate but can't"), very poor precision (fragments, dashes, abandoned thoughts).

## Your Task

Rate the journal entry below by placing it relative to these three reference points. Use the same dimensions and scales:

### Dimensions (each 1-5):
1. **PREDICTION_COHERENCE**: How smoothly thoughts flow, how predictable the next thought is (1=random jumps, 5=smooth flow)
2. **MODEL_COMPLEXITY**: How well experience is compressed into few themes (1=too many scattered topics, 5=focused compression)
3. **COMPRESSION_PROGRESS**: Whether writing moves from description toward insight (1=no insight, 5=genuine understanding reached)
4. **BELIEF_INTEGRATION**: How well contradictions are handled with awareness (1=unresolved, 5=integrated or no contradictions)
5. **PRECISION_WEIGHTING**: How confidently and clearly thoughts are expressed (1=fragments/hedging, 5=decisive/clear)

### Overall (1-10):
6. **OVERALL_ENTROPY**: Holistic assessment of mental model functioning (1=excellent like Reference 1, 10=very poor like Reference 3)

## Output Format

Respond with ONLY a JSON object (no markdown, no explanation before/after):

{
    "prediction_coherence": <1-5>,
    "model_complexity": <1-5>,
    "compression_progress": <1-5>,
    "belief_integration": <1-5>,
    "precision_weighting": <1-5>,
    "overall_entropy": <1-10>,
    "reasoning": "<2-3 sentences explaining where this entry falls relative to the reference points>"
}
"""

PASS_RUBRICS: dict[str, str] = {
    "a": RUBRIC_A,
    "b": RUBRIC_B,
    "c": RUBRIC_C,
}


# ---------------------------------------------------------------------------
# Consolidated input creation
# ---------------------------------------------------------------------------

def create_consolidated_input() -> list[dict[str, str]]:
    """Read all source CSVs and create a single consolidated list.

    Returns list of dicts with keys: index, journal, source, source_file.
    Also writes to ALL_ENTRIES_CSV.
    """
    MULTIRATER_DIR.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, str]] = []
    idx = 0

    for csv_path, source, source_name in SOURCE_CSVS:
        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found, skipping.")
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Find journal text column
                journal_text = None
                for col in JOURNAL_COLUMNS:
                    if col in row and row[col].strip():
                        journal_text = row[col].strip()
                        break

                if not journal_text:
                    continue

                entries.append({
                    "index": str(idx),
                    "journal": journal_text,
                    "source": source,
                    "source_file": source_name,
                })
                idx += 1

    # Write consolidated CSV
    with open(ALL_ENTRIES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "journal", "source", "source_file"])
        writer.writeheader()
        writer.writerows(entries)

    print(f"  Consolidated {len(entries)} entries to {ALL_ENTRIES_CSV}")
    n_human = sum(1 for e in entries if e["source"] == "human")
    n_synth = sum(1 for e in entries if e["source"] == "synthetic")
    print(f"  Human: {n_human}, Synthetic: {n_synth}")

    return entries


def load_consolidated_input() -> list[dict[str, str]]:
    """Load consolidated entries (create if not exists)."""
    if ALL_ENTRIES_CSV.exists():
        with open(ALL_ENTRIES_CSV, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return create_consolidated_input()


# ---------------------------------------------------------------------------
# Per-pass labeling
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: Path) -> dict[int, dict]:
    """Load checkpoint entries, keyed by index."""
    if not checkpoint_path.exists():
        return {}
    results: dict[int, dict] = {}
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results[int(data["_index"])] = data
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def run_pass(
    pass_name: str,
    entries: list[dict[str, str]],
    model: str,
    delay: float,
    limit: int | None = None,
    retry: int = 3,
) -> None:
    """Run a single labeling pass with the given rubric variant."""
    rubric = PASS_RUBRICS[pass_name]
    output_csv = MULTIRATER_DIR / f"pass_{pass_name}_labeled.csv"
    checkpoint_path = MULTIRATER_DIR / f"pass_{pass_name}.checkpoint.jsonl"

    print(f"\n{'='*72}")
    print(f"PASS {pass_name.upper()}: {['Structure Analyst', 'Reader Experience', 'Comparative Anchor'][ord(pass_name) - ord('a')]}")
    print(f"{'='*72}")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    print(f"  Loaded {len(checkpoint)} entries from checkpoint")

    # Initialize labeler
    import anthropic  # noqa: F811
    labeler = AnthropicLabeler(model=model, rubric=rubric)
    print(f"  Model: {model}")
    print(f"  Rubric: Pass {pass_name.upper()} ({len(rubric)} chars)")

    # Determine entries to process
    to_process = entries[:limit] if limit else entries
    total = len(to_process)

    for i, entry in enumerate(to_process):
        idx = int(entry["index"])

        # Skip if already in checkpoint
        if idx in checkpoint:
            continue

        pct = (i + 1) / total * 100
        print(f"\r  [{i+1}/{total}] ({pct:.1f}%) Processing entry {idx}...", end="", flush=True)

        # Label with retry logic
        for attempt in range(retry):
            try:
                label = labeler.label(entry["journal"])
                break
            except Exception as e:
                if attempt < retry - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"\n    Retry {attempt+1}/{retry} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"\n    FAILED entry {idx} after {retry} attempts: {e}")
                    label = None

        if label is None:
            continue

        # Save to checkpoint
        result = {
            "journal": entry["journal"],
            "source": entry["source"],
            "source_file": entry["source_file"],
            "prediction_coherence": label.prediction_coherence,
            "model_complexity": label.model_complexity,
            "compression_progress": label.compression_progress,
            "belief_integration": label.belief_integration,
            "precision_weighting": label.precision_weighting,
            "overall_entropy": label.overall_entropy,
            "reasoning": label.reasoning,
            "_index": idx,
        }
        checkpoint[idx] = result

        with open(checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Print score inline
        print(f" entropy={label.overall_entropy}", end="")

        time.sleep(delay)

    print(f"\n  Pass {pass_name.upper()} complete: {len(checkpoint)} entries labeled")

    # Write final CSV
    fieldnames = [
        "journal", "source", "source_file",
        "prediction_coherence", "model_complexity", "compression_progress",
        "belief_integration", "precision_weighting", "overall_entropy", "reasoning",
    ]

    # Sort by index to maintain order
    sorted_results = sorted(checkpoint.values(), key=lambda x: int(x["_index"]))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_results)

    print(f"  Saved {len(sorted_results)} entries to {output_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-rater MES labeling")
    parser.add_argument("--pass", dest="pass_name", choices=["a", "b", "c", "all"],
                        default="all", help="Which pass to run (default: all)")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Anthropic model name (default: claude-sonnet-4-6)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between API calls in seconds (default: 0.3)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N entries (for testing)")
    parser.add_argument("--retry", type=int, default=3,
                        help="Max retries per entry (default: 3)")
    parser.add_argument("--rebuild-input", action="store_true",
                        help="Force rebuild of consolidated input CSV")
    args = parser.parse_args()

    print("=" * 72)
    print("MES Multi-Rater Labeling")
    print("=" * 72)

    # Create or load consolidated input
    if args.rebuild_input or not ALL_ENTRIES_CSV.exists():
        print("\nBuilding consolidated input...")
        entries = create_consolidated_input()
    else:
        print(f"\nLoading consolidated input from {ALL_ENTRIES_CSV}...")
        entries = load_consolidated_input()
        n_human = sum(1 for e in entries if e["source"] == "human")
        n_synth = sum(1 for e in entries if e["source"] == "synthetic")
        print(f"  {len(entries)} entries ({n_human} human, {n_synth} synthetic)")

    # Run passes
    passes = ["a", "b", "c"] if args.pass_name == "all" else [args.pass_name]

    for pass_name in passes:
        run_pass(
            pass_name=pass_name,
            entries=entries,
            model=args.model,
            delay=args.delay,
            limit=args.limit,
            retry=args.retry,
        )

    print(f"\n{'='*72}")
    print("All passes complete!")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
