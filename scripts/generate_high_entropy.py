#!/usr/bin/env python3
"""
Generate high-entropy synthetic journals using few-shot examples from real data.

Uses GPT-5.4 with real high-entropy journal examples as demonstrations,
then labels with Claude (cross-model) to avoid circular labeling.

Usage:
    OPENAI_API_KEY=sk-... uv run python scripts/generate_high_entropy.py \
        --output data/high_entropy_synthetic_for_labeling.csv \
        --total 250

Requires:
    - OPENAI_API_KEY environment variable
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Few-shot example extraction
# ---------------------------------------------------------------------------

JOURNALS_CSV = Path(__file__).parent.parent / "journals_labeled.csv"


def extract_few_shot_examples(
    journals_csv: Path,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Extract real high-entropy and low-entropy examples for few-shot prompting."""
    high_candidates: list[tuple[int, str]] = []
    low_candidates: list[tuple[int, str]] = []

    with open(journals_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entropy = int(row["overall_entropy"])
            text = row["journal"].strip()
            length = len(text)

            if entropy >= 7 and 150 <= length <= 1600:
                high_candidates.append((entropy, text))
            elif entropy == 2 and 200 <= length <= 700:
                low_candidates.append((entropy, text))

    # Prioritize highest entropy, deduplicate
    seen = set()
    unique_high = []
    for e, t in sorted(high_candidates, key=lambda x: -x[0]):
        if t not in seen:
            seen.add(t)
            unique_high.append((e, t))

    # Select diverse examples: 2x entropy-8+, rest entropy-7
    high_examples = [x for x in unique_high if x[0] >= 8][:3]
    high_examples += [x for x in unique_high if x[0] == 7][:9]
    low_examples = low_candidates[:3]

    return high_examples[:12], low_examples[:3]


# ---------------------------------------------------------------------------
# Persona and situation pools
# ---------------------------------------------------------------------------

OCCUPATIONS = [
    "student", "nurse", "warehouse worker", "barista", "teacher",
    "uber driver", "stay-at-home parent", "retail worker", "construction worker",
    "server", "security guard", "freelancer", "janitor", "caregiver",
    "fast food worker", "delivery driver", "receptionist",
]

HIGH_ENTROPY_STATES = [
    "spiraling with anxiety",
    "hasn't slept in 36 hours",
    "mid-panic attack",
    "drunk at 2am",
    "overwhelmed and shutting down",
    "racing thoughts won't stop",
    "dissociating slightly",
    "grief-stricken and raw",
    "manic energy, can't focus",
    "rage-journaling after a fight",
    "crying while writing",
    "brain fog from medication",
]

SITUATIONS = [
    "just had a terrible fight with partner",
    "got fired today, can't process it",
    "multiple crises happening at once — bills, kids, car broke down",
    "breakup and moving out at the same time",
    "writing at 3am after a nightmare",
    "on hold with hospital while kids are screaming",
    "texting and journaling at the same time during an argument",
    "found out partner cheated, world is collapsing",
    "deadline at work plus family emergency",
    "first day after a major loss, nothing makes sense",
    "stuck in a waiting room with bad news coming",
    "venting mid-conversation, half-texting half-journaling",
    "post-binge guilty and scattered",
    "trying to process something traumatic that just happened",
    "holiday alone, spiraling about everything",
]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You generate realistic journal entries that exhibit specific levels of mental disorganization. You are NOT writing ABOUT entropy or disorganization -- you are writing AS someone whose thoughts are structurally disorganized.

CRITICAL DISTINCTION:
- A well-written entry ABOUT stress = LOW entropy (organized narrative about hard topic)
- A STRUCTURALLY fragmented entry = HIGH entropy (the writing itself is disorganized)

Writing about multiple problems clearly and coherently is NOT high entropy.
Mid-thought breaks, topic crashes, missing punctuation, unresolved loops = high entropy.

You must NOT mention entropy, mental health scores, coherence, or any meta-analysis terms.
Do NOT use em dashes (—). Use regular dashes, ellipses, or just crash topics together.
Output ONLY a JSON object: {"journal_text": "..."}"""


def build_level_instructions(level: int) -> str:
    """Return level-specific structural requirements."""
    if level == 6:
        return (
            "Some topic drift and a few incomplete thoughts. One or two abrupt "
            "transitions where you jump topics without connecting them. Mostly "
            "coherent but with notable cracks in the flow. Maybe one sentence "
            "that trails off or restarts."
        )
    elif level == 7:
        return (
            "Frequent topic jumps WITHOUT transitions. Several incomplete sentences. "
            "Poor punctuation in places (run-ons, missing periods). Thoughts that "
            "loop back without resolution. Mix of short fragments and rambling "
            "sentences. Feels like the person keeps getting derailed."
        )
    elif level == 8:
        return (
            "Severe run-on sentences. Missing punctuation throughout. Thoughts crash "
            "into each other without any transition. Multiple restarts ('wait no I mean'). "
            "No clear narrative arc. Topics appear and disappear randomly. "
            "Entry ends abruptly mid-thought or trails off."
        )
    elif level == 9:
        return (
            "Near stream-of-consciousness. Sentences barely connect to each other. "
            "Extreme punctuation chaos (excessive ellipses, random caps, no periods). "
            "Emotional outbursts interrupting unrelated thoughts. Words and phrases "
            "without complete sentence structure. No closure whatsoever. "
            "Feels like someone dumping raw unfiltered brain noise."
        )
    elif level == 10:
        return (
            "Extreme fragmentation. Telegram-style fragments. Barely coherent "
            "connection between any two adjacent sentences. Could be a list of "
            "disconnected thoughts, half-finished sentences, or raw emotional "
            "fragments. Very brief, chaotic, or feels like random notes. "
            "Absolutely no narrative structure."
        )
    return ""


def build_user_prompt(
    target_level: int,
    persona: str,
    situation: str,
    high_examples: list[tuple[int, str]],
    low_examples: list[tuple[int, str]],
) -> str:
    """Build the generation prompt with few-shot examples."""
    parts = []

    parts.append("## What makes writing HIGH entropy (structurally disorganized):\n")
    parts.append(
        "1. MID-SENTENCE BREAKS: Thoughts stop and restart before completion\n"
        '   "I was going to call her but then wait did I leave the stove on"\n\n'
        "2. TOPIC COLLISIONS: Unrelated topics crash together without transition\n"
        '   NOT: "Work was hard. Now about my relationship..." (that\'s organized)\n'
        '   YES: "Work was hard and I keep thinking about mom called and I don\'t should exercise"\n\n'
        "3. MISSING/CHAOTIC PUNCTUATION: Run-on sentences, excessive ellipses, no periods\n\n"
        "4. UNRESOLVED LOOPS: Circling back without progression\n"
        '   "I should call him... anyway the groceries... but I should really call him"\n\n'
        '5. HEDGING AND RESTARTS: "I mean, no, wait, what I meant was..."\n\n'
        "6. ABSENT CLOSURE: Entry ends mid-thought or just stops\n\n"
        "7. SURFACE-LEVEL TOPIC HOPPING: Brief mentions of many things, development of none\n"
    )

    parts.append("\n## Real examples of HIGH entropy writing:\n")
    for entropy, text in high_examples:
        parts.append(f"\n[Entropy {entropy}/10]:\n{text}\n")

    parts.append("\n## Real examples of LOW entropy writing (for contrast):\n")
    for entropy, text in low_examples:
        parts.append(f"\n[Entropy {entropy}/10]:\n{text}\n")

    word_range = {6: "100-300", 7: "80-400", 8: "60-400", 9: "40-300", 10: "20-200"}

    parts.append(f"\n## Your task:\n\n")
    parts.append(
        f"Generate ONE journal entry at approximately entropy level {target_level}/10.\n\n"
        f"Persona: {persona}\n"
        f"Situation: {situation}\n\n"
        f"The entry should be {word_range.get(target_level, '80-300')} words.\n\n"
        f"Structural requirements for entropy {target_level}:\n"
        f"{build_level_instructions(target_level)}\n\n"
        "IMPORTANT: Do NOT fix grammar, punctuation, or sentence structure. "
        "Real high-entropy journals have typos, missing periods, run-on sentences, "
        "and incomplete thoughts. Your output should too. Do NOT write a polished "
        "reflection on difficult emotions — write as someone whose thoughts are "
        "genuinely scattered and disorganized.\n\n"
        'Return ONLY: {"journal_text": "..."}'
    )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

BANNED_TERMS = frozenset({
    "entropy", "coherence score", "mes score", "mental entropy",
    "disorganization score", "fragmentation level",
    "as an ai", "i'm an ai", "language model",
})


def validate_entry(text: str, target_level: int) -> bool:
    """Validate a generated journal entry."""
    if not text or len(text) < 50:
        return False
    if len(text) > 3000:
        return False

    text_lower = text.lower()
    for term in BANNED_TERMS:
        if term in text_lower:
            return False

    return True


def extract_journal_text(response_text: str) -> str | None:
    """Extract journal_text from LLM JSON response."""
    text = response_text.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
        return data.get("journal_text", "").strip()
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
                return data.get("journal_text", "").strip()
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate high-entropy synthetic journals using few-shot examples"
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default=Path("data/high_entropy_synthetic_for_labeling.csv"),
        help="Output CSV file (default: data/high_entropy_synthetic_for_labeling.csv)",
    )
    parser.add_argument(
        "--journals-csv", type=Path, default=JOURNALS_CSV,
        help="Source CSV for few-shot examples (default: journals_labeled.csv)",
    )
    parser.add_argument(
        "--total", "-n", type=int, default=250,
        help="Total entries to generate (split across levels 6-10)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="gpt-5.4",
        help="OpenAI model to use (default: gpt-5.4)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries per entry (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for persona/situation sampling (default: 42)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Checkpoint file (default: <output>.checkpoint.jsonl)",
    )

    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = args.output.with_suffix(".checkpoint.jsonl")

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    try:
        import openai
    except ImportError:
        print("Error: openai package not installed. Run: uv add openai")
        sys.exit(1)

    client = openai.OpenAI(timeout=60.0, max_retries=3)
    rng = random.Random(args.seed)

    # Extract few-shot examples
    print(f"Extracting few-shot examples from {args.journals_csv}...")
    high_examples, low_examples = extract_few_shot_examples(args.journals_csv)
    print(f"  {len(high_examples)} high-entropy examples (levels {set(e for e, _ in high_examples)})")
    print(f"  {len(low_examples)} low-entropy contrast examples")

    if len(high_examples) < 5:
        print("Error: Not enough high-entropy examples found. Need at least 5.")
        sys.exit(1)

    # Load checkpoint
    completed: dict[str, str] = {}  # key -> journal_text
    if args.checkpoint.exists():
        with open(args.checkpoint, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    completed[data["key"]] = data["journal"]
        print(f"Loaded {len(completed)} entries from checkpoint")

    # Build generation plan: split total across levels 6-10
    per_level = args.total // 5
    remainder = args.total % 5
    target_counts = {}
    for i, level in enumerate(range(6, 11)):
        target_counts[level] = per_level + (1 if i < remainder else 0)

    print(f"\nGeneration plan: {args.total} total entries")
    for level, count in target_counts.items():
        print(f"  Level {level}: {count} entries")
    print()

    # Generate
    all_entries: list[dict[str, str | int]] = []
    total_generated = 0
    total_failed = 0

    for target_level, count in target_counts.items():
        print(f"--- Generating {count} entries at entropy level {target_level} ---")

        for i in range(count):
            key = f"L{target_level}_{i}"

            # Skip if already in checkpoint
            if key in completed:
                all_entries.append({
                    "journal": completed[key],
                    "target_entropy": target_level,
                })
                total_generated += 1
                continue

            # Sample persona and situation
            occupation = rng.choice(OCCUPATIONS)
            state = rng.choice(HIGH_ENTROPY_STATES)
            situation = rng.choice(SITUATIONS)
            age = rng.choice(["19", "23", "28", "34", "42", "55"])
            persona = f"{age}-year-old {occupation}, {state}"

            prompt = build_user_prompt(
                target_level, persona, situation,
                high_examples, low_examples,
            )

            # Generate with retries
            success = False
            for attempt in range(args.max_retries):
                try:
                    token_param = "max_completion_tokens" if args.model.startswith("gpt-5") else "max_tokens"
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=args.temperature,
                        **{token_param: 1200},
                    )

                    raw_text = response.choices[0].message.content.strip()
                    journal_text = extract_journal_text(raw_text)

                    if journal_text and validate_entry(journal_text, target_level):
                        # Save to checkpoint
                        with open(args.checkpoint, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "key": key,
                                "journal": journal_text,
                                "target_entropy": target_level,
                            }) + "\n")

                        all_entries.append({
                            "journal": journal_text,
                            "target_entropy": target_level,
                        })
                        total_generated += 1
                        success = True

                        pct = (total_generated) / args.total * 100
                        print(f"  [{total_generated}/{args.total}] ({pct:.0f}%) L{target_level}_{i} "
                              f"len={len(journal_text)} chars")
                        break
                    else:
                        reason = "too short" if journal_text and len(journal_text) < 50 else "parse failed" if not journal_text else "validation failed"
                        if attempt < args.max_retries - 1:
                            print(f"  L{target_level}_{i} retry {attempt+1} ({reason})")

                except Exception as e:
                    if attempt < args.max_retries - 1:
                        print(f"  L{target_level}_{i} retry {attempt+1} (error: {e})")
                        time.sleep(2 ** attempt)
                    else:
                        print(f"  L{target_level}_{i} FAILED: {e}")

            if not success:
                total_failed += 1

            time.sleep(args.delay)

    # Save final CSV
    print(f"\n{'='*60}")
    print(f"Generated: {total_generated}, Failed: {total_failed}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["journal"])
        writer.writeheader()
        for entry in all_entries:
            writer.writerow({"journal": entry["journal"]})

    print(f"Saved {len(all_entries)} entries to {args.output}")

    # Distribution summary
    level_counts: dict[int, int] = {}
    for entry in all_entries:
        lvl = entry["target_entropy"]
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    print("\nTarget entropy distribution:")
    for lvl in sorted(level_counts):
        print(f"  Level {lvl}: {level_counts[lvl]}")


if __name__ == "__main__":
    main()
