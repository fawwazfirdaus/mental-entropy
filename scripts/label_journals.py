#!/usr/bin/env python3
"""
Synthetic labeling script for journal entries using LLM.

This script labels journal entries with Mental Entropy Score (MES) dimensions
to create training data for the MES model.

Usage:
    uv run python scripts/label_journals.py --input journals_raw.csv --output journals_labeled.csv

Requires:
    - ANTHROPIC_API_KEY environment variable (for Claude)
    - OR OPENAI_API_KEY environment variable (for GPT-4)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Try to import API clients
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# =============================================================================
# LABELING RUBRIC - Aligned with MES Philosophy
# =============================================================================

LABELING_RUBRIC = """
You are an expert in cognitive science and information processing theory, specifically Karl Friston's Free Energy Principle. Your task is to analyze journal entries and rate them on dimensions related to Mental Entropy - a measure of how well the mind's internal model is functioning, as revealed through writing.

## Core Concept

The mind is an information-processing system that builds predictive models of reality. Mental Entropy measures how well that system is functioning based on writing structure. A well-functioning mind produces writing that is:
- Predictable in flow (good generative model)
- Compressed around key themes (parsimonious model)
- Moving toward insight (compression progress)
- Internally consistent (no contradictory beliefs)
- Expressed with appropriate confidence (well-calibrated precision)

You are measuring the ORGANIZATION of the mind's information processing, not judging the person or their feelings.

## Key Distinctions

### Intentional Topic Switches vs Fragmentation
- ORGANIZED: "Work was stressful. [coherent paragraph about work] Now about my relationship... [coherent paragraph about relationship]"
- FRAGMENTED: "Work was stressful and I keep thinking about—mom called and I don't know why I—should exercise but anyway the meeting—"

Multiple organized topics = LOW entropy. Mid-thought fragmentation = HIGH entropy.

### Insight vs Surface Description
- TRUE INSIGHT: The writer connects experiences, identifies patterns, reaches a new understanding
- SURFACE: Lists events or states feelings without synthesizing or making meaning

## Rating Dimensions

### 1. PREDICTION_COHERENCE (1-5)
How smoothly do thoughts flow from one to the next? A good generative model produces connected, predictable mental output. Breaks in coherence represent prediction errors the model failed to suppress.

1 = No prediction: Nearly random sequence of thoughts, each sentence seems independent, no discernible flow
2 = Poor prediction: Frequent abrupt shifts, many sentences come from nowhere, reader cannot predict what comes next
3 = Moderate prediction: Some unexpected jumps between thoughts, but overall direction is maintained
4 = Good prediction: Mostly smooth flow with occasional small topic shifts that are still comprehensible
5 = Excellent prediction: Every sentence follows naturally from the previous, reader can anticipate direction

### 2. MODEL_COMPLEXITY (1-5)
How many distinct mental threads is the writer juggling? A parsimonious model compresses experience into few coherent themes. High complexity means many unrelated threads competing for attention — the model hasn't found a unifying compression.

1 = Uncompressed: So many topics that no structure is apparent, stream of random associations
2 = High complexity: Many distinct topics with unclear relationships, writer pulled in many directions
3 = Moderate complexity: Several distinct topics, some organization but the entry sprawls
4 = Well compressed: Two to three themes, clearly delineated, with purposeful transitions
5 = Maximally compressed: One or two clear themes everything connects to, focused and purposeful

### 3. COMPRESSION_PROGRESS (1-5)
Does the writing move from raw experience toward understanding? This captures whether the mind is actively compressing reality — learning, integrating, making meaning. The narrative arc IS compression progress.

1 = No compression: Pure stream-of-consciousness or fragmented listing, no meaning-making, entry ends no closer to insight than it began
2 = Minimal compression: Mostly description or narration without reflection, writer recounts events without synthesizing
3 = Moderate compression: Attempts at reflection or synthesis, but incomplete — writer may identify a question without resolving it
4 = Good compression: Some insight or synthesis emerges, writer makes connections or draws conclusions, ending feels more resolved
5 = Strong compression: Clear movement from description to insight, writer connects experiences, identifies patterns, reaches new understanding

### 4. BELIEF_INTEGRATION (1-5)
How well does the writer handle conflicting thoughts, feelings, or beliefs? Not whether contradictions exist (they are normal), but whether they are acknowledged, held with awareness, and potentially integrated. Unresolved contradictions indicate a fragmented internal model.

1 = No integration: Multiple unresolved contradictions with zero awareness, opposing beliefs stated as absolute truths in sequence
2 = Poor integration: Contradictions stated but not acknowledged, feels like two different people wrote different sections
3 = Partial integration: Some contradictions acknowledged, others left hanging, mixed awareness
4 = Good integration: Contradictions acknowledged with awareness, even if not fully resolved ("I know these feelings don't quite fit together, but...")
5 = Full integration: Contradictions explicitly acknowledged AND integrated ("I feel both grateful and resentful, and both are valid"). Or genuinely no contradictions present.

### 5. PRECISION_WEIGHTING (1-5)
How confidently and clearly does the writer express their thoughts? This measures the stability and decisiveness of expression — not content, but how settled the mind is on its own model. Fragments, restarts, hedging, and erratic structure indicate poor precision estimation.

1 = No precision: Almost entirely fragments, false starts, and abandoned thoughts, extreme hedging or wild swings, punctuation noise
2 = Poor precision: Frequent fragments, restarts, hedging ("maybe", "I don't know", "I guess"), erratic sentence length, many abandoned thoughts
3 = Moderate precision: Noticeable hedging, some fragments or restarts, variable sentence length, writer seems to be thinking out loud
4 = Good precision: Mostly clear expression with occasional hedging or incomplete thoughts, minor processing difficulty
5 = Well-calibrated precision: Clear, decisive expression, complete sentences, appropriate confidence, no restarts or abandoned thoughts

### 6. OVERALL_ENTROPY (1-10)
Holistic assessment of mental entropy (1 = very organized, 10 = very disorganized)

Consider all dimensions above. This is NOT an average — use your judgment about how well this person's mind is functioning as an information-processing system, as revealed through their writing.

1-2 = Highly integrated: Smooth flow, compressed themes, genuine insight, resolved tensions, stable expression
3-4 = Mostly organized: Good structure with minor lapses, some compression progress, generally coherent
5-6 = Mixed: Some organized sections, some fragmented, inconsistent coherence
7-8 = Notably fragmented: Significant disorganization, hard to follow, unresolved contradictions, processing noise
9-10 = Severely fragmented: Chaotic expression, extreme disconnection, no compression, high cognitive noise

## Output Format

Respond with ONLY a JSON object (no markdown, no explanation before/after):

{
    "prediction_coherence": <1-5>,
    "model_complexity": <1-5>,
    "compression_progress": <1-5>,
    "belief_integration": <1-5>,
    "precision_weighting": <1-5>,
    "overall_entropy": <1-10>,
    "reasoning": "<2-3 sentences explaining your ratings, focusing on information-processing observations>"
}
"""


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class JournalLabel:
    """Labels for a single journal entry."""
    prediction_coherence: int
    model_complexity: int
    compression_progress: int
    belief_integration: int
    precision_weighting: int
    overall_entropy: int
    reasoning: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LabeledEntry:
    """A journal entry with its labels."""
    original_text: str
    labels: JournalLabel

    def to_dict(self) -> dict:
        return {
            "journal": self.original_text,
            **self.labels.to_dict()
        }


# =============================================================================
# LLM Clients
# =============================================================================

class AnthropicLabeler:
    """Label journals using Claude API."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", rubric: str | None = None):
        self.client = anthropic.Anthropic()
        self.model = model
        self.rubric = rubric or LABELING_RUBRIC

    def label(self, journal_text: str) -> JournalLabel:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"{self.rubric}\n\n## Journal Entry to Analyze:\n\n{journal_text}"
                }
            ]
        )

        response_text = message.content[0].text.strip()

        # Parse JSON response
        try:
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            data = json.loads(response_text)
            return JournalLabel(
                prediction_coherence=int(data.get("prediction_coherence", 3)),
                model_complexity=int(data.get("model_complexity", 3)),
                compression_progress=int(data.get("compression_progress", 3)),
                belief_integration=int(data.get("belief_integration", 5)),  # Default 5 = no contradictions
                precision_weighting=int(data.get("precision_weighting", 3)),
                overall_entropy=int(data.get("overall_entropy", 5)),
                reasoning=str(data.get("reasoning", "No reasoning provided"))
            )
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse LLM response: {e}\nResponse: {response_text}")


class OpenAILabeler:
    """Label journals using OpenAI API."""

    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI(timeout=60.0, max_retries=5)
        self.model = model

    def label(self, journal_text: str) -> JournalLabel:
        # GPT-5+ models use max_completion_tokens instead of max_tokens
        token_param = "max_completion_tokens" if self.model.startswith("gpt-5") else "max_tokens"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": LABELING_RUBRIC
                },
                {
                    "role": "user",
                    "content": f"## Journal Entry to Analyze:\n\n{journal_text}"
                }
            ],
            **{token_param: 1024},
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content.strip()

        try:
            data = json.loads(response_text)
            return JournalLabel(
                prediction_coherence=int(data.get("prediction_coherence", 3)),
                model_complexity=int(data.get("model_complexity", 3)),
                compression_progress=int(data.get("compression_progress", 3)),
                belief_integration=int(data.get("belief_integration", 5)),  # Default 5 = no contradictions
                precision_weighting=int(data.get("precision_weighting", 3)),
                overall_entropy=int(data.get("overall_entropy", 5)),
                reasoning=str(data.get("reasoning", "No reasoning provided"))
            )
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse LLM response: {e}\nResponse: {response_text}")


# =============================================================================
# Main Processing
# =============================================================================

def load_journals(input_path: Path) -> list[str]:
    """Load journal entries from CSV."""
    journals = []
    with open(input_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle common column names
            text = row.get('journal') or row.get('journal_text') or row.get('full_text') or row.get('text') or row.get('entry') or row.get('content')
            if text:
                journals.append(text.strip())
    return journals


def load_checkpoint(checkpoint_path: Path) -> dict[int, LabeledEntry]:
    """Load existing progress from checkpoint file."""
    if not checkpoint_path.exists():
        return {}

    labeled = {}
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                idx = data.pop('_index')
                journal = data.pop('journal')
                labels = JournalLabel(**{k: v for k, v in data.items() if k != '_index'})
                labeled[idx] = LabeledEntry(original_text=journal, labels=labels)
    return labeled


def save_checkpoint(checkpoint_path: Path, idx: int, entry: LabeledEntry):
    """Append a labeled entry to checkpoint file."""
    with open(checkpoint_path, 'a', encoding='utf-8') as f:
        data = entry.to_dict()
        data['_index'] = idx
        f.write(json.dumps(data) + '\n')


def save_final_output(output_path: Path, labeled_entries: list[LabeledEntry]):
    """Save all labeled entries to final CSV."""
    if not labeled_entries:
        return

    fieldnames = ['journal', 'prediction_coherence', 'model_complexity', 'compression_progress',
                  'belief_integration', 'precision_weighting', 'overall_entropy', 'reasoning']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in labeled_entries:
            writer.writerow(entry.to_dict())


def get_labeler(provider: str, model: str | None = None):
    """Get the appropriate labeler based on provider."""
    if provider == "anthropic":
        if not HAS_ANTHROPIC:
            print("Error: anthropic package not installed. Run: uv add anthropic")
            sys.exit(1)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)
        return AnthropicLabeler()

    elif provider == "openai":
        if not HAS_OPENAI:
            print("Error: openai package not installed. Run: uv add openai")
            sys.exit(1)
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set")
            sys.exit(1)
        return OpenAILabeler(model=model) if model else OpenAILabeler()

    else:
        raise ValueError(f"Unknown provider: {provider}")


def main():
    parser = argparse.ArgumentParser(
        description="Label journal entries with MES dimensions using LLM"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("journals_raw.csv"),
        help="Input CSV file with journal entries"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("journals_labeled.csv"),
        help="Output CSV file for labeled entries"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider to use (default: anthropic)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file for resuming (default: <output>.checkpoint.jsonl)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of entries to process (for testing)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Number of retries for failed API calls (default: 3)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name override (default: provider-specific default)"
    )

    args = parser.parse_args()

    # Set default checkpoint path
    if args.checkpoint is None:
        args.checkpoint = args.output.with_suffix('.checkpoint.jsonl')

    # Load input
    print(f"Loading journals from {args.input}...")
    journals = load_journals(args.input)
    print(f"Found {len(journals)} journal entries")

    if args.limit:
        journals = journals[:args.limit]
        print(f"Limited to {len(journals)} entries")

    # Load checkpoint
    labeled = load_checkpoint(args.checkpoint)
    print(f"Loaded {len(labeled)} previously labeled entries from checkpoint")

    # Initialize labeler
    print(f"Using {args.provider} as LLM provider")
    labeler = get_labeler(args.provider, model=args.model)

    # Process entries
    total = len(journals)
    for idx, journal_text in enumerate(journals):
        if idx in labeled:
            continue

        # Progress indicator
        pct = (idx + 1) / total * 100
        print(f"[{idx + 1}/{total}] ({pct:.1f}%) Processing entry...", end=" ", flush=True)

        # Retry logic
        for attempt in range(args.retry):
            try:
                labels = labeler.label(journal_text)
                entry = LabeledEntry(original_text=journal_text, labels=labels)
                labeled[idx] = entry
                save_checkpoint(args.checkpoint, idx, entry)
                print(f"entropy={labels.overall_entropy}")
                break
            except Exception as e:
                if attempt < args.retry - 1:
                    print(f"retry {attempt + 1}...", end=" ", flush=True)
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"FAILED: {e}")

        # Rate limiting
        time.sleep(args.delay)

    # Save final output
    print(f"\nSaving {len(labeled)} labeled entries to {args.output}...")

    # Sort by original index
    sorted_entries = [labeled[i] for i in sorted(labeled.keys())]
    save_final_output(args.output, sorted_entries)

    # Summary statistics
    if labeled:
        entropies = [e.labels.overall_entropy for e in labeled.values()]
        avg_entropy = sum(entropies) / len(entropies)
        print(f"\nSummary:")
        print(f"  Total labeled: {len(labeled)}")
        print(f"  Average entropy: {avg_entropy:.2f}")
        print(f"  Entropy distribution:")
        for bucket in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]:
            count = sum(1 for e in entropies if bucket[0] <= e <= bucket[1])
            print(f"    {bucket[0]}-{bucket[1]}: {count} ({count/len(entropies)*100:.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
