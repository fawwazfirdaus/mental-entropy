"""Global Disorder (GD) feature extraction for MES.

This module computes deterministic text-only features for global writing
disorganization that local embedding coherence can miss: topic resets, dangling
referents, abandoned setup, corrupted text residue, loops, and run-on chains.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mental_entropy.embedding.types import EmbeddingResult


GD_FEATURE_KEYS: frozenset[str] = frozenset(
    [
        "gd_topic_reset_rate",
        "gd_dangling_referent_rate",
        "gd_abandoned_setup_rate",
        "gd_block_reset_rate",
        "gd_abandoned_setup_count_norm",
        "gd_no_compression_ending",
        "gd_loop_repetition_rate",
        "gd_corruption_rate",
        "gd_runon_chain_rate",
        "gd_meta_scaffold_rate",
        "gd_unanchored_question_rate",
        "gd_disorder_marker_count_norm",
        "gd_disorder_markers_per_500w",
        "gd_single_use_entity_rate",
        "gd_entity_drift_score",
        "gd_compression_failure_score",
        "gd_global_disorder_score",
    ]
)

TOPIC_RESET_RE = re.compile(
    r"\b(anyway|also|wait|oh|speaking of|unrelated|side note|back to|"
    r"just remembered|nevermind|never mind)\b",
    re.IGNORECASE,
)
REFERENT_START_RE = re.compile(
    r"^\s*(it|this|that|these|those|he|she|they|him|her|them|his|their|"
    r"there|then)\b",
    re.IGNORECASE,
)
NAMED_ANCHOR_RE = re.compile(r"\b[A-Z][a-z]{2,}\b|\b(my|mom|dad|friend|partner|husband|wife|boyfriend|girlfriend)\b")
ABANDONED_SETUP_RE = re.compile(
    r"\b(throwaway|sorry for|apologies for|english isn't|english is not|"
    r"mobile|format|you'?ll see why|long story|for context|background)\b",
    re.IGNORECASE,
)
ENDING_NO_COMPRESSION_RE = re.compile(
    r"\b(i don't know|idk|whatever|anyway|i won't elaborate|nothing makes sense|"
    r"leave me alone|can't explain|cannot explain)\b",
    re.IGNORECASE,
)
CORRUPTION_RE = re.compile(
    r"(<[^>]+>|alt=|border=|href=|&[a-z]+;|take the .*quiz|quiz result|"
    r"\bhttp[s]?://|www\.|\\u[0-9a-fA-F]{4})",
    re.IGNORECASE,
)
QUESTION_RE = re.compile(r"\?")
WORD_RE = re.compile(r"[A-Za-z']+")
ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
REFLECTION_RE = re.compile(
    r"\b(realized?|understand|because|means?|learned|therefore|so now|"
    r"i know|i need|that helps|this tells me)\b",
    re.IGNORECASE,
)
ENTITY_STOPWORDS = frozenset(
    {
        "Also",
        "Anyway",
        "Apologies",
        "Background",
        "Daniel",
        "English",
        "For",
        "Jennifer",
        "Mark",
        "Reddit",
        "Sarah",
        "That",
        "There",
        "This",
        "Wait",
    }
) - {"Daniel", "Jennifer", "Mark", "Sarah"}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count / total)


def _has_dangling_referent(text: str, seen_anchor: bool) -> bool:
    if not REFERENT_START_RE.search(text):
        return False
    if REFLECTION_RE.search(text):
        return False
    return not seen_anchor


def _has_loop_repetition(text: str) -> bool:
    tokens = _tokens(text)
    if len(tokens) < 6:
        return False

    # Adjacent repeated words or short phrases.
    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i + 1]:
            return True
    for i in range(len(tokens) - 3):
        if tokens[i : i + 2] == tokens[i + 2 : i + 4]:
            return True

    # Excessive reuse of one content token within a sentence.
    counts: dict[str, int] = {}
    for token in tokens:
        if len(token) <= 3:
            continue
        counts[token] = counts.get(token, 0) + 1
    return any(count >= 3 for count in counts.values())


def _has_runon_chain(text: str) -> bool:
    tokens = _tokens(text)
    if len(tokens) < 25:
        return False
    lower = text.lower()
    connector_count = sum(
        lower.count(f" {connector} ")
        for connector in ("and", "but", "because", "so", "then", "like")
    )
    punctuation_count = sum(text.count(mark) for mark in ".;:")
    return connector_count >= 5 or punctuation_count == 0


def _is_meta_scaffold(text: str) -> bool:
    return bool(ABANDONED_SETUP_RE.search(text))


def _is_unanchored_question(text: str) -> bool:
    if not QUESTION_RE.search(text):
        return False
    tokens = _tokens(text)
    return len(tokens) <= 12


def _split_blocks(sentences: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for sentence in sentences:
        if not sentence.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(sentence)
    if current:
        blocks.append(current)
    return blocks


def _block_reset_rate(blocks: list[list[str]]) -> float:
    if len(blocks) <= 1:
        return 0.0

    resets = 0
    for block in blocks[1:]:
        first_sentence = block[0]
        if (
            TOPIC_RESET_RE.search(first_sentence)
            or REFERENT_START_RE.search(first_sentence)
            or ABANDONED_SETUP_RE.search(first_sentence)
        ):
            resets += 1
    return _rate(resets, len(blocks) - 1)


def _entity_counts(sentences: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sentence in sentences:
        for match in ENTITY_RE.findall(sentence):
            if match in ENTITY_STOPWORDS:
                continue
            counts[match] = counts.get(match, 0) + 1
    return counts


def gd_features(sentences: list[str]) -> dict[str, float]:
    """Compute GD features from sentence text.

    Args:
        sentences: Ordered sentence strings.

    Returns:
        A dict with exactly the keys in GD_FEATURE_KEYS.
    """
    content_sentences = [sentence for sentence in sentences if sentence.strip()]
    n = len(content_sentences)
    if n == 0:
        return _empty_features()

    blocks = _split_blocks(sentences)
    topic_resets = 0
    dangling_referents = 0
    abandoned_setups = 0
    loops = 0
    corruptions = 0
    runons = 0
    meta_scaffolds = 0
    unanchored_questions = 0
    seen_anchor = False

    for sentence in content_sentences:
        if TOPIC_RESET_RE.search(sentence):
            topic_resets += 1
        if _has_dangling_referent(sentence, seen_anchor):
            dangling_referents += 1
        if ABANDONED_SETUP_RE.search(sentence):
            abandoned_setups += 1
        if _has_loop_repetition(sentence):
            loops += 1
        if CORRUPTION_RE.search(sentence):
            corruptions += 1
        if _has_runon_chain(sentence):
            runons += 1
        if _is_meta_scaffold(sentence):
            meta_scaffolds += 1
        if _is_unanchored_question(sentence):
            unanchored_questions += 1
        if NAMED_ANCHOR_RE.search(sentence):
            seen_anchor = True

    ending = content_sentences[-1] if content_sentences else ""
    no_compression_ending = 1.0 if ENDING_NO_COMPRESSION_RE.search(ending) else 0.0
    block_reset_rate = _block_reset_rate(blocks)
    abandoned_setup_count_norm = float(min(abandoned_setups / 4.0, 1.0))
    word_count = max(sum(len(_tokens(sentence)) for sentence in content_sentences), 1)
    marker_count = (
        topic_resets
        + dangling_referents
        + abandoned_setups
        + loops
        + corruptions
        + runons
        + meta_scaffolds
        + unanchored_questions
        + int(no_compression_ending)
    )
    disorder_markers_per_500w = float(marker_count / word_count * 500.0)
    entity_counts = _entity_counts(content_sentences)
    entity_total = sum(entity_counts.values())
    single_use_entities = sum(1 for count in entity_counts.values() if count == 1)
    single_use_entity_rate = _rate(single_use_entities, entity_total)
    entity_drift_score = float(
        min(1.0, single_use_entity_rate * min(entity_total / 4.0, 1.0))
    )
    reflection_count = sum(
        1 for sentence in content_sentences if REFLECTION_RE.search(sentence)
    )
    reflection_absence = 1.0 if reflection_count == 0 else 0.0
    compression_failure_score = float(
        min(
            1.0,
            (0.45 * no_compression_ending)
            + (0.25 * abandoned_setup_count_norm)
            + (0.20 * block_reset_rate)
            + (0.10 * reflection_absence),
        )
    )

    values = {
        "gd_topic_reset_rate": _rate(topic_resets, n),
        "gd_dangling_referent_rate": _rate(dangling_referents, n),
        "gd_abandoned_setup_rate": _rate(abandoned_setups, n),
        "gd_block_reset_rate": block_reset_rate,
        "gd_abandoned_setup_count_norm": abandoned_setup_count_norm,
        "gd_no_compression_ending": no_compression_ending,
        "gd_loop_repetition_rate": _rate(loops, n),
        "gd_corruption_rate": _rate(corruptions, n),
        "gd_runon_chain_rate": _rate(runons, n),
        "gd_meta_scaffold_rate": _rate(meta_scaffolds, n),
        "gd_unanchored_question_rate": _rate(unanchored_questions, n),
        "gd_disorder_markers_per_500w": disorder_markers_per_500w,
        "gd_single_use_entity_rate": single_use_entity_rate,
        "gd_entity_drift_score": entity_drift_score,
        "gd_compression_failure_score": compression_failure_score,
    }
    values["gd_disorder_marker_count_norm"] = float(min(marker_count / 8.0, 1.0))
    values["gd_global_disorder_score"] = _global_disorder_score(values)
    return values


def gd_features_from_result(result: "EmbeddingResult") -> dict[str, float]:
    """Compute GD features from an EmbeddingResult."""
    return gd_features([sentence.text for sentence in result.sentences])


def _global_disorder_score(values: dict[str, float]) -> float:
    weighted = (
        0.16 * values["gd_topic_reset_rate"]
        + 0.13 * values["gd_dangling_referent_rate"]
        + 0.12 * values["gd_abandoned_setup_rate"]
        + 0.13 * values["gd_no_compression_ending"]
        + 0.13 * values["gd_loop_repetition_rate"]
        + 0.30 * values["gd_corruption_rate"]
        + 0.10 * values["gd_runon_chain_rate"]
        + 0.05 * values["gd_meta_scaffold_rate"]
        + 0.03 * values["gd_unanchored_question_rate"]
        + 0.20 * values["gd_disorder_marker_count_norm"]
        + 0.10 * values["gd_block_reset_rate"]
        + 0.08 * values["gd_abandoned_setup_count_norm"]
        + 0.12 * values["gd_entity_drift_score"]
        + 0.16 * values["gd_compression_failure_score"]
        + 0.15 * min(values["gd_disorder_markers_per_500w"] / 20.0, 1.0)
    )
    return float(min(1.0, max(0.0, weighted)))


def _empty_features() -> dict[str, float]:
    return {key: 0.0 for key in GD_FEATURE_KEYS}
