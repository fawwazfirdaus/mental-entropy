"""Text normalization and sentence splitting for the embedding layer."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import TYPE_CHECKING

import spacy

if TYPE_CHECKING:
    from spacy.language import Language

_logger = logging.getLogger(__name__)

_nlp: Language | None = None
_using_trained_model: bool = False


def _get_nlp() -> Language:
    """
    Lazily initialize the spaCy pipeline.

    Attempts to load en_core_web_sm (trained sentence boundary detection).
    Falls back to blank multilingual + sentencizer if model unavailable.
    """
    global _nlp, _using_trained_model

    if _nlp is not None:
        return _nlp

    try:
        _nlp = spacy.load(
            "en_core_web_sm",
            disable=["tagger", "lemmatizer", "ner", "attribute_ruler"],
        )
        _using_trained_model = True
        _logger.debug("Loaded en_core_web_sm for sentence splitting")
    except OSError:
        _nlp = spacy.blank("xx")
        _nlp.add_pipe("sentencizer")
        _using_trained_model = False
        _logger.warning(
            "en_core_web_sm not found, falling back to rule-based sentencizer. "
            "Install with: uv add en-core-web-sm"
        )

    return _nlp


def is_using_trained_model() -> bool:
    """Return True if using en_core_web_sm, False if using fallback sentencizer."""
    _get_nlp()
    return _using_trained_model


def normalize_text(raw: str) -> str:
    """
    Lightly normalize raw journal text.

    - Strip leading and trailing whitespace.
    - Normalize line endings (CRLF -> LF, CR -> LF).
    - Collapse repeated spaces and tabs into a single space.
    - Preserve punctuation, emojis, capitalization, and newlines.
    """
    text = raw.strip()
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse spaces and tabs (not newlines) into single space
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _is_punctuation_only(s: str) -> bool:
    """
    Check if a string contains only Unicode punctuation (and whitespace).

    Returns False for strings containing emojis, letters, numbers, or symbols.
    """
    for char in s:
        if char.isspace():
            continue
        category = unicodedata.category(char)
        # P* = punctuation categories
        if not category.startswith("P"):
            return False
    return True


def _split_block_with_spacy(block: str, nlp: Language) -> list[str]:
    """Split a text block into sentences using spaCy sentencizer."""
    doc = nlp(block)
    sentences = [sent.text.strip() for sent in doc.sents]
    return [s for s in sentences if s]


def _split_block_with_regex(block: str) -> list[str]:
    """
    Fallback: split a text block on sentence-ending punctuation.

    Splits on . ? ! followed by whitespace.
    """
    parts = re.split(r"(?<=[.!?])\s+", block)
    return [p.strip() for p in parts if p.strip()]


# Patterns for fragment detection
_DASH_END_PATTERN = re.compile(r"(—|--)\s*$")
_ELLIPSIS_END_PATTERN = re.compile(r"\.{3,}\s*$")
_NO_TERMINAL_PUNCT_PATTERN = re.compile(r"[^.!?:;]\s*$")


def _split_on_fragments(text: str) -> list[str]:
    """
    Pre-split text on newlines where fragments/interruptions are detected.

    This handles cases where:
    - Line ends with em-dash (—) or double-dash (--) followed by newline
    - Line ends with ellipsis (...) followed by newline
    - Line lacks terminal punctuation followed by newline

    We ONLY split when the next line starts with a CAPITAL letter, indicating
    a new thought/sentence. This preserves legitimate continuations like:
    "I was thinking—\\nand then realized..." (lowercase = continuation)

    While correctly splitting fragments like:
    "I was thinking—\\nActually, never mind." (capital = new thought)
    """
    if not text:
        return []

    lines = text.split("\n")
    chunks: list[str] = []
    current_chunk: list[str] = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            # Blank line - flush current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            continue

        current_chunk.append(line_stripped)
        should_split = False

        # Find next non-empty line to check capitalization
        next_line = None
        for j in range(i + 1, len(lines)):
            nl = lines[j].strip()
            if nl:
                next_line = nl
                break

        # Only split if next line starts with capital letter (indicates new thought)
        if next_line and next_line[0].isupper():
            # Rule 1: Line ends with dash (— or --)
            if _DASH_END_PATTERN.search(line_stripped):
                should_split = True

            # Rule 2: Line ends with ellipsis (...)
            elif _ELLIPSIS_END_PATTERN.search(line_stripped):
                should_split = True

            # Rule 3: Line lacks terminal punctuation
            elif _NO_TERMINAL_PUNCT_PATTERN.search(line_stripped):
                should_split = True

        if should_split:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Flush remaining content
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def split_sentences(normalized_text: str) -> list[str]:
    """
    Split normalized text into sentence-like units.

    Strategy:
    1. Pre-split on newlines where fragments/interruptions are detected
       (lines ending with —, --, or ... followed by capital letter)
    2. Split by blank lines (double newlines) into blocks
    3. For each block, apply spaCy sentence segmentation
    4. If spaCy yields no sentences, fall back to regex split on . ? !
    5. Post-filter: remove empty strings and punctuation-only strings

    Returns an ordered list of clean sentence strings.
    """
    if not normalized_text:
        return []

    nlp = _get_nlp()

    # Pre-split on fragment boundaries (handles interrupted thoughts)
    pre_split_chunks = _split_on_fragments(normalized_text)

    sentences: list[str] = []

    for chunk in pre_split_chunks:
        # Split by blank lines within each chunk
        blocks = re.split(r"\n(?:[ \t]*\n)+", chunk)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Try spaCy first
            block_sentences = _split_block_with_spacy(block, nlp)

            # Fallback to regex if spaCy yields nothing
            if not block_sentences:
                block_sentences = _split_block_with_regex(block)

            # If still nothing, treat the whole block as a sentence
            if not block_sentences and block:
                block_sentences = [block]

            sentences.extend(block_sentences)

    # Post-filter: remove empty and punctuation-only strings
    filtered: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if _is_punctuation_only(s):
            continue
        filtered.append(s)

    return filtered


def split_sentences_with_blocks(normalized_text: str) -> list[tuple[str, int]]:
    """
    Split normalized text into sentence-like units, preserving block (paragraph) IDs.

    A block is a section of text separated by blank lines. Block IDs are 0-indexed
    and increment at each blank line boundary.

    This enables block-aware feature extraction:
    - Coherence breaks WITHIN a block suggest fragmentation (bad signal)
    - Coherence breaks BETWEEN blocks are expected transitions (neutral)

    Strategy:
    1. Split by blank lines into blocks
    2. For each block, extract sentences using spaCy (with regex fallback)
    3. Return (sentence_text, block_id) tuples

    Returns an ordered list of (sentence, block_id) tuples.
    """
    if not normalized_text:
        return []

    nlp = _get_nlp()

    # Split by blank lines into blocks (one or more blank lines = block boundary)
    raw_blocks = re.split(r"\n(?:[ \t]*\n)+", normalized_text)

    result: list[tuple[str, int]] = []
    block_id = 0

    for raw_block in raw_blocks:
        raw_block = raw_block.strip()
        if not raw_block:
            continue

        # Pre-split this block on fragment boundaries
        pre_split_chunks = _split_on_fragments(raw_block)

        block_sentences: list[str] = []
        for chunk in pre_split_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # Try spaCy first
            chunk_sentences = _split_block_with_spacy(chunk, nlp)

            # Fallback to regex if spaCy yields nothing
            if not chunk_sentences:
                chunk_sentences = _split_block_with_regex(chunk)

            # If still nothing, treat the whole chunk as a sentence
            if not chunk_sentences and chunk:
                chunk_sentences = [chunk]

            block_sentences.extend(chunk_sentences)

        # Post-filter and add to result with block_id
        sentences_added = 0
        for s in block_sentences:
            s = s.strip()
            if not s:
                continue
            if _is_punctuation_only(s):
                continue
            result.append((s, block_id))
            sentences_added += 1

        # Only increment block_id if we actually added sentences from this block
        if sentences_added > 0:
            block_id += 1

    return result
