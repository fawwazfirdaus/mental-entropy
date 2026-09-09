# Codebase Mental Model

This document captures how the code currently behaves, based on source and tests.
It is intentionally implementation-focused and complements `README.md` and `docs/mes_architecture.md`.

## What Is Implemented

The repository currently implements:

1. Embedding pipeline: raw text -> sentence embeddings + document embedding
2. Feature extraction: CE, SE, NE, CLE feature dicts
3. Synthetic data utilities: generate synthetic journal JSONL and extract/transform records

Not implemented yet:

1. Learned subscore models (CE/SE/NE/CLE -> scores)
2. Final weighted MES combiner API

## Runtime Data Flow

1. `embed_journal_entry(raw_text)` in `src/mental_entropy/embedding/embed.py`
2. `normalize_text()` and `split_sentences()` in `src/mental_entropy/embedding/text.py`
3. `embed_sentences()` in `src/mental_entropy/embedding/model.py`
4. `compute_doc_embedding()` in `src/mental_entropy/embedding/model.py`
5. Feature extraction from `EmbeddingResult` via:
   - `ce_features_from_result()`
   - `se_features_from_result()`
   - `ne_features_from_result()`
   - `cle_features_from_result()`

All public APIs are re-exported from `src/mental_entropy/__init__.py`.

## Core Contracts And Invariants

1. Deterministic behavior is expected for same inputs.
2. Sentence and document embeddings are L2-normalized.
3. Sentence order is preserved with sequential `SentenceEmbedding.id`.
4. Feature outputs are JSON-serializable Python scalars (`int`/`float`), not NumPy scalars.
5. Feature modules validate numeric inputs and raise `ValueError` on NaN/Inf.
6. Edge cases (`n=0`, `n=1`) are handled with safe defaults, not exceptions.

## Sentence Segmentation Behavior

`split_sentences()` uses a hybrid strategy:

1. Pre-split newline fragments when line endings look interrupted (dash/ellipsis/no terminal punctuation) and next non-empty line starts with a capital letter.
2. Split blocks by blank lines.
3. Apply spaCy sentence segmentation.
4. Fallback to regex splitting if spaCy yields nothing.
5. Remove empty and punctuation-only segments.

spaCy behavior:

1. Tries `en_core_web_sm` first.
2. Falls back to `spacy.blank("xx")` + sentencizer if model is missing.

## Feature Module Semantics

### CE (`src/mental_entropy/features/ce.py`)

1. Uses adjacent cosine similarity sequence plus skip connections.
2. Key thresholds:
   - `BREAK_T` for coherence breaks
   - `SHARP_DROP_T` for relative drops
3. Produces 24 features.

### SE (`src/mental_entropy/features/se.py`)

1. Uses NumPy agglomerative clustering with average linkage and distance threshold (`CLUSTER_T`).
2. Tracks topic dispersion and switching with both count/rate and jump-weighted metrics.
3. Produces 9 features.

### NE (`src/mental_entropy/features/ne.py`)

1. Mixes embedding arc metrics with regex/phrase-based narrative cues.
2. Text cues include temporal markers/jumps, connectors, reflection, fragment-like sentences, punctuation breaks.
3. Produces 13 features.

### CLE (`src/mental_entropy/features/cle.py`)

1. Mixes surface cognitive-load markers with local embedding instability features.
2. Key thresholds:
   - `LOW_T`, `HIGH_T`, `REP_T`
3. Includes semantic break and semantic isolation rates as explicit features.
4. Produces 14 features.

## Utilities

### Thresholds (`src/mental_entropy/utils/thresholds.py`)

Centralized thresholds for CE/SE/CLE. Modules import from here rather than hardcoding values.

### Linguistic (`src/mental_entropy/utils/linguistic.py`)

Shared regex and phrase-list logic used by NE/CLE:

1. Verb-pattern heuristics
2. Punctuation pattern detection
3. Phrase matching via cached compiled regex

## Synthetic Data Utilities

### Generator (`src/mental_entropy/datagen/synthetic_journals.py`)

1. Deterministic spec sampling (persona/context/entropy bucket) with seeded RNG.
2. OpenAI-compatible chat completions client over `urllib`.
3. Strict post-generation validation:
   - disallowed terms/PII-like patterns
   - sentence/word/paragraph constraints
   - bullet constraints
4. Writes validated JSONL with metadata.

### Extraction (`src/mental_entropy/datagen/extract_journal_texts.py`)

1. Iterates JSONL records with optional invalid-line skipping.
2. Extracts `journal_text`.
3. Writes plain text or JSONL outputs.
4. Optional embedding export mode for extracted records.

## Test-Validated Expectations

Current test suite verifies:

1. Embedding path basics, normalization, determinism, sentence-order preservation, long-input handling.
2. Exact feature keysets and JSON-serializable scalar types.
3. Correct behavior for edge cases and numeric corner cases.
4. Wrapper parity (`*_features_from_result` matches direct feature calls).
5. Datagen utility correctness for JSONL extraction and core generation helpers.

At the time of writing, running `uv run pytest tests -q` passes all tests.
