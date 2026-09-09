# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mental Entropy Score (MES) pipeline for journal text analysis. The system computes numeric features from sentence embeddings to measure semantic coherence, topic fragmentation, narrative structure, and cognitive load in journal entries.

**Stack**: Python 3.12, PyTorch, Transformers (mxbai-embed-large-v1), spaCy 3.8.0, pytest

## Doc Routing

Use this read order to keep context small and task-focused:

1. `AGENTS.md` (entry-point contract)
2. `docs/agent_docs_index.md` (task-based doc map)
3. Only load the documents required for the current task

## Essential Commands

### Setup and Installation

```bash
# With uv (recommended):
uv sync                    # Production dependencies
uv sync --all-extras       # With dev dependencies (pytest)

# Without uv (using venv directly):
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Verify package loads correctly
PYTHONPATH=src .venv/bin/python -c "from mental_entropy import embed_journal_entry; print('OK')"
```

### Testing

```bash
# Run all tests (PYTHONPATH required for module resolution)
PYTHONPATH=src .venv/bin/pytest tests/ -v

# Run a single test file
PYTHONPATH=src .venv/bin/pytest tests/test_embedding_smoke.py -v
PYTHONPATH=src .venv/bin/pytest tests/features/test_ce_features.py -v
PYTHONPATH=src .venv/bin/pytest tests/features/test_bc_features.py -v
PYTHONPATH=src .venv/bin/pytest tests/temporal/test_te_features.py -v

# Run a single test function
PYTHONPATH=src .venv/bin/pytest tests/features/test_ce_features.py::test_ce_features_empty_input -v

# Run tests matching a pattern
PYTHONPATH=src .venv/bin/pytest tests/ -v -k "empty"

# With uv (if installed):
uv run pytest tests/ -v
```

**Note**: No linter/formatter is configured. Match existing code style manually.

### API Server

```bash
# Install with API dependencies
uv sync --extra api

# Run the MES API server
PYTHONPATH=src .venv/bin/python -m mental_entropy.api

# Or with custom port/host
MES_PORT=8000 MES_HOST=0.0.0.0 PYTHONPATH=src .venv/bin/python -m mental_entropy.api
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/score` | Score a single journal entry → MES score + features |
| POST | `/score/batch` | Score up to 50 entries in one request |
| POST | `/temporal` | Compute 21 temporal entropy features from history |
| GET | `/health` | Health check + model info |

**Example:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"text": "Today I felt scattered, jumping between tasks without finishing any."}'
```

**API Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MES_PORT` | 8000 | API server port |
| `MES_HOST` | 0.0.0.0 | Bind address |
| `MES_WORKERS` | 1 | Uvicorn workers (keep at 1 — model is shared) |
| `MES_CORS_ORIGINS` | * | Comma-separated allowed CORS origins |

### Synthetic Data Generation

```bash
# Generate synthetic journal JSONL (OpenAI-compatible endpoint)
OPENAI_API_KEY=... \
uv run python -m mental_entropy.datagen.synthetic_journals \
  --output data/synthetic_journals.jsonl \
  --total-entries 800 \
  --batch-size 40
```

```bash
# Extract journal_text from JSONL
uv run python -m mental_entropy.datagen.extract_journal_texts \
  --input data/synthetic_journals.jsonl \
  --output data/journal_texts.txt \
  --format txt
```

## Architecture Overview

### High-Level Pipeline

The MES pipeline has evolved through v1-v6. The **current production model is hybrid v6**.

**Production scoring (hybrid v6):**
1. **Embedding Layer** (`embedding/`): Text → 1024-dim mxbai embedding
2. **Feature Layer** (`features/`): Embeddings → 77 numeric features (top 10 used)
3. **Hybrid Ridge** (`models/`): Ridge regression on [1024-dim embedding + 10 features] → MES score (0-100)

**Legacy scoring (still available as fallback):**
- **Subscore models** (v3/v5): XGBoost dimension models + Ridge combiner on 77 features
- **Linear** (v1): Hand-tuned 13-feature weighted formula

The hybrid v6 model was trained on 4240 entries (2995 human + 1245 synthetic) with
multi-rater consensus labels (3 labeling passes, inter-rater correlation 0.83-0.92).
human_corr = 0.636, cv_corr = 0.774.

**API Layer** (`api.py`): FastAPI REST service exposing scoring + temporal analysis

### Data Flow

Current implemented flow:

```
Raw journal text
    ↓
embed_journal_entry()
    ├─ normalize_text() → clean text
    ├─ split_sentences() → sentence boundaries (spaCy + regex fallback)
    ├─ embed_sentences() → mxbai embeddings (mean pooling + L2 norm)
    └─ compute_doc_embedding() → mean of sentence embeddings
    ↓
EmbeddingResult (sentences + doc_embedding)
    ↓
Feature extraction (choose one or more):
    ├─ ce_features_from_result() → 29 Coherence Entropy features
    ├─ se_features_from_result() → 9 Semantic Entropy features
    ├─ ne_features_from_result() → 13 Narrative Entropy features
    ├─ cle_features_from_result() → 14 Cognitive Load Entropy features
    └─ bc_features_from_result() → 12 Belief Conflict features
    ↓
Score aggregation (optional):
    └─ compute_mes_from_text() → MES score (0-100) + all 77 features

---

Longitudinal analysis (multi-entry):
list[ScoredEntry] (timestamp, mes_score, features)
    ↓
te_features() → 21 Temporal Entropy features (trajectory, baseline, stability)
```

Subscore model inference (method="auto" or "subscore"):

```
All 77 features (ce_* + se_* + ne_* + cle_* + bc_*)
    ↓
5 dimension-aligned XGBoost models (each takes ALL 77 features):
    ├─ continuity_model         → continuity subscore (1-5)
    ├─ topic_focus_model        → topic_focus subscore (1-5)
    ├─ contradiction_model      → contradiction_integration subscore (1-5)
    ├─ cognitive_clarity_model   → cognitive_clarity subscore (1-5)
    └─ narrative_closure_model   → narrative_closure subscore (1-5)
    ↓
Ridge combiner → overall_entropy (1-10) → MES score (0-100)
```

### Key Modules

#### Embedding Layer (`embedding/`)

- **`embed.py`**: Main API entry point (`embed_journal_entry()`)
- **`text.py`**: Text normalization and sentence splitting
  - Normalizes whitespace, line endings, Unicode quotes
  - Uses blank-line blocks + spaCy sentencizer + regex fallback
- **`model.py`**: Transformer model management
  - Loads `mixedbread-ai/mxbai-embed-large-v1` (1024-dim)
  - Mean pooling + L2 normalization
  - Model is frozen (no gradients, deterministic)
- **`types.py`**: Data structures (`SentenceEmbedding`, `EmbeddingResult`)

#### Feature Layer (`features/`)

All feature modules follow the same pattern:
- Accept `list[list[float]]` (raw embeddings) OR `EmbeddingResult`
- Return `dict[str, int | float]` (JSON-serializable)
- Define expected keys as `{MODULE}_FEATURE_KEYS: frozenset[str]`
- Handle edge cases (empty input, single sentence) gracefully

**Five feature modules:**

1. **CE (Coherence Entropy)** - 29 features
   - Measures flow and breaks in adjacent sentence similarity
   - Key signals: breaks (similarity < 0.45), sharp drops, coherent runs
   - Block-aware features distinguish intra-block (mid-paragraph) vs inter-block (topic transition) breaks
   - Uses thresholds: `BREAK_T`, `SHARP_DROP_T`

2. **SE (Semantic Entropy)** - 9 features
   - Measures topic dispersion via agglomerative clustering
   - Key signals: cluster count, cluster entropy, topic switches
   - Uses threshold: `CLUSTER_T` (0.52 cosine distance)

3. **NE (Narrative Entropy)** - 13 features
   - Measures narrative arc vs fragmentation
   - Combines embedding analysis (start-end similarity, arc linearity) with text patterns (temporal markers, reflection cues)
   - Text analysis uses regex patterns from `utils/linguistic.py`

4. **CLE (Cognitive Load Entropy)** - 14 features
   - Measures cognitive overload and thought instability
   - Surface signals: fragments, restarts, hedging
   - Embedding signals: similarity CV, zigzags, semantic breaks
   - Uses thresholds: `LOW_T`, `HIGH_T`, `REP_T`

5. **BC (Belief Conflict)** - 12 features
   - Measures unresolved contradictions in belief statements
   - Detects belief sentences (identity, value, modal statements)
   - Finds conflict pairs via embedding similarity + polarity opposition
   - Tracks integration markers that resolve contradictions
   - Key signals: unresolved conflict rate, belief sentence rate
   - Uses threshold: `CONFLICT_SIM_T`

#### Score Aggregation (`score.py`)

- **`compute_mes_from_text()`**: End-to-end scoring from raw text
- **`compute_mes_score()`**: Weighted aggregation of features → 0-100 score
- **`interpret_mes_score()`**: Human-readable interpretation

Three scoring methods (via `method` parameter):
- **`"auto"`** (default): Uses subscore models if available, else falls back to linear
- **`"subscore"`**: Uses dimension-aligned XGBoost models + Ridge combiner
- **`"linear"`**: Hand-tuned 13-feature weighted formula (fallback)

The linear fallback uses tier-based weighting calibrated against LLM labels:
- Tier 1 (58%): SE/NE/CE top features (dominant_cluster_frac, start_end_sim, adj_min)
- Tier 2 (22%): Mid-range features (adj_p25, fragment_rate)
- Tier 3 (13%): Supporting features (cluster_entropy, length_cv)
- Tier 4 (7%): BC features (unresolved_rate, belief_rate)

Score interpretation:
- 0-20: Very low entropy (highly organized)
- 21-35: Low entropy (well-organized)
- 36-50: Moderate entropy (some fragmentation)
- 51-65: Elevated entropy (noticeable fragmentation)
- 66-80: High entropy (significant disorganization)
- 81-100: Very high entropy (severely fragmented)

#### Subscore Models (`models/`)

Dimension-aligned XGBoost models (v3 architecture). Auto-detected from `combiner.json`.

- **`_registry.py`**: Model loading, architecture detection, inference
  - `DIMENSION_NAMES`: 5 rubric dimensions (continuity, topic_focus, etc.)
  - `ALL_FEATURE_KEYS`: All 77 features in sorted order
  - `predict_subscores()`: Returns `{dim}_subscore` for each dimension
  - `predict_mes()`: Full pipeline → MES score (0-100)
  - `subscore_models_available()`: Check if artifacts exist + xgboost installed
- **`_artifacts/`**: Serialized model files
  - `{dimension}_model.json`: Per-dimension XGBoost models
  - `combiner.json`: Ridge coefficients + architecture marker
  - `manifest.json`: Training metadata and metrics

Architecture is backward-compatible: if old module-aligned artifacts are present
(no `"architecture"` key in combiner.json), the registry loads them as v1.

#### Temporal Layer (`temporal/`)

Stateless longitudinal analysis for tracking MES trajectory over time.

- **`types.py`**: `ScoredEntry` dataclass (timestamp, mes_score, optional features)
- **`te.py`**: Temporal Entropy feature extraction

**TE (Temporal Entropy)** - 21 features:

*Trajectory metrics (11):*
- `te_n_entries`, `te_time_span_days`, `te_mean_gap_days`, `te_max_gap_days`
- `te_slope`, `te_slope_per_week` (linear regression of score over time)
- `te_velocity_mean`, `te_velocity_std` (points per day change)
- `te_acceleration_mean` (change in velocity)
- `te_rolling_7d_mean`, `te_rolling_7d_std` (recent 7-day window)

*Baseline comparison (6):*
- `te_baseline_mean`, `te_baseline_std` (all-time personal stats)
- `te_latest_zscore`, `te_latest_percentile` (current vs history)
- `te_days_since_high`, `te_days_since_low`

*Stability metrics (4):*
- `te_trend_direction` (-1 improving, 0 flat, +1 worsening)
- `te_volatility` (coefficient of variation)
- `te_streak_improving`, `te_streak_worsening` (consecutive entries)

**Usage:**
```python
from datetime import datetime
from mental_entropy import te_features, ScoredEntry

# Raw tuple input (backend passes this)
entries = [
    (datetime(2024, 1, 1), 45.2, None),
    (datetime(2024, 1, 2), 52.1, None),
]
features = te_features(entries)

# Or with ScoredEntry objects
entries = [ScoredEntry(datetime(2024, 1, 1), 45.2), ...]
features = te_features_from_entries(entries)
```

#### Utilities (`utils/`)

- **`thresholds.py`**: All calibrated threshold constants
  - Centralized to ensure consistency across modules
  - Calibrated specifically for mxbai embeddings (wider similarity distribution than E5)
  - Includes: `BREAK_T`, `SHARP_DROP_T`, `CLUSTER_T`, `LOW_T`, `HIGH_T`, `REP_T`, `CONFLICT_SIM_T`
- **`linguistic.py`**: Regex patterns, word lists, linguistic analysis functions
  - Temporal markers, discourse connectors, reflection phrases
  - Verb detection, fragment detection, punctuation break detection
  - Belief patterns, integration markers, negation words, polar word pairs (for BC)

#### Data Generation (`datagen/`)

- **`synthetic_journals.py`**: Synthetic journal generator
  - Deterministic persona/context sampling
  - OpenAI-compatible prompt generation
  - JSONL output with validation and metadata
- **`extract_journal_texts.py`**: Extract journal_text fields from JSONL

### Design Principles

1. **Determinism**: Same input always produces same output (frozen model, no randomness)
2. **L2 normalization**: All embeddings are unit vectors (cosine similarity = dot product)
3. **Order preservation**: Sentence IDs match original text order
4. **JSON serializability**: All features are Python `int` or `float` (explicit casts from NumPy types)
5. **Pure functions**: Feature extractors have no external state

### Common Gotchas

- **Embedding pooling**: Uses CLS pooling (not mean pooling) for sentence embeddings, but mean pooling for document embedding
- **Feature prefixes**: All feature keys use module prefix (`ce_`, `se_`, `ne_`, `cle_`, `bc_`, `te_`)
- **Empty handling**: Each feature module defines `_empty_features()` helper for edge cases
- **NumPy types**: Always cast to Python types for JSON: `int(n)`, `float(x)`
- **Thresholds**: Import from `utils/thresholds.py`, not hardcoded values

## Environment Variables

```bash
MES_DEVICE=cpu          # Device for inference (default: cpu, use 'cuda' for GPU)
MES_MODEL_REVISION=...  # Pin model to specific commit hash for reproducibility
```

Synthetic data generation:

```bash
OPENAI_API_KEY=...               # API key for OpenAI-compatible endpoint
OPENAI_MODEL=...                 # Model name (default: gpt-4.1-mini)
OPENAI_BASE_URL=...              # Chat completions URL
```

## Code Style Reference

See `docs/agent_style_guide.md` for detailed style/testing conventions.
`AGENTS.md` remains the primary entry-point contract.

Key points:

- Use `from __future__ import annotations` at top of files
- Type annotate all function signatures
- Use Python 3.10+ syntax: `list[X]`, `dict[str, int | float]`, `X | None`
- Frozen, slotted dataclasses for immutable data
- Google-style docstrings for public functions
- Constants in `UPPER_SNAKE_CASE`
- Feature keys as `frozenset[str]`

## Design System
Always read DESIGN.md before making any visual or UI decisions.
All font choices, colors, spacing, and aesthetic direction are defined there.
Do not deviate without explicit user approval.
In QA mode, flag any code that doesn't match DESIGN.md.
