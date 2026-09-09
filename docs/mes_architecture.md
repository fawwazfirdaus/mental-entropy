# MES Model Architecture (Current Implementation)

## Scope

This repository implements the full MES pipeline through all four layers:
embedding, feature extraction, subscore models, and the MES combiner.

## Architecture Overview

The MES model is a four-layer, interpretable pipeline:

1. **Journal Text** (input)
2. **Text Embeddings** (transformer encoder)
3. **NLP Features** (77 hand-engineered, theory-driven features)
4. **Subscore Models** (5 per-module XGBoost regressors)
5. **Ridge MES Combiner** (learned weighted combination)

The combiner's learned weights (from Ridge regression on 2,209 entries):

```
MES = Ridge(CE_subscore, SE_subscore, NE_subscore, CLE_subscore, BC_subscore)
    = 0.37*CE + (-0.09)*SE + 0.55*NE + 0.29*CLE + 0.04*BC + intercept
```

Note: The system also supports a linear fallback method (hand-tuned weighted sum
of 13 features, cv_corr ~ 0.627) when XGBoost is not installed.

## Current Implementation (Code Mapping)

### Layer 1: Text Embeddings

Implemented in `src/mental_entropy/embedding/`.

- Sentence segmentation and normalization: `text.py`
- Embedding model: `model.py` uses `mixedbread-ai/mxbai-embed-large-v1`
- Pooling: mean pooling for sentence embeddings, L2-normalized
- Document embedding: mean of sentence embeddings, L2-normalized
- Public API: `embed_journal_entry()` in `embed.py`

Notes:
- Embedding dimension is 1024 (mxbai-embed-large-v1)
- Deterministic by default (model frozen, no gradients, CPU by default)

### Layer 2: NLP Features

Implemented in `src/mental_entropy/features/` and `src/mental_entropy/utils/`.

- CE: `features/ce.py` (29 features) - Coherence Entropy
- SE: `features/se.py` (9 features) - Semantic Entropy
- NE: `features/ne.py` (13 features) - Narrative Entropy
- CLE: `features/cle.py` (14 features) - Cognitive Load Entropy
- BC: `features/bc.py` (12 features) - Belief Conflict
- Thresholds: `utils/thresholds.py` (calibrated for mxbai mean pooling)
- Linguistic cues: `utils/linguistic.py` (regex + phrase lists)

All feature functions are deterministic, pure, and JSON-serializable.

### Layer 3: Subscore Models

Implemented in `src/mental_entropy/models/`.

Five per-module XGBoost regressors, each trained on its module's features:

| Module | Features | cv_corr | Role |
|--------|----------|---------|------|
| CE | 29 | 0.670 | Coherence and flow |
| SE | 9 | 0.378 | Topic fragmentation |
| NE | 13 | 0.736 | Narrative structure |
| CLE | 14 | 0.678 | Cognitive load |
| BC | 12 | 0.308 | Belief conflict |

Training uses stacked generalization:
1. 5-fold out-of-fold (OOF) predictions from each module model
2. Ridge combiner trained on the 5 OOF subscore columns

Trained model artifacts stored in `src/mental_entropy/models/_artifacts/`:
- `{module}_model.json` - XGBoost native JSON format (5 files)
- `combiner.json` - Ridge coefficients and intercept
- `manifest.json` - Training metadata and metrics

Key files:
- `models/_registry.py` - Model loading, feature-group mapping, inference
- `models/__init__.py` - Public API: `predict_subscores()`, `predict_mes()`

### Layer 4: MES Combiner

Implemented in `src/mental_entropy/score.py`.

The `compute_mes_score()` function supports three methods:
- `"auto"` (default): Uses subscore models if available, else linear fallback
- `"linear"`: Hand-tuned weighted sum of 13 features (cv_corr ~ 0.627)
- `"subscore"`: Trained per-module models + Ridge combiner (cv_corr ~ 0.769)

## Performance Metrics

Evaluated on 2,209 entries (964 LLM-labeled + 1,245 synthetic):

| Method | cv_corr | Description |
|--------|---------|-------------|
| Linear (13 features) | 0.627 | Hand-tuned weighted sum |
| **Subscore (77 features)** | **0.769** | Per-module XGBoost + Ridge |
| XGBoost monolithic (77 feat) | 0.794 | Single model, diagnostic upper bound |

## Feature-to-Subscore Mapping

| Subscore | Primary Feature Signals | Module |
|----------|------------------------|--------|
| CE | Adjacent similarity stats, coherence breaks, sharp drops, coherent runs | `features/ce.py` |
| SE | Cluster count, cluster entropy, topic switches, intra/inter similarity | `features/se.py` |
| NE | Arc linearity, start/end similarity, temporal markers, reflection cues | `features/ne.py` |
| CLE | Fragments, restarts, hedges, similarity jitter, repetition | `features/cle.py` |
| BC | Belief sentences, conflict pairs, integration markers, unresolved rate | `features/bc.py` |

## Dependencies

Core package dependencies (always required):
- torch, transformers, spacy, en-core-web-sm

Optional model dependencies (`pip install mental-entropy[models]`):
- xgboost >= 2.0.0
- scikit-learn >= 1.3.0

When xgboost is not installed, `compute_mes_score()` automatically falls back
to the linear weighted method.

## Quick Code Entry Points

- Embedding: `src/mental_entropy/embedding/embed.py`
- Features: `src/mental_entropy/features/`
- Models: `src/mental_entropy/models/`
- Scoring: `src/mental_entropy/score.py`
- Training: `autoresearch-macos/train_subscores.py`
- Thresholds: `src/mental_entropy/utils/thresholds.py`
- Linguistic cues: `src/mental_entropy/utils/linguistic.py`

## Potential Improvements

1. **SHAP interpretability**: Add per-feature SHAP values for each subscore model
2. **Human correlation**: Current subscore OOF human_corr (0.235) is below linear (0.301); tuning hyperparameters or weighting could improve this
3. **Threshold recalibration**: Recalibrate the 7 thresholds in `utils/thresholds.py` on the expanded dataset
4. **More labeled data**: Additional real human-labeled journals would improve all models
