# Mental Entropy: Predicting Cognitive Disorganization from Journal Text

An experimental NLP/ML pipeline that uses journal text to estimate how scattered, tangled, or disorganized someone’s thinking is. Higher scores indicate greater disorganization; lower scores indicate clearer, more organized thinking.

It combines transformer embeddings, interpretable linguistic features, and supervised regression into a 0–100 research score.

The central question: **can learned representations predict cognitive disorganization from journal text?** “Mental entropy” is the project's rubric-defined target, not a validated measure of mental health or information-theoretic entropy.

## Methods

| Component | Implementation |
| --- | --- |
| Representation | `mxbai-embed-large-v1` document and sentence embeddings; 1,024 dimensions |
| Feature engineering | Semantic continuity, topic clustering, narrative progression, linguistic load, belief conflict, and experimental global-disorder cues |
| Default model (v7) | Five Ridge regressors predict rubric dimensions from document embeddings; a second Ridge combines those predictions with ten selected linguistic features |
| Comparisons | Direct embedding/feature Ridge, XGBoost dimension models, and a hand-weighted feature baseline |
| Label pipeline | Multiple LLM labeling passes, trimmed consensus, and retained rater disagreement |
| Evaluation | Source-stratified cross-validation, control examples, error analysis, calibration experiments, and label-provenance tests |

The five intermediate targets are prediction coherence, model complexity, compression progress, belief integration, and precision weighting. These are operational rubric dimensions; their theoretical names do not establish cognitive validity.

```text
Journal text → sentence/document embeddings + linguistic features
                           ↓
             five embedding-based Ridge regressors
                           ↓
         predicted dimensions + ten selected features
                           ↓
                    Ridge combiner → MES
```

Start with [the scorer](src/mental_entropy/score.py), [feature modules](src/mental_entropy/features), [v7 training](autoresearch-macos/train_v7_fep.py), and [evaluation](scripts/evaluate_locked_human_eval.py).

## Recorded results and their limits

The [packaged v7 manifest](src/mental_entropy/models/_artifacts/manifest.json) records training on **4,240 entries: 2,995 human-written and 1,245 synthetic**.

| Historical experiment | Pearson correlation on human-written subset |
| --- | ---: |
| Direct embedding + feature Ridge | 0.6161 |
| Two-stage v7 model | 0.6167 |

These are historical development results from the saved manifest, not a fresh benchmark. The small difference does not establish a performance improvement; v7 exposes intermediate rubric predictions for inspection.

**Human-written does not mean human-annotated.** The consensus targets are model-generated. Correlation measures agreement with those targets, not clinical validity or independent expert agreement. Cross-validation was also used for model selection; feature selection occurs before the fold loop. A nested, source-grouped evaluation is needed for an unbiased comparison.

The current preparation code excludes designated evaluation rows by source index. The packaged model predates those safeguards and later label corrections, so it must not be presented as an independently validated model on that evaluation set. Regenerate caches and retrain before making new claims. See the [research notes](docs/research_notes.md).

## Run locally

```bash
uv sync --all-extras
PYTHONPATH=src .venv/bin/python - <<'PY'
from mental_entropy.score import compute_mes_from_text

result = compute_mes_from_text(
    "I planned three tasks this morning. I finished the first, then revised "
    "my schedule so I could complete the remaining work tomorrow.",
    method="embedding",
)
print(result["mes_score"])
PY
```

The first inference downloads pretrained embedding weights. Scoring runs locally and needs no service account or LLM API key. `method="embedding"` requests the learned path explicitly; `auto` can fall back when model artifacts are unavailable. The linear baseline still uses embedding-derived features.

### Verification

```bash
PYTHONPATH=src .venv/bin/pytest tests/ -q
```

Tests cover feature edge cases, model loading, API behavior, temporal analysis, evaluation controls, and label provenance. Passing tests does not establish predictive validity.

### Experiments

See [experiment instructions](autoresearch-macos/README.md) for cache preparation, retraining, and evaluation. Training overwrites packaged artifacts, so run it in a separate checkout when preserving a baseline. Generated experiment outputs belong in ignored `artifacts/`.

### Optional local API

```bash
PYTHONPATH=src .venv/bin/python -m mental_entropy.api
# Interactive endpoint documentation: http://localhost:8000/docs
```

The standalone FastAPI service exposes scoring, batch scoring, and longitudinal analysis. Legacy insight-prompt and feedback utilities remain as experimental application code; their outputs are not evidence of therapeutic effectiveness. There is no required connection to a hosted application.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/mental_entropy/embedding/` | Transformer inference and text segmentation |
| `src/mental_entropy/features/` | Interpretable feature extraction |
| `src/mental_entropy/models/` | Model registry and packaged parameters |
| `autoresearch-macos/` | Regression comparisons and training experiments |
| `scripts/` | Label aggregation, evaluation, and failure analysis |
| `tests/` | Unit and integration tests |
| `docs/research_notes.md` | Evaluation caveats and next experiments |

## Data and scope

The repository contains journal text and derived labels. Source files and labels are research inputs, not a newly licensed dataset release. Redistribution permissions and de-identification are not established by this README; verify them before reusing or redistributing the data.

This project uses pretrained models and includes AI-assisted experimentation. It does not claim to train a language model from scratch. Stronger next steps are an independently annotated test set, grouped/nested evaluation, and ablations that separate sentiment, text length, and narrative disorder.
