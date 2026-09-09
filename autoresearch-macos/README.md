# MES training experiments

This directory contains the project's feature-weight optimization and supervised regression experiments. Despite its historical name, it is not a language-model pretraining framework and the regression scripts are not macOS-specific.

## Reproduce a training run

Run from the repository root in a separate checkout: training writes into `src/mental_entropy/models/_artifacts/`.

```bash
uv sync --all-extras

# Compute features and document embeddings from the consensus CSV.
# This can take substantial time; pretrained weights download on first use.
PYTHONPATH=src .venv/bin/python autoresearch-macos/prepare.py

# Fit the five dimension regressors and second-stage combiner.
PYTHONPATH=src .venv/bin/python autoresearch-macos/train_v7_fep.py

# Produce evaluation metrics and control-example diagnostics.
PYTHONPATH=src .venv/bin/python scripts/evaluate_locked_human_eval.py \
  --output artifacts/eval/local_report.json --fail-on-gates
```

The evaluation command returns a nonzero exit status when acceptance gates fail. It is a diagnostic, not a promise that the packaged or retrained model passes. Outputs are local and ignored by Git.

Preparation reads `data/multirater/consensus_labels.csv`, excludes designated evaluation indices, and writes features and embedding arrays into `features_cache.json`. Regenerate caches after changing labels or evaluation membership. Training consumes those caches; older cached labels must not be mixed with refreshed labels.

## Experiment map

- `train.py`: hand-crafted feature-weight search.
- `train_embedding_regression.py`: direct embedding regression and feature comparisons.
- `train_subscores*.py`: feature-based dimension models and ensemble variants.
- `train_v7_fep.py`: packaged two-stage embedding model.
- `train_v8_gd_calibrated.py`: global-disorder and calibration experiments.
- `train_v9_gd_correction.py`: correction-model experiment.
- `train_v10_constrained_combiner.py`: sign-constrained combiner experiment.
- `diagnose*.py`: diagnostic analyses from earlier iterations.

The later version numbers identify experiments, not automatically promoted improvements. The packaged manifest identifies the inference baseline. Historical result files may refer to earlier labels, data membership, or model choices and should not be compared without checking provenance.

The training script selects features and hyperparameters using development data. Its reported correlations are not a substitute for nested cross-validation and an untouched test set. See [research notes](../docs/research_notes.md).
