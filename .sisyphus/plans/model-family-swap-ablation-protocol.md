# Model Family Swap Ablation Protocol

## Goal

Compare baseline and swap model-family assignments for CE/SE/NE/CLE under fixed data splits and a pre-registered decision rule.

## Variants

- Baseline profile:
  - CE -> GAM
  - SE -> XGBoost
  - NE -> GAM
  - CLE -> XGBoost
- Swap profile:
  - CE -> XGBoost
  - SE -> GAM
  - NE -> XGBoost
  - CLE -> GAM

## Fixed Data Splits

- Train: `data/train_labeled.jsonl`
- Validation: `data/val_labeled.jsonl`
- Test: `data/test_labeled.jsonl`

No resplitting is allowed.

## Metrics

- Per-subscore: MAE, RMSE, Spearman
- Aggregate:
  - mean MAE across CE/SE/NE/CLE
  - mean Spearman across CE/SE/NE/CLE

## Decision Rule (Pre-Registered)

1. Primary selector on validation set: lower mean MAE wins.
2. If mean MAE ties within 1e-6: higher mean Spearman wins.
3. If still tied: keep baseline.
4. Test set is used once for confirmation only.

## Test Confirmation Rule

- Do not tune hyperparameters after reading test metrics.
- If test direction conflicts with val direction, recommendation output must include an instability warning.

## Strict Label Source Rule

- Training/evaluation must use `labels.{ce,se,ne,cle}` only.
- Any fallback to top-level label keys or `entropy_analysis` is forbidden in ablation mode.

## Reproducibility

- Include git commit hash, run timestamp, profile id, and data paths in each report.
- Persist all outputs under `artifacts/ablation/`.
