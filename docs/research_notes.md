# Research notes

Read when: interpreting results, planning ablations, or reproducing training.

## What the project demonstrates

The implementation connects representation extraction, linguistic feature engineering, noisy-label aggregation, regression modeling, and diagnostic evaluation. The intermediate dimensions make the prediction path inspectable, although they remain learned proxies for a labeling rubric.

## Evaluation caveats

- The saved manifest is a historical record. It does not describe a rerun on the current corrected labels.
- “Human” in training metrics identifies the text source, not an expert labeler.
- v7 selects top features before the fold loop and tunes model choices against development correlations. Those correlations can be optimistic.
- Source stratification balances human/synthetic membership; it does not establish author-level or duplicate-group separation.
- Evaluation indices are excluded by current preparation code. That does not retroactively remove them from older model training, or undo repeated development against the same controls.
- Same-model labeling passes share systematic biases. Rater agreement alone is insufficient evidence of target validity.
- Temporal confidence and insight feedback are application heuristics, not calibrated uncertainty or causal treatment-effect estimates.

## Next experiments

1. Establish data provenance, reuse permissions, and a de-identified evaluation release with independent annotations.
2. Freeze a new test set by author/source and near-duplicate group before further tuning.
3. Move feature selection, scaling, and hyperparameter search inside training folds; use nested cross-validation.
4. Compare features only, embeddings only, direct hybrid regression, and two-stage regression under identical splits.
5. Report MAE and rank/linear correlation with uncertainty intervals, broken down by source, length, and label disagreement.
6. Test whether emotionally negative but coherent text is distinguished from neutral fragmented text; report representative failures.

## Reproduction boundaries

Local scoring needs pretrained embedding weights and the packaged regression parameters. Label generation is a separate, optional API-backed workflow. Model training needs regenerated feature/embedding caches. Preserve the original model and manifest before retraining, and save split membership, label version, configuration, and metrics with each new run.
