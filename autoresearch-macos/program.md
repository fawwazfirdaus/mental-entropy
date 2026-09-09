# autoresearch — MES Weight Optimization

This is an experiment to autonomously optimize the Mental Entropy Score (MES) aggregator weights.

## Background

The MES pipeline computes 77 numeric features from journal entry text (measuring coherence, topic fragmentation, narrative structure, cognitive load, and belief conflict). These features are combined into a single 0-100 score via a weighted linear aggregation in `train.py`.

The dataset contains **~1,050 labeled journal entries** from two sources:
- **245 human-written** real journal entries (overall_entropy 1-10, labeled by LLMs)
- **~809 synthetic** scored entries (entropy_overall 0-1, mapped to 1-10 scale via percentile-rank scoring from the datagen pipeline)

The agent optimizes against the combined dataset. The output reports separate correlations (`human_corr`, `synth_corr`) so you can monitor whether improvements generalize across both data sources. Your job is to improve cv_corr.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `program.md` — this file, the experiment instructions.
   - `prepare.py` — fixed evaluation harness, data loading, correlation math. Do not modify.
   - `train.py` — the file you modify. Weight configuration only.
4. **Verify cache exists**: Check that `features_cache.json` exists in the repo directory. If not, tell the human to run:
   ```
   cd /Users/karanpatil/Desktop/mental-entropy
   PYTHONPATH=src .venv/bin/python autoresearch-macos/prepare.py
   ```
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment evaluates a weight configuration against 245 labeled journal entries. It runs in **under 1 second** (pure numpy math on cached features). You launch it as: `uv run train.py`.

**What you CAN do:**
- Modify the `WEIGHTS` list in `train.py`. Everything above the "EVALUATION" line is fair game: add features, remove features, change weights, change transforms, change normalization parameters.

**What you CANNOT do:**
- Modify `prepare.py`. It contains the fixed evaluation harness.
- Modify anything below the "EVALUATION" comment line in `train.py`.
- Install new packages or add dependencies.

**The goal is simple: get the highest cv_corr.** This is the 5-fold cross-validated Pearson correlation between your MES scores and the LLM entropy labels. Cross-validation guards against overfitting to the 245 samples.

**Secondary goals:**
- Keep `val_corr` and `cv_corr` close (gap > 0.05 suggests overfitting).
- Keep `weight_sum` close to 1.0 (the scoring formula is `sum(w * transform(x)) * 100`).
- Lower MSE is better when correlations are tied.

**Simplicity criterion**: All else being equal, fewer features is better. If you can match performance with 12 features instead of 17, that's a win. Removing a feature and maintaining cv_corr is a great outcome.

**The first run**: Always establish the baseline first by running train.py as-is.

## Output format

The script prints a summary like this:

```
---
val_corr:         0.420000
cv_corr:          0.395000
cv_std:           0.045000
mse:              412.3400
score_mean:       36.2
score_std:        10.5
n_features:       17
weight_sum:       1.000
human_corr:       0.420000  (n=245)
synth_corr:       0.380000  (n=809)
```

Extract the key metric: `grep "^cv_corr:" run.log`

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header and 5 columns:

```
commit	cv_corr	val_corr	status	description
```

1. git commit hash (short, 7 chars)
2. cv_corr achieved (e.g. 0.423456)
3. val_corr achieved (e.g. 0.445678)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	cv_corr	val_corr	status	description
a1b2c3d	0.395000	0.420000	keep	baseline
b2c3d4e	0.401000	0.428000	keep	increase se_dominant_cluster_frac to 0.18
c3d4e5f	0.388000	0.415000	discard	remove ce_adj_min
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Edit the `WEIGHTS` list in `train.py` with an experimental idea.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read out the results: `grep "^cv_corr:\|^val_corr:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 20 run.log` to read the error and fix it.
7. Record the results in the tsv
8. If cv_corr improved (higher), you "advance" the branch, keeping the git commit
9. If cv_corr is equal or worse, you git reset back to where you started

**Timeout**: Each experiment takes < 1 second. If a run takes more than 30 seconds, something is broken — kill it and debug.

**Crashes**: Usually a typo or bad feature name. Fix and re-run.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder. The loop runs until the human interrupts you.

## Experiment ideas

Here are directions to explore, roughly ordered by expected impact:

### Weight redistribution
- The current weights were hand-assigned proportional to |correlation|. Try different proportions.
- Increase weight on features with |r| > 0.30 (the top tier matters most).
- Decrease or zero-out features with |r| < 0.10 (they may just add noise).

### Add unused features
There are 77 features available but only 17 are used. Some promising unused ones:
- `se_inter_mean` (r=-0.304) — topic separation, very strong signal
- `ce_adj_p10` (r=-0.302) — bottom 10th percentile coherence
- `ce_adj_mean` (r=-0.291) — average transition quality
- `ce_adj_median` (r=-0.271) — median transition quality
- `ne_consolidation_delta` (r=-0.155) — narrative convergence

### Remove weak features
- `bc_unresolved_rate` (r=+0.041) and `bc_belief_sentence_rate` (r=-0.045) have very weak correlations. Try removing them.
- `ce_intra_block_break_rate` (r=+0.080) is also weak. Does removing it help?

### Transform tuning
- The `dir_norm` params (2.5 for cluster_entropy, 1.0 for length_cv) were eyeballed. Try different values.
- Some "inv" features might work better as `inv_clip` with a specific clipping value.

### Aggressive feature selection
- Try using ONLY the top 5-8 features by |correlation|. Sometimes less is more with small datasets.
- If a small feature set performs well, it's more robust and interpretable.

### Feature substitution
- `ce_adj_min` vs `ce_adj_p10` vs `ce_adj_p25` — these are correlated with each other. Which subset works best?
- `ne_start_end_sim` vs centroid features — try different narrative proximity measures.

### Weight scale experiments
- Does it matter if weights sum to 1.0 or 0.8 or 1.2? (It shouldn't affect correlation, but affects MSE/score range.)
- Try concentrating more weight in fewer features vs spreading across many.
