# Full Training Pipeline: Mini → Production (Revised with Human Data)

## TL;DR

> **Quick Summary**: Transition MES models from 20-row mini-training to production using ~1024 entries (809 synthetic + ~215 human journals) labeled by gpt-4.1. Human-only val/test splits ensure real-world evaluation quality.
> 
> **Deliverables**:
> - Cleaned human journals: `data/human_journals_clean.jsonl`
> - Combined clean corpus: `data/all_journals_clean.jsonl`
> - GPT-4.1 labeled dataset: `data/all_journals_labeled.jsonl`
> - Train/Val/Test splits: `data/train_labeled.jsonl`, `data/val_labeled.jsonl`, `data/test_labeled.jsonl`
> - Production GAM models: `gam_ce.joblib`, `gam_ne.joblib` (CE/NE)
> - Production XGBoost models: `xgb_se.json`, `xgb_cle.json` (SE/CLE)
> - Updated metadata JSONs with production sample counts and CV metrics
> - Val set evaluation report
> 
> **Estimated Effort**: Medium-Large (labeling ~45 min + 2× training ~30-60 min each)
> **Parallel Execution**: YES — limited parallelism (pipeline is mostly sequential; GAM+XGB train in parallel)
> **Critical Path**: T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8/T9 → T10 → T11 → T12

---

## Context

### Original Request
User wants to transition from mini-trained models (20 synthetic random rows, negative R² values) to real production models. Additionally, user added ~247 human journal entries (`data/journals_raw.csv`) which are higher value than synthetic data. Val and test sets must be strictly human-only for realistic evaluation.

### Interview Summary
**Key Discussions**:
- Clean data without previous scoring to avoid `train_xgb.py` fallback contamination
- Combine both synthetic datasets (legacy 409 + v3 400 = 809 entries)
- Use gpt-4.1 (not mini) for higher label quality
- Human journals: convert CSV→JSONL, filter ≥30 words, assign `human_` prefix IDs
- Split strategy: 60/20/20 for human data; all synthetic in train; val/test human-only
- Equal treatment (no sample weights between synthetic and human)
- Label ALL entries first, then split (stateless LLM labeling = no cross-contamination)

**Research Findings**:
- `label_journals.py` processes entries independently (temperature=0.0, no cross-entry context) — labeling before splitting is safe
- `train_xgb.py` has 3-tier label fallback including `entropy_analysis` — clean input eliminates risk
- Both training scripts compute features on-the-fly from `journal_text`
- Labels are ordinal: `{0.0, 0.25, 0.5, 0.75, 1.0}` — 5 anchors only
- Current models trained on `.sisyphus/evidence/task-12-13-mini.jsonl` — 20 random rows

### Metis Review (Revision)
**Identified Gaps** (addressed):
- **Usable count is 216, not 234**: 31 entries fail the 30-word filter (not ~13 as initially estimated). All downstream counts updated.
- **1 exact duplicate entry**: Dedup added to conversion step to prevent train/val leakage.
- **CSV parsing hazard**: File has 59 multiline entries + mixed CRLF/LF. Must use `csv.DictReader`, not line splitting.
- **CRLF contamination**: `\r` must be stripped from journal text during conversion.
- **Field name mismatch**: CSV column is `journal`, JSONL schema expects `journal_text`. Rename during conversion.
- **Non-journal content** (~8 entries ≥30 words): Chat-seeking posts, ASL-style intros. Included — 30-word filter catches the worst; remaining will receive honest labels.
- **Arabic/mixed-language entries** (3): Included — real-world diversity. Embedding model handles multilingual text.
- **Truncated entries** (33 without terminal punctuation): Included — real human writing patterns, valuable signal.
- **Em-dash convention**: No em-dash removal for human data — the v3 `_no_emdash` cleanup addressed a synthetic generation artifact.
- **Validation script incompatibility**: Human entries lack `entropy_analysis`. Validate synthetic portion only via positional matching.
- **Very long entries** (12 entries >1000 words, max 1805): Included — GPT-4.1 handles 128K context. Feature extraction tested on shorter text but should generalize.
- **Duplicate IDs (pre-existing)**: Legacy and v3 share 380 IDs. Resolved: v3 offset +1000, human entries get `human_N` string IDs.
- **Fixed random seed**: Split uses seed=42 for reproducibility.

---

## Work Objectives

### Core Objective
Replace mini-trained placeholder models (20 synthetic rows, negative R²) with production models trained on ~949 entries (809 synthetic + ~130 human), evaluated on ~43 human-only val entries, with ~43 human-only test entries held out.

### Concrete Deliverables
- `data/human_journals_clean.jsonl` — Filtered, deduped, converted human journals
- `data/all_journals_clean.jsonl` — Combined corpus (legacy + v3 + human, no `entropy_analysis`)
- `data/all_journals_labeled.jsonl` — Full gpt-4.1 labeled dataset
- `data/train_labeled.jsonl` — Training split (all synthetic + 60% human)
- `data/val_labeled.jsonl` — Validation split (20% human, human-only)
- `data/test_labeled.jsonl` — Test split (20% human, human-only, held out)
- `src/mental_entropy/models/gam_ce.joblib` — Production CE model
- `src/mental_entropy/models/gam_ne.joblib` — Production NE model
- `src/mental_entropy/models/xgb_se.json` — Production SE model
- `src/mental_entropy/models/xgb_cle.json` — Production CLE model
- `src/mental_entropy/models/gam_metadata.json` — Updated with production metrics
- `src/mental_entropy/models/xgb_metadata.json` — Updated with production metrics

### Definition of Done
- [ ] All 4 model artifacts replaced with production versions
- [ ] Metadata shows n_samples ≥ 900 (train split: ~949, not 20)
- [ ] `uv run pytest tests/ -v` passes (exit 0)
- [ ] Pipeline smoke test: embed → features → all subscores → MES → valid finite values
- [ ] Val set evaluation report generated with per-subscore MAE

### Must Have
- Clean human JSONL without CSV artifacts (no CRLF, proper field names, deduped)
- Globally unique IDs across all sources (legacy ints, v3 offset ints, `human_N` strings)
- Pre-training data validation (completeness + no contamination)
- Human-only val/test splits with fixed random seed
- Automated pass/fail gate after label validation step
- Full test suite pass after model replacement
- Val set evaluation with per-subscore MAE

### Must NOT Have (Guardrails)
- MUST NOT modify `labeling_rubric.py` — rubric is working as-is
- MUST NOT touch `src/mental_entropy/features/` — feature extraction is out of scope
- MUST NOT modify `combiner.py` — MES weights are not changing
- MUST NOT use `synthetic_journals_800_scored.jsonl` as training input
- MUST NOT weaken any test assertions to match new model outputs
- MUST NOT add model versioning/registry (DVC, MLflow) — not in scope
- MUST NOT commit `.env` or API keys
- MUST NOT modify `label_journals.py`, `validate_labels.py`, or `train_*.py` scripts
- MUST NOT hardcode entry counts (234, 1043) — derive from actual data
- MUST NOT use `open().readlines()` or `splitlines()` for CSV parsing — use `csv.DictReader`
- MUST NOT apply em-dash removal to human entries

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.
> Acceptance criteria requiring "user manually tests/confirms" are FORBIDDEN.

### Test Decision
- **Infrastructure exists**: YES
- **Automated tests**: Tests-after (existing test suite covers model loading + scoring)
- **Framework**: pytest via `uv run pytest tests/ -v`

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

| Deliverable Type | Verification Tool | Method |
|------------------|-------------------|--------|
| Data files | Bash (python one-liners) | Parse JSONL, count rows, check keys, validate values |
| Model artifacts | Bash (pytest + python) | Run test suite + smoke test |
| Validation report | Bash (capture stdout) | Run validate_labels.py, parse output |
| Split files | Bash (python one-liners) | Verify counts, uniqueness, human-only constraints |

---

## Execution Strategy

### Sequential Pipeline with Parallel Training

> This pipeline is inherently sequential (each step depends on the previous).
> Parallelism opportunities: GAM + XGB training simultaneous; post-training eval + pytest.

```
Wave 1 (Data Preparation):
├── Task 1: Convert CSV → JSONL + filter + dedup [quick]
└── Task 2: Concatenate 3 sources into clean JSONL [quick] (after T1)

Wave 2 (Labeling — requires OPENAI_API_KEY):
├── Task 3: Pilot label run (50 entries) [unspecified-high]
└── Task 4: Full labeling of all entries [unspecified-high] (after T3)

Wave 3 (Validation + Go/No-Go):
└── Task 5: Run validation + present results to user [unspecified-high]

Wave 4 (Splitting):
└── Task 6: Split labeled data into train/val/test [quick]

Wave 5 (Pre-training + Training — PARALLEL):
├── Task 7: Pre-training data validation [quick]
├── Task 8: Train GAM models (CE/NE) on train split [unspecified-high] ← after T7
└── Task 9: Train XGBoost models (SE/CLE) on train split [unspecified-high] ← after T7, parallel with T8

Wave 6 (Evaluation + Verification):
├── Task 10: Post-training evaluation on val set [unspecified-high]
├── Task 11: Full verification (pytest + smoke test + metadata) [unspecified-high] (parallel with T10)
└── Task 12: Commit everything [quick] (after T10 + T11)

Critical Path: T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8/T9 → T10/T11 → T12
Parallel Speedup: T8+T9 run simultaneously (~30-60 min saved); T10+T11 in parallel
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|------------|--------|------|
| 1 | — | 2 | 1 |
| 2 | 1 | 3 | 1 |
| 3 | 2 | 4 | 2 |
| 4 | 3 | 5 | 2 |
| 5 | 4 | 6 | 3 |
| 6 | 5 | 7 | 4 |
| 7 | 6 | 8, 9 | 5 |
| 8 | 7 | 10, 11 | 5 |
| 9 | 7 | 10, 11 | 5 |
| 10 | 8, 9 | 12 | 6 |
| 11 | 8, 9 | 12 | 6 |
| 12 | 10, 11 | — | 6 |

### Agent Dispatch Summary

| Wave | # Parallel | Tasks → Agent Category |
|------|------------|----------------------|
| 1 | 1 | T1 → `quick`, T2 → `quick` (sequential) |
| 2 | 1 | T3 → `unspecified-high`, T4 → `unspecified-high` (sequential) |
| 3 | 1 | T5 → `unspecified-high` |
| 4 | 1 | T6 → `quick` |
| 5 | 2 | T7 → `quick`, T8 → `unspecified-high`, T9 → `unspecified-high` (T8+T9 parallel after T7) |
| 6 | 2 | T10 → `unspecified-high`, T11 → `unspecified-high` (parallel), T12 → `quick` (after) |

---

## TODOs

- [ ] 1. Convert human journals CSV → JSONL with filtering and dedup

  **What to do**:
  - Write an inline Python script (via bash, not a permanent file) that:
    1. Reads `data/journals_raw.csv` using `csv.DictReader` (MUST — file has 59 multiline entries that break with line splitting)
    2. Strips `\r` from journal text (CRLF contamination)
    3. Renames field `journal` → `journal_text` to match JSONL schema
    4. Filters entries with fewer than 30 words (word count via `len(text.split())`)
    5. Deduplicates by exact `journal_text` match (1 known duplicate exists)
    6. Assigns sequential string IDs: `human_1`, `human_2`, ...
    7. Adds `source: "human"` field for traceability
    8. Writes output to `data/human_journals_clean.jsonl` (one JSON object per line, UTF-8, LF)
  - Report: total input rows, filtered count, dedup count, final output count

  **Must NOT do**:
  - Do NOT modify the source CSV file
  - Do NOT use `open().readlines()` or `splitlines()` — use `csv.DictReader`
  - Do NOT apply em-dash removal (that was a synthetic data artifact)
  - Do NOT create a permanent script file — use inline python via bash
  - Do NOT hardcode expected row counts — derive from actual data

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple data conversion, one file output, well-defined logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (first task)
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `data/synthetic_journals_400_v3_no_emdash.jsonl:1-3` — Target JSONL schema to match: `{id, journal_text, ...}`. Human entries will be sparse (no persona/context/metadata).
  - `data/journals_raw.csv:1-5` — Source CSV: single column `journal`, 247 rows, mixed CRLF/LF, 59 multiline entries

  **API/Type References**:
  - `scripts/label_journals.py:178-185` — `_normalize_entry_id()`: handles both int and str IDs. The `human_N` format works fine here.
  - `scripts/label_journals.py:326-334` — Resume logic uses ID-based dedup. Unique string IDs ensure safe resume.

  **WHY Each Reference Matters**:
  - The JSONL schema must be compatible with `label_journals.py` which reads `id` + `journal_text` fields
  - `csv.DictReader` is MANDATORY because 59 entries span multiple lines (quoted newlines in CSV). Naive line splitting will corrupt these entries, producing wrong row count and mangled text.
  - CRLF stripping prevents `\r` characters in journal text which could affect tokenization and embedding

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Clean JSONL has correct schema and no CSV artifacts
    Tool: Bash (python one-liner)
    Preconditions: data/journals_raw.csv exists
    Steps:
      1. Run the conversion script
      2. Run: python3 -c "
         import json
         rows = [json.loads(l) for l in open('data/human_journals_clean.jsonl') if l.strip()]
         assert len(rows) >= 210, f'Expected >= 210 rows, got {len(rows)}'
         for r in rows:
             assert 'id' in r and 'journal_text' in r, f'Missing fields in {r.get(\"id\")}'
             assert isinstance(r['id'], str) and r['id'].startswith('human_'), f'Bad ID: {r[\"id\"]}'
             assert isinstance(r['journal_text'], str) and len(r['journal_text'].split()) >= 30, f'{r[\"id\"]}: below 30 words'
             assert '\\r' not in r['journal_text'], f'{r[\"id\"]}: contains CR'
             assert r.get('source') == 'human', f'{r[\"id\"]}: missing source field'
         assert len(set(r['id'] for r in rows)) == len(rows), 'Duplicate IDs'
         assert len(set(r['journal_text'] for r in rows)) == len(rows), 'Duplicate texts'
         print(f'PASS: {len(rows)} clean human journal entries')
         "
    Expected Result: "PASS: N clean human journal entries" with N >= 210
    Failure Indicators: AssertionError about missing fields, bad IDs, CRLF, or duplicates
    Evidence: .sisyphus/evidence/task-1-human-clean.txt

  Scenario: Filtered entries are truly below 30 words
    Tool: Bash (python one-liner)
    Preconditions: Conversion completed
    Steps:
      1. Run: python3 -c "
         import csv, json
         with open('data/journals_raw.csv', newline='') as f:
             all_rows = list(csv.DictReader(f))
         clean = [json.loads(l) for l in open('data/human_journals_clean.jsonl') if l.strip()]
         filtered_count = len(all_rows) - len(clean)
         print(f'Total CSV: {len(all_rows)}, Clean: {len(clean)}, Filtered: {filtered_count}')
         assert filtered_count >= 30, f'Expected >= 30 filtered, got {filtered_count}'
         print('PASS: Filtering applied correctly')
         "
    Expected Result: ~31+ entries filtered (below 30 words + duplicates)
    Failure Indicators: Too few filtered (CSV parsing issue) or too many (over-aggressive filter)
    Evidence: .sisyphus/evidence/task-1-filter-report.txt
  ```

  **Commit**: NO (groups with final commit)

---

- [ ] 2. Concatenate all 3 sources into clean combined JSONL

  **What to do**:
  - Write an inline Python script that:
    1. Reads `data/synthetic_journals_400_legacy.jsonl` (409 rows) — keep IDs as-is
    2. Reads `data/synthetic_journals_400_v3_no_emdash.jsonl` (400 rows) — offset IDs by +1000
    3. Reads `data/human_journals_clean.jsonl` (from Task 1) — IDs already formatted
    4. Adds `source` field: `"legacy"`, `"v3"`, or `"human"` (human already has it)
    5. Strips `entropy_analysis` field if present (should not be, but defensive)
    6. Writes combined output to `data/all_journals_clean.jsonl`
  - Verify: total rows = 409 + 400 + N_human, all unique IDs, all have `journal_text`, NONE have `entropy_analysis`

  **Must NOT do**:
  - Do NOT modify the source files
  - Do NOT add `entropy_analysis` or any scoring fields
  - Do NOT create a permanent script file — use inline python via bash
  - Do NOT use `synthetic_journals_800_scored.jsonl` as input

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple data concatenation, one file output
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (after T1)
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `data/synthetic_journals_400_legacy.jsonl:1-3` — Source schema: `{id, persona, context, journal_text, metadata}`. IDs are integers 1-410.
  - `data/synthetic_journals_400_v3_no_emdash.jsonl:1-3` — Same schema. IDs are integers 1-429 (380 overlap with legacy!).
  - `data/human_journals_clean.jsonl` — From Task 1. Sparse schema: `{id, journal_text, source}`.

  **API/Type References**:
  - `scripts/label_journals.py:22` — `DEFAULT_INPUT_PATH` shows expected input format
  - `scripts/label_journals.py:326-334` — Resume logic uses ID-based dedup; unique IDs critical

  **WHY Each Reference Matters**:
  - 380 IDs overlap between legacy and v3. Without the +1000 offset, resume-after-interruption silently drops up to 380 entries during labeling.
  - The `_normalize_entry_id` function accepts both int and str IDs, so mixed ID types (int for synthetic, str for human) work fine.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Combined file has correct total with unique IDs and no contamination
    Tool: Bash (python one-liner)
    Preconditions: human_journals_clean.jsonl exists from T1
    Steps:
      1. Run the concatenation script
      2. Run: python3 -c "
         import json
         with open('data/human_journals_clean.jsonl') as f:
             n_human = sum(1 for l in f if l.strip())
         expected = 409 + 400 + n_human
         ids = set()
         with open('data/all_journals_clean.jsonl') as f:
             rows = [json.loads(l) for l in f if l.strip()]
         assert len(rows) == expected, f'Expected {expected}, got {len(rows)}'
         for r in rows:
             assert 'entropy_analysis' not in r, f'Row {r[\"id\"]} has entropy_analysis'
             assert r.get('id') not in ids, f'Duplicate ID: {r[\"id\"]}'
             ids.add(r['id'])
             assert isinstance(r.get('journal_text'), str) and r['journal_text'].strip()
         print(f'PASS: {len(rows)} rows ({expected} expected), all unique IDs, no contamination')
         "
    Expected Result: "PASS: N rows (N expected)" with N = 409 + 400 + n_human
    Failure Indicators: Wrong count, duplicate IDs, entropy_analysis present
    Evidence: .sisyphus/evidence/task-2-combined-validation.txt

  Scenario: ID ranges are correct per source
    Tool: Bash (python one-liner)
    Preconditions: Combined file exists
    Steps:
      1. Run: python3 -c "
         import json
         with open('data/all_journals_clean.jsonl') as f:
             rows = [json.loads(l) for l in f if l.strip()]
         legacy = [r for r in rows if r.get('source') == 'legacy']
         v3 = [r for r in rows if r.get('source') == 'v3']
         human = [r for r in rows if r.get('source') == 'human']
         assert len(legacy) == 409, f'Legacy: {len(legacy)}'
         assert len(v3) == 400, f'V3: {len(v3)}'
         assert len(human) >= 210, f'Human: {len(human)}'
         assert all(isinstance(r['id'], int) and r['id'] < 1000 for r in legacy), 'Legacy ID >= 1000'
         assert all(isinstance(r['id'], int) and r['id'] >= 1001 for r in v3), 'V3 ID < 1001'
         assert all(isinstance(r['id'], str) and r['id'].startswith('human_') for r in human), 'Bad human ID'
         print(f'PASS: {len(legacy)} legacy, {len(v3)} v3, {len(human)} human')
         "
    Expected Result: 409 legacy, 400 v3, >=210 human
    Failure Indicators: Wrong counts, wrong ID ranges
    Evidence: .sisyphus/evidence/task-2-source-breakdown.txt
  ```

  **Commit**: NO (groups with final commit)

---

- [ ] 3. Pilot label run (50 entries) to validate pipeline

  **What to do**:
  - Run `label_journals.py` with `--max-entries 50` to label a small batch first
  - This validates: API key works with gpt-4.1, rubric produces valid labels, label distribution looks reasonable
  - Command:
    ```bash
    OPENAI_MODEL=gpt-4.1 uv run python scripts/label_journals.py \
      --input data/all_journals_clean.jsonl \
      --output data/all_journals_labeled.jsonl \
      --max-entries 50
    ```
  - After completion, check:
    1. Output file has ~50 rows (some may fail and be skipped)
    2. All rows have `labels` dict with all 4 keys
    3. All label values are in `{0.0, 0.25, 0.5, 0.75, 1.0}`
    4. Label distribution is not degenerate (not all same value)
    5. No `entropy_analysis` in output rows

  **Must NOT do**:
  - Do NOT proceed to full labeling if pilot shows problems (quality/API issues)
  - Do NOT modify `label_journals.py` or `labeling_rubric.py`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Long-running API calls, needs monitoring and result analysis
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential with T4)
  - **Blocks**: Task 4
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `scripts/label_journals.py:368-409` — CLI args: `--input`, `--output`, `--batch-size`, `--max-entries`
  - `scripts/label_journals.py:22-28` — Defaults: input=scored file (we override), output=labeled_journals.jsonl, model=gpt-4.1-mini (we override via env)

  **API/Type References**:
  - `src/mental_entropy/datagen/labeling_rubric.py` — Contains `LABELING_PROMPT`, `RESPONSE_SCHEMA`, `LABELING_FEW_SHOT_EXAMPLES`. Labels constrained to `{0.0, 0.25, 0.5, 0.75, 1.0}`.

  **WHY Each Reference Matters**:
  - `--max-entries 50` caps the run for a cheap validation pass
  - `OPENAI_MODEL` env var overrides the default `gpt-4.1-mini` to `gpt-4.1`
  - The resume logic means running full labeling (T4) after pilot will pick up where pilot left off

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Pilot produces ~50 valid labeled rows
    Tool: Bash (python one-liner)
    Preconditions: OPENAI_API_KEY is set, combined clean file exists from T2
    Steps:
      1. Run the labeling command (as specified above)
      2. Run: python3 -c "
         import json
         with open('data/all_journals_labeled.jsonl') as f:
             rows = [json.loads(l) for l in f if l.strip()]
         valid = [r for r in rows if isinstance(r.get('labels'), dict) and all(k in r['labels'] for k in ('ce','se','ne','cle'))]
         print(f'Total rows: {len(rows)}, Valid: {len(valid)}')
         assert len(valid) >= 45, f'Too few valid: {len(valid)}'
         anchors = {0.0, 0.25, 0.5, 0.75, 1.0}
         for r in valid:
             for k in ('ce','se','ne','cle'):
                 assert r['labels'][k] in anchors, f'Bad anchor: {k}={r[\"labels\"][k]}'
         print('PASS: All labels valid anchors')
         "
    Expected Result: "PASS: All labels valid anchors" with ≥45 valid rows
    Failure Indicators: Too few rows, invalid anchor values, API errors in stderr
    Evidence: .sisyphus/evidence/task-3-pilot-validation.txt

  Scenario: Label distribution is not degenerate
    Tool: Bash (python one-liner)
    Preconditions: Pilot labeled file exists
    Steps:
      1. Run: python3 -c "
         import json
         from collections import Counter
         with open('data/all_journals_labeled.jsonl') as f:
             rows = [json.loads(l) for l in f if l.strip()]
         for key in ('ce','se','ne','cle'):
             dist = Counter(r['labels'][key] for r in rows if 'labels' in r)
             print(f'{key}: {dict(sorted(dist.items()))}')
             assert len(dist) >= 2, f'{key} has only {len(dist)} unique values — degenerate!'
         print('PASS: All subscores have diverse label distribution')
         "
    Expected Result: Each subscore shows at least 2 different anchor values
    Failure Indicators: Any subscore has all entries at the same anchor
    Evidence: .sisyphus/evidence/task-3-pilot-distribution.txt

  Scenario: No entropy_analysis contamination in pilot output
    Tool: Bash (python one-liner)
    Preconditions: Pilot labeled file exists
    Steps:
      1. Run: python3 -c "
         import json
         with open('data/all_journals_labeled.jsonl') as f:
             for i, line in enumerate(f):
                 row = json.loads(line)
                 assert 'entropy_analysis' not in row, f'Line {i+1} has entropy_analysis'
         print('PASS: No entropy_analysis contamination')
         "
    Expected Result: "PASS: No entropy_analysis contamination"
    Failure Indicators: AssertionError with line number
    Evidence: .sisyphus/evidence/task-3-pilot-no-contamination.txt
  ```

  **Commit**: NO (groups with final commit)

---

- [ ] 4. Full labeling of all entries

  **What to do**:
  - Run `label_journals.py` WITHOUT `--max-entries` to label remaining entries (the resume logic picks up after the pilot's 50)
  - Command:
    ```bash
    OPENAI_MODEL=gpt-4.1 uv run python scripts/label_journals.py \
      --input data/all_journals_clean.jsonl \
      --output data/all_journals_labeled.jsonl
    ```
  - This will label the remaining entries (50 already done in pilot)
  - After completion, verify coverage is ≥ 95% of total input
  - Report exact count and any failures

  **Must NOT do**:
  - Do NOT delete the pilot output first — the script appends and skips already-labeled IDs
  - Do NOT modify any scripts

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Very long-running API operation (~45 min for ~1000 entries), needs patience and post-run verification
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after T3)
  - **Blocks**: Task 5
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - `scripts/label_journals.py:319-365` — `run_labeling()`: loads existing IDs from output, skips already-labeled, processes remaining in batches, prints progress
  - `scripts/label_journals.py:352` — Output opened in append mode (`"a"`)
  - `scripts/label_journals.py:306-309` — Failed rows print warning to stderr but don't abort

  **WHY Each Reference Matters**:
  - The append + ID-skip resume logic means we just run the same command again — it picks up where the pilot left off
  - Failed rows are warned in stderr, not fatal. Count the output to know actual success rate.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Full labeling produces ≥95% coverage
    Tool: Bash (python one-liner)
    Preconditions: Pilot completed (T3), OPENAI_API_KEY set
    Steps:
      1. Run the full labeling command (as specified above)
      2. Run: python3 -c "
         import json
         with open('data/all_journals_clean.jsonl') as f:
             total_input = sum(1 for l in f if l.strip())
         with open('data/all_journals_labeled.jsonl') as f:
             rows = [json.loads(l) for l in f if l.strip()]
         valid = [r for r in rows if isinstance(r.get('labels'), dict) and all(k in r['labels'] for k in ('ce','se','ne','cle'))]
         coverage = len(valid) / total_input
         print(f'Total input: {total_input}, Labeled: {len(rows)}, Valid: {len(valid)}, Coverage: {coverage:.1%}')
         assert coverage >= 0.95, f'Coverage too low: {coverage:.1%}'
         print('PASS')
         "
    Expected Result: Coverage ≥ 95% of input entries
    Failure Indicators: Coverage below 95% — indicates systematic labeling failures
    Evidence: .sisyphus/evidence/task-4-full-coverage.txt

  Scenario: Full label distribution report
    Tool: Bash (python one-liner)
    Preconditions: Full labeled file exists
    Steps:
      1. Run: python3 -c "
         import json
         from collections import Counter
         with open('data/all_journals_labeled.jsonl') as f:
             rows = [json.loads(l) for l in f if l.strip()]
         for key in ('ce','se','ne','cle'):
             dist = Counter(r['labels'][key] for r in rows if 'labels' in r)
             total = sum(dist.values())
             print(f'{key}: ' + ', '.join(f'{v:.2f}={c} ({c/total:.0%})' for v, c in sorted(dist.items())))
             assert len(dist) >= 3, f'{key} has only {len(dist)} anchors used — may be too narrow'
         print('PASS: Good label diversity')
         "
    Expected Result: Each subscore uses at least 3 of 5 anchor values
    Failure Indicators: Degenerate distribution
    Evidence: .sisyphus/evidence/task-4-full-distribution.txt

  Scenario: Human entries are labeled correctly
    Tool: Bash (python one-liner)
    Preconditions: Full labeled file exists
    Steps:
      1. Run: python3 -c "
         import json
         with open('data/all_journals_labeled.jsonl') as f:
             rows = [json.loads(l) for l in f if l.strip()]
         human = [r for r in rows if isinstance(r.get('id'), str) and r['id'].startswith('human_')]
         valid_human = [r for r in human if isinstance(r.get('labels'), dict)]
         print(f'Human entries: {len(human)}, With labels: {len(valid_human)}')
         assert len(valid_human) >= len(human) * 0.95, f'Too many human entries missing labels'
         print('PASS: Human entries labeled')
         "
    Expected Result: ≥95% of human entries have labels
    Failure Indicators: Many human entries missing labels
    Evidence: .sisyphus/evidence/task-4-human-labels.txt
  ```

  **Commit**: NO (groups with final commit)

---

- [ ] 5. Run synthetic-only validation with ID remap and auto gate decision

  **What to do**:
  - Build two temporary validation inputs before running `validate_labels.py`:
    1. `data/labeled_synthetic_only.jsonl` — filter `data/all_journals_labeled.jsonl` to synthetic rows only (`source in {"legacy","v3"}`)
    2. `data/synthetic_journals_800_scored_remapped.jsonl` — copy `data/synthetic_journals_800_scored.jsonl` and normalize IDs to match training corpus:
       - First 409 rows = legacy, keep IDs as-is
       - Next 400 rows = v3, set `id = original_id + 1000`
  - Verify remapped scored file has 809 rows and unique IDs
  - Run `validate_labels.py` on the normalized files:
    ```bash
    uv run python scripts/validate_labels.py \
      --labeled-input data/labeled_synthetic_only.jsonl \
      --scored-input data/synthetic_journals_800_scored_remapped.jsonl
    ```
  - Capture full output and save as evidence
  - Analyze results against thresholds:
    - Spearman rho >= 0.3 per subscore
    - Disagreement rate <= 25% per subscore (|delta| > 0.3)
  - **Auto gate rule**:
    - If all thresholds pass: mark task PASS and proceed to Task 6
    - If any threshold fails: mark task FAIL and halt execution for remediation

  **Must NOT do**:
  - Do NOT run validator against mixed human+synthetic labeled file
  - Do NOT modify `validate_labels.py`
  - Do NOT mutate source datasets (`all_journals_labeled.jsonl` or `synthetic_journals_800_scored.jsonl`)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires preprocessing + metric interpretation + pass/fail gate decision
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (gate checkpoint)
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References**:

  **Pattern References**:
  - `scripts/validate_labels.py:135-147` — scored file is indexed by unique ID; duplicate IDs are dropped (`duplicate_scored_ids`)
  - `scripts/validate_labels.py:170-173` — positional fallback only uses `scored_no_id`, not ID mismatches
  - `scripts/validate_labels.py:198-205` — prints `compared_rows`, matching stats, and ID health

  **API/Type References**:
  - `scripts/validate_labels.py:55-63` — labels are read from `labels` dict
  - `scripts/validate_labels.py:66-73` — heuristics are read from `entropy_analysis.entropy_{key}`
  - `scripts/validate_labels.py:20` — disagreement threshold constant (0.3)

  **WHY Each Reference Matters**:
  - The scored synthetic file has duplicate IDs across legacy/v3. Without remapping, `validate_labels.py` drops duplicates and under-compares rows.
  - Positional fallback does NOT rescue ID mismatches when IDs are present, so explicit ID normalization is required.
  - Synthetic-only validation avoids mismatch with human rows that have no heuristic entropy fields.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Remapped scored input has full synthetic coverage and unique IDs
    Tool: Bash (python one-liner)
    Preconditions: all_journals_labeled + synthetic_journals_800_scored exist
    Steps:
      1. Build labeled_synthetic_only.jsonl and synthetic_journals_800_scored_remapped.jsonl
      2. Run: python3 -c "
         import json
         scored = [json.loads(l) for l in open('data/synthetic_journals_800_scored_remapped.jsonl') if l.strip()]
         labeled = [json.loads(l) for l in open('data/labeled_synthetic_only.jsonl') if l.strip()]
         assert len(scored) == 809, f'scored rows={len(scored)} != 809'
         assert len(set(str(r['id']) for r in scored)) == len(scored), 'duplicate IDs remain in scored remap'
         assert len(labeled) >= 760, f'labeled synthetic too low: {len(labeled)}'
         print(f'PASS: scored={len(scored)} unique, labeled_synth={len(labeled)}')
         "
    Expected Result: scored=809 unique IDs, labeled synthetic >=760
    Failure Indicators: duplicate IDs in scored remap, unexpectedly low labeled synthetic count
    Evidence: .sisyphus/evidence/task-5-remap-health.txt

  Scenario: Validation report generated and gate decision computed
    Tool: Bash (capture stdout + parser)
    Preconditions: Remapped inputs built
    Steps:
      1. Run validation command (as specified above), capture full stdout
      2. Parse `compared_rows` and per-subscore metrics
      3. Assert compared_rows >= 760 (>=95% of 809)
      4. For each subscore assert: spearman >= 0.3 AND disagreement_rate <= 25%
      5. Emit gate decision: PROCEED or FAIL
    Expected Result: All 4 subscores pass thresholds => PROCEED
    Failure Indicators: compared_rows < 760, any spearman < 0.3, any disagreement_rate > 25%
    Evidence: .sisyphus/evidence/task-5-validation-report.txt
  ```

  **Commit**: NO (groups with final commit)

---

- [ ] 6. Split labeled data into train/val/test

  **What to do**:
  - Write an inline Python script that:
    1. Reads `data/all_journals_labeled.jsonl`
    2. Separates entries by source: synthetic (int IDs) vs human (string IDs starting with `human_`)
    3. Shuffles human entries with `random.seed(42)` for reproducibility
    4. Splits human entries 60/20/20: ~130 train, ~43 val, ~43 test
    5. Combines ALL synthetic entries + 60% human entries → `data/train_labeled.jsonl`
    6. Writes 20% human → `data/val_labeled.jsonl`
    7. Writes 20% human → `data/test_labeled.jsonl`
  - Report: total entries, synthetic count, human count, per-split counts
  - Verify: no ID overlap between splits, val/test are 100% human-only

  **Must NOT do**:
  - Do NOT put any synthetic entries in val or test sets
  - Do NOT modify the labeled data (just partition it)
  - Do NOT use a non-deterministic seed

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple data partitioning with well-defined rules
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (solo)
  - **Blocks**: Task 7
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - `scripts/train_gam.py:77-90` — `_extract_labels()`: expects `labels.ce` and `labels.ne` in each row
  - `scripts/train_xgb.py:85-100` — `_extract_label()`: 3-tier fallback. With clean labeled data, only tier-1 (`labels` dict) is used.

  **WHY Each Reference Matters**:
  - Training scripts read `labels` dict from each row. The split files must preserve this field exactly as produced by the labeler.
  - The `source` field is used to determine synthetic vs human, so it must be present in all rows.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Splits are correct, non-overlapping, and val/test are human-only
    Tool: Bash (python one-liner)
    Preconditions: Labeled file exists from T4
    Steps:
      1. Run the splitting script
      2. Run: python3 -c "
         import json
         train = [json.loads(l) for l in open('data/train_labeled.jsonl') if l.strip()]
         val = [json.loads(l) for l in open('data/val_labeled.jsonl') if l.strip()]
         test = [json.loads(l) for l in open('data/test_labeled.jsonl') if l.strip()]
         train_ids = set(r['id'] for r in train)
         val_ids = set(r['id'] for r in val)
         test_ids = set(r['id'] for r in test)
         # No overlap
         assert not (train_ids & val_ids), 'Train/Val overlap'
         assert not (train_ids & test_ids), 'Train/Test overlap'
         assert not (val_ids & test_ids), 'Val/Test overlap'
         # Val and Test are human-only
         assert all(isinstance(r['id'], str) and r['id'].startswith('human_') for r in val), 'Val has non-human'
         assert all(isinstance(r['id'], str) and r['id'].startswith('human_') for r in test), 'Test has non-human'
         # Train has all 809 synthetic
         synth_in_train = [r for r in train if isinstance(r['id'], int)]
         assert len(synth_in_train) == 809, f'Expected 809 synthetic in train, got {len(synth_in_train)}'
         # All entries accounted for
         total = len(train) + len(val) + len(test)
         with open('data/all_journals_labeled.jsonl') as f:
             expected = sum(1 for l in f if l.strip())
         assert total == expected, f'Split total {total} != labeled total {expected}'
         print(f'Train: {len(train)} (synth: {len(synth_in_train)}, human: {len(train)-len(synth_in_train)})')
         print(f'Val: {len(val)} (all human)')
         print(f'Test: {len(test)} (all human)')
         print(f'Total: {total}')
         print('PASS')
         "
    Expected Result: Train ≈ 949, Val ≈ 43, Test ≈ 43, no overlap, val/test human-only
    Failure Indicators: Overlap between splits, non-human in val/test, wrong synthetic count
    Evidence: .sisyphus/evidence/task-6-split-validation.txt

  Scenario: All split entries have valid labels
    Tool: Bash (python one-liner)
    Preconditions: Split files exist
    Steps:
      1. Run: python3 -c "
         import json
         anchors = {0.0, 0.25, 0.5, 0.75, 1.0}
         for name in ('train', 'val', 'test'):
             with open(f'data/{name}_labeled.jsonl') as f:
                 rows = [json.loads(l) for l in f if l.strip()]
             for r in rows:
                 labels = r.get('labels', {})
                 for k in ('ce','se','ne','cle'):
                     assert k in labels, f'{name}/{r[\"id\"]}: missing labels.{k}'
                     assert labels[k] in anchors, f'{name}/{r[\"id\"]}: bad value {labels[k]}'
             print(f'{name}: {len(rows)} rows, all labels valid')
         print('PASS')
         "
    Expected Result: All rows in all splits have valid labels
    Failure Indicators: Missing or invalid labels
    Evidence: .sisyphus/evidence/task-6-split-labels.txt
  ```

  **Commit**: NO (groups with final commit)

---

- [ ] 7. Pre-training data validation

  **What to do**:
  - Before training, programmatically verify `data/train_labeled.jsonl` is safe for training:
    1. Every row has `labels` dict with all 4 keys (`ce`, `se`, `ne`, `cle`)
    2. All label values are in `{0.0, 0.25, 0.5, 0.75, 1.0}`
    3. No row has `entropy_analysis` key (prevents `train_xgb.py` fallback contamination)
    4. Every row has non-empty `journal_text` (needed for on-the-fly feature computation)
    5. Total valid rows ≥ 900 (expected ~949)
  - If any check fails, report the issue and STOP — do not proceed to training

  **Must NOT do**:
  - Do NOT proceed to training if validation fails
  - Do NOT modify the split data — filter to a new file if needed

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple data validation, no complex logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 5 (gates T8 and T9)
  - **Blocks**: Tasks 8, 9
  - **Blocked By**: Task 6

  **References**:

  **Pattern References**:
  - `scripts/train_xgb.py:85-100` — `_extract_label()`: 3-tier fallback including `entropy_analysis`. This is WHY we validate no `entropy_analysis` exists.
  - `scripts/train_gam.py:77-90` — `_extract_labels()`: strict, only reads `labels.ce` and `labels.ne`. No fallback.
  - `scripts/label_journals.py:100` — `ANCHOR_VALUES` = `{0.0, 0.25, 0.5, 0.75, 1.0}`

  **WHY Each Reference Matters**:
  - `train_xgb.py`'s fallback is the primary risk. If `entropy_analysis` is present AND `labels` is missing for a row, training silently uses heuristic scores as ground truth.
  - `train_gam.py` is strict (good), but we validate anyway for consistency.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All pre-training validations pass on train split
    Tool: Bash (python one-liner)
    Preconditions: train_labeled.jsonl exists from T6
    Steps:
      1. Run: python3 -c "
         import json
         anchors = {0.0, 0.25, 0.5, 0.75, 1.0}
         valid = 0
         with open('data/train_labeled.jsonl') as f:
             for i, line in enumerate(f, 1):
                 row = json.loads(line)
                 assert 'entropy_analysis' not in row, f'Row {i}: has entropy_analysis!'
                 labels = row.get('labels')
                 assert isinstance(labels, dict), f'Row {i}: missing labels dict'
                 for k in ('ce','se','ne','cle'):
                     assert k in labels, f'Row {i}: missing labels.{k}'
                     assert labels[k] in anchors, f'Row {i}: labels.{k}={labels[k]} not in anchors'
                 text = row.get('journal_text', '')
                 assert isinstance(text, str) and text.strip(), f'Row {i}: empty journal_text'
                 valid += 1
         assert valid >= 900, f'Only {valid} valid rows (need >= 900)'
         print(f'PASS: {valid} rows validated — all labels complete, no contamination, all have text')
         "
    Expected Result: "PASS: N rows validated" with N ≥ 900
    Failure Indicators: AssertionError with specific row and issue
    Evidence: .sisyphus/evidence/task-7-pretrain-validation.txt
  ```

  **Commit**: NO (groups with final commit)

---

- [ ] 8. Train GAM models (CE/NE) on train split

  **What to do**:
  - Run the GAM training script on the TRAIN SPLIT (not the full labeled file):
    ```bash
    uv run python scripts/train_gam.py \
      --labeled-data data/train_labeled.jsonl \
      --output-dir src/mental_entropy/models \
      --k-folds 5
    ```
  - This will:
    1. Load all ~949 train split rows
    2. Compute CE and NE features from `journal_text` on-the-fly (embedding each entry)
    3. Train CE and NE pyGAM regressors with monotonic constraints
    4. Run 5-fold cross-validation
    5. Overwrite `gam_ce.joblib`, `gam_ne.joblib`, `gam_metadata.json`
  - **NOTE**: Feature computation is expensive (~30-60 min on CPU for ~949 embeddings)
  - After completion, review `gam_metadata.json` for:
    - `n_samples` should be ≥ 900 (not 20)
    - CV R² should be > 0 (better than mean prediction)
    - CV MAE should be < 0.25 (reasonable for 5-anchor ordinal labels)

  **Must NOT do**:
  - Do NOT train on `data/all_journals_labeled.jsonl` — use ONLY `data/train_labeled.jsonl`
  - Do NOT change the feature order or monotonicity constraints
  - Do NOT modify the training script
  - Do NOT reduce k-folds below 5

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Long-running computation (~30-60 min), needs patience and post-run analysis
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES — parallel with Task 9
  - **Parallel Group**: Wave 5 (with T9, after T7)
  - **Blocks**: Tasks 10, 11
  - **Blocked By**: Task 7

  **References**:

  **Pattern References**:
  - `scripts/train_gam.py:273-368` — `main()`: full training flow from CLI args to artifact output
  - `scripts/train_gam.py:93-111` — `_extract_row_features()`: computes features from `journal_text` on-the-fly via `embed_journal_entry()` if not precomputed
  - `scripts/train_gam.py:186-206` — `_fit_gam()`: gridsearch over lambda values, monotonic constraints

  **API/Type References**:
  - `src/mental_entropy/scoring/constants.py` — `CE_FEATURE_ORDER` (24 features), `NE_FEATURE_ORDER` (13 features), `CE_MONOTONICITY`, `NE_MONOTONICITY`
  - `src/mental_entropy/models/gam_metadata.json` — Current metadata (n_samples=20) to compare against

  **WHY Each Reference Matters**:
  - Feature computation from `journal_text` is the bottleneck — each of ~949 entries needs embedding model inference
  - Monotonicity constraints ensure models behave sensibly

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: GAM training produces valid production artifacts
    Tool: Bash (python)
    Preconditions: Pre-training validation passed (T7)
    Steps:
      1. Run training command (as specified above), capture stdout
      2. Run: python3 -c "
         import json
         m = json.load(open('src/mental_entropy/models/gam_metadata.json'))
         ce_n = m['models']['ce']['n_samples']
         ne_n = m['models']['ne']['n_samples']
         ce_r2 = m['models']['ce']['cv']['mean']['r2']
         ne_r2 = m['models']['ne']['cv']['mean']['r2']
         ce_mae = m['models']['ce']['cv']['mean']['mae']
         ne_mae = m['models']['ne']['cv']['mean']['mae']
         print(f'CE: n={ce_n}, R2={ce_r2:.4f}, MAE={ce_mae:.4f}')
         print(f'NE: n={ne_n}, R2={ne_r2:.4f}, MAE={ne_mae:.4f}')
         assert ce_n >= 900, f'CE samples too low: {ce_n}'
         assert ne_n >= 900, f'NE samples too low: {ne_n}'
         print('PASS: Production GAM models trained')
         "
    Expected Result: n_samples ≥ 900 for both CE and NE
    Failure Indicators: Low sample count, training errors
    Evidence: .sisyphus/evidence/task-8-gam-training.txt

  Scenario: GAM artifacts are loadable
    Tool: Bash (python)
    Preconditions: Training completed
    Steps:
      1. Run: python3 -c "
         import joblib
         ce = joblib.load('src/mental_entropy/models/gam_ce.joblib')
         ne = joblib.load('src/mental_entropy/models/gam_ne.joblib')
         print(f'CE model type: {type(ce).__name__}')
         print(f'NE model type: {type(ne).__name__}')
         print('PASS: Both models loadable')
         "
    Expected Result: Both models load without error
    Failure Indicators: Import/load errors, wrong model type
    Evidence: .sisyphus/evidence/task-8-gam-loadable.txt
  ```

  **Commit**: NO (groups with final commit)

---

- [ ] 9. Train XGBoost models (SE/CLE) on train split

  **What to do**:
  - Run the XGBoost training script on the TRAIN SPLIT:
    ```bash
    uv run python scripts/train_xgb.py \
      --labeled-data data/train_labeled.jsonl \
      --output-dir src/mental_entropy/models \
      --k-folds 5
    ```
  - This will:
    1. Load all ~949 train split rows
    2. Compute SE and CLE features from `journal_text` on-the-fly
    3. Train SE and CLE XGBoost regressors with monotone constraints
    4. Run 5-fold cross-validation
    5. Overwrite `xgb_se.json`, `xgb_cle.json`, `xgb_metadata.json`
  - **NOTE**: If running in parallel with T8, both scripts will independently embed all ~949 entries. This is expected (no shared state).
  - After completion, review `xgb_metadata.json` for:
    - `n_rows` should be ≥ 900 (not 20)
    - CV MAE should be < 0.25

  **Must NOT do**:
  - Do NOT train on `data/all_journals_labeled.jsonl` — use ONLY `data/train_labeled.jsonl`
  - Do NOT change monotone constraints or hyperparameters
  - Do NOT modify the training script

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Long-running computation, needs patience and post-run analysis
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES — parallel with Task 8
  - **Parallel Group**: Wave 5 (with T8, after T7)
  - **Blocks**: Tasks 10, 11
  - **Blocked By**: Task 7

  **References**:

  **Pattern References**:
  - `scripts/train_xgb.py:314-379` — `main()`: full training flow
  - `scripts/train_xgb.py:121-167` — `_extract_or_compute_features()`: computes SE/CLE features from `journal_text` on-the-fly
  - `scripts/train_xgb.py:170-189` — `_load_rows()`: loads and validates all rows; fail-fast on bad rows

  **API/Type References**:
  - `src/mental_entropy/scoring/constants.py` — `SE_FEATURE_ORDER` (9 features), `CLE_FEATURE_ORDER` (14 features), `SE_MONOTONE_CONSTRAINTS`, `CLE_MONOTONE_CONSTRAINTS`
  - `scripts/train_xgb.py:32-42` — `FIXED_PARAMS`: XGBoost hyperparameters (max_depth=3, lr=0.05, n_estimators=200)

  **WHY Each Reference Matters**:
  - `_load_rows()` is fail-fast — any invalid row aborts. Pre-training validation (T7) ensures this won't happen.
  - FIXED_PARAMS are tuned for small-data regularization.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: XGBoost training produces valid production artifacts
    Tool: Bash (python)
    Preconditions: Pre-training validation passed (T7)
    Steps:
      1. Run training command (as specified above), capture stdout
      2. Run: python3 -c "
         import json
         m = json.load(open('src/mental_entropy/models/xgb_metadata.json'))
         n = m['n_rows']
         se_mae = m['models']['se']['cv']['mean_mae']
         cle_mae = m['models']['cle']['cv']['mean_mae']
         print(f'Rows: {n}')
         print(f'SE MAE: {se_mae:.4f}')
         print(f'CLE MAE: {cle_mae:.4f}')
         assert n >= 900, f'Rows too low: {n}'
         print('PASS: Production XGB models trained')
         "
    Expected Result: n_rows ≥ 900, MAE values reported
    Failure Indicators: Low row count, training errors
    Evidence: .sisyphus/evidence/task-9-xgb-training.txt

  Scenario: XGBoost artifacts are loadable
    Tool: Bash (python)
    Preconditions: Training completed
    Steps:
      1. Run: uv run python -c "
         import xgboost as xgb
         se = xgb.Booster()
         se.load_model('src/mental_entropy/models/xgb_se.json')
         cle = xgb.Booster()
         cle.load_model('src/mental_entropy/models/xgb_cle.json')
         print(f'SE model loaded, features: {se.num_features()}')
         print(f'CLE model loaded, features: {cle.num_features()}')
         print('PASS: Both XGB models loadable')
         "
    Expected Result: Both models load and report correct feature counts (SE=9, CLE=14)
    Failure Indicators: Load errors, wrong feature counts
    Evidence: .sisyphus/evidence/task-9-xgb-loadable.txt
  ```

  **Commit**: NO (groups with final commit)

---

- [ ] 10. Post-training evaluation on val set

  **What to do**:
  - Score every entry in `data/val_labeled.jsonl` using the newly trained models
  - Compare predicted subscores against gpt-4.1 labels
  - Compute per-subscore MAE and RMSE on the val set
  - This is the first real quality signal on human data — the val set is human-only
  - Write an inline Python script that:
    1. Loads each val entry
    2. Runs `embed_journal_entry(journal_text)`
    3. Computes all 4 subscores using `score_*_from_result()`
    4. Compares each predicted score to the label
    5. Reports per-subscore MAE, RMSE, and Spearman ρ
  - **NOTE**: This requires `uv run` since it imports `mental_entropy` scoring functions. ~43 entries × embedding = ~5 min on CPU.
  - Save full evaluation report as evidence
  - This is INFORMATIONAL — results don't block the commit, but are reported for transparency

  **Must NOT do**:
  - Do NOT retrain based on val results (that defeats the purpose)
  - Do NOT modify val data or models
  - Do NOT use val data for any model selection or tuning

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires running the scoring pipeline on ~43 entries and computing evaluation metrics
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES — parallel with Task 11
  - **Parallel Group**: Wave 6 (with T11, after T8+T9)
  - **Blocks**: Task 12
  - **Blocked By**: Tasks 8, 9

  **References**:

  **Pattern References**:
  - `src/mental_entropy/__init__.py` — Public API: `embed_journal_entry`, `score_ce_from_result`, `score_se_from_result`, `score_ne_from_result`, `score_cle_from_result`
  - `src/mental_entropy/scoring/` — Scoring module that loads model artifacts and produces SubScore objects with `.score` attribute

  **API/Type References**:
  - `src/mental_entropy/scoring/gam_scorers.py` — GAM scoring implementation used by `score_ce_from_result` and `score_ne_from_result`
  - `src/mental_entropy/scoring/xgb_scorers.py` — XGBoost scoring implementation used by `score_se_from_result` and `score_cle_from_result`
  - `src/mental_entropy/scoring/types.py` — `SubScore` type with `.score` float field in [0, 1]

  **WHY Each Reference Matters**:
  - The scoring functions load model artifacts from `src/mental_entropy/models/` — which were just overwritten by T8 and T9. This validates that the new models produce reasonable predictions on unseen human data.
  - MAE < 0.25 would indicate the models generalize adequately from synthetic+human train to human-only val.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Val set evaluation produces per-subscore metrics
    Tool: Bash (uv run python)
    Preconditions: Models trained (T8, T9), val split exists (T6)
    Steps:
      1. Run: uv run python -c "
         import json, math
         from mental_entropy import embed_journal_entry, score_ce_from_result, score_se_from_result, score_ne_from_result, score_cle_from_result
         with open('data/val_labeled.jsonl') as f:
             rows = [json.loads(l) for l in f if l.strip()]
         errors = {k: [] for k in ('ce','se','ne','cle')}
         scorers = {'ce': score_ce_from_result, 'se': score_se_from_result, 'ne': score_ne_from_result, 'cle': score_cle_from_result}
         for i, r in enumerate(rows):
             result = embed_journal_entry(r['journal_text'])
             for k, scorer in scorers.items():
                 pred = scorer(result).score
                 label = r['labels'][k]
                 errors[k].append(abs(pred - label))
             if (i+1) % 10 == 0:
                 print(f'Scored {i+1}/{len(rows)}...')
         print(f'\\n=== Val Set Evaluation ({len(rows)} human entries) ===')
         for k in ('ce','se','ne','cle'):
             mae = sum(errors[k]) / len(errors[k])
             rmse = math.sqrt(sum(e**2 for e in errors[k]) / len(errors[k]))
             print(f'{k.upper()}: MAE={mae:.4f}, RMSE={rmse:.4f}')
         print('\\nEvaluation complete')
         "
    Expected Result: Per-subscore MAE and RMSE reported. MAE < 0.25 per subscore is ideal but not blocking.
    Failure Indicators: Exceptions during scoring, all-zero predictions, MAE > 0.5 (extremely poor)
    Evidence: .sisyphus/evidence/task-10-val-evaluation.txt

  Scenario: All val entries produce valid finite scores
    Tool: Bash (uv run python)
    Preconditions: Same as above
    Steps:
      1. Run: uv run python -c "
         import json, math
         from mental_entropy import embed_journal_entry, score_ce_from_result, score_se_from_result, score_ne_from_result, score_cle_from_result, mes_score
         with open('data/val_labeled.jsonl') as f:
             rows = [json.loads(l) for l in f if l.strip()]
         for r in rows:
             result = embed_journal_entry(r['journal_text'])
             ce = score_ce_from_result(result)
             se = score_se_from_result(result)
             ne = score_ne_from_result(result)
             cle = score_cle_from_result(result)
             m = mes_score(ce.score, se.score, ne.score, cle.score)
             assert all(0 <= s.score <= 1 for s in (ce, se, ne, cle)), f'{r[\"id\"]}: score out of [0,1]'
             assert 0 <= m.score <= 100, f'{r[\"id\"]}: MES out of [0,100]'
             assert all(math.isfinite(s.score) for s in (ce, se, ne, cle)), f'{r[\"id\"]}: non-finite score'
         print(f'PASS: All {len(rows)} val entries produce valid finite scores')
         "
    Expected Result: "PASS: All N val entries produce valid finite scores"
    Failure Indicators: Score out of range, non-finite values, exceptions
    Evidence: .sisyphus/evidence/task-10-val-finite.txt
  ```

  **Commit**: NO (groups with final commit)

---

- [ ] 11. Full verification (pytest + smoke test + metadata check)

  **What to do**:
  - Run the full test suite:
    ```bash
    uv run pytest tests/ -v
    ```
  - Run end-to-end smoke test:
    ```bash
    uv run python -c "
    from mental_entropy import embed_journal_entry, score_ce_from_result, score_se_from_result, score_ne_from_result, score_cle_from_result, mes_score
    r = embed_journal_entry('Today was scattered but I found focus after lunch. The evening felt calmer.')
    ce = score_ce_from_result(r)
    se = score_se_from_result(r)
    ne = score_ne_from_result(r)
    cle = score_cle_from_result(r)
    m = mes_score(ce.score, se.score, ne.score, cle.score)
    print(f'CE={ce.score:.3f} SE={se.score:.3f} NE={ne.score:.3f} CLE={cle.score:.3f} MES={m.score}')
    assert all(0 <= s.score <= 1 for s in (ce, se, ne, cle))
    assert 0 <= m.score <= 100
    print('PASS')
    "
    ```
  - Verify metadata shows production counts:
    ```bash
    python3 -c "
    import json
    gam = json.load(open('src/mental_entropy/models/gam_metadata.json'))
    xgb_m = json.load(open('src/mental_entropy/models/xgb_metadata.json'))
    print(f'GAM CE samples: {gam[\"models\"][\"ce\"][\"n_samples\"]}')
    print(f'GAM NE samples: {gam[\"models\"][\"ne\"][\"n_samples\"]}')
    print(f'XGB rows: {xgb_m[\"n_rows\"]}')
    assert gam['models']['ce']['n_samples'] >= 900
    assert xgb_m['n_rows'] >= 900
    print('PASS: All models production-trained')
    "
    ```

  **Must NOT do**:
  - Do NOT modify test assertions to make them pass
  - Do NOT modify scoring code
  - If tests fail, report the failures — they indicate real regressions

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Comprehensive verification with multiple validation steps
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES — parallel with Task 10
  - **Parallel Group**: Wave 6 (with T10, after T8+T9)
  - **Blocks**: Task 12
  - **Blocked By**: Tasks 8, 9

  **References**:

  **Pattern References**:
  - `tests/` — Full test suite (11 embedding + 18 CE + 23 SE + 37 NE + 51 CLE tests)
  - `src/mental_entropy/scoring/loader.py` — Model loading uses `importlib.resources`, module-level cache

  **WHY Each Reference Matters**:
  - Tests load real model artifacts — they'll catch format/loading regressions from new models
  - Module-level cache means pytest (which creates fresh process) will load the new models correctly

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Full test suite passes
    Tool: Bash
    Preconditions: All model artifacts updated (T8, T9)
    Steps:
      1. Run: uv run pytest tests/ -v
      2. Check exit code is 0
    Expected Result: All tests pass (exit 0)
    Failure Indicators: Any test failures or errors
    Evidence: .sisyphus/evidence/task-11-pytest-results.txt

  Scenario: End-to-end pipeline smoke test
    Tool: Bash (python)
    Preconditions: Models updated
    Steps:
      1. Run the smoke test command (as specified above)
    Expected Result: All subscores in [0,1], MES in [0,100], "PASS" printed
    Failure Indicators: Scores out of range, exceptions, non-finite values
    Evidence: .sisyphus/evidence/task-11-smoke-test.txt

  Scenario: Metadata shows production counts
    Tool: Bash (python)
    Preconditions: Metadata files updated
    Steps:
      1. Run the metadata check command (as specified above)
    Expected Result: "PASS: All models production-trained" with counts ≥ 900
    Failure Indicators: Counts still showing 20 (old mini-trained values)
    Evidence: .sisyphus/evidence/task-11-metadata-check.txt
  ```

  **Commit**: NO (groups with T12)

---

- [ ] 12. Commit everything

  **What to do**:
  - Stage and commit all new/changed files:
    - `data/human_journals_clean.jsonl` (new — converted/filtered human journals)
    - `data/all_journals_clean.jsonl` (new — combined clean corpus)
    - `data/all_journals_labeled.jsonl` (new — gpt-4.1 labels)
    - `data/train_labeled.jsonl` (new — training split)
    - `data/val_labeled.jsonl` (new — validation split, human-only)
    - `data/test_labeled.jsonl` (new — test split, human-only)
    - `src/mental_entropy/models/gam_ce.joblib` (updated)
    - `src/mental_entropy/models/gam_ne.joblib` (updated)
    - `src/mental_entropy/models/xgb_se.json` (updated)
    - `src/mental_entropy/models/xgb_cle.json` (updated)
    - `src/mental_entropy/models/gam_metadata.json` (updated)
    - `src/mental_entropy/models/xgb_metadata.json` (updated)
  - Commit message should include CV metrics summary and dataset composition:
    ```
    feat(models): retrain all subscore models on N entries (809 synthetic + M human)

    - Data: 809 synthetic + M human journals (30-word min, deduped)
    - Labels: gpt-4.1, 5-anchor ordinal {0.0, 0.25, 0.5, 0.75, 1.0}
    - Train: 809 synthetic + ~130 human | Val: ~43 human | Test: ~43 human
    - CE GAM: N samples, R²=X.XX, MAE=X.XX
    - NE GAM: N samples, R²=X.XX, MAE=X.XX
    - SE XGB: N samples, MAE=X.XX
    - CLE XGB: N samples, MAE=X.XX
    - Val MAE: CE=X.XX, SE=X.XX, NE=X.XX, CLE=X.XX

    Previous: 20-row mini-trained placeholders (negative R²)
    ```
  - Do NOT push unless user explicitly asks

  **Must NOT do**:
  - Do NOT commit `.env` or any file containing API keys
  - Do NOT push to remote without explicit user request
  - Do NOT amend existing commits

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple git operations
  - **Skills**: [`git-master`]
    - `git-master`: Safe commit workflow with proper staging and message formatting

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 6 (after T10 and T11)
  - **Blocks**: None
  - **Blocked By**: Tasks 10, 11

  **References**:

  **Pattern References**:
  - `scripts/committer` — Repo-local safe commit helper
  - `.gitignore` — Check that `.env` and API keys are excluded

  **WHY Each Reference Matters**:
  - Must verify `.env` is gitignored before staging

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Commit includes all expected files and no secrets
    Tool: Bash (git)
    Preconditions: All tasks T1-T11 completed
    Steps:
      1. Run: git status to verify expected files are changed/added
      2. Verify .env is NOT in staged files
      3. Create commit with specified message format
      4. Run: git log -1 --stat to verify commit contents
    Expected Result: Commit created with all data files + model artifacts, no .env
    Failure Indicators: Missing files, .env included, commit fails
    Evidence: .sisyphus/evidence/task-12-commit.txt
  ```

  **Commit**: YES
  - Message: `feat(models): retrain all subscore models on N entries (809 synthetic + M human)`
  - Files: `data/human_journals_clean.jsonl`, `data/all_journals_clean.jsonl`, `data/all_journals_labeled.jsonl`, `data/train_labeled.jsonl`, `data/val_labeled.jsonl`, `data/test_labeled.jsonl`, `src/mental_entropy/models/*`
  - Pre-commit: `uv run pytest tests/ -v`

---

---

## Final Verification Wave

> Task 11 serves as the final verification wave — it runs pytest, smoke tests, and metadata checks. Task 10 provides the human-data quality signal. Together they cover F1-F4 equivalent checks.

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 12 | `feat(models): retrain all subscore models on N entries (809 synthetic + M human) with gpt-4.1 labels` | `data/human_journals_clean.jsonl`, `data/all_journals_clean.jsonl`, `data/all_journals_labeled.jsonl`, `data/train_labeled.jsonl`, `data/val_labeled.jsonl`, `data/test_labeled.jsonl`, `src/mental_entropy/models/*` | `uv run pytest tests/ -v` |

---

## Success Criteria

### Verification Commands
```bash
# All tests pass
uv run pytest tests/ -v  # Expected: exit 0, all tests pass

# Metadata shows production counts
python3 -c "import json; m=json.load(open('src/mental_entropy/models/gam_metadata.json')); print(f'CE samples: {m[\"models\"][\"ce\"][\"n_samples\"]}')"
# Expected: >= 900 (train split, not 20)

python3 -c "import json; m=json.load(open('src/mental_entropy/models/xgb_metadata.json')); print(f'XGB rows: {m[\"n_rows\"]}')"
# Expected: >= 900 (train split, not 20)

# End-to-end smoke test
uv run python -c "
from mental_entropy import embed_journal_entry, score_ce_from_result, score_se_from_result, score_ne_from_result, score_cle_from_result, mes_score
r = embed_journal_entry('Today was scattered but I found focus after lunch.')
ce = score_ce_from_result(r); se = score_se_from_result(r); ne = score_ne_from_result(r); cle = score_cle_from_result(r)
m = mes_score(ce.score, se.score, ne.score, cle.score)
assert all(0 <= s.score <= 1 for s in (ce, se, ne, cle))
assert 0 <= m.score <= 100
print(f'CE={ce.score:.3f} SE={se.score:.3f} NE={ne.score:.3f} CLE={cle.score:.3f} MES={m.score}')
print('PASS')
"

# Val set evaluation exists
test -f .sisyphus/evidence/task-10-val-evaluation.txt && echo "Val evaluation exists" || echo "MISSING"
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] Model metadata shows production sample counts (≥ 900)
- [ ] CV metrics improved over mini-trained baselines (R² > 0)
- [ ] Val set MAE < 0.25 per subscore (directional, not blocking)
- [ ] Train/val/test splits have no ID overlap
- [ ] Val and test are 100% human-only entries
