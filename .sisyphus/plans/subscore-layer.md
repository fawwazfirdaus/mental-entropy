# Subscore Model Layer + MES Combiner

## TL;DR

> **Quick Summary**: Implement the subscore model layer (Layer 3) and MES combiner (Layer 4) for the Mental Entropy pipeline. This adds LLM-as-judge labeling, GAM/XGBoost subscore training, a scoring API with feature contribution explanations, and a deterministic MES combiner — completing the full pipeline from journal text to MES score.
> 
> **Deliverables**:
> - LLM labeling pipeline for generating per-subscore 0-1 labels
> - Training scripts for 4 subscore models (2 GAM, 2 XGBoost)
> - `src/mental_entropy/scoring/` package with `score_ce()`, `score_se()`, `score_ne()`, `score_cle()`, `mes_score()`
> - Frozen dataclass returns (`SubscoreResult`, `MESResult`) with feature contributions
> - Bundled model artifacts in `src/mental_entropy/models/`
> - Tests and validation suite matching existing quality standards
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 5 waves + FINAL review
> **Critical Path**: T1 → T6 → T12 → T14 → T17 → F1-F4

---

## Context

### Original Request
Implement the subscore model layer — the third and fourth layers of the MES architecture as described in `docs/mes_architecture.md`. This covers all 4 documented gaps: (1) subscore model training pipeline, (2) persisted model artifacts + versioning, (3) `mes_score()` API + deterministic combiner, (4) validation suite.

### Interview Summary
**Key Discussions**:
- **Training labels**: LLM-as-judge approach chosen over heuristic percentile-rank labels. Extend existing datagen OpenAI-compatible endpoint pattern.
- **SHAP in public API**: Exposed via frozen dataclass — every subscore call returns both score and per-feature explanations.
- **Return type**: Frozen dataclass (`SubscoreResult`) matching `EmbeddingResult` pattern, not plain dict.
- **Model artifacts**: Bundled in package as `src/mental_entropy/models/` using hatchling package data config.
- **Tests**: Tests after implementation, matching existing patterns (keyset checks, determinism, JSON serializability, boundary cases).
- **Scope**: All 4 architecture gaps in one plan.

**Research Findings**:
- **pyGAM** is the right GAM library — only one with native monotonic constraints (`s(i, constraints="monotonic_inc")`). InterpretML EBM lacks this.
- **SHAP TreeExplainer + GAM mismatch** (Metis finding): TreeExplainer only works with tree-based models. For GAM models (CE/NE), use GAM's native term contributions which are mathematically equivalent to SHAP for additive models. Field renamed to `feature_contributions` not `shap_values`.
- **XGBoost JSON** format is stable and cross-version compatible for serialization. Joblib for GAMs.
- **809 synthetic journals** exist with per-subscore heuristic scores, but labels are derived from percentile-ranking the same features — insufficient for training. LLM labels needed.
- **Existing datagen infrastructure** (env vars, OpenAI-compatible client, retries, JSONL) can be extended for labeling.
- **HIGH_FEATURES / LOW_FEATURES** in `scripts/entropy_bin_and_balance.py` define monotonicity directions — source of truth for GAM constraints.

### Metis Review
**Identified Gaps** (addressed in plan):
- **SHAP+GAM incompatibility**: Resolved by using GAM term contributions for CE/NE, SHAP TreeExplainer for SE/CLE only. Unified under `feature_contributions` field name.
- **pyGAM Python 3.12 risk**: Addressed by making pyGAM validation the very first task (T1). Fallback: XGBoost with monotonic constraints for all 4 models.
- **Feature ordering sensitivity**: XGBoost requires frozen feature ordering. Added `*_FEATURE_ORDER` tuples to constants.
- **Edge cases**: All-zero features, missing keys, int→float casting, MES clamping — all addressed in scoring function specs.
- **New deps as optional**: All new deps under `[project.optional-dependencies].scoring`, not core.
- **Package data bundling**: Novel for this repo — needs explicit hatchling config and import testing.
- **Training scripts in `scripts/`**: Not in package source — training is offline.
- **Small dataset (809)**: K-fold cross-validation, shallow trees, regularization recommended.

---

## Work Objectives

### Core Objective
Complete the MES pipeline by implementing learned subscore models (Layer 3) and a deterministic combiner (Layer 4), enabling `journal_text → embedding → features → subscores → MES score` end-to-end.

### Concrete Deliverables
- `src/mental_entropy/scoring/` — new package with types, loader, scoring functions, combiner
- `src/mental_entropy/models/` — bundled model artifacts (4 models + metadata)
- `scripts/label_journals.py` — LLM labeling pipeline
- `scripts/train_subscore_models.py` — unified training script
- `scripts/validate_labels.py` — label quality validation
- `tests/scoring/` — comprehensive test suite
- Updated `pyproject.toml` with `scoring` optional dep group
- Updated docs: `mes_architecture.md`, `README.md`, `CLAUDE.md`

### Definition of Done
- [x] `uv run python -c "from mental_entropy.scoring import score_ce, score_se, score_ne, score_cle, mes_score; print('OK')"` → prints `OK`
- [x] `uv run pytest tests/ -v` → all tests pass (existing + new)
- [x] Full pipeline: `embed_journal_entry(text) → features → subscores → mes_score()` runs without error
- [x] Model artifacts exist in `src/mental_entropy/models/` and load at import time
- [x] `json.dumps(dataclasses.asdict(result))` succeeds for all return types

### Must Have
- 4 subscore functions: `score_ce()`, `score_se()`, `score_ne()`, `score_cle()`
- `mes_score()` combiner with weights 0.35/0.15/0.20/0.30
- `SubscoreResult` and `MESResult` frozen dataclasses
- `feature_contributions: dict[str, float]` in `SubscoreResult`
- Feature key validation (frozenset check) before model input
- Edge case handling: all-zero features → valid score in [0,1]
- MES score clamped to [0, 100]
- Deterministic output (same input → same output)
- JSON-serializable returns (no numpy types)
- Model artifacts bundled as package data
- LLM labeling script extending existing datagen pattern
- Training script producing exportable artifacts
- Tests matching existing quality patterns

### Must NOT Have (Guardrails)
- MUST NOT modify any file in `features/`, `embedding/`, `utils/`, or existing tests
- MUST NOT add scoring deps (`xgboost`, `pygam`, `shap`, `joblib`) to core dependencies — optional only
- MUST NOT build automated hyperparameter search — manual config is sufficient for v1
- MUST NOT build model registry/versioning system — single artifact set, versioned via git
- MUST NOT build feature selection/importance analysis beyond what SHAP/GAM contributions provide
- MUST NOT build automated retraining pipeline — training is a manual offline step
- MUST NOT add calibration layer on top of combiner
- MUST NOT build dashboards, plots, or interactive visualization tools
- MUST NOT abstract LLM provider beyond existing OpenAI-compatible pattern
- MUST NOT bundle SHAP background/training data in the package
- MUST NOT use `pkg_resources` or `__file__` for artifact loading — use `importlib.resources`

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest >= 8.0)
- **Automated tests**: YES (tests after implementation)
- **Framework**: pytest (existing)
- **Pattern**: Match existing test style — keyset checks, determinism, JSON serializability, boundary cases

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

| Deliverable Type | Verification Tool | Method |
|------------------|-------------------|--------|
| Python module | Bash (uv run python) | Import, call functions, assert output |
| Training script | Bash (uv run python scripts/...) | Run with small dataset, verify output |
| Config (pyproject.toml) | Bash (uv sync) | Install + import check |
| Package data | Bash (uv run python -c) | importlib.resources access |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — 5 parallel, all quick/writing):
├── Task 1:  Dependencies + pyGAM validation [quick]
├── Task 2:  Scoring types (SubscoreResult, MESResult) [quick]
├── Task 3:  Feature ordering constants + monotonicity [quick]
├── Task 4:  Hatchling package data config [quick]
└── Task 5:  LLM labeling rubric design [writing]

Wave 2 (Infrastructure + Scripts — 6 parallel):
├── Task 6:  Model loader + scoring package scaffold (depends: 1,2,3,4) [unspecified-high]
├── Task 7:  MES combiner function (depends: 2) [quick]
├── Task 8:  LLM labeling script (depends: 5) [unspecified-high]
├── Task 9:  Label validation script (depends: 5) [quick]
├── Task 10: GAM training script (depends: 1,3) [deep]
└── Task 11: XGBoost training script (depends: 1,3) [deep]

Wave 3 (Scoring Functions + API — 3 parallel):
├── Task 12: GAM scoring: score_ce + score_ne (depends: 3,6) [deep]
├── Task 13: XGBoost scoring: score_se + score_cle (depends: 3,6) [deep]
└── Task 14: Public API exports + __init__.py (depends: 7,12,13) [quick]

Wave 4 (Testing + Docs — 5 parallel):
├── Task 15: Tests: scoring types, constants, model loader (depends: 2,3,6) [unspecified-high]
├── Task 16: Tests: scoring functions + feature contributions (depends: 12,13) [unspecified-high]
├── Task 17: Tests: MES combiner + integration + e2e (depends: 7,14) [deep]
├── Task 18: Tests: validation suite + edge cases (depends: 14) [deep]
└── Task 19: Documentation updates (depends: 14) [writing]

Wave FINAL (Review — 4 parallel):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)

Critical Path: T1 → T6 → T12 → T14 → T17 → F1-F4
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 6 (Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|------------|--------|------|
| 1 | — | 6, 10, 11 | 1 |
| 2 | — | 6, 7, 12, 13, 14, 15 | 1 |
| 3 | — | 6, 10, 11, 12, 13, 15 | 1 |
| 4 | — | 6 | 1 |
| 5 | — | 8, 9 | 1 |
| 6 | 1, 2, 3, 4 | 12, 13, 15 | 2 |
| 7 | 2 | 14, 17 | 2 |
| 8 | 5 | — | 2 |
| 9 | 5 | — | 2 |
| 10 | 1, 3 | — | 2 |
| 11 | 1, 3 | — | 2 |
| 12 | 3, 6 | 14, 16 | 3 |
| 13 | 3, 6 | 14, 16 | 3 |
| 14 | 7, 12, 13 | 17, 18, 19 | 3 |
| 15 | 2, 3, 6 | — | 4 |
| 16 | 12, 13 | — | 4 |
| 17 | 7, 14 | — | 4 |
| 18 | 14 | — | 4 |
| 19 | 14 | — | 4 |
| F1-F4 | 15-19 | — | FINAL |

### Agent Dispatch Summary

| Wave | # Parallel | Tasks → Agent Category |
|------|------------|----------------------|
| 1 | **5** | T1 → `quick`, T2 → `quick`, T3 → `quick`, T4 → `quick`, T5 → `writing` |
| 2 | **6** | T6 → `unspecified-high`, T7 → `quick`, T8 → `unspecified-high`, T9 → `quick`, T10 → `deep`, T11 → `deep` |
| 3 | **3** | T12 → `deep`, T13 → `deep`, T14 → `quick` |
| 4 | **5** | T15 → `unspecified-high`, T16 → `unspecified-high`, T17 → `deep`, T18 → `deep`, T19 → `writing` |
| FINAL | **4** | F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep` |

---

## TODOs

- [x] 1. Dependencies + pyGAM Python 3.12 Validation

  **What to do**:
  - Add `[project.optional-dependencies].scoring` to `pyproject.toml` with: `pygam>=0.8,<1`, `xgboost>=2.0,<3`, `shap>=0.43,<1`, `joblib>=1.3,<2`
  - Run `uv sync --extra scoring` to install
  - Validate pyGAM works on Python 3.12: `from pygam import LinearGAM, s; gam = LinearGAM(s(0)); print("pyGAM OK")`
  - If pyGAM fails on 3.12: document the failure and note that T10/T12 must use XGBoost with `monotone_constraints` as fallback for ALL 4 models
  - Do NOT add scoring deps to the core `dependencies` list

  **Must NOT do**:
  - Do not modify existing dependencies
  - Do not add deps to core `dependencies` (they must be optional)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-file edit + smoke test, no complex logic
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None applicable

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: Tasks 6, 10, 11
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `pyproject.toml:17-20` — Existing `[project.optional-dependencies].dev` pattern. Follow this exact structure for the `scoring` group.

  **API/Type References**:
  - `pyproject.toml:22-27` — Build system is hatchling, not setuptools. Ensure no setuptools-specific config is added.

  **WHY Each Reference Matters**:
  - `pyproject.toml:17-20` — Shows the exact syntax for optional dependency groups in this project. Copy this pattern.
  - `pyproject.toml:22-27` — The build system determines how package data is configured (Task 4). Don't confuse with setuptools.

  **Acceptance Criteria**:

  - [ ] `pyproject.toml` has `[project.optional-dependencies].scoring` section with 4 deps
  - [ ] `uv sync --extra scoring` succeeds (exit code 0)
  - [ ] `uv run python -c "from pygam import LinearGAM; from xgboost import XGBRegressor; import shap; import joblib; print('OK')"` prints `OK`
  - [ ] If pyGAM fails: a file `.sisyphus/evidence/task-1-pygam-fallback.txt` documents the error

  **QA Scenarios**:

  ```
  Scenario: Scoring deps install and import successfully
    Tool: Bash (uv run)
    Preconditions: Clean environment, uv.lock exists
    Steps:
      1. Run `uv sync --extra scoring`
      2. Run `uv run python -c "from pygam import LinearGAM, s; from xgboost import XGBRegressor; import shap; import joblib; print('ALL_DEPS_OK')"`
    Expected Result: Both commands exit 0, second prints "ALL_DEPS_OK"
    Failure Indicators: Import error, missing module, version conflict
    Evidence: .sisyphus/evidence/task-1-deps-install.txt

  Scenario: pyGAM monotonic constraint smoke test
    Tool: Bash (uv run python)
    Preconditions: Scoring deps installed
    Steps:
      1. Run `uv run python -c "
         import numpy as np
         from pygam import LinearGAM, s
         X = np.random.rand(50, 3)
         y = np.random.rand(50)
         gam = LinearGAM(s(0, constraints='monotonic_inc') + s(1) + s(2))
         gam.fit(X, y)
         pred = gam.predict(X[:1])
         assert 0 < len(pred) == 1
         print('PYGAM_MONOTONIC_OK')
         "`
    Expected Result: Prints "PYGAM_MONOTONIC_OK"
    Failure Indicators: ImportError, RuntimeError, constraint not supported
    Evidence: .sisyphus/evidence/task-1-pygam-monotonic.txt

  Scenario: Existing deps unaffected
    Tool: Bash (uv run)
    Preconditions: Scoring deps installed
    Steps:
      1. Run `uv run python -c "from mental_entropy import embed_journal_entry; print('EXISTING_OK')"`
    Expected Result: Prints "EXISTING_OK"
    Failure Indicators: Import error in existing package
    Evidence: .sisyphus/evidence/task-1-existing-deps.txt
  ```

  **Commit**: YES
  - Message: `build(scoring): add optional scoring dependency group`
  - Files: `pyproject.toml`
  - Pre-commit: `uv sync --extra scoring`

---

- [x] 2. Scoring Types (SubscoreResult, MESResult)

  **What to do**:
  - Create `src/mental_entropy/scoring/types.py`
  - Define `SubscoreResult` frozen dataclass:
    ```python
    @dataclass(frozen=True, slots=True)
    class SubscoreResult:
        score: float  # 0.0 to 1.0
        feature_contributions: dict[str, float]  # per-feature contribution to score
    ```
  - Define `MESResult` frozen dataclass:
    ```python
    @dataclass(frozen=True, slots=True)
    class MESResult:
        score: int  # 0 to 100
        raw: float  # 0.0 to 1.0 (before rounding)
        ce: float  # CE subscore
        se: float  # SE subscore
        ne: float  # NE subscore
        cle: float  # CLE subscore
    ```
  - Both must follow `@dataclass(frozen=True, slots=True)` pattern from `embedding/types.py`
  - All fields must be JSON-serializable Python types (int, float, dict, str) — no numpy

  **Must NOT do**:
  - Do not name the field `shap_values` — use `feature_contributions` (GAM models produce term contributions, not SHAP)
  - Do not import from scoring deps (xgboost, pygam) — types are pure Python

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file, small dataclass definitions, follows existing pattern exactly
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5)
  - **Blocks**: Tasks 6, 7, 12, 13, 14, 15
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/mental_entropy/embedding/types.py:8-31` — Canonical frozen dataclass pattern. Copy this style exactly: `@dataclass(frozen=True, slots=True)`, docstrings on each field, `from __future__ import annotations`.

  **WHY Each Reference Matters**:
  - `embedding/types.py` — This is THE reference for how structured types are defined in this project. Match `from __future__ import annotations`, field-level docstrings, and the frozen+slots decorators exactly.

  **Acceptance Criteria**:

  - [ ] `src/mental_entropy/scoring/types.py` exists with `SubscoreResult` and `MESResult`
  - [ ] `src/mental_entropy/scoring/__init__.py` exists (can be empty or minimal)
  - [ ] `uv run python -c "from mental_entropy.scoring.types import SubscoreResult, MESResult; print('OK')"` prints `OK`
  - [ ] Both dataclasses are frozen and slotted
  - [ ] `json.dumps(dataclasses.asdict(SubscoreResult(score=0.5, feature_contributions={'a': 0.1})))` succeeds

  **QA Scenarios**:

  ```
  Scenario: Types are importable and frozen
    Tool: Bash (uv run python)
    Preconditions: Package installed
    Steps:
      1. Run `uv run python -c "
         import dataclasses, json
         from mental_entropy.scoring.types import SubscoreResult, MESResult
         sr = SubscoreResult(score=0.72, feature_contributions={'ce_break_rate': 0.15, 'ce_adj_mean': -0.08})
         mr = MESResult(score=72, raw=0.72, ce=0.8, se=0.3, ne=0.6, cle=0.9)
         assert sr.score == 0.72
         assert sr.feature_contributions['ce_break_rate'] == 0.15
         assert mr.score == 72
         # Frozen check
         try:
             sr.score = 0.5
             assert False, 'Should be frozen'
         except dataclasses.FrozenInstanceError:
             pass
         # JSON serializable
         json.dumps(dataclasses.asdict(sr))
         json.dumps(dataclasses.asdict(mr))
         print('TYPES_OK')
         "`
    Expected Result: Prints "TYPES_OK"
    Failure Indicators: ImportError, FrozenInstanceError not raised, json.dumps fails
    Evidence: .sisyphus/evidence/task-2-types-ok.txt
  ```

  **Commit**: YES
  - Message: `feat(scoring): add SubscoreResult and MESResult type definitions`
  - Files: `src/mental_entropy/scoring/__init__.py`, `src/mental_entropy/scoring/types.py`
  - Pre-commit: `uv run python -c "from mental_entropy.scoring.types import SubscoreResult, MESResult; print('OK')"`

---

- [x] 3. Feature Ordering Constants + Monotonicity Mappings

  **What to do**:
  - Create `src/mental_entropy/scoring/constants.py`
  - Define frozen feature ordering tuples for each subscore model. Order MUST match the order used during training. Use alphabetical sort of the existing `*_FEATURE_KEYS` frozensets for deterministic ordering:
    ```python
    CE_FEATURE_ORDER: tuple[str, ...] = tuple(sorted(CE_FEATURE_KEYS))
    SE_FEATURE_ORDER: tuple[str, ...] = tuple(sorted(SE_FEATURE_KEYS))
    NE_FEATURE_ORDER: tuple[str, ...] = tuple(sorted(NE_FEATURE_KEYS))
    CLE_FEATURE_ORDER: tuple[str, ...] = tuple(sorted(CLE_FEATURE_KEYS))
    ```
  - Define monotonicity direction mappings for GAM models, derived from `HIGH_FEATURES` / `LOW_FEATURES` in `scripts/entropy_bin_and_balance.py`:
    - Features in `HIGH_FEATURES` → `"monotonic_inc"` (higher value = higher entropy)
    - Features in `LOW_FEATURES` → `"monotonic_dec"` (higher value = lower entropy)
    - Features in neither → no constraint (unconstrained spline)
    ```python
    CE_MONOTONICITY: dict[str, str] = {
        "ce_break_rate": "monotonic_inc",
        "ce_low_mass": "monotonic_inc",
        "ce_adj_std": "monotonic_inc",
        "ce_sharp_drop_rate": "monotonic_inc",
        "ce_adj_mean": "monotonic_dec",
        "ce_longest_coherent_run": "monotonic_dec",
        # remaining CE features: unconstrained
    }
    ```
  - Define XGBoost monotone_constraints tuples for SE/CLE (same source of truth)

  **Must NOT do**:
  - Do not hardcode feature lists — import from existing `*_FEATURE_KEYS` frozensets
  - Do not invent new monotonicity directions — use exactly `HIGH_FEATURES` / `LOW_FEATURES` from `scripts/entropy_bin_and_balance.py`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mapping existing constants into new format, no complex logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5)
  - **Blocks**: Tasks 6, 10, 11, 12, 13, 15
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/mental_entropy/features/ce.py:24-57` — `CE_FEATURE_KEYS` frozenset. Import this, don't duplicate.
  - `src/mental_entropy/features/se.py` — `SE_FEATURE_KEYS` (look for the frozenset near top of file)
  - `src/mental_entropy/features/ne.py` — `NE_FEATURE_KEYS`
  - `src/mental_entropy/features/cle.py` — `CLE_FEATURE_KEYS`

  **API/Type References**:
  - `scripts/entropy_bin_and_balance.py:30-62` — `HIGH_FEATURES` and `LOW_FEATURES` tuples. These are the source of truth for monotonicity directions.

  **WHY Each Reference Matters**:
  - `*_FEATURE_KEYS` — Must import, not copy. If feature sets ever change, ordering must stay in sync.
  - `HIGH_FEATURES/LOW_FEATURES` — Defines which features drive entropy up vs down. This directly maps to GAM constraint direction and XGBoost monotone_constraints.

  **Acceptance Criteria**:

  - [ ] `src/mental_entropy/scoring/constants.py` exists
  - [ ] Each `*_FEATURE_ORDER` tuple has the same length as its `*_FEATURE_KEYS` frozenset
  - [ ] `CE_MONOTONICITY` and `NE_MONOTONICITY` dicts exist with entries for features in HIGH/LOW lists
  - [ ] `SE_MONOTONE_CONSTRAINTS` and `CLE_MONOTONE_CONSTRAINTS` tuples exist (for XGBoost)
  - [ ] All constants are importable: `from mental_entropy.scoring.constants import CE_FEATURE_ORDER`

  **QA Scenarios**:

  ```
  Scenario: Feature orders match feature keys
    Tool: Bash (uv run python)
    Preconditions: Package installed
    Steps:
      1. Run `uv run python -c "
         from mental_entropy.features import CE_FEATURE_KEYS, SE_FEATURE_KEYS, NE_FEATURE_KEYS, CLE_FEATURE_KEYS
         from mental_entropy.scoring.constants import CE_FEATURE_ORDER, SE_FEATURE_ORDER, NE_FEATURE_ORDER, CLE_FEATURE_ORDER
         assert set(CE_FEATURE_ORDER) == CE_FEATURE_KEYS, f'CE mismatch'
         assert set(SE_FEATURE_ORDER) == SE_FEATURE_KEYS, f'SE mismatch'
         assert set(NE_FEATURE_ORDER) == NE_FEATURE_KEYS, f'NE mismatch'
         assert set(CLE_FEATURE_ORDER) == CLE_FEATURE_KEYS, f'CLE mismatch'
         assert len(CE_FEATURE_ORDER) == 24
         assert len(SE_FEATURE_ORDER) == 9
         assert len(NE_FEATURE_ORDER) == 13
         assert len(CLE_FEATURE_ORDER) == 14
         print('ORDERING_OK')
         "`
    Expected Result: Prints "ORDERING_OK"
    Failure Indicators: AssertionError on set mismatch or wrong length
    Evidence: .sisyphus/evidence/task-3-ordering-ok.txt

  Scenario: Monotonicity mappings reference real features
    Tool: Bash (uv run python)
    Preconditions: Package installed
    Steps:
      1. Run `uv run python -c "
         from mental_entropy.features import CE_FEATURE_KEYS, NE_FEATURE_KEYS
         from mental_entropy.scoring.constants import CE_MONOTONICITY, NE_MONOTONICITY
         for k in CE_MONOTONICITY:
             assert k in CE_FEATURE_KEYS, f'{k} not in CE_FEATURE_KEYS'
         for k in NE_MONOTONICITY:
             assert k in NE_FEATURE_KEYS, f'{k} not in NE_FEATURE_KEYS'
         print('MONOTONICITY_OK')
         "`
    Expected Result: Prints "MONOTONICITY_OK"
    Failure Indicators: Feature name not found in keys
    Evidence: .sisyphus/evidence/task-3-monotonicity-ok.txt
  ```

  **Commit**: YES
  - Message: `feat(scoring): add feature ordering and monotonicity constants`
  - Files: `src/mental_entropy/scoring/constants.py`
  - Pre-commit: `uv run python -c "from mental_entropy.scoring.constants import CE_FEATURE_ORDER; print('OK')"`

---

- [x] 4. Hatchling Package Data Config

  **What to do**:
  - Create `src/mental_entropy/models/` directory
  - Add `src/mental_entropy/models/__init__.py` (empty, makes it a package for importlib.resources)
  - Add a placeholder `src/mental_entropy/models/.gitkeep` or placeholder metadata JSON
  - Update `pyproject.toml` hatchling config to include model files:
    ```toml
    [tool.hatch.build.targets.wheel]
    packages = ["src/mental_entropy"]
    ```
    This already includes all files under `src/mental_entropy/`, which will include `models/`. Verify this is the case.
  - Test that `importlib.resources` can access the models directory after install

  **Must NOT do**:
  - Do not use `pkg_resources` (deprecated)
  - Do not use `__file__` path hacking
  - Do not use `MANIFEST.in` (hatchling doesn't use it)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Directory creation + config verification, minimal logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5)
  - **Blocks**: Task 6
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `pyproject.toml:26-27` — Current hatchling wheel config. The `packages = ["src/mental_entropy"]` line already includes all subpackages.

  **WHY Each Reference Matters**:
  - The existing wheel config should already bundle `models/` since it includes the entire `src/mental_entropy/` tree. This task verifies that assumption and ensures non-`.py` files (`.json`, `.joblib`) are included.

  **Acceptance Criteria**:

  - [ ] `src/mental_entropy/models/__init__.py` exists
  - [ ] `importlib.resources` can locate the models package
  - [ ] Non-Python files in `models/` are accessible after install

  **QA Scenarios**:

  ```
  Scenario: Models directory is accessible via importlib.resources
    Tool: Bash (uv run python)
    Preconditions: Package installed
    Steps:
      1. Run `uv run python -c "
         import importlib.resources as resources
         ref = resources.files('mental_entropy.models')
         print(f'Models path: {ref}')
         print('PKG_DATA_OK')
         "`
    Expected Result: Prints path and "PKG_DATA_OK"
    Failure Indicators: ModuleNotFoundError, resource not found
    Evidence: .sisyphus/evidence/task-4-pkg-data.txt
  ```

  **Commit**: YES (groups with T2)
  - Message: `feat(scoring): scaffold models directory for artifact bundling`
  - Files: `src/mental_entropy/models/__init__.py`

---

- [x] 5. LLM Labeling Rubric Design

  **What to do**:
  - Create `src/mental_entropy/datagen/labeling_rubric.py` containing the prompt template and format specification
  - Design a rubric for each subscore (CE, SE, NE, CLE) that tells the LLM what 0.0, 0.25, 0.5, 0.75, 1.0 looks like in a journal
  - CE rubric: rate how coherent/disjointed the writing is (0 = perfectly coherent, 1 = completely fragmented)
  - SE rubric: rate how many unrelated topics appear (0 = single focused topic, 1 = many scattered topics)
  - NE rubric: rate narrative structure (0 = clear arc with reflection, 1 = no arc, erratic, fragmented)
  - CLE rubric: rate cognitive overload signals (0 = clear/calm expression, 1 = chaotic/overwhelmed/uncertain)
  - Output format: JSON with 4 float scores in [0, 1] and brief justification per score
  - Define the expected LLM response schema
  - Include examples of low/medium/high journals with expected scores (few-shot)
  - Use ordinal anchors (0.0, 0.25, 0.5, 0.75, 1.0) to reduce continuous scale noise

  **Must NOT do**:
  - Do not implement the LLM calling logic here — that's Task 8
  - Do not use more than 3 few-shot examples (token efficiency)
  - Do not reference the feature values in the rubric — the LLM judges from text only

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Prompt engineering and rubric design is primarily a writing task
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4)
  - **Blocks**: Tasks 8, 9
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/mental_entropy/datagen/synthetic_journals.py:31-66` — Frozen dataclass config pattern for EntrySpec/Persona/Context. Follow this style if defining config types.
  - `docs/mes_architecture.md:79-84` — Feature-to-subscore mapping table. Use this to understand what each subscore measures conceptually.

  **External References**:
  - README.md feature descriptions — The README has detailed descriptions of what each feature set captures. Use these to inform the rubric definitions (e.g., CE = coherence breaks, SE = topic drift, NE = narrative arc, CLE = cognitive load).

  **WHY Each Reference Matters**:
  - `mes_architecture.md:79-84` — Defines what CE/SE/NE/CLE conceptually measure. The rubric must align with these definitions.
  - README feature descriptions — Provide concrete signal descriptions (e.g., "high CLE = chaotic, interrupted, uncertain thinking") that should inform rubric anchors.

  **Acceptance Criteria**:

  - [ ] `src/mental_entropy/datagen/labeling_rubric.py` exists
  - [ ] Contains prompt template with rubric for all 4 subscores
  - [ ] Each subscore has 5 ordinal anchors (0.0, 0.25, 0.5, 0.75, 1.0) with descriptions
  - [ ] Expected response format is defined (JSON schema)
  - [ ] At least 2 few-shot examples included
  - [ ] Module is importable: `from mental_entropy.datagen.labeling_rubric import LABELING_PROMPT`

  **QA Scenarios**:

  ```
  Scenario: Rubric module imports and contains expected exports
    Tool: Bash (uv run python)
    Preconditions: Package installed
    Steps:
      1. Run `uv run python -c "
         from mental_entropy.datagen.labeling_rubric import LABELING_PROMPT, RESPONSE_SCHEMA
         assert 'CE' in LABELING_PROMPT or 'coherence' in LABELING_PROMPT.lower()
         assert 'SE' in LABELING_PROMPT or 'semantic' in LABELING_PROMPT.lower()
         assert 'NE' in LABELING_PROMPT or 'narrative' in LABELING_PROMPT.lower()
         assert 'CLE' in LABELING_PROMPT or 'cognitive' in LABELING_PROMPT.lower()
         assert len(LABELING_PROMPT) > 500, 'Rubric too short'
         print('RUBRIC_OK')
         "`
    Expected Result: Prints "RUBRIC_OK"
    Failure Indicators: ImportError, assertion failure
    Evidence: .sisyphus/evidence/task-5-rubric-ok.txt
  ```

  **Commit**: YES
  - Message: `feat(datagen): add LLM labeling rubric for subscore annotation`
  - Files: `src/mental_entropy/datagen/labeling_rubric.py`

---

- [x] 6. Model Loader + Scoring Package Scaffold

  **What to do**:
  - Create `src/mental_entropy/scoring/loader.py` with model loading infrastructure
  - Use `importlib.resources` to locate model artifacts in `mental_entropy.models`
  - Implement lazy loading: models loaded on first `score_*()` call, then cached in module-level dict
  - Support loading both `.joblib` (GAM models) and `.json` (XGBoost models)
  - Validate model artifact exists at load time; raise clear `FileNotFoundError` with message like `"CE model artifact not found. Install scoring extras: pip install mental-entropy[scoring]"`
  - Define a `_load_model(name: str, model_type: str) -> Any` internal function
  - Create `src/mental_entropy/scoring/__init__.py` that imports and re-exports public API from submodules
  - Feature dict validation helper: `_validate_features(features: dict, expected_keys: frozenset, feature_order: tuple) -> list[float]` that:
    - Checks all expected keys are present (raise `ValueError` if missing)
    - Raises `ValueError` on extra/unexpected keys
    - Casts int values to float
    - Returns values in the frozen feature order

  **Must NOT do**:
  - Do not load models at module import time (lazy loading only)
  - Do not use `pkg_resources` or `__file__` path hacking
  - Do not import `xgboost`/`pygam`/`shap` at module level — import inside functions (for users who only want features)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Core infrastructure piece with lazy loading, error handling, validation logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 9, 10, 11)
  - **Blocks**: Tasks 12, 13, 15
  - **Blocked By**: Tasks 1, 2, 3, 4

  **References**:

  **Pattern References**:
  - `src/mental_entropy/embedding/types.py:8-31` — Dataclass pattern for return types (imported from T2)
  - `src/mental_entropy/features/__init__.py:1-63` — Package `__init__.py` export pattern. Follow the same structure: import from submodules, define `__all__`.
  - `src/mental_entropy/embedding/model.py` — Lazy model loading pattern. The embedding model is loaded once and cached. Follow similar caching approach.

  **API/Type References**:
  - `src/mental_entropy/scoring/types.py` — (from T2) SubscoreResult, MESResult to import and re-export
  - `src/mental_entropy/scoring/constants.py` — (from T3) Feature ordering tuples for validation

  **WHY Each Reference Matters**:
  - `embedding/model.py` — Shows how this project handles lazy model loading. The scoring loader should follow the same pattern.
  - `features/__init__.py` — The scoring package needs the same export structure to match project conventions.

  **Acceptance Criteria**:

  - [ ] `src/mental_entropy/scoring/loader.py` exists with `_load_model()` and `_validate_features()`
  - [ ] Models are loaded lazily (not at import time)
  - [ ] Missing model file raises `FileNotFoundError` with helpful message
  - [ ] `_validate_features({'ce_adj_mean': 0.5, ...}, CE_FEATURE_KEYS, CE_FEATURE_ORDER)` returns list of floats in correct order
  - [ ] Missing feature key raises `ValueError`
  - [ ] Int values are cast to float without error

  **QA Scenarios**:

  ```
  Scenario: Feature validation accepts valid dict and returns ordered list
    Tool: Bash (uv run python)
    Preconditions: Package installed
    Steps:
      1. Run `uv run python -c "
         from mental_entropy.features import CE_FEATURE_KEYS
         from mental_entropy.scoring.constants import CE_FEATURE_ORDER
         from mental_entropy.scoring.loader import _validate_features
         features = {k: 0.5 for k in CE_FEATURE_KEYS}
         features['ce_n_sentences'] = 5  # int value
         result = _validate_features(features, CE_FEATURE_KEYS, CE_FEATURE_ORDER)
         assert len(result) == 24
         assert all(isinstance(v, float) for v in result)
         print('VALIDATE_OK')
         "`
    Expected Result: Prints "VALIDATE_OK"
    Failure Indicators: TypeError on int, wrong length
    Evidence: .sisyphus/evidence/task-6-validate-ok.txt

  Scenario: Missing feature key raises ValueError
    Tool: Bash (uv run python)
    Preconditions: Package installed
    Steps:
      1. Run `uv run python -c "
         from mental_entropy.features import CE_FEATURE_KEYS
         from mental_entropy.scoring.constants import CE_FEATURE_ORDER
         from mental_entropy.scoring.loader import _validate_features
         features = {k: 0.5 for k in list(CE_FEATURE_KEYS)[:10]}  # missing keys
         try:
             _validate_features(features, CE_FEATURE_KEYS, CE_FEATURE_ORDER)
             print('FAIL: should have raised')
         except ValueError as e:
             assert 'missing' in str(e).lower() or 'key' in str(e).lower()
             print('MISSING_KEY_ERROR_OK')
         "`
    Expected Result: Prints "MISSING_KEY_ERROR_OK"
    Failure Indicators: No error raised, wrong error type
    Evidence: .sisyphus/evidence/task-6-missing-key.txt

  Scenario: Missing model artifact gives clear error
    Tool: Bash (uv run python)
    Preconditions: Package installed, no model artifacts exist yet
    Steps:
      1. Run `uv run python -c "
         from mental_entropy.scoring.loader import _load_model
         try:
             _load_model('nonexistent', 'joblib')
             print('FAIL')
         except FileNotFoundError as e:
             assert 'not found' in str(e).lower()
             print('MISSING_MODEL_ERROR_OK')
         "`
    Expected Result: Prints "MISSING_MODEL_ERROR_OK"
    Failure Indicators: Wrong error type, no error
    Evidence: .sisyphus/evidence/task-6-missing-model.txt
  ```

  **Commit**: YES
  - Message: `feat(scoring): add model loader and feature validation infrastructure`
  - Files: `src/mental_entropy/scoring/loader.py`, `src/mental_entropy/scoring/__init__.py`

---

- [x] 7. MES Combiner Function

  **What to do**:
  - Create `src/mental_entropy/scoring/combiner.py`
  - Implement `mes_score(ce: float, se: float, ne: float, cle: float) -> MESResult`:
    ```python
    def mes_score(ce: float, se: float, ne: float, cle: float) -> MESResult:
        raw = 0.35 * ce + 0.15 * se + 0.20 * ne + 0.30 * cle
        raw = max(0.0, min(1.0, raw))  # clamp
        score = round(raw * 100)
        score = max(0, min(100, score))  # clamp integer
        return MESResult(score=score, raw=raw, ce=ce, se=se, ne=ne, cle=cle)
    ```
  - Validate inputs: all subscores must be float in [0, 1], raise `ValueError` otherwise
  - The combiner is deterministic and pure — no model loading, no side effects
  - Weights: CE=0.35, SE=0.15, NE=0.20, CLE=0.30 (from `docs/mes_architecture.md:19`)

  **Must NOT do**:
  - Do not add calibration or scaling on top of the weighted sum
  - Do not make weights configurable (they're fixed by architecture spec)
  - Do not accept `SubscoreResult` objects — take raw floats for flexibility

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single pure function, simple math, clear spec
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 8, 9, 10, 11)
  - **Blocks**: Tasks 14, 17
  - **Blocked By**: Task 2

  **References**:

  **API/Type References**:
  - `docs/mes_architecture.md:19-20` — `MES_raw = 0.35 * CE + 0.15 * SE + 0.20 * NE + 0.30 * CLE; MES_score = round(MES_raw * 100)`. This is the exact formula.
  - `src/mental_entropy/scoring/types.py` — (from T2) `MESResult` dataclass to return.

  **WHY Each Reference Matters**:
  - `mes_architecture.md:19-20` — The weights and formula are the specification. Do not deviate.

  **Acceptance Criteria**:

  - [ ] `src/mental_entropy/scoring/combiner.py` exists with `mes_score()`
  - [ ] `mes_score(0.8, 0.3, 0.6, 0.9)` returns `MESResult` with `score` in [0, 100]
  - [ ] Formula matches spec: `round((0.35*0.8 + 0.15*0.3 + 0.20*0.6 + 0.30*0.9) * 100)` = `round(0.715 * 100)` = 72
  - [ ] Edge case: `mes_score(0.0, 0.0, 0.0, 0.0)` → score=0
  - [ ] Edge case: `mes_score(1.0, 1.0, 1.0, 1.0)` → score=100
  - [ ] Invalid input: `mes_score(-0.1, 0.5, 0.5, 0.5)` raises `ValueError`

  **QA Scenarios**:

  ```
  Scenario: MES combiner produces correct score
    Tool: Bash (uv run python)
    Preconditions: Package installed
    Steps:
      1. Run `uv run python -c "
         from mental_entropy.scoring.combiner import mes_score
         result = mes_score(0.8, 0.3, 0.6, 0.9)
         expected = round((0.35*0.8 + 0.15*0.3 + 0.20*0.6 + 0.30*0.9) * 100)
         assert result.score == expected, f'{result.score} != {expected}'
         assert result.raw == 0.35*0.8 + 0.15*0.3 + 0.20*0.6 + 0.30*0.9
         assert result.ce == 0.8
         assert result.se == 0.3
         assert isinstance(result.score, int)
         print(f'MES={result.score}, COMBINER_OK')
         "`
    Expected Result: Prints "MES=72, COMBINER_OK" (or similar)
    Failure Indicators: Wrong score, wrong type
    Evidence: .sisyphus/evidence/task-7-combiner-ok.txt

  Scenario: Boundary values and clamping
    Tool: Bash (uv run python)
    Preconditions: Package installed
    Steps:
      1. Run `uv run python -c "
         from mental_entropy.scoring.combiner import mes_score
         assert mes_score(0.0, 0.0, 0.0, 0.0).score == 0
         assert mes_score(1.0, 1.0, 1.0, 1.0).score == 100
         try:
             mes_score(-0.1, 0.5, 0.5, 0.5)
             print('FAIL: should reject negative')
         except ValueError:
             pass
         try:
             mes_score(0.5, 1.1, 0.5, 0.5)
             print('FAIL: should reject >1')
         except ValueError:
             pass
         print('BOUNDARY_OK')
         "`
    Expected Result: Prints "BOUNDARY_OK"
    Evidence: .sisyphus/evidence/task-7-boundary-ok.txt
  ```

  **Commit**: YES
  - Message: `feat(scoring): add deterministic MES combiner`
  - Files: `src/mental_entropy/scoring/combiner.py`

---

- [x] 8. LLM Labeling Script

  **What to do**:
  - Create `scripts/label_journals.py` that reads journals from JSONL, sends each to an LLM for per-subscore labeling, and writes labeled output
  - Extend existing datagen pattern: use `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` env vars
  - Reuse HTTP client pattern from `synthetic_journals.py` (urllib, retries, error handling)
  - Import rubric from `mental_entropy.datagen.labeling_rubric`
  - Input: `data/synthetic_journals_800_scored.jsonl` (reads `journal_text` field)
  - Output: `data/labeled_journals.jsonl` with added `labels` block: `{"ce": 0.5, "se": 0.25, "ne": 0.75, "cle": 0.5}`
  - CLI args: `--input`, `--output`, `--batch-size`, `--max-entries` (for test runs)
  - Parse LLM JSON response, validate scores are in [0, 1], retry on parse failure
  - Progress reporting: print every N entries
  - Resume support: skip entries already in output file (append mode)

  **Must NOT do**:
  - Do not abstract beyond OpenAI-compatible pattern
  - Do not run the full labeling (that's manual) — just build the script
  - Do not add new API client libraries — use urllib like existing code

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Script with HTTP client, JSON parsing, retry logic, resume support
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 9, 10, 11)
  - **Blocks**: None (labeling script is standalone)
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - `src/mental_entropy/datagen/synthetic_journals.py:467-513` — OpenAI-compatible HTTP client with retry logic. Copy this pattern exactly for the labeling endpoint calls.
  - `src/mental_entropy/datagen/synthetic_journals.py:236-293` — Main CLI pattern with argparse, progress reporting, JSONL output.
  - `scripts/entropy_bin_and_balance.py:85-104` — JSONL loading pattern.

  **API/Type References**:
  - `src/mental_entropy/datagen/labeling_rubric.py` — (from T5) Prompt template and response schema.

  **WHY Each Reference Matters**:
  - `synthetic_journals.py:467-513` — The project has an established HTTP client pattern. Don't reinvent it.
  - `synthetic_journals.py:236-293` — CLI and progress patterns that match project conventions.

  **Acceptance Criteria**:

  - [ ] `scripts/label_journals.py` exists and is executable
  - [ ] `uv run python scripts/label_journals.py --help` prints usage without error
  - [ ] Script uses `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` env vars
  - [ ] With `--max-entries 1`, script makes one API call and writes one labeled line
  - [ ] Output JSONL includes `labels: {ce: float, se: float, ne: float, cle: float}`

  **QA Scenarios**:

  ```
  Scenario: Script help works
    Tool: Bash
    Preconditions: Package installed
    Steps:
      1. Run `uv run python scripts/label_journals.py --help`
    Expected Result: Prints usage with --input, --output, --batch-size flags
    Evidence: .sisyphus/evidence/task-8-help.txt

  Scenario: Dry run with mock (no API key needed)
    Tool: Bash (uv run python)
    Preconditions: Package installed, data/synthetic_journals_800_scored.jsonl exists
    Steps:
      1. Verify the script parses and validates without making API calls by testing the loading logic:
         `uv run python -c "
         import json
         lines = open('data/synthetic_journals_800_scored.jsonl').readlines()[:3]
         for line in lines:
             entry = json.loads(line)
             assert 'journal_text' in entry
         print(f'LOADED {len(lines)} entries, PARSE_OK')
         "`
    Expected Result: Prints "LOADED 3 entries, PARSE_OK"
    Evidence: .sisyphus/evidence/task-8-parse-ok.txt
  ```

  **Commit**: YES
  - Message: `feat(datagen): add LLM-based journal labeling script`
  - Files: `scripts/label_journals.py`

---

- [x] 9. Label Validation Script

  **What to do**:
  - Create `scripts/validate_labels.py` that checks LLM label quality
  - Compare LLM labels against existing heuristic scores in `entropy_analysis`
  - Compute rank correlation (Spearman) between LLM labels and heuristic scores per subscore
  - Flag entries where |LLM_label - heuristic_score| > 0.3 as disagreements
  - Print summary: correlation per subscore, count of disagreements, score distributions
  - CLI args: `--labeled-input`, `--scored-input`

  **Must NOT do**:
  - Do not reject labels automatically — just report
  - Do not import ML libraries — use scipy for correlation only

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small standalone script, straightforward statistics
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 8, 10, 11)
  - **Blocks**: None
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - `scripts/entropy_bin_and_balance.py:139-171` — Score computation and comparison pattern.

  **Acceptance Criteria**:

  - [ ] `scripts/validate_labels.py` exists
  - [ ] `uv run python scripts/validate_labels.py --help` works

  **QA Scenarios**:

  ```
  Scenario: Script runs with help flag
    Tool: Bash
    Steps:
      1. Run `uv run python scripts/validate_labels.py --help`
    Expected Result: Prints usage
    Evidence: .sisyphus/evidence/task-9-help.txt
  ```

  **Commit**: YES (groups with T8)
  - Message: `feat(scripts): add label validation script`
  - Files: `scripts/validate_labels.py`

---

- [x] 10. GAM Training Script (CE + NE)

  **What to do**:
  - Create `scripts/train_gam.py` (or section within `scripts/train_subscore_models.py`)
  - Load labeled JSONL, extract features, train pyGAM models for CE and NE
  - Apply monotonicity constraints from `scoring/constants.py`:
    - Features in `CE_MONOTONICITY` / `NE_MONOTONICITY` get `constraints="monotonic_inc"` or `"monotonic_dec"`
    - Other features get unconstrained splines
  - Build GAM formula: `LinearGAM(s(0, constraints='monotonic_inc') + s(1) + ...)` using the frozen feature order
  - K-fold cross-validation (k=5) for model selection, then retrain on full data for production artifact
  - Export trained model via `joblib.dump()` to `src/mental_entropy/models/gam_ce.joblib` and `gam_ne.joblib`
  - Export feature contribution function: for a given input X, compute per-term partial dependence
  - Save metadata JSON with: feature order, monotonicity constraints, CV metrics, training date
  - CLI: `--labeled-data`, `--output-dir`, `--k-folds`

  **Must NOT do**:
  - Do not put training code in the package source — scripts only
  - Do not build hyperparameter search — use default smoothing with gridsearch for lambda only
  - Do not import scoring functions — training is independent

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: ML training pipeline with cross-validation, monotonic constraints, artifact export
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 8, 9, 11)
  - **Blocks**: None (artifacts used by T12 but T12 can use mock/placeholder)
  - **Blocked By**: Tasks 1, 3

  **References**:

  **Pattern References**:
  - `scripts/entropy_bin_and_balance.py:107-136` — Feature extraction + processing loop pattern.
  - `scripts/entropy_bin_and_balance.py:30-62` — `HIGH_FEATURES` / `LOW_FEATURES` for understanding what drives monotonicity.

  **API/Type References**:
  - `src/mental_entropy/scoring/constants.py` — (from T3) `CE_FEATURE_ORDER`, `CE_MONOTONICITY`, `NE_FEATURE_ORDER`, `NE_MONOTONICITY`
  - `src/mental_entropy/features/ce.py:24-57` — `CE_FEATURE_KEYS` frozenset for validation

  **External References**:
  - pyGAM docs: `https://pygam.readthedocs.io/` — `LinearGAM`, `s()` term, `constraints` parameter, `gridsearch()`

  **WHY Each Reference Matters**:
  - `scoring/constants.py` — Provides the frozen feature order (must match between training and inference) and monotonicity directions.
  - `entropy_bin_and_balance.py:30-62` — Source of truth for which features drive entropy up/down.

  **Acceptance Criteria**:

  - [ ] Script exists and `--help` works
  - [ ] With a small labeled dataset (5 entries), script runs to completion
  - [ ] Produces `.joblib` files in the output directory
  - [ ] Produces metadata JSON with feature order, constraints, and CV metrics
  - [ ] Model can be loaded: `joblib.load('gam_ce.joblib')`

  **QA Scenarios**:

  ```
  Scenario: Training script help and structure
    Tool: Bash
    Steps:
      1. Run `uv run python scripts/train_gam.py --help` (or equivalent)
    Expected Result: Shows --labeled-data, --output-dir flags
    Evidence: .sisyphus/evidence/task-10-help.txt

  Scenario: Smoke run with synthetic mini-dataset
    Tool: Bash (uv run python)
    Steps:
      1. Create a 10-entry labeled JSONL with random features and labels
      2. Run training with --k-folds 2 and this mini dataset
      3. Verify .joblib files are created
      4. Load model and make one prediction
    Expected Result: Training completes, model files exist, prediction returns float
    Evidence: .sisyphus/evidence/task-10-smoke.txt
  ```

  **Commit**: YES
  - Message: `feat(scripts): add GAM training pipeline for CE/NE subscores`
  - Files: `scripts/train_gam.py` (or `scripts/train_subscore_models.py`)

---

- [x] 11. XGBoost Training Script (SE + CLE)

  **What to do**:
  - Create `scripts/train_xgb.py` (or section within `scripts/train_subscore_models.py`)
  - Load labeled JSONL, extract features, train XGBoost models for SE and CLE
  - Apply monotonic constraints from constants: build `monotone_constraints` tuple from `SE_MONOTONE_CONSTRAINTS` / `CLE_MONOTONE_CONSTRAINTS`
  - Use conservative hyperparameters for small dataset: `max_depth=3`, `learning_rate=0.05`, `n_estimators=200`, `reg_alpha=0.1`, `reg_lambda=1.0`, `subsample=0.8`
  - K-fold cross-validation (k=5) for model selection, retrain on full data for production
  - Export model via XGBoost native `save_model()` to `.json` format: `xgb_se.json`, `xgb_cle.json`
  - Save metadata JSON with: feature order, monotone constraints, CV metrics, hyperparameters
  - CLI: `--labeled-data`, `--output-dir`, `--k-folds`

  **Must NOT do**:
  - Do not build hyperparameter search — use fixed conservative params
  - Do not use pickle for XGBoost — use native JSON format only
  - Do not put training code in the package

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: ML training with cross-validation, constraints, multiple output formats
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 8, 9, 10)
  - **Blocks**: None
  - **Blocked By**: Tasks 1, 3

  **References**:

  **API/Type References**:
  - `src/mental_entropy/scoring/constants.py` — (from T3) `SE_FEATURE_ORDER`, `SE_MONOTONE_CONSTRAINTS`, `CLE_FEATURE_ORDER`, `CLE_MONOTONE_CONSTRAINTS`

  **External References**:
  - XGBoost docs: monotone_constraints parameter
  - XGBoost save_model JSON format

  **Acceptance Criteria**:

  - [ ] Script exists and `--help` works
  - [ ] With small labeled dataset, script runs to completion
  - [ ] Produces `.json` model files
  - [ ] Produces metadata JSON with hyperparameters and CV metrics
  - [ ] Model loadable: `xgb.XGBRegressor(); model.load_model('xgb_se.json')`

  **QA Scenarios**:

  ```
  Scenario: XGBoost smoke training
    Tool: Bash (uv run python)
    Steps:
      1. Create 10-entry mini dataset with random features + labels
      2. Run training with --k-folds 2
      3. Verify .json model files created
      4. Load model, make prediction, assert float output
    Expected Result: Training completes, model loadable
    Evidence: .sisyphus/evidence/task-11-smoke.txt
  ```

  **Commit**: YES (groups with T10 if in same file)
  - Message: `feat(scripts): add XGBoost training pipeline for SE/CLE subscores`
  - Files: `scripts/train_xgb.py`

---

- [x] 12. GAM Scoring Functions (score_ce + score_ne)

  **What to do**:
  - Create `src/mental_entropy/scoring/gam_scorers.py`
  - Implement `score_ce(features: dict[str, int | float]) -> SubscoreResult`:
    1. Validate features against `CE_FEATURE_KEYS` using `_validate_features()`
    2. Load GAM model (lazy, cached via loader)
    3. Convert feature dict to ordered numpy array using `CE_FEATURE_ORDER`
    4. Predict: `score = float(gam.predict(X)[0])`
    5. Clamp score to [0, 1]
    6. Compute feature contributions using GAM's per-term partial dependence:
       ```python
       contributions = {}
       for i, name in enumerate(CE_FEATURE_ORDER):
           contributions[name] = float(gam.partial_dependence(term=i, X=X)[0])
       ```
    7. Return `SubscoreResult(score=score, feature_contributions=contributions)`
  - Implement `score_ne(features: dict[str, float]) -> SubscoreResult` — same pattern
  - Add convenience wrappers: `score_ce_from_result(result: EmbeddingResult) -> SubscoreResult` that calls `ce_features_from_result()` then `score_ce()`
  - Handle edge case: all-zero features → valid score (whatever the model produces, clamped to [0,1])

  **Must NOT do**:
  - Do not use SHAP for GAM models — use native term contributions
  - Do not import pygam at module level — lazy import inside the function
  - Do not modify any existing feature module

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: ML model inference, GAM-specific API (partial_dependence), lazy loading, edge cases
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 13, 14)
  - **Blocks**: Tasks 14, 16
  - **Blocked By**: Tasks 3, 6

  **References**:

  **Pattern References**:
  - `src/mental_entropy/features/ce.py:269-282` — `ce_features_from_result()` wrapper pattern. Follow this for `score_ce_from_result()`.
  - `src/mental_entropy/scoring/loader.py` — (from T6) `_validate_features()` and `_load_model()`

  **API/Type References**:
  - `src/mental_entropy/scoring/types.py` — (from T2) `SubscoreResult`
  - `src/mental_entropy/scoring/constants.py` — (from T3) `CE_FEATURE_ORDER`, `CE_MONOTONICITY`

  **External References**:
  - pyGAM `partial_dependence()` — Returns per-term contribution for a given input. This is the GAM-native equivalent of SHAP values.

  **WHY Each Reference Matters**:
  - `ce.py:269-282` — The `_from_result()` wrapper pattern. The scoring functions need identical wrappers for pipeline convenience.
  - `loader.py` — Provides the infrastructure (validation, loading) that these functions consume.

  **Acceptance Criteria**:

  - [ ] `src/mental_entropy/scoring/gam_scorers.py` exists with `score_ce()` and `score_ne()`
  - [ ] Both return `SubscoreResult` with `score` in [0, 1] and `feature_contributions` dict
  - [ ] Feature contributions keys match `CE_FEATURE_KEYS` / `NE_FEATURE_KEYS`
  - [ ] `score_ce_from_result()` and `score_ne_from_result()` wrappers exist
  - [ ] All-zero feature input produces valid result without error
  - [ ] Result is JSON-serializable

  **QA Scenarios**:

  ```
  Scenario: GAM scoring produces valid SubscoreResult
    Tool: Bash (uv run python)
    Preconditions: Model artifacts exist in models/ dir
    Steps:
      1. Run `uv run python -c "
         from mental_entropy.features import CE_FEATURE_KEYS
         from mental_entropy.scoring.gam_scorers import score_ce
         features = {k: 0.5 for k in CE_FEATURE_KEYS}
         features['ce_n_sentences'] = 5
         features['ce_n_adj'] = 4
         result = score_ce(features)
         assert 0.0 <= result.score <= 1.0, f'Score out of range: {result.score}'
         assert set(result.feature_contributions.keys()) == CE_FEATURE_KEYS
         assert all(isinstance(v, float) for v in result.feature_contributions.values())
         print(f'CE={result.score:.3f}, GAM_SCORE_OK')
         "`
    Expected Result: Prints score and "GAM_SCORE_OK"
    Failure Indicators: Score out of range, missing contribution keys
    Evidence: .sisyphus/evidence/task-12-gam-score.txt

  Scenario: All-zero features edge case
    Tool: Bash (uv run python)
    Steps:
      1. Run `uv run python -c "
         from mental_entropy.features import CE_FEATURE_KEYS
         from mental_entropy.scoring.gam_scorers import score_ce
         features = {k: 0.0 for k in CE_FEATURE_KEYS}
         result = score_ce(features)
         assert 0.0 <= result.score <= 1.0
         print('ZERO_FEATURES_OK')
         "`
    Expected Result: Prints "ZERO_FEATURES_OK"
    Evidence: .sisyphus/evidence/task-12-zero-features.txt
  ```

  **Commit**: YES
  - Message: `feat(scoring): add GAM-based CE and NE scoring functions`
  - Files: `src/mental_entropy/scoring/gam_scorers.py`

---

- [x] 13. XGBoost Scoring Functions (score_se + score_cle)

  **What to do**:
  - Create `src/mental_entropy/scoring/xgb_scorers.py`
  - Implement `score_se(features: dict[str, int | float]) -> SubscoreResult`:
    1. Validate features against `SE_FEATURE_KEYS` using `_validate_features()`
    2. Load XGBoost model (lazy, cached)
    3. Convert feature dict to ordered numpy array using `SE_FEATURE_ORDER`
    4. Predict: `score = float(model.predict(X)[0])`
    5. Clamp to [0, 1]
    6. Compute SHAP values using `shap.TreeExplainer`:
       ```python
       explainer = shap.TreeExplainer(model)
       shap_values = explainer.shap_values(X)
       contributions = {name: float(shap_values[0][i]) for i, name in enumerate(SE_FEATURE_ORDER)}
       ```
    7. Return `SubscoreResult(score=score, feature_contributions=contributions)`
  - Implement `score_cle()` — same pattern
  - Add `score_se_from_result()` and `score_cle_from_result()` convenience wrappers
  - Cache the TreeExplainer alongside the model (create once per model load)

  **Must NOT do**:
  - Do not import xgboost/shap at module level — lazy import
  - Do not store background data for TreeExplainer (tree_path_dependent doesn't need it)
  - Do not modify existing feature modules

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: ML inference + SHAP integration, caching, lazy imports
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 12, 14)
  - **Blocks**: Tasks 14, 16
  - **Blocked By**: Tasks 3, 6

  **References**:

  **Pattern References**:
  - `src/mental_entropy/features/se.py` — `se_features_from_result()` wrapper pattern
  - `src/mental_entropy/scoring/loader.py` — (from T6) Validation and loading

  **API/Type References**:
  - `src/mental_entropy/scoring/types.py` — `SubscoreResult`
  - `src/mental_entropy/scoring/constants.py` — `SE_FEATURE_ORDER`, `CLE_FEATURE_ORDER`

  **Acceptance Criteria**:

  - [ ] `src/mental_entropy/scoring/xgb_scorers.py` exists with `score_se()` and `score_cle()`
  - [ ] Both return `SubscoreResult` with valid score and feature_contributions
  - [ ] SHAP values sum approximately to `score - baseline` (within floating point tolerance)
  - [ ] Convenience wrappers `score_se_from_result()` and `score_cle_from_result()` exist
  - [ ] All-zero input produces valid result

  **QA Scenarios**:

  ```
  Scenario: XGBoost scoring with SHAP contributions
    Tool: Bash (uv run python)
    Preconditions: Model artifacts exist
    Steps:
      1. Run `uv run python -c "
         from mental_entropy.features import SE_FEATURE_KEYS
         from mental_entropy.scoring.xgb_scorers import score_se
         features = {k: 0.5 for k in SE_FEATURE_KEYS}
         features['se_n_clusters'] = 3
         features['se_switch_count'] = 2
         result = score_se(features)
         assert 0.0 <= result.score <= 1.0
         assert set(result.feature_contributions.keys()) == SE_FEATURE_KEYS
         print(f'SE={result.score:.3f}, XGB_SCORE_OK')
         "`
    Expected Result: Prints score and "XGB_SCORE_OK"
    Evidence: .sisyphus/evidence/task-13-xgb-score.txt
  ```

  **Commit**: YES
  - Message: `feat(scoring): add XGBoost-based SE and CLE scoring with SHAP`
  - Files: `src/mental_entropy/scoring/xgb_scorers.py`

---

- [x] 14. Public API Exports + __init__.py Updates

  **What to do**:
  - Update `src/mental_entropy/scoring/__init__.py` to export all public scoring functions:
    ```python
    from mental_entropy.scoring.types import SubscoreResult, MESResult
    from mental_entropy.scoring.gam_scorers import score_ce, score_ne, score_ce_from_result, score_ne_from_result
    from mental_entropy.scoring.xgb_scorers import score_se, score_cle, score_se_from_result, score_cle_from_result
    from mental_entropy.scoring.combiner import mes_score
    ```
  - Update `src/mental_entropy/__init__.py` to re-export scoring functions at the top level:
    ```python
    from mental_entropy.scoring import (
        SubscoreResult, MESResult,
        score_ce, score_se, score_ne, score_cle,
        score_ce_from_result, score_se_from_result, score_ne_from_result, score_cle_from_result,
        mes_score,
    )
    ```
  - Add all new names to `__all__`
  - Guard scoring imports with try/except for users without scoring extras installed:
    ```python
    try:
        from mental_entropy.scoring import ...
    except ImportError:
        pass  # scoring extras not installed
    ```

  **Must NOT do**:
  - Do not remove existing exports
  - Do not make scoring imports mandatory (guard with try/except)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Import wiring only
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after T7, T12, T13)
  - **Blocks**: Tasks 17, 18, 19
  - **Blocked By**: Tasks 7, 12, 13

  **References**:

  **Pattern References**:
  - `src/mental_entropy/__init__.py:1-27` — Current top-level exports. Follow this exact pattern.
  - `src/mental_entropy/features/__init__.py:1-63` — Feature package exports pattern.

  **Acceptance Criteria**:

  - [ ] `from mental_entropy.scoring import score_ce, score_se, score_ne, score_cle, mes_score` works
  - [ ] `from mental_entropy import score_ce, mes_score` works (top-level re-export)
  - [ ] Without scoring extras: `from mental_entropy import embed_journal_entry` still works
  - [ ] Existing imports unchanged: `from mental_entropy import ce_features` works

  **QA Scenarios**:

  ```
  Scenario: All scoring functions importable from top level
    Tool: Bash (uv run python)
    Steps:
      1. Run `uv run python -c "
         from mental_entropy import score_ce, score_se, score_ne, score_cle, mes_score
         from mental_entropy import SubscoreResult, MESResult
         from mental_entropy import score_ce_from_result, score_se_from_result
         print('ALL_EXPORTS_OK')
         "`
    Expected Result: Prints "ALL_EXPORTS_OK"
    Evidence: .sisyphus/evidence/task-14-exports.txt

  Scenario: Existing imports unaffected
    Tool: Bash (uv run python)
    Steps:
      1. Run `uv run python -c "
         from mental_entropy import embed_journal_entry, ce_features, se_features, ne_features, cle_features
         from mental_entropy import ce_features_from_result, se_features_from_result
         print('EXISTING_EXPORTS_OK')
         "`
    Expected Result: Prints "EXISTING_EXPORTS_OK"
    Evidence: .sisyphus/evidence/task-14-existing.txt
  ```

  **Commit**: YES
  - Message: `feat(scoring): wire up public API exports`
  - Files: `src/mental_entropy/scoring/__init__.py`, `src/mental_entropy/__init__.py`

---

- [x] 15. Tests: Scoring Types, Constants, Model Loader

  **What to do**:
  - Create `tests/scoring/test_scoring_types.py`
  - Create `tests/scoring/test_scoring_constants.py`
  - Create `tests/scoring/test_scoring_loader.py`
  - Test types: frozen check, slots, JSON serializability via `dataclasses.asdict()` + `json.dumps()`
  - Test constants: feature order lengths match feature key frozensets, monotonicity keys subset of feature keys, feature orders are sorted tuples
  - Test loader: `_validate_features()` with valid dict, missing keys, extra keys, int→float cast; `_load_model()` with missing file
  - Follow existing test patterns: exact assertions, boundary cases, parametrize where appropriate

  **Must NOT do**:
  - Do not modify existing tests
  - Do not require model artifacts for loader tests (test error path)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multiple test files, comprehensive test coverage
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 16, 17, 18, 19)
  - **Blocks**: None
  - **Blocked By**: Tasks 2, 3, 6

  **References**:

  **Pattern References**:
  - `tests/features/test_ce_features.py:66-91` — Exact keyset validation test pattern. Follow this for constants tests.
  - `tests/features/test_se_features.py` — JSON serializability and type checking patterns.

  **Acceptance Criteria**:

  - [ ] `uv run pytest tests/scoring/test_scoring_types.py -v` passes
  - [ ] `uv run pytest tests/scoring/test_scoring_constants.py -v` passes
  - [ ] `uv run pytest tests/scoring/test_scoring_loader.py -v` passes
  - [ ] Tests cover: frozen instances, JSON round-trip, keyset matching, validation errors

  **QA Scenarios**:

  ```
  Scenario: All scoring infrastructure tests pass
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/scoring/ -v --tb=short`
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-15-tests.txt
  ```

  **Commit**: YES
  - Message: `test(scoring): add tests for types, constants, and model loader`
  - Files: `tests/scoring/test_scoring_types.py`, `tests/scoring/test_scoring_constants.py`, `tests/scoring/test_scoring_loader.py`

---

- [x] 16. Tests: Scoring Functions + Feature Contributions

  **What to do**:
  - Create `tests/scoring/test_gam_scorers.py`
  - Create `tests/scoring/test_xgb_scorers.py`
  - Test: valid input → SubscoreResult with score in [0,1], contributions dict with correct keys
  - Test: all-zero features → valid result
  - Test: determinism (same input, same output, 2 calls)
  - Test: JSON serializability of results
  - Test: `_from_result()` wrappers produce same output as direct call
  - If no model artifacts available: mock the model loading, test the validation and formatting logic
  - Test: feature contributions keys match expected feature keys exactly

  **Must NOT do**:
  - Do not modify existing tests

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 15, 17, 18, 19)
  - **Blocks**: None
  - **Blocked By**: Tasks 12, 13

  **References**:

  **Pattern References**:
  - `tests/features/test_ce_features.py` — Type checking, determinism, keyset validation patterns
  - `tests/features/test_cle_features.py` — Boundary case patterns (empty, single, all-zero)

  **Acceptance Criteria**:

  - [ ] `uv run pytest tests/scoring/test_gam_scorers.py -v` passes
  - [ ] `uv run pytest tests/scoring/test_xgb_scorers.py -v` passes
  - [ ] Tests cover: valid scoring, determinism, edge cases, JSON serializability

  **QA Scenarios**:

  ```
  Scenario: Scorer tests pass
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/scoring/test_gam_scorers.py tests/scoring/test_xgb_scorers.py -v`
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-16-tests.txt
  ```

  **Commit**: YES
  - Message: `test(scoring): add tests for GAM and XGBoost scoring functions`
  - Files: `tests/scoring/test_gam_scorers.py`, `tests/scoring/test_xgb_scorers.py`

---

- [x] 17. Tests: MES Combiner + Integration + E2E Pipeline

  **What to do**:
  - Create `tests/scoring/test_combiner.py`
  - Create `tests/scoring/test_integration.py`
  - Combiner tests: exact formula verification, boundary values (0,0,0,0 → 0; 1,1,1,1 → 100), invalid input rejection, clamping, determinism, type checks (`score` is int, `raw` is float)
  - Integration tests: full pipeline `text → embed → features → subscores → MES`. Use a known journal text, verify all intermediate types are correct and final MES is in [0, 100]
  - Round-trip serialization test: all result types → `json.dumps(dataclasses.asdict())` → `json.loads()` → verify values preserved
  - Package import test: `from mental_entropy import mes_score` works

  **Must NOT do**:
  - Do not modify existing tests
  - Do not hardcode specific MES values (models may change) — test ranges and types

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Integration testing across multiple layers, pipeline verification
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 15, 16, 18, 19)
  - **Blocks**: None
  - **Blocked By**: Tasks 7, 14

  **References**:

  **Pattern References**:
  - `tests/test_embedding_smoke.py` — Integration test pattern for the embedding layer. Follow this style for pipeline tests.

  **Acceptance Criteria**:

  - [ ] `uv run pytest tests/scoring/test_combiner.py tests/scoring/test_integration.py -v` passes
  - [ ] Combiner tests verify exact formula for known inputs
  - [ ] Integration test runs full pipeline without error
  - [ ] All existing tests still pass: `uv run pytest tests/ -v`

  **QA Scenarios**:

  ```
  Scenario: Full test suite (existing + new) passes
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/ -v`
    Expected Result: All tests pass (existing 140+ and new scoring tests)
    Evidence: .sisyphus/evidence/task-17-full-suite.txt
  ```

  **Commit**: YES
  - Message: `test(scoring): add combiner and integration tests`
  - Files: `tests/scoring/test_combiner.py`, `tests/scoring/test_integration.py`, `tests/scoring/__init__.py`

---

- [x] 18. Validation Suite + Edge Cases

  **What to do**:
  - Create `tests/scoring/test_validation.py` — comprehensive validation
  - Test stability: run scoring 100 times on same input, verify identical output each time
  - Test all edge cases from Metis review:
    - Empty feature dict → clear error
    - All-zero features → valid score
    - Features with int values (e.g., `ce_n_sentences=5`) → works without type error
    - Missing feature key → clear `ValueError`
    - Extra feature key → clear `ValueError`
    - Score output always in [0, 1] for subscores, [0, 100] for MES
  - Test model artifact integrity: load, predict, verify no NaN/Inf in output
  - Test deterministic round-trip: serialize result → deserialize → compare

  **Must NOT do**:
  - Do not modify existing tests
  - Do not test model accuracy (that's the training pipeline's job)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Comprehensive edge case testing, stability verification
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 15, 16, 17, 19)
  - **Blocks**: None
  - **Blocked By**: Task 14

  **References**:

  **Pattern References**:
  - `tests/features/test_ce_features.py` — Empty input and NaN/Inf validation patterns
  - `tests/features/test_cle_features.py` — Edge case patterns

  **Acceptance Criteria**:

  - [ ] `uv run pytest tests/scoring/test_validation.py -v` passes
  - [ ] Stability test: 100 identical calls produce identical results
  - [ ] All 6 Metis edge cases covered

  **QA Scenarios**:

  ```
  Scenario: Validation suite passes
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/scoring/test_validation.py -v --tb=short`
    Expected Result: All validation tests pass
    Evidence: .sisyphus/evidence/task-18-validation.txt
  ```

  **Commit**: YES
  - Message: `test(scoring): add validation suite and edge case tests`
  - Files: `tests/scoring/test_validation.py`

---

- [x] 19. Documentation Updates

  **What to do**:
  - Update `docs/mes_architecture.md`:
    - Mark Gaps 1-4 as implemented
    - Add code mapping for Layer 3 (scoring/) and Layer 4 (combiner)
    - Document model types: CE/NE = pyGAM, SE/CLE = XGBoost
    - Document artifact locations
    - Update "Quick Code Entry Points"
  - Update `README.md`:
    - Add "Scoring a Journal Entry" usage section showing the full pipeline
    - Add `scoring` extras to installation: `uv sync --extra scoring`
    - Add `mes_score()` usage example
    - Update project structure tree
  - Update `CLAUDE.md`:
    - Add scoring module to architecture section
    - Add scoring commands to common commands
    - Note scoring optional deps
  - Update `docs/agent_docs_index.md` with new scoring docs routing

  **Must NOT do**:
  - Do not create new documentation files (update existing ones only)
  - Do not add emojis

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation updates across multiple files
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 15, 16, 17, 18)
  - **Blocks**: None
  - **Blocked By**: Task 14

  **References**:

  **Pattern References**:
  - `docs/mes_architecture.md:1-100` — Existing architecture doc structure. Add new sections following the same format.
  - `README.md` — Existing usage examples and structure tree. Follow same style.

  **Acceptance Criteria**:

  - [ ] `docs/mes_architecture.md` marks gaps as implemented
  - [ ] `README.md` has scoring usage example
  - [ ] `CLAUDE.md` references scoring module
  - [ ] All updated docs are valid markdown

  **QA Scenarios**:

  ```
  Scenario: Docs reference real code paths
    Tool: Bash (uv run python)
    Steps:
      1. Run `uv run python -c "
         # Verify the usage example from README actually works
         from mental_entropy.scoring import score_ce, mes_score
         from mental_entropy.scoring.types import SubscoreResult, MESResult
         print('DOCS_IMPORTS_OK')
         "`
    Expected Result: Prints "DOCS_IMPORTS_OK"
    Evidence: .sisyphus/evidence/task-19-docs.txt
  ```

  **Commit**: YES
  - Message: `docs: update architecture, README, and CLAUDE.md for scoring layer`
  - Files: `docs/mes_architecture.md`, `README.md`, `CLAUDE.md`, `docs/agent_docs_index.md`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Rejection → fix → re-run.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, import, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run pytest tests/ -v`. Review all changed files for: `as Any` casts, empty catches, print statements in library code, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic variable names. Verify all new code follows existing conventions (frozen dataclasses, JSON serializability, determinism).
  Output: `Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Run full pipeline: `embed_journal_entry(text) → features → subscores → mes_score()`. Verify each step produces correct types. Test with 3 different journal texts (coherent, scattered, chaotic). Test without scoring extras installed (features still work). Save evidence to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Verify no files in `features/`, `embedding/`, `utils/` were modified. Detect unaccounted changes. Flag any scoring deps that leaked into core dependencies.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| After Task(s) | Message | Key Files |
|---------------|---------|-----------|
| 1 | `build(scoring): add optional scoring dependency group` | `pyproject.toml` |
| 2, 4 | `feat(scoring): add types and models directory scaffold` | `scoring/types.py`, `models/__init__.py` |
| 3 | `feat(scoring): add feature ordering and monotonicity constants` | `scoring/constants.py` |
| 5 | `feat(datagen): add LLM labeling rubric` | `datagen/labeling_rubric.py` |
| 6 | `feat(scoring): add model loader infrastructure` | `scoring/loader.py`, `scoring/__init__.py` |
| 7 | `feat(scoring): add MES combiner` | `scoring/combiner.py` |
| 8, 9 | `feat(scripts): add labeling and validation scripts` | `scripts/label_journals.py`, `scripts/validate_labels.py` |
| 10 | `feat(scripts): add GAM training pipeline` | `scripts/train_gam.py` |
| 11 | `feat(scripts): add XGBoost training pipeline` | `scripts/train_xgb.py` |
| 12 | `feat(scoring): add GAM scoring for CE/NE` | `scoring/gam_scorers.py` |
| 13 | `feat(scoring): add XGBoost scoring for SE/CLE` | `scoring/xgb_scorers.py` |
| 14 | `feat(scoring): wire up public API exports` | `scoring/__init__.py`, `__init__.py` |
| 15-18 | `test(scoring): add comprehensive scoring test suite` | `tests/scoring/` |
| 19 | `docs: update architecture and README for scoring layer` | `docs/`, `README.md`, `CLAUDE.md` |

---

## Success Criteria

### Verification Commands
```bash
# 1. Deps install
uv sync --extra scoring  # Expected: exit 0

# 2. Package import
uv run python -c "from mental_entropy.scoring import score_ce, score_se, score_ne, score_cle, mes_score; print('OK')"
# Expected: OK

# 3. Full pipeline
uv run python -c "
from mental_entropy import embed_journal_entry, ce_features_from_result, se_features_from_result, ne_features_from_result, cle_features_from_result
from mental_entropy.scoring import score_ce, score_se, score_ne, score_cle, mes_score
r = embed_journal_entry('Today was a good day. I felt calm and clear-headed throughout.')
ce = score_ce(ce_features_from_result(r))
se = score_se(se_features_from_result(r))
ne = score_ne(ne_features_from_result(r))
cle = score_cle(cle_features_from_result(r))
m = mes_score(ce.score, se.score, ne.score, cle.score)
print(f'MES={m.score}, CE={ce.score:.2f}, SE={se.score:.2f}, NE={ne.score:.2f}, CLE={cle.score:.2f}')
assert 0 <= m.score <= 100
print('PIPELINE_OK')
"
# Expected: PIPELINE_OK

# 4. Tests
uv run pytest tests/ -v
# Expected: all pass

# 5. Existing functionality preserved
uv run python -c "from mental_entropy import embed_journal_entry, ce_features; print('EXISTING_OK')"
# Expected: EXISTING_OK
```

### Final Checklist
- [x] All "Must Have" items present and verified
- [x] All "Must NOT Have" guardrails respected
- [x] All existing tests still pass
- [x] No files in `features/`, `embedding/`, `utils/` modified
- [x] Scoring deps are optional only
- [x] Model artifacts bundled and loadable
- [x] Documentation updated
