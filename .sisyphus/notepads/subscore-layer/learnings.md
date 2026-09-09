# Learnings

- Added `scoring` optional extra in `pyproject.toml` under `[project.optional-dependencies]` with pyGAM/xgboost/shap/joblib pinned to requested ranges.
- `uv sync --extra scoring` installs scoring deps without moving them into core `[project].dependencies`.
- Import smoke for `pygam`, `xgboost`, `shap`, and `joblib` succeeded (`ALL_DEPS_OK`).
- `src/mental_entropy/scoring/types.py` now defines `SubscoreResult` and `MESResult` as `@dataclass(frozen=True, slots=True)` with JSON-serializable field types only.
- `src/mental_entropy/scoring/__init__.py` exports `SubscoreResult` and `MESResult` via `__all__` using existing package export conventions.
- Type smoke test passed for importability, frozen behavior, and `dataclasses.asdict(...)` JSON serialization; evidence saved to `.sisyphus/evidence/task-2-types-ok.txt`.
- Added `src/mental_entropy/scoring/constants.py` with `tuple(sorted(...))` feature ordering derived directly from `*_FEATURE_KEYS` imports to keep ordering deterministic and auto-synced with feature modules.
- Monotonic constraints now derive from `HIGH_FEATURES`/`LOW_FEATURES` (`scripts/entropy_bin_and_balance.py`) via a shared mapping, which keeps GAM direction dictionaries and XGBoost constraint tuples consistent with the scoring source of truth.
- Added `src/mental_entropy/datagen/labeling_rubric.py` with stable exported symbols for Task 8 imports: `LABELING_SYSTEM_PROMPT`, `LABELING_OUTPUT_SCHEMA`, `LABELING_FEW_SHOT_EXAMPLES`, `LABELING_USER_PROMPT_TEMPLATE`.
- Rubric anchors are explicit at `0.0/0.25/0.5/0.75/1.0` across CE/SE/NE/CLE and constrain labeling to text-only judgment.

- Creating `src/mental_entropy/models/__init__.py` is sufficient to make `mental_entropy.models` discoverable via `importlib.resources.files(...)`.
- Hatchling wheel config `packages = ["src/mental_entropy"]` already covers `mental_entropy.models`; no extra package-data/manifest config needed for this scaffold step.
- Acceptance compatibility can be preserved with symbol aliases: `LABELING_PROMPT = LABELING_SYSTEM_PROMPT` and `RESPONSE_SCHEMA = LABELING_OUTPUT_SCHEMA`.
- Export compatibility for downstream imports should include both legacy and required names in `__all__`.
- Added `src/mental_entropy/scoring/loader.py` with module-level `_MODEL_CACHE` and `importlib.resources.files("mental_entropy.models")` artifact resolution, so model loading stays lazy and package-safe.
- `_load_model(name, model_type)` now supports `joblib` and `json` artifact loading with function-local heavy imports (`joblib`, `xgboost`) and a missing-artifact `FileNotFoundError` that includes scoring extras guidance.
- `_validate_features(...)` now enforces exact key matching (missing/extra raise `ValueError`), accepts `int` inputs, and returns ordered `list[float]` values via `feature_order`.

- Added `src/mental_entropy/scoring/combiner.py` with deterministic `mes_score(ce, se, ne, cle) -> MESResult` using fixed weights `0.35/0.15/0.20/0.30`, raw clamp `[0.0, 1.0]`, and integer score clamp `[0, 100]` after `round(raw * 100)`.
- Input validation now rejects bools and non-numeric values and raises field-specific `ValueError` messages for type/range violations (`ce`, `se`, `ne`, `cle`).
- Task-7 validation evidence saved to `.sisyphus/evidence/task-7-combiner-ok.txt` and `.sisyphus/evidence/task-7-boundary-ok.txt`.

- Added `scripts/train_xgb.py` for Task 11 with CLI (`--labeled-data`, `--output-dir`, `--k-folds`) and fixed conservative XGBoost params (`max_depth=3`, `learning_rate=0.05`, `n_estimators=200`, `reg_alpha=0.1`, `reg_lambda=1.0`, `subsample=0.8`).
- Training script uses frozen feature orders and monotone constraints from `mental_entropy.scoring.constants` (`SE_FEATURE_ORDER`/`CLE_FEATURE_ORDER`, `SE_MONOTONE_CONSTRAINTS`/`CLE_MONOTONE_CONSTRAINTS`) and exports native JSON models (`xgb_se.json`, `xgb_cle.json`) plus `xgb_metadata.json`.
- Robust tiny-dataset handling: CV uses effective folds `min(requested_k, n_samples)`, skips with explicit metadata note when fewer than 2 folds are possible, and still retrains final models on full data.
- Smoke QA passed for help, mini train run, and model load/predict; evidence saved to `.sisyphus/evidence/task-11-smoke.txt`.

- Added `scripts/label_journals.py` for Task 8 with OpenAI-compatible env vars (`OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`), urllib-based request handling, retry logic, and resume behavior that skips already-labeled IDs.
- Label response parsing now enforces strict JSON object validation with required keys, schema-aligned anchors, and score bounds in `[0, 1]` before writing output.
- Task-8 evidence generated: `.sisyphus/evidence/task-8-help.txt` and `.sisyphus/evidence/task-8-parse-ok.txt`.

- Added `scripts/validate_labels.py` as a deterministic JSONL validator that computes per-subscore Spearman correlation via `scipy.stats.spearmanr` and disagreement counts for `abs(label - heuristic) > 0.3` without rejecting labels.
- Validator gracefully handles missing IDs by attempting ID-based matching first (`id`, `entry_id`, `journal_id`) then positional fallback for scored rows without IDs, while reporting missing/duplicate ID health metrics.
- Validation output is concise but complete: run-level matching stats, per-subscore correlation + missing counts, and side-by-side LLM vs heuristic distributions (`min/p25/median/p75/max/mean`).

- Added `scripts/train_gam.py` for Task 10 with CLI (`--labeled-data`, `--output-dir`, `--k-folds`), CE/NE feature extraction fallback (`ce_features`/`ne_features` dicts first, `journal_text` embedding fallback second), monotonic GAM term construction from `CE_MONOTONICITY`/`NE_MONOTONICITY`, and `joblib.dump` exports for `gam_ce.joblib` + `gam_ne.joblib`.
- GAM training metadata now records UTC timestamp, frozen feature order, per-feature constraint mapping, requested/used CV folds, fold metrics (`mse`, `mae`, `r2`), and artifact filenames in `gam_metadata.json`.
- Script bootstrap now prepends repo root + `src/` to `sys.path` so `uv run python scripts/train_gam.py ...` can import `mental_entropy.scoring.constants` even though that module references `scripts.entropy_bin_and_balance`.
- Tiny-dataset smoke runs can emit repeated `did not converge` lines from pyGAM while still producing valid artifacts and loadable models; current script keeps this non-fatal and completes export.

- Added `src/mental_entropy/scoring/gam_scorers.py` for Task 12 with `score_ce`, `score_ne`, and `_from_result` wrappers, reusing `_validate_features(...)`, lazy `_load_model(..., 'joblib')`, deterministic `numpy` row shaping, and clamped `[0, 1]` outputs.
- GAM feature contributions now use native `partial_dependence(term=i, X=X)` per frozen feature order with robust scalar coercion (`np.asarray(...).reshape(-1)[0]`) to keep JSON-safe Python floats and stable contribution key ordering.

- Added `src/mental_entropy/scoring/xgb_scorers.py` for Task 13 with `score_se`, `score_cle`, `score_se_from_result`, and `score_cle_from_result`, using `_validate_features(...)`, lazy `_load_model(..., 'json')`, and XGBoost Booster prediction through `xgboost.DMatrix` built from ordered `numpy.float32` rows.
- XGBoost scorer outputs now clamp scores to `[0, 1]`, compute SHAP contributions via `shap.TreeExplainer(model)` (no background dataset), and return deterministic contribution dicts keyed exactly by `SE_FEATURE_ORDER` / `CLE_FEATURE_ORDER`.
- Added module-local TreeExplainer cache (`_TREE_EXPLAINER_CACHE`) keyed by model name to avoid repeated explainer initialization per model.

- Task 14 export wiring is complete: `mental_entropy.scoring` now re-exports scorer APIs (`score_ce/score_se/score_ne/score_cle`, `_from_result` variants), `mes_score`, and result dataclasses while preserving existing constants exports.
- Top-level `mental_entropy.__init__` now conditionally re-exports scoring symbols behind `try/except ImportError` and extends `__all__` only when scoring imports succeed, keeping non-scoring package imports unaffected.

- Added `tests/scoring/test_gam_scorers.py` and `tests/scoring/test_xgb_scorers.py` to validate scorer contracts for valid features, all-zero features, determinism, JSON-safe scalar types, exact contribution keysets/order, and `_from_result` wrapper parity with direct scoring calls.
- Task 16 test evidence updated at `.sisyphus/evidence/task-16-tests.txt` with `20 passed` (`uv run pytest tests/scoring/test_gam_scorers.py tests/scoring/test_xgb_scorers.py -v`).

- Added Task 17 tests in `tests/scoring/test_combiner.py` to cover exact MES weighted formula behavior, boundary outputs (`0` and `100`), invalid input rejection, deterministic repeated calls, and output type guarantees (`score` int, `raw` float).
- Added Task 17 integration coverage in `tests/scoring/test_integration.py` for end-to-end flow (`text -> embed -> features -> subscore models -> mes_score`), dataclass JSON roundtrip checks via `asdict/json.dumps/json.loads`, and top-level `from mental_entropy import mes_score` import verification.
- Added `tests/scoring/__init__.py` to ensure repo-root `scripts.*` imports are resolvable during full-suite collection so scoring top-level re-exports can load consistently.
- Full suite evidence for Task 17 saved to `.sisyphus/evidence/task-17-full-suite.txt` (`220 passed`).
- Added Task 15 test coverage in `tests/scoring/test_scoring_types.py`, `tests/scoring/test_scoring_constants.py`, and `tests/scoring/test_scoring_loader.py` for frozen/slots dataclasses, JSON serialization via `asdict()` + `json.dumps()`, constants ordering/keyset/monotonic constraints, and loader validation + missing-artifact error paths without requiring real model files.
- Task 15 QA run passed with `18 passed`; evidence saved to `.sisyphus/evidence/task-15-tests.txt`.

- Added `tests/scoring/test_validation.py` for Task 18 with deterministic validation coverage for: empty/missing/extra feature key errors, int-value acceptance in `_validate_features`, all-zero feature scoring, score clamp/range enforcement, 100-call stability, artifact output finiteness (no NaN/Inf), and JSON round-trip determinism for `SubscoreResult`/`MESResult`.
- Added repo-root `sys.path` bootstrap in the Task 18 test module so `scripts.entropy_bin_and_balance` remains importable in pytest `src/` layout when `mental_entropy.scoring.constants` is imported.
- Task 18 QA run passed: `uv run pytest tests/scoring/test_validation.py -v --tb=short` with evidence saved to `.sisyphus/evidence/task-18-validation.txt`.
- Task 19 docs now map scoring and combiner implementation paths in `docs/mes_architecture.md`, including `scoring/gam_scorers.py`, `scoring/xgb_scorers.py`, `scoring/combiner.py`, `scoring/loader.py`, and `src/mental_entropy/models/` artifacts.
- `README.md` now includes scoring install guidance (`uv sync --extra scoring`) plus end-to-end pipeline examples that produce CE/SE/NE/CLE subscores and final `mes_score`.
- `CLAUDE.md` and `docs/agent_docs_index.md` now route scoring tasks to the right docs and include scoring import verification command coverage.
