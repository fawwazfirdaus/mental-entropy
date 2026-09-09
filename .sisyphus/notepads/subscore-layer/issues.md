# Issues

- pyGAM monotonic smoke run printed `did not converge` before `PYGAM_MONOTONIC_OK`; command exited successfully but convergence behavior should be revisited with a more stable fitting setup if strict convergence is required.
- `lsp_diagnostics` could not execute because the configured Python LSP server (`basedpyright-langserver`) is not installed in this environment; `uv run pip install basedpyright` failed due PEP 668 externally-managed-environment restrictions.
- No new blockers in Task 3; ordering and monotonicity validation checks passed and evidence files were written.
- No blockers in Task 5 rubric design; only adjustment was removing a non-essential module docstring to satisfy comment/docstring hook policy.

- No task-4 blockers encountered; `importlib.resources.files('mental_entropy.models')` resolved successfully after package scaffold.
- No blockers in Task 5 acceptance fix; minimal alias patch resolved import mismatch without changing rubric semantics.
- No blockers in Task 6 implementation; all three required validation scenarios completed and evidence files were generated.

- No blockers in Task 7 combiner implementation; formula and validation behavior matched acceptance checks.

- While implementing Task 11 smoke, `XGBRegressor.save_model(...)` raised `TypeError: _estimator_type undefined` in this environment; resolved by exporting via native booster (`model.get_booster().save_model(...)`), which still loads correctly through `xgb.XGBRegressor().load_model(...)`.

- No blockers in Task 8 implementation; `--help` and parse smoke checks passed and evidence files were written.

- No blockers in Task 9 implementation; script help and smoke validation run succeeded with evidence files in `.sisyphus/evidence/`.
- Environment limitation still applies: Python LSP diagnostics (`basedpyright-langserver`) are unavailable, so `lsp_diagnostics` cannot be executed locally until that tool is installed in a writable Python environment.

- Task 12 QA scenarios are currently blocked by missing model artifacts in `src/mental_entropy/models` (`gam_ce.joblib` not found), so evidence files record the blocker instead of a scored output.

- Task 13 real-model runtime verification is artifact-dependent: `xgb_se.json` / `xgb_cle.json` are not present in `src/mental_entropy/models`, so functional QA used patched loader/dependency smoke to validate inference flow and API behavior without artifact files.

- No blockers in Task 14 export wiring; import smoke checks passed for both `mental_entropy.scoring` and top-level `mental_entropy` scoring re-exports.

- No blockers in Task 16 scorer test implementation; both new files passed targeted pytest run (`20 passed`) and diagnostics are clean on changed test files.

- Task 17 full-suite collection initially failed when `mental_entropy` was imported before scoring-path setup, so `mes_score` was absent from the already-cached top-level module; resolved by adding `tests/scoring/__init__.py` repo-root path bootstrap and reloading `mental_entropy` in the explicit top-level import test.
- No remaining blockers for Task 17; targeted scoring tests and full suite now pass.
- No blockers in Task 15 test authoring; all required scoring tests passed and evidence file was written.

- Task 18 test collection initially failed with `ModuleNotFoundError: No module named 'scripts'` because pytest `src/` path setup does not include repo root by default; resolved by prepending repo root to `sys.path` in `tests/scoring/test_validation.py` before importing `mental_entropy.scoring`.
- No remaining blockers in Task 18 after path bootstrap; validation suite passes and evidence file was generated.
- No blockers in Task 19 docs update; required scoring import QA passed and evidence saved to `.sisyphus/evidence/task-19-docs.txt`.
- `lsp_diagnostics` cannot validate `.md`/`.txt` files in this environment because no Markdown or plain text LSP server is configured.
