# Decisions

- Task 16 scorer tests use real packaged model artifacts (`gam_ce.joblib`, `gam_ne.joblib`, `xgb_se.json`, `xgb_cle.json`) instead of mocked loaders so assertions cover runtime scoring paths and contribution key ordering end-to-end.

- Task 15 loader missing-artifact tests mock `importlib.resources.files` in `mental_entropy.scoring.loader` so `_load_model()` error-path behavior is validated without depending on real `mental_entropy.models` artifacts.
