# Decisions

| # | Decision | Why |
|---|---|---|
| 1 | Default data is a seeded synthetic generator; the IBM Telco CSV is local/private only | The IBM CSV has no confirmed license (IBM's repo licenses code only). Synthetic data is safe to commit and fully deterministic. |
| 2 | Stratified random split, not temporal | The data has no time column; stratification keeps the churn rate equal in both splits. |
| 3 | Decision threshold 0.5, candidate uses `class_weight=balanced` | Simple, documented default; ~28% positive class. Threshold is configurable and untuned. |
| 4 | `params.yaml` removed; DVC reads `configs/*.yaml` directly as params | One source of truth for settings. A separate `params.yaml` would duplicate every value. |
| 5 | MLflow tracking with a local SQLite backend (`mlflow.db`), no model registry | Meets tracking requirements without registry infrastructure. MLflow pinned `>=3.17,<4`. |
| 6 | Model artifact logged to MLflow only when the candidate passes its gates | Rejected runs keep metrics/reports/failure reasons but no deployable model. |
| 7 | `numpy.dtype` is the only type trusted in MLflow's skops model serialization | MLflow 3 refuses untrusted types; trusting one named type is narrower than switching to pickle. |
| 8 | `provenance.json` has no timestamp | Identical inputs must give byte-identical outputs; run time is already in the MLflow run. |
| 9 | Run MLflow as `python -m mlflow` | On this Windows machine an Application Control policy blocks the `mlflow.exe` launcher. |
| 10 | Identical rows (ignoring `customer_id`) are counted and capped at 2%, not rejected one by one | Found when validating the real IBM Telco data: 57 of 7043 rows (0.81%) are legitimately identical on the retained features. Duplicate `customer_id` still fails outright. |
| 11 | `adapters.py` maps the IBM Telco CSV onto our schema; unmapped values are left visible so validation reports them | Private local testing only; the CSV has no confirmed license and is never committed. |
| 12 | When local Windows blocks Python's `_multiprocessing` (Application Control), run checks and `dvc repro` in a `python:3.12-slim` container with the project mounted | Same Python version and locked deps as CI; also yields a Linux-built `dvc.lock` that matches CI's model hash. |
