# Decisions

| # | Decision | Why |
|---|---|---|
| 1 | Default data is a seeded synthetic generator; the IBM Telco CSV is local/private only | The IBM CSV has no confirmed license (IBM's repo licenses code only). Synthetic data is safe to commit and fully deterministic. |
| 2 | Stratified random split, not temporal | The data has no time column; stratification keeps the churn rate equal in both splits. |
| 3 | Decision threshold 0.5, candidate uses `class_weight=balanced` | Simple, documented default; ~28% positive class. Threshold is configurable and untuned. |
| 4 | `params.yaml` removed; DVC reads `configs/*.yaml` directly as params | One source of truth for settings. A separate `params.yaml` would duplicate every value. Deviation from the spec's suggested tree. |
| 5 | MLflow tracking with a local SQLite backend (`mlflow.db`), no model registry | Meets tracking requirements without registry infrastructure. MLflow pinned `>=3.17,<4`. |
| 6 | Model artifact logged to MLflow only when the candidate passes its gates | Rejected runs keep metrics/reports/failure reasons but no deployable model. |
| 7 | `numpy.dtype` is the only type trusted in MLflow's skops model serialization | MLflow 3 refuses untrusted types; trusting one named type is narrower than switching to pickle. |
| 8 | `provenance.json` has no timestamp | Identical inputs must give byte-identical outputs; run time is already in the MLflow run. |
| 9 | Run MLflow as `python -m mlflow` | On this Windows machine an Application Control policy blocks the `mlflow.exe` launcher. |
