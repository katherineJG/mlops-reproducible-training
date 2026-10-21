# Architecture

```text
ingest -> validate -> split -> [ColumnTransformer + estimator Pipeline] -> evaluate/gate -> track (MLflow)
   \__ provenance.json (source, fingerprint, rows, schema version)        \__ metrics.json + report.md
```

| Module | Responsibility |
|---|---|
| `ingest.py` | Generate seeded synthetic data or load a local CSV; write provenance and fingerprint |
| `validate.py` | Pandera contract; failures name the violated rule; no silent drops |
| `features.py` | `ColumnTransformer` (impute/encode/scale), fitted only inside the pipeline on train data |
| `train.py` | Stratified split, baseline + candidate fit, seeds from config |
| `evaluate.py` | P/R/F1/ROC-AUC/PR-AUC, threshold gates, JSON + Markdown report |
| `tracking.py` | MLflow params/metrics/artifacts, git SHA, dataset fingerprint, environment |

Stages are wired in `dvc.yaml` (added in M3). Config: `configs/data.yaml`, `configs/model.yaml`, validated at startup (M1/M2).
