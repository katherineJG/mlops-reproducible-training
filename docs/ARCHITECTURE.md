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

## Reproducibility

`dvc.yaml` has two stages: `ingest` (data/raw) and `train` (models, reports). Deps are the source files and
the raw data; params are `configs/data.yaml` and `configs/model.yaml`. `dvc repro` skips unchanged stages.

Deterministic (verified): a forced re-run on one machine gave an identical `dvc.lock`. Across operating
systems (Windows-generated lock vs a Linux CI run), the raw data, provenance and fingerprint, split
(`split.json`), report and `metrics.json` were identical; CI checks this on every push.
Where variation remains: the pickled model files (`models/`) are same-size but not byte-identical across
OS (observed in CI), so do not compare them by hash across machines. Other CPUs, BLAS or library versions
may also change floating-point results in the last digits; use a tolerance (the tests use 1e-9).
MLflow run ids, timestamps and the git SHA differ per run by design.

Stages are wired in `dvc.yaml`. Config: `configs/data.yaml`, `configs/model.yaml`, validated at startup (M1/M2).
