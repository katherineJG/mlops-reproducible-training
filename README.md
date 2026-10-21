# mlops-reproducible-training

Reproducible, leakage-safe customer-churn training pipeline. Work in progress; see
`Reproducible ML Training Pipeline Project.md` (spec) and `docs/TASK_PLAN.md`.

## Quick start (Python 3.12, [uv](https://docs.astral.sh/uv/))

```bash
uv sync          # creates .venv with locked dependencies
make check       # ruff + mypy + pytest (or run the `uv run ...` lines in the Makefile)
```

## Commands

```bash
uv run python -m churn_pipeline.ingest                       # writes data/raw/raw.csv + provenance.json
uv run python -m churn_pipeline.validate data/raw/raw.csv    # exit 1 + named rules on bad data
uv run python -m churn_pipeline.train                        # fit baseline + candidate, apply gates
```

Or run the whole versioned pipeline with DVC (skips unchanged stages):

```bash
uv run dvc repro
```

## Experiment tracking (MLflow)

Each `train` run logs params, baseline/candidate metrics, git SHA (+ dirty flag), dataset fingerprint,
configs, reports, provenance, a pip freeze, and (only if the candidate passes its gates) the model.
The default backend is local SQLite (`mlflow.db`, git-ignored). To browse runs:

```bash
uv run python -m mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

`train` writes `models/` and `reports/` (DVC-tracked, not in Git). Exit code 2 means the candidate failed an
evaluation gate in `configs/model.yaml`; the report still lists which gate failed.

`data/sample/churn_sample.csv` is a small committed synthetic fixture (200 rows, seed 42) used by tests.

## Data

Default data is a seeded synthetic generator (added in M1). No third-party dataset is committed.
The IBM Telco churn CSV has no confirmed license, so it is only ever used locally and privately
(`data/private/`, git-ignored).
