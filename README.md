# mlops-reproducible-training

Reproducible, leakage-safe customer-churn training pipeline. Work in progress; see
`Reproducible ML Training Pipeline Project.md` (spec) and `docs/TASK_PLAN.md`.

## Quick start (Python 3.12, [uv](https://docs.astral.sh/uv/))

```bash
uv sync          # creates .venv with locked dependencies
make check       # ruff + mypy + pytest (or run the `uv run ...` lines in the Makefile)
```

## Data

Default data is a seeded synthetic generator (added in M1). No third-party dataset is committed.
The IBM Telco churn CSV has no confirmed license, so it is only ever used locally and privately
(`data/private/`, git-ignored).
