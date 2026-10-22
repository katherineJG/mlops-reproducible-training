# Task Plan — mlops-reproducible-training

Python 3.12 pinned everywhere (pyproject `requires-python`, `.python-version`, Docker base `python:3.12-slim`, CI `setup-python`).

## Data decision (needs user sign-off)
- Default path: **deterministic synthetic churn generator** (`ingest.py generate`), seeded, Telco-like schema. No licence risk, no download, used by CI and tests.
- Optional path: real IBM Telco Churn CSV, **user-supplied/downloaded locally, never committed**. Licence is unconfirmed (see README data notes); personal, non-commercial use only.

## M0 — Setup
Files: pyproject.toml (uv or pip-tools lock), Makefile, ruff/mypy/pytest config, docs/ARCHITECTURE.md, configs/{data,model}.yaml, params.yaml, empty package skeleton, CI stub.
Done when: `make lint typecheck test` run green on the empty skeleton.

## M1 — Ingestion and validation
Files: ingest.py, validate.py (Pandera), tests/unit/test_validate.py, data/sample fixture.
Behavior: generate/load, provenance JSON (source, fingerprint, rows, schema version, method), rule-named errors, counted explicit remediation only.

## M2 — Leakage-safe model
Files: features.py, train.py, evaluate.py + tests.
Behavior: stratified random split (no time column → random is justified; documented), ColumnTransformer inside Pipeline, dummy baseline vs logistic regression / gradient boosting, seeds from config, metrics (P/R/F1/ROC-AUC/PR-AUC), configurable gates, JSON + Markdown report. Test: preprocessing fitted on train only.

## M3 — Tracking and DVC
Files: tracking.py, dvc.yaml, params.yaml. MLflow local file/sqlite backend (pin version), log params/metrics/artifacts/git SHA/fingerprint/env. Verify clean run + unchanged repeat run.

## M4 — CI, Docker, docs
ci.yml (ruff, mypy, pytest unit+integration), CPU Dockerfile, README, MODEL_CARD.md, DECISIONS.md, one rejected-candidate demo, results only from real runs.

## Open items
- Git repo not initialised (folder is not a repo); `git init` needed for SHA logging and PR workflow.
- Remote GitHub MCP failed to connect; CI can only be verified once pushed.
