# AGENTS.md

Spec: `Reproducible ML Training Pipeline Project.md`. Plan: `docs/TASK_PLAN.md`.

- Python 3.12 only. Dependencies via `uv` (`uv.lock` committed). Run tools with `uv run`.
- Checks: `make check` (ruff, ruff format, mypy strict, pytest). No make? Run the `uv run` lines in the Makefile.
- After changing anything under `src/` or `configs/`, run `uv run dvc repro` and commit `dvc.lock` (CI fails on a stale lock).
- One milestone at a time; tests with each feature; fix root causes, never weaken tests.
- Hyperparameters and thresholds live in `configs/*.yaml`, not in source.
- Never commit data other than synthetic fixtures. `data/private/` is git-ignored (IBM Telco CSV: unlicensed, local use only).
- Never claim a check passed unless it was run. No pushes, publishing or paid services without approval.
