# Acceptance criteria status

Statuses are from checks actually run on 2026-10-21 (Windows 11 + Python 3.12, Docker Desktop, GitHub Actions on ubuntu-latest).

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | New developer can install and run the pipeline | PASS | Fresh `git clone` into an empty folder, then `uv sync`, `uv run dvc repro`, `train`, ruff, mypy, pytest (35 passed at the time; 40 now). Done on Windows only; Linux covered by CI steps. |
| 2 | Invalid data fails with an actionable error | PASS | `docs/DEMOS.md` (exit 1, rule + column + row); 9 negative validation tests. |
| 3 | Preprocessing fitted only on training data | PASS | `tests/unit/test_model.py::test_preprocessing_is_fitted_on_train_only`. |
| 4 | Baseline and candidate metrics reported | PASS | `reports/metrics.json`, `reports/report.md`, `MODEL_CARD.md`. |
| 5 | Thresholds gate eligibility | PASS | Unit tests for gates; integration test; demo with exit code 2 (`docs/DEMOS.md`). |
| 6 | MLflow has params, metrics, artifacts, provenance | PASS | `tests/integration/test_tracking.py` reads them back; UI/REST query returned the run. Browser view of the UI not checked. |
| 7 | Data/pipeline dependencies versioned | PASS | `dvc.yaml` + `dvc.lock`; clean run, unchanged-repeat skip, forced re-run identical on one machine. No DVC remote (fresh clones recompute outputs). |
| 8 | Unit and integration tests run in CI | PASS | GitHub Actions run green; `make check` runs all 35 tests, then `dvc repro` and a reproducibility check. |
| 9 | Docker build and execution tested | PASS | `docker build` + `docker run` succeeded locally; container metrics, split and `raw.csv` matched the local run. CI also builds and runs the image on every push. |
| 10 | README: architecture, data source/license, metrics, limitations, troubleshooting | PASS with caveat | All sections present. The Mermaid diagram has not been viewed rendered on GitHub. |
| 11 | `MODEL_CARD.md` documents use, limits, methodology | PASS | `MODEL_CARD.md`. |

## GitHub deliverables not done
- Short demo recording or screenshots (not produced).
- CI badge is present, but the repository is private so it only renders for signed-in collaborators.

## Known risks
- All results use synthetic data with an invented signal; they show the pipeline works, not real-world performance.
- Pickled model files are not byte-identical across operating systems (observed in CI).
