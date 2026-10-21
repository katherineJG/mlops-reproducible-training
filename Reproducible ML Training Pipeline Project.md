# Project 01 — Reproducible ML Training Pipeline

> Repository name: `mlops-reproducible-training`  
> Level: Beginner–intermediate  
> Estimated effort: 1–2 weeks at 10–15 hours/week

## 1. Overview

Build a reproducible machine-learning pipeline that predicts customer churn from tabular data. The main objective is not model novelty; it is reliable data handling, leakage-safe training, traceable experiments, versioned inputs, measurable evaluation, and repeatable execution.

Use a public dataset whose license permits the intended use. Document the source and license. Include a small synthetic fixture or deterministic test dataset so tests do not rely on a live download. Do not commit a large or sensitive dataset.

## 2. Goals and non-goals

### Goals
- Ingest data and validate its schema and quality.
- Produce a leakage-safe training and evaluation workflow.
- Establish a baseline and compare at least one candidate model.
- Track parameters, metrics, artifacts, source revision, dataset fingerprint, and environment metadata.
- Make data/pipeline dependencies reproducible.
- Enforce an evaluation gate before a candidate is marked eligible.
- Run tests and quality checks in GitHub Actions.
- Provide a clean local quick start.

### Non-goals
- Production API serving (Project 02).
- Live drift monitoring (Project 03).
- Kubernetes orchestration (Project 04).
- Automated production deployment or retraining (Project 05).

## 3. Required stack

- Python, pandas, scikit-learn
- MLflow for experiment tracking and model artifact management
- DVC for data/pipeline versioning where appropriate
- Pydantic or Pandera for data contracts
- pytest, Ruff, mypy (or an explicitly justified alternative)
- Docker for reproducible execution
- GitHub Actions for CI
- Makefile or equivalent task runner

Choose a supported Python version and pin it consistently across local development, CI, and Docker. Keep the first implementation CPU-only.

## 4. Functional requirements

### Data ingestion and contracts
- Provide a documented command to obtain or generate the data.
- Validate required columns, data types, target values, missingness, duplicate rows/IDs, numeric ranges, and impossible values.
- Produce useful validation errors that identify the failed rule.
- Do not silently drop rows or columns. Any remediation must be explicit and counted.
- Record dataset source, version or fingerprint, row count, schema version, and generation/download method.

### Training
- Split data before fitting preprocessing transforms.
- Put imputers, encoders, scalers and estimator in a scikit-learn `Pipeline` / `ColumnTransformer`.
- Avoid target leakage and document the split strategy. If time is relevant to the dataset, explain whether a temporal split is more appropriate than a random split.
- Set random seeds where applicable.
- Train a simple baseline and at least one candidate model.
- Keep hyperparameters in version-controlled configuration, not scattered through source files.

### Evaluation
- Select metrics appropriate to the problem. For churn classification, report at least precision, recall, F1, ROC-AUC and PR-AUC when calculable; explain class imbalance and the decision threshold.
- Compare candidate results with the baseline.
- Make minimum acceptance thresholds configurable.
- Fail the pipeline or mark the candidate ineligible when thresholds are not met.
- Save machine-readable metrics and a human-readable report.

### Experiment tracking
- Log parameters, metrics, artifacts, Git commit SHA, dataset fingerprint, configuration and dependency/environment information to MLflow.
- Record enough metadata to associate a model artifact with its training run.
- Document how to run a local MLflow tracking server or use a supported local tracking configuration.
- If using a model registry, pin and document the MLflow version and registry backend requirements.

### Pipeline reproducibility
- Define pipeline stages and dependencies with DVC or a documented equivalent.
- A clean run must not depend on undocumented local files.
- Explain which outputs are deterministic and where numerical variation may remain.
- Keep generated artifacts out of Git unless small and deliberately included as examples.

## 5. Suggested architecture

1. Ingestion: obtain or generate raw data and record provenance.
2. Validation: enforce schema and quality rules.
3. Feature preparation: define transformations inside the estimator pipeline.
4. Training: fit baseline and candidate.
5. Evaluation: compute metrics and apply gates.
6. Tracking: log parameters, metrics and artifacts.
7. CI: run static checks and a lightweight end-to-end test.

## 6. Repository structure

```text
mlops-reproducible-training/
├── .github/workflows/ci.yml
├── configs/
│   ├── data.yaml
│   └── model.yaml
├── data/
│   └── sample/
├── docs/
│   ├── ARCHITECTURE.md
│   └── DECISIONS.md
├── reports/
├── src/churn_pipeline/
│   ├── __init__.py
│   ├── ingest.py
│   ├── validate.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── tracking.py
├── tests/
│   ├── unit/
│   └── integration/
├── AGENTS.md
├── dvc.yaml
├── params.yaml
├── Dockerfile
├── Makefile
├── pyproject.toml
├── README.md
└── MODEL_CARD.md
```

Adapt the structure only when there is a clear reason; keep package names and documented commands consistent.

## 7. Implementation milestones

### M0 — Design and repository setup
- Create `AGENTS.md`, `docs/PROJECT_SPEC.md` (this file may serve that purpose), `docs/ARCHITECTURE.md`, and a task plan.
- Set up package metadata, dependency lock, formatting/linting, type checks, tests and Makefile.
- Define configuration schema and documented CLI commands.

### M1 — Data ingestion and validation
- Implement download/generation and provenance metadata.
- Add schema and quality validation.
- Test valid data and representative invalid cases.

### M2 — Leakage-safe model pipeline
- Implement split strategy, preprocessing pipeline, baseline and candidate.
- Test that preprocessing is fitted only on training data.
- Save fitted model and evaluation artifacts.

### M3 — Tracking and pipeline reproducibility
- Integrate MLflow.
- Add DVC stages and dependencies, or document a justified alternative.
- Verify a clean pipeline run and a repeat run with unchanged inputs.

### M4 — CI, container and documentation
- Build a CPU-friendly Docker image.
- Configure pull-request checks.
- Verify README instructions from a clean environment.
- Produce sample results only from actual runs.

## 8. Testing strategy

- Unit tests for schema rules, preprocessing, metrics, configuration and threshold logic.
- Regression tests for stable input/output contracts.
- Integration test that runs the pipeline on a small deterministic fixture.
- Negative tests for missing columns, invalid target values, empty data and failed evaluation gates.
- Reproducibility test comparing dataset fingerprint, split assignment and metrics within a documented tolerance.
- CI test path must not require a paid service or a live external dataset.

## 9. Acceptance criteria

- [ ] A new developer can install dependencies and run the documented pipeline.
- [ ] Invalid data fails explicitly with an actionable error.
- [ ] Preprocessing is fitted only on training data.
- [ ] Baseline and candidate metrics are reported.
- [ ] Configured evaluation thresholds gate candidate eligibility.
- [ ] MLflow contains run parameters, metrics, artifacts and provenance.
- [ ] Data/pipeline dependencies are versioned or otherwise reproducibly declared.
- [ ] Unit and integration tests run in CI.
- [ ] Docker build and execution instructions are tested.
- [ ] README includes architecture, data source/license, metrics, limitations and troubleshooting.
- [ ] `MODEL_CARD.md` documents intended use, limitations and evaluation methodology.

## 10. GitHub deliverables

- Architecture diagram
- Clean quick-start instructions
- Example evaluation report and actual metrics
- CI badge (only if CI is configured and passing)
- Model card and data provenance notes
- One documented failed-validation or rejected-candidate demonstration
- Short demo recording or screenshots


## Engineering standards

These standards apply throughout this project.

- Prefer small, cohesive modules and explicit interfaces over a monolithic script.
- Keep configuration outside application logic. Validate configuration at startup.
- Pin dependencies with a lockfile or equivalent reproducible dependency mechanism.
- Use type hints, clear docstrings for public interfaces, structured logging, and actionable errors.
- Never commit credentials, tokens, private datasets, or sensitive production payloads.
- Never claim a test, deployment, benchmark, or security check passed unless it was actually run.
- Do not introduce infrastructure or dependencies that are not justified by a stated requirement.
- Keep the default path runnable locally with free or low-cost tools. Cloud deployment is optional unless explicitly listed as a requirement.
- Make random seeds and dataset versions explicit where reproducibility is expected.
- Add tests with each feature rather than postponing all tests until the end.
- Update documentation whenever commands, configuration, interfaces, or architecture change.
- Use a pull-request workflow: small changes, reviewed diffs, automated checks, and meaningful commit messages.

## Claude Code operating instructions

1. Read this file fully before making changes. Inspect the repository and existing files first.
2. If the repository is empty, propose a brief implementation plan before writing code. If it is not empty, identify existing conventions and avoid overwriting user work.
3. Implement one milestone at a time. Before each milestone, state the intended files and expected behavior.
4. For each milestone, write or update tests, run relevant checks, inspect failures, and fix root causes rather than weakening tests.
5. Do not silently change the agreed architecture, acceptance criteria, API contracts, or dependency choices. Explain trade-offs and ask before material scope changes.
6. Never fabricate test output, metrics, screenshots, model results, or deployment status.
7. Do not run destructive commands, publish packages/images, deploy to external infrastructure, incur cloud costs, or use credentials without explicit authorization.
8. Keep secrets out of logs and test fixtures. Use synthetic/sample data and safe local defaults.
9. When complete, report: files changed, commands actually executed, test results, unmet criteria, known risks, and the next recommended task.
10. Treat acceptance criteria as the definition of done. Mark each item `PASS`, `FAIL`, or `NOT VERIFIED` with evidence.

## Definition of done

The project is complete only when:
- A new contributor can follow the README from a clean environment.
- The documented happy path works end to end.
- Automated tests and quality checks pass in the available environment, or blockers are explicitly documented.
- At least one meaningful failure scenario is demonstrated.
- Architecture, configuration, security assumptions, limitations, and operational recovery are documented.
- CI runs the agreed checks on pull requests.
- No secrets or private data are committed.
- All acceptance criteria have evidence-backed statuses.
