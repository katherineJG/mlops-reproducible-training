# Model card — churn classifier (portfolio pipeline)

## Intended use
Demonstrates a reproducible, leakage-safe training and evaluation workflow for tabular churn
classification. It is a learning/portfolio artifact. **It is not intended for real customer decisions.**

## Model
`LogisticRegression` (`class_weight=balanced`, `C=1.0`) behind a `ColumnTransformer`: median-impute and
standardize numeric features; most-frequent-impute and one-hot encode categoricals. Baseline: a
`DummyClassifier` (class prior) behind the same preprocessing. All settings are in `configs/model.yaml`.

## Data
Seeded **synthetic** Telco-like data (`churn_pipeline.ingest.generate_synthetic`, seed 42, 5000 rows). The
churn signal is a hand-written logistic function of tenure, charges, contract, payment method and
paperless billing, so results show that the pipeline works, **not** how a model would perform on real
customers. The IBM Telco churn CSV has no confirmed license and is not committed or distributed; it may
only be used locally and privately.

## Evaluation methodology
- Stratified 80/20 random split (`test_size=0.2`, seed 42). The data has no time column, so a temporal split
  does not apply; for real customer data with dates, a temporal split would be preferable.
- Preprocessing is fitted on training data only (inside the sklearn `Pipeline`; covered by a unit test).
- Decision threshold 0.5 (untuned). Test churn rate is 27.7%, so PR-AUC is reported next to ROC-AUC.
- Gates (`configs/model.yaml`): ROC-AUC ≥ 0.70, PR-AUC ≥ 0.40, recall ≥ 0.50, and ROC-AUC at least 0.05
  above the baseline.

## Results (one run, synthetic data, test split of 1000 rows)

| metric | baseline | candidate |
|---|---|---|
| precision | 0.000 | 0.533 |
| recall | 0.000 | 0.751 |
| f1 | 0.000 | 0.624 |
| roc_auc | 0.500 | 0.831 |
| pr_auc | 0.277 | 0.670 |

The baseline never predicts churn at threshold 0.5, hence precision/recall of 0. Source: `reports/metrics.json`.

## Limitations
- Synthetic data; no real-world validity, no fairness or subgroup analysis, no calibration check.
- Single split, no confidence intervals or cross-validation; differences of a few points are not meaningful.
- Threshold and gate values are illustrative, not derived from a business cost model.
- Numerical results can differ in the last digits across operating systems or library versions.
