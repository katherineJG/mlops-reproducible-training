# Failure demonstrations

Both outputs below are copied from real runs on the synthetic dataset (seed 42, 5000 rows).

## 1. Failed validation

A copy of `data/raw/raw.csv` was edited: row 3 `monthly_charges = -20`, row 7 `churn = 2`, and row 9 given row 8's `customer_id`.

```text
$ python -m churn_pipeline.validate bad.csv
ERROR __main__: 4 contract violation(s):
rule customer_id_unique on column customer_id: C000008 (row index 8) -> customer_id must be unique
rule customer_id_unique on column customer_id: C000008 (row index 9) -> customer_id must be unique
rule in_range(0, 500) on column monthly_charges: -20.0 (row index 3) -> value is outside the allowed range (0, 500)
rule isin([0, 1]) on column churn: 2 (row index 7) -> value is not one of the allowed values [0, 1]
exit code 1
```

With the `partner` column removed as well, the first line becomes
`rule column_in_dataframe on column None: partner (row index None) -> required column is missing from the data`. Training never starts on invalid data
(`test_invalid_data_stops_before_training`).

## 2. Rejected candidate

`min_roc_auc` raised from 0.70 to 0.90 in a temporary copy of `configs/model.yaml`:

```text
$ python -m churn_pipeline.train --model-config strict.yaml
INFO  MLflow run 464c3c4992c14a1db80e58daeebe6032 logged
ERROR candidate INELIGIBLE: roc_auc 0.831 < required 0.900
exit code 2
```

The run is still logged to MLflow with `eligible=False` and the failure reason, but no model artifact is
stored. `reports/report.md` ends with `**Candidate verdict: INELIGIBLE**` and the failed gate.
