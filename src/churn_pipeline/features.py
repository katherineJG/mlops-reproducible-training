"""Preprocessing lives inside the estimator pipeline so it is only ever fitted on training data."""

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["senior_citizen", "tenure_months", "monthly_charges", "total_charges"]
CATEGORICAL = [
    "gender",
    "partner",
    "contract",
    "internet_service",
    "payment_method",
    "paperless_billing",
]
FEATURES = NUMERIC + CATEGORICAL


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Impute/scale numerics, impute/one-hot categoricals, then the estimator."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    prep = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    return Pipeline([("prep", prep), ("model", estimator)])
