from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier

from churn_pipeline.config import GatesConfig, ModelConfig, load_model_config
from churn_pipeline.evaluate import check_gates, compute_metrics
from churn_pipeline.features import FEATURES, build_pipeline
from churn_pipeline.ingest import generate_synthetic
from churn_pipeline.train import split


@pytest.fixture
def cfg() -> ModelConfig:
    return load_model_config(Path("configs/model.yaml"))


GATES = GatesConfig(
    min_roc_auc=0.7, min_pr_auc=0.4, min_recall=0.5, min_roc_auc_lift_over_baseline=0.05
)


def test_shipped_config_valid(cfg: ModelConfig) -> None:
    assert cfg.candidate.type == "logistic_regression"


def test_bad_config_rejected(cfg: ModelConfig) -> None:
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(cfg.model_dump() | {"test_size": 1.5})


def test_preprocessing_is_fitted_on_train_only(cfg: ModelConfig) -> None:
    df = generate_synthetic(1000, 5)
    train, test = split(df, "churn", cfg)
    pipe = build_pipeline(DummyClassifier()).fit(train[FEATURES], train["churn"])
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    cols = ["senior_citizen", "tenure_months", "monthly_charges", "total_charges"]
    np.testing.assert_allclose(scaler.mean_, train[cols].mean().to_numpy())
    assert not np.allclose(scaler.mean_, df[cols].mean().to_numpy())
    assert len(test) + len(train) == len(df)


def test_split_is_deterministic_disjoint_and_stratified(cfg: ModelConfig) -> None:
    df = generate_synthetic(1000, 5)
    a_train, a_test = split(df, "churn", cfg)
    b_train, b_test = split(df, "churn", cfg)
    assert list(a_test["customer_id"]) == list(b_test["customer_id"])
    assert set(a_train["customer_id"]).isdisjoint(a_test["customer_id"])
    assert abs(a_train["churn"].mean() - a_test["churn"].mean()) < 0.01


def test_metrics_known_values() -> None:
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.6, 0.4, 0.9]), 0.5)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["roc_auc"] == 0.75


def test_metrics_single_class_not_calculable() -> None:
    m = compute_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), 0.5)
    assert m["roc_auc"] is None and m["pr_auc"] is None


def _m(roc: float | None, pr: float | None = 0.6, rec: float = 0.7) -> dict[str, float | None]:
    return {"roc_auc": roc, "pr_auc": pr, "recall": rec}


def test_gates_pass_and_fail() -> None:
    assert check_gates(_m(0.8), _m(0.5), GATES) == []
    reasons = check_gates(_m(0.72, 0.3, 0.4), _m(0.5), GATES)
    assert any("pr_auc" in r for r in reasons) and any("recall" in r for r in reasons)


def test_gate_fails_on_insufficient_lift_and_uncalculable() -> None:
    assert any("lift" in r for r in check_gates(_m(0.72), _m(0.70), GATES))
    assert any("not calculable" in r for r in check_gates(_m(None, None), _m(0.5), GATES))
