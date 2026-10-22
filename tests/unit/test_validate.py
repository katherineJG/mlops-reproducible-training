from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from churn_pipeline.ingest import generate_synthetic
from churn_pipeline.validate import DataValidationError, validate


@pytest.fixture
def df() -> pd.DataFrame:
    return generate_synthetic(300, seed=1)


def test_valid_data_passes_unchanged(df: pd.DataFrame) -> None:
    before = df.copy()
    pd.testing.assert_frame_equal(validate(df), before)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d.drop(columns="partner"), "required column is missing"),
        (lambda d: d.assign(surprise=1), "unexpected extra column"),
        (lambda d: d.assign(tenure_months=d["tenure_months"].astype(float)), "wrong data type"),
        (lambda d: d.assign(gender=["other"] + list(d["gender"][1:])), "not one of the allowed"),
        (lambda d: d.assign(monthly_charges=[-1.0] + list(d["monthly_charges"][1:])), ">= 0|range"),
    ],
)
def test_errors_explain_the_rule_in_plain_english(
    df: pd.DataFrame, mutate: Callable[[pd.DataFrame], pd.DataFrame], expected: str
) -> None:
    with pytest.raises(DataValidationError, match=expected):
        validate(mutate(df))


def test_empty_data_fails() -> None:
    with pytest.raises(DataValidationError, match="non_empty"):
        validate(generate_synthetic(5, seed=1).iloc[0:0])


def test_missing_column_fails(df: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="churn"):
        validate(df.drop(columns="churn"))


def test_unexpected_column_fails(df: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="surprise"):
        validate(df.assign(surprise=1))


def test_invalid_target_value_fails(df: pd.DataFrame) -> None:
    df.loc[0, "churn"] = 2
    with pytest.raises(DataValidationError, match="isin"):
        validate(df)


def test_duplicate_id_fails(df: pd.DataFrame) -> None:
    df.loc[1, "customer_id"] = df.loc[0, "customer_id"]
    with pytest.raises(DataValidationError, match="customer_id_unique"):
        validate(df)


def test_many_duplicate_rows_fail(df: pd.DataFrame) -> None:
    dups = df.iloc[:20].assign(customer_id=[f"D{i}" for i in range(20)])
    with pytest.raises(DataValidationError, match="duplicate_row_rate"):
        validate(pd.concat([df, dups], ignore_index=True))


def test_a_few_duplicate_rows_are_tolerated_and_counted(
    df: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    dup = df.iloc[[0]].assign(customer_id="C999999")
    validate(pd.concat([df, dup], ignore_index=True))
    assert "identical apart from customer_id" in caplog.text


def test_negative_charges_fail(df: pd.DataFrame) -> None:
    df.loc[0, "monthly_charges"] = -5.0
    with pytest.raises(DataValidationError, match="monthly_charges"):
        validate(df)


def test_impossible_value_fails(df: pd.DataFrame) -> None:
    df.loc[0, ["tenure_months", "total_charges"]] = [0, 100.0]
    with pytest.raises(DataValidationError, match="no_charges_before_first_month"):
        validate(df)


def test_excess_missingness_fails(df: pd.DataFrame) -> None:
    df.loc[df.index[:10], "total_charges"] = np.nan
    with pytest.raises(DataValidationError, match="total_charges_missing_rate"):
        validate(df)


def test_missing_in_required_column_fails(df: pd.DataFrame) -> None:
    df["contract"] = df["contract"].astype(object)
    df.loc[0, "contract"] = None
    with pytest.raises(DataValidationError, match="contract"):
        validate(df)
