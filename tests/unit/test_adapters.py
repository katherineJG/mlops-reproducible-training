import pandas as pd
import pytest

from churn_pipeline.adapters import from_ibm_telco
from churn_pipeline.validate import SCHEMA, DataValidationError, validate


def _ibm(**over: object) -> pd.DataFrame:
    """Invented rows in the IBM layout (not real IBM data)."""
    rows = {
        "customerID": ["A-1", "A-2", "A-3"],
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [1, 34, 0],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Credit card (automatic)",
            "Bank transfer (automatic)",
        ],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "MonthlyCharges": [29.85, 80.5, 20.0],
        "TotalCharges": ["29.85", "2737.0", " "],
        "Churn": ["No", "Yes", "No"],
    }
    return pd.DataFrame(rows | over)


def test_ibm_layout_maps_to_the_schema_columns() -> None:
    out = from_ibm_telco(_ibm())
    # (not validate(): 1 blank in 3 rows would trip the 1% missing-rate cap by design)
    assert list(out.columns) == list(SCHEMA.columns)
    assert "Dependents" not in out.columns
    assert out["churn"].tolist() == [0, 1, 0]
    assert out["total_charges"].isna().tolist() == [False, False, True]
    assert out["internet_service"].tolist() == ["dsl", "fiber", "none"]


def test_unknown_category_is_reported_by_validation_not_hidden() -> None:
    with pytest.raises(DataValidationError, match="contract"):
        validate(from_ibm_telco(_ibm(Contract=["Weekly", "One year", "Two year"])))


def test_garbage_total_charges_is_an_error() -> None:
    with pytest.raises(ValueError, match="non-numeric total_charges"):
        from_ibm_telco(_ibm(TotalCharges=["12.5", "abc", " "]))


def test_wrong_file_is_rejected() -> None:
    with pytest.raises(ValueError, match="not an IBM Telco CSV"):
        from_ibm_telco(pd.DataFrame({"a": [1]}))
