"""Map the public IBM Telco churn CSV onto this project's schema (for private, local use)."""

import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RENAME = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "tenure": "tenure_months",
    "Contract": "contract",
    "InternetService": "internet_service",
    "PaymentMethod": "payment_method",
    "PaperlessBilling": "paperless_billing",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}
# Unmapped values are left as-is so validation reports them instead of hiding them as NaN.
VALUES = {
    "gender": {"Female": "female", "Male": "male"},
    "partner": {"Yes": "yes", "No": "no"},
    "paperless_billing": {"Yes": "yes", "No": "no"},
    "contract": {
        "Month-to-month": "month-to-month",
        "One year": "one-year",
        "Two year": "two-year",
    },
    "internet_service": {"DSL": "dsl", "Fiber optic": "fiber", "No": "none"},
    "payment_method": {
        "Electronic check": "electronic-check",
        "Mailed check": "mailed-check",
        "Bank transfer (automatic)": "bank-transfer",
        "Credit card (automatic)": "card",
    },
    "churn": {"Yes": 1, "No": 0},
}


def from_ibm_telco(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename, recode and select the columns the pipeline uses; log what is dropped or converted."""
    missing = sorted(set(RENAME) - set(raw.columns))
    if missing:
        raise ValueError(f"not an IBM Telco CSV, missing columns: {missing}")
    dropped = sorted(set(raw.columns) - set(RENAME))
    log.info("columns not used by this pipeline (dropped explicitly): %s", dropped)
    df = raw[list(RENAME)].rename(columns=RENAME)
    # Blank TotalCharges (new customers) become NaN; any other non-numeric text is an error.
    total = df["total_charges"].astype("string").str.strip()
    blank = total == ""
    bad = total[~blank & pd.to_numeric(total, errors="coerce").isna()]
    if len(bad):
        raise ValueError(f"non-numeric total_charges values: {sorted(set(bad))[:5]}")
    df["total_charges"] = pd.to_numeric(total.mask(blank), errors="coerce")
    log.info("blank total_charges converted to missing: %d", int(blank.sum()))
    for col, mapping in VALUES.items():
        df[col] = df[col].replace(mapping)
    if df["churn"].isin([0, 1]).all():  # otherwise leave as-is so validation shows the bad value
        df["churn"] = df["churn"].astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an IBM Telco churn CSV to this schema.")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from_ibm_telco(pd.read_csv(args.src)).to_csv(args.dst, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
