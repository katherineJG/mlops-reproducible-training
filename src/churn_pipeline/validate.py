"""Data contract: every rule is named so failures say exactly what broke. Nothing is dropped."""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from churn_pipeline.ingest import CONTRACTS, INTERNET, PAYMENT

log = logging.getLogger(__name__)

MAX_TOTAL_CHARGES_MISSING = 0.01  # tolerated fraction of missing total_charges
# Customers can legitimately share every feature (seen in real data: ~0.8% of rows), so identical
# rows ignoring customer_id are counted and capped rather than rejected one by one.
MAX_DUPLICATE_ROW_RATE = 0.02


class DataValidationError(ValueError):
    """Raised when data violates the contract; the message lists each failed rule."""


def _no_dupes(name: str) -> pa.Check:
    return pa.Check(lambda s: ~s.duplicated(keep=False), name=name, error=name)


SCHEMA = pa.DataFrameSchema(
    {
        "customer_id": pa.Column(str, [_no_dupes("customer_id_unique")], nullable=False),
        "gender": pa.Column(str, pa.Check.isin(["female", "male"]), nullable=False),
        "senior_citizen": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "partner": pa.Column(str, pa.Check.isin(["yes", "no"]), nullable=False),
        "tenure_months": pa.Column(int, pa.Check.in_range(0, 120), nullable=False),
        "contract": pa.Column(str, pa.Check.isin(CONTRACTS), nullable=False),
        "internet_service": pa.Column(str, pa.Check.isin(INTERNET), nullable=False),
        "payment_method": pa.Column(str, pa.Check.isin(PAYMENT), nullable=False),
        "paperless_billing": pa.Column(str, pa.Check.isin(["yes", "no"]), nullable=False),
        "monthly_charges": pa.Column(float, pa.Check.in_range(0, 500), nullable=False),
        "total_charges": pa.Column(float, pa.Check.ge(0), nullable=True),
        "churn": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    },
    strict=True,  # unexpected extra columns fail instead of being silently ignored
    coerce=False,
    checks=[
        pa.Check(
            lambda df: ~((df["tenure_months"] == 0) & (df["total_charges"].fillna(0) > 0)),
            name="no_charges_before_first_month",
            error="no_charges_before_first_month",
        ),
    ],
)


_EXPLANATIONS = [
    (r"^isin\((.*)\)$", "value is not one of the allowed values {0}"),
    (r"^in_range\((.*)\)$", "value is outside the allowed range ({0})"),
    (r"^greater_than_or_equal_to\((.*)\)$", "value must be >= {0}"),
    (r"^dtype\((.*)\)$", "column has the wrong data type, expected {0}"),
    (r"^not_nullable$", "missing values are not allowed in this column"),
    (r"^column_in_dataframe$", "required column is missing from the data"),
    (r"^column_in_schema$", "column is not part of the contract (unexpected extra column)"),
    (r"^customer_id_unique$", "customer_id must be unique"),
    (r"^no_charges_before_first_month$", "tenure is 0 months but total charges are above 0"),
]


def _explain(check: str) -> str:
    """Plain-English meaning of a Pandera check name; falls back to the name itself."""
    for pattern, text in _EXPLANATIONS:
        if m := re.match(pattern, check):
            return text.format(*m.groups())
    return check


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Return df unchanged if valid; otherwise raise DataValidationError naming each failed rule."""
    if df.empty:
        raise DataValidationError("rule 'non_empty': dataset has no rows")
    try:
        SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        fails = exc.failure_cases
        lines = [
            f"rule {row.check!s} on column {row.column!s}: "
            f"{row.failure_case!s} (row index {row.index!s}) -> {_explain(str(row.check))}"
            for row in fails.head(20).itertuples()
        ]
        more = len(fails) - len(lines)
        suffix = f"\n... and {more} more failures" if more > 0 else ""
        raise DataValidationError(
            f"{len(fails)} contract violation(s):\n" + "\n".join(lines) + suffix
        ) from exc
    dup_rows = int(df.drop(columns="customer_id").duplicated(keep=False).sum())
    if dup_rows:
        log.warning(
            "%d rows (%.2f%%) are identical apart from customer_id",
            dup_rows,
            100 * dup_rows / len(df),
        )
    if dup_rows / len(df) > MAX_DUPLICATE_ROW_RATE:
        raise DataValidationError(
            f"rule 'duplicate_row_rate': {dup_rows} of {len(df)} rows are identical apart from "
            f"customer_id, above the {MAX_DUPLICATE_ROW_RATE:.0%} cap"
        )
    missing = float(df["total_charges"].isna().mean())
    if missing > MAX_TOTAL_CHARGES_MISSING:
        raise DataValidationError(
            f"rule 'total_charges_missing_rate': {missing:.3f} exceeds {MAX_TOTAL_CHARGES_MISSING}"
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a churn CSV against the contract.")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        validate(pd.read_csv(args.csv))
    except DataValidationError as exc:
        log.error("%s", exc)
        sys.exit(1)
    log.info("validation passed")


if __name__ == "__main__":
    main()
