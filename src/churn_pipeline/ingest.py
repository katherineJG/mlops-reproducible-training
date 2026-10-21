"""Obtain data (seeded synthetic generator or a local CSV) and record its provenance."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churn_pipeline.config import DataConfig, load_data_config

log = logging.getLogger(__name__)

CONTRACTS = ["month-to-month", "one-year", "two-year"]
INTERNET = ["dsl", "fiber", "none"]
PAYMENT = ["card", "bank-transfer", "electronic-check", "mailed-check"]


def generate_synthetic(n_rows: int, seed: int) -> pd.DataFrame:
    """Deterministic Telco-like churn data. Same (n_rows, seed) always gives the same frame."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n_rows)
    contract = rng.choice(CONTRACTS, n_rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, n_rows, p=[0.35, 0.45, 0.20])
    base = np.select([internet == "dsl", internet == "fiber"], [45.0, 80.0], default=20.0)
    monthly = np.round(base + rng.normal(0, 5, n_rows).clip(-10, 10), 2)
    total = np.round(monthly * tenure + rng.normal(0, 3, n_rows).clip(0, None) * (tenure > 0), 2)
    paperless = rng.choice(["yes", "no"], n_rows, p=[0.6, 0.4])
    payment = rng.choice(PAYMENT, n_rows)
    logit = (
        -0.5
        - 0.045 * tenure
        + 0.02 * (monthly - 60)
        + 1.2 * (contract == "month-to-month")
        - 0.8 * (contract == "two-year")
        + 0.5 * (payment == "electronic-check")
        + 0.3 * (paperless == "yes")
    )
    churn = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n_rows)],
            "gender": rng.choice(["female", "male"], n_rows),
            "senior_citizen": rng.choice([0, 1], n_rows, p=[0.84, 0.16]),
            "partner": rng.choice(["yes", "no"], n_rows),
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "paperless_billing": paperless,
            "monthly_charges": monthly,
            "total_charges": total,
            "churn": churn,
        }
    )


def load_csv(path: Path) -> pd.DataFrame:
    """Read a local CSV as-is; no rows or columns are dropped (validation decides)."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 over the canonical CSV bytes: sensitive to values, order and columns.

    Line endings are forced to LF; pandas otherwise uses the OS default and the hash would differ
    between Windows and Linux.
    """
    return hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def build_provenance(df: pd.DataFrame, cfg: DataConfig) -> dict[str, Any]:
    """Describe how the data was obtained. No timestamp: identical inputs give identical bytes."""
    method = (
        f"synthetic generator seed={cfg.seed} n_rows={cfg.n_rows}"
        if cfg.source == "synthetic"
        else f"local csv {cfg.csv_path}"
    )
    return {
        "source": cfg.source,
        "method": method,
        "fingerprint_sha256": fingerprint(df),
        "row_count": len(df),
        "columns": list(df.columns),
        "schema_version": cfg.schema_version,
    }


def ingest(cfg: DataConfig, out_dir: Path) -> pd.DataFrame:
    """Obtain data, write `raw.csv` and `provenance.json` into out_dir, return the frame."""
    if cfg.source == "synthetic":
        df = generate_synthetic(cfg.n_rows, cfg.seed)
    else:
        assert cfg.csv_path is not None  # guaranteed by DataConfig validation
        df = load_csv(cfg.csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "raw.csv", index=False, lineterminator="\n")
    prov = build_provenance(df, cfg)
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    log.info("ingested %d rows, fingerprint %s", len(df), prov["fingerprint_sha256"][:12])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Obtain data and record provenance.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ingest(load_data_config(args.config), args.out)


if __name__ == "__main__":
    main()
