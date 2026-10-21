import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from churn_pipeline.config import DataConfig, load_data_config
from churn_pipeline.ingest import fingerprint, generate_synthetic, ingest, load_csv
from churn_pipeline.validate import validate


def _cfg(**over: object) -> DataConfig:
    base: dict[str, object] = {
        "source": "synthetic",
        "seed": 7,
        "n_rows": 100,
        "schema_version": "1",
        "target": "churn",
        "id_column": "customer_id",
    }
    return DataConfig.model_validate(base | over)


def test_generator_is_deterministic() -> None:
    a, b = generate_synthetic(200, 3), generate_synthetic(200, 3)
    pd.testing.assert_frame_equal(a, b)
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(generate_synthetic(200, 4))


def test_fingerprint_is_pinned_across_operating_systems() -> None:
    # Same value on Windows, Linux and in Docker; changes only if the generator or hashing changes.
    pinned = "ac72783d014ada5f7b692124eefb0d22b41a0702451790e2778f93e5d28f3428"
    assert fingerprint(generate_synthetic(50, 1)) == pinned


def test_generated_data_is_valid_and_has_both_classes() -> None:
    df = validate(generate_synthetic(5000, 42))
    assert 0.1 < df["churn"].mean() < 0.6


def test_ingest_writes_provenance(tmp_path: Path) -> None:
    df = ingest(_cfg(), tmp_path)
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["row_count"] == len(df) == 100
    assert prov["fingerprint_sha256"] == fingerprint(df)
    assert prov["schema_version"] == "1"
    assert "seed=7" in prov["method"]
    assert fingerprint(pd.read_csv(tmp_path / "raw.csv")) == prov["fingerprint_sha256"]


def test_csv_source_requires_path() -> None:
    with pytest.raises(ValidationError, match="csv_path"):
        _cfg(source="csv")


def test_unknown_config_key_rejected() -> None:
    with pytest.raises(ValidationError):
        _cfg(typo=1)


def test_missing_csv_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CSV not found"):
        load_csv(tmp_path / "nope.csv")


def test_shipped_config_is_valid() -> None:
    assert load_data_config(Path("configs/data.yaml")).source == "synthetic"
