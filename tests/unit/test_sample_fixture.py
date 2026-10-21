from pathlib import Path

import pandas as pd

from churn_pipeline.ingest import fingerprint, generate_synthetic, load_csv
from churn_pipeline.validate import validate


def test_committed_sample_is_valid_and_matches_generator() -> None:
    sample = load_csv(Path("data/sample/churn_sample.csv"))
    validate(sample)
    # Regenerate: generate_synthetic(200, 42).to_csv(<sample path>, index=False)
    assert fingerprint(sample) == fingerprint(generate_synthetic(200, 42))
    assert isinstance(sample, pd.DataFrame)
