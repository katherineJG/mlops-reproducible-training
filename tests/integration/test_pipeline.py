import json
from pathlib import Path

import pytest

from churn_pipeline.config import ModelConfig, load_model_config
from churn_pipeline.ingest import fingerprint, generate_synthetic
from churn_pipeline.train import run
from churn_pipeline.validate import DataValidationError

pytestmark = pytest.mark.integration

PROV = {"fingerprint_sha256": "x", "row_count": 2000, "source": "synthetic"}


@pytest.fixture
def cfg() -> ModelConfig:
    return load_model_config(Path("configs/model.yaml"))


def _run(cfg: ModelConfig, out: Path, seed: int = 42) -> list[str]:
    df = generate_synthetic(2000, seed)
    return run(
        df,
        cfg,
        "churn",
        "customer_id",
        PROV | {"fingerprint_sha256": fingerprint(df)},
        out / "models",
        out / "reports",
    )


def test_end_to_end_is_eligible_and_writes_outputs(cfg: ModelConfig, tmp_path: Path) -> None:
    assert _run(cfg, tmp_path) == []
    m = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert m["eligible"] and m["candidate"]["roc_auc"] > m["baseline"]["roc_auc"]
    assert (tmp_path / "reports" / "report.md").read_text().startswith("# Evaluation report")
    assert (tmp_path / "models" / "candidate.joblib").is_file()


def test_repeat_run_is_reproducible(cfg: ModelConfig, tmp_path: Path) -> None:
    _run(cfg, tmp_path / "a")
    _run(cfg, tmp_path / "b")
    a, b = (json.loads((tmp_path / d / "reports" / "metrics.json").read_text()) for d in "ab")
    assert a["candidate"] == pytest.approx(b["candidate"], abs=1e-9)
    assert (tmp_path / "a/reports/split.json").read_text() == (
        tmp_path / "b/reports/split.json"
    ).read_text()


def test_impossible_gate_marks_candidate_ineligible(cfg: ModelConfig, tmp_path: Path) -> None:
    strict = cfg.model_copy(update={"gates": cfg.gates.model_copy(update={"min_roc_auc": 0.999})})
    failures = _run(strict, tmp_path)
    assert failures and "roc_auc" in failures[0]
    assert json.loads((tmp_path / "reports" / "metrics.json").read_text())["eligible"] is False
    report = (tmp_path / "reports" / "report.md").read_text()
    assert "**Candidate verdict: INELIGIBLE**\n\n- roc_auc" in report


def test_invalid_data_stops_before_training(cfg: ModelConfig, tmp_path: Path) -> None:
    df = generate_synthetic(500, 1)
    df.loc[0, "churn"] = 9
    with pytest.raises(DataValidationError):
        run(df, cfg, "churn", "customer_id", PROV, tmp_path / "m", tmp_path / "r")
    assert not (tmp_path / "m").exists()
