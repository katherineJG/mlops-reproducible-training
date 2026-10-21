import json
from pathlib import Path

import mlflow
import pytest

from churn_pipeline.config import load_data_config, load_model_config
from churn_pipeline.ingest import fingerprint, generate_synthetic
from churn_pipeline.tracking import log_run
from churn_pipeline.train import run

pytestmark = pytest.mark.integration


def _train_and_log(tmp_path: Path, model_yaml: Path = Path("configs/model.yaml")) -> str:
    data_cfg = load_data_config(Path("configs/data.yaml"))
    model_cfg = load_model_config(model_yaml)
    df = generate_synthetic(2000, 42)
    prov = {
        "fingerprint_sha256": fingerprint(df),
        "row_count": len(df),
        "source": "synthetic",
        "schema_version": "1",
    }
    models, reports = tmp_path / "models", tmp_path / "reports"
    failures = run(df, model_cfg, "churn", "customer_id", prov, models, reports)
    m = json.loads((reports / "metrics.json").read_text())
    return log_run(
        f"sqlite:///{tmp_path / 'mlflow.db'}",
        "test",
        data_cfg,
        model_cfg,
        prov,
        m["baseline"],
        m["candidate"],
        failures,
        models,
        reports,
        [Path("configs/data.yaml"), model_yaml],
    )


def test_run_has_params_metrics_provenance_and_model(tmp_path: Path) -> None:
    run_id = _train_and_log(tmp_path)
    client = mlflow.MlflowClient(f"sqlite:///{tmp_path / 'mlflow.db'}")
    r = client.get_run(run_id)
    assert r.data.tags["dataset_fingerprint"] == fingerprint(generate_synthetic(2000, 42))
    assert r.data.tags["git_sha"] and r.data.tags["eligible"] == "True"
    assert r.data.params["model.seed"] == "42"
    assert r.data.metrics["candidate_roc_auc"] > r.data.metrics["baseline_roc_auc"]
    paths = {a.path for a in client.list_artifacts(run_id)}
    assert {"config", "reports", "environment", "provenance.json"} <= paths
    assert client.search_logged_models(experiment_ids=[r.info.experiment_id])


def test_rejected_candidate_is_logged_without_model(tmp_path: Path) -> None:
    strict = tmp_path / "strict.yaml"
    strict.write_text(
        Path("configs/model.yaml").read_text().replace("min_roc_auc: 0.70", "min_roc_auc: 0.99")
    )
    run_id = _train_and_log(tmp_path, strict)
    client = mlflow.MlflowClient(f"sqlite:///{tmp_path / 'mlflow.db'}")
    r = client.get_run(run_id)
    assert r.data.tags["eligible"] == "False" and "roc_auc" in r.data.tags["gate_failures"]
    assert not client.search_logged_models(experiment_ids=[r.info.experiment_id])
