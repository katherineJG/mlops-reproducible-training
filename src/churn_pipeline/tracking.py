"""Log a training run to MLflow: params, metrics, provenance, git SHA, environment, model."""

import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn

from churn_pipeline.config import DataConfig, ModelConfig
from churn_pipeline.evaluate import Metrics


def git_state() -> tuple[str, bool]:
    """(HEAD SHA, has uncommitted changes). ('unknown', False) outside a git checkout."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return sha, dirty


def _flat(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(_flat(key, v) if isinstance(v, dict) else {key: v})
    return out


def environment_freeze() -> str:
    """Installed distributions as `name==version` lines, sorted."""
    return "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()))


def log_run(
    tracking_uri: str,
    experiment: str,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    provenance: dict[str, Any],
    baseline: Metrics,
    candidate: Metrics,
    failures: list[str],
    models_dir: Path,
    reports_dir: Path,
    config_files: list[Path],
) -> str:
    """Record one run and return its MLflow run id. Metrics that are None are not logged."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    sha, dirty = git_state()
    with mlflow.start_run() as run:
        mlflow.log_params(_flat("data", data_cfg.model_dump(mode="json")))
        mlflow.log_params(_flat("model", model_cfg.model_dump(mode="json")))
        for prefix, m in (("baseline", baseline), ("candidate", candidate)):
            mlflow.log_metrics({f"{prefix}_{k}": v for k, v in m.items() if v is not None})
        mlflow.set_tags(
            {
                "git_sha": sha,
                "git_dirty": str(dirty),
                "dataset_fingerprint": provenance["fingerprint_sha256"],
                "dataset_source": provenance["source"],
                "schema_version": provenance["schema_version"],
                "eligible": str(not failures),
                "gate_failures": "; ".join(failures),
                "python": platform.python_version(),
                "mlflow": mlflow.__version__,
            }
        )
        for f in config_files:
            mlflow.log_artifact(str(f), "config")
        for name in ("metrics.json", "report.md", "split.json"):
            mlflow.log_artifact(str(reports_dir / name), "reports")
        mlflow.log_text(environment_freeze(), "environment/requirements-freeze.txt")
        mlflow.log_dict(provenance, "provenance.json")
        # Only eligible candidates get a model artifact; rejected runs stay inspectable via reports.
        if not failures:
            pipe = joblib.load(models_dir / "candidate.joblib")
            # skops (MLflow 3 default) rejects numpy.dtype unless trusted; trust only that type.
            mlflow.sklearn.log_model(pipe, name="model", skops_trusted_types=["numpy.dtype"])
        return str(run.info.run_id)
