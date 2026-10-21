"""Split first, then fit baseline and candidate pipelines, evaluate and gate."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_pipeline.config import ModelConfig, load_data_config, load_model_config
from churn_pipeline.evaluate import check_gates, compute_metrics, write_outputs
from churn_pipeline.features import FEATURES, build_pipeline
from churn_pipeline.tracking import log_run
from churn_pipeline.validate import DataValidationError, validate

log = logging.getLogger(__name__)


def split(df: pd.DataFrame, target: str, cfg: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random split. The data has no time column, so a temporal split does not apply."""
    train, test = train_test_split(
        df, test_size=cfg.test_size, stratify=df[target], random_state=cfg.seed
    )
    return train, test


def make_baseline(cfg: ModelConfig) -> Pipeline:
    return build_pipeline(DummyClassifier(strategy="prior"))


def make_candidate(cfg: ModelConfig) -> Pipeline:
    est: BaseEstimator = LogisticRegression(random_state=cfg.seed, **cfg.candidate.params)
    return build_pipeline(est)


def run(
    df: pd.DataFrame,
    cfg: ModelConfig,
    target: str,
    id_column: str,
    provenance: dict[str, Any],
    models_dir: Path,
    reports_dir: Path,
) -> list[str]:
    """Train and evaluate on validated data; returns gate failures (empty = eligible)."""
    validate(df)
    train, test = split(df, target, cfg)
    metrics = {}
    for name, pipe in {"baseline": make_baseline(cfg), "candidate": make_candidate(cfg)}.items():
        pipe.fit(train[FEATURES], train[target])
        proba = pipe.predict_proba(test[FEATURES])[:, 1]
        metrics[name] = compute_metrics(
            test[target].to_numpy(), np.asarray(proba), cfg.decision_threshold
        )
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, models_dir / f"{name}.joblib")
    failures = check_gates(metrics["candidate"], metrics["baseline"], cfg.gates)
    write_outputs(
        reports_dir,
        metrics["baseline"],
        metrics["candidate"],
        failures,
        cfg.decision_threshold,
        provenance,
    )
    (reports_dir / "split.json").write_text(
        json.dumps({"test_ids": sorted(test[id_column])}), encoding="utf-8", newline="\n"
    )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, evaluate and gate the churn model.")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="churn-training")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    data_cfg = load_data_config(args.data_config)
    model_cfg = load_model_config(args.model_config)
    provenance = json.loads((args.raw / "provenance.json").read_text(encoding="utf-8"))
    try:
        failures = run(
            pd.read_csv(args.raw / "raw.csv"),
            model_cfg,
            data_cfg.target,
            data_cfg.id_column,
            provenance,
            args.models,
            args.reports,
        )
    except DataValidationError as exc:
        log.error("%s", exc)
        sys.exit(1)
    results = json.loads((args.reports / "metrics.json").read_text(encoding="utf-8"))
    run_id = log_run(
        args.tracking_uri,
        args.experiment,
        data_cfg,
        model_cfg,
        provenance,
        results["baseline"],
        results["candidate"],
        failures,
        args.models,
        args.reports,
        [args.data_config, args.model_config],
    )
    log.info("MLflow run %s logged to %s", run_id, args.tracking_uri)
    if failures:
        log.error("candidate INELIGIBLE: %s", "; ".join(failures))
        sys.exit(2)
    log.info("candidate eligible; see %s", args.reports / "report.md")


if __name__ == "__main__":
    main()
