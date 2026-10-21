"""Metrics, acceptance gates and report rendering."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_pipeline.config import GatesConfig

Metrics = dict[str, float | None]


def compute_metrics(y_true: npt.NDArray[Any], proba: npt.NDArray[Any], threshold: float) -> Metrics:
    """P/R/F1 at the decision threshold, plus threshold-free ROC-AUC and PR-AUC.

    AUCs are None when only one class is present (not calculable).
    """
    pred = (proba >= threshold).astype(int)
    both = len(np.unique(y_true)) == 2
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both else None,
        "pr_auc": float(average_precision_score(y_true, proba)) if both else None,
        "positive_rate": float(np.mean(y_true)),
    }


def check_gates(candidate: Metrics, baseline: Metrics, gates: GatesConfig) -> list[str]:
    """Return a human-readable reason for every gate the candidate fails (empty = eligible)."""
    failures: list[str] = []

    def need(name: str, value: float | None, minimum: float) -> None:
        if value is None:
            failures.append(f"{name} not calculable")
        elif value < minimum:
            failures.append(f"{name} {value:.3f} < required {minimum:.3f}")

    need("roc_auc", candidate["roc_auc"], gates.min_roc_auc)
    need("pr_auc", candidate["pr_auc"], gates.min_pr_auc)
    need("recall", candidate["recall"], gates.min_recall)
    base_auc = baseline["roc_auc"]
    if base_auc is not None:
        need(
            "roc_auc lift over baseline",
            None if candidate["roc_auc"] is None else candidate["roc_auc"] - base_auc,
            gates.min_roc_auc_lift_over_baseline,
        )
    return failures


def render_report(
    baseline: Metrics,
    candidate: Metrics,
    failures: list[str],
    threshold: float,
    provenance: dict[str, Any],
) -> str:
    """Human-readable Markdown report."""
    keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    rows = "\n".join(f"| {k} | {fmt(baseline[k])} | {fmt(candidate[k])} |" for k in keys)
    verdict = "ELIGIBLE" if not failures else "INELIGIBLE"
    reasons = "".join(f"\n- {f}" for f in failures)
    return (
        "# Evaluation report\n\n"
        f"Dataset fingerprint: `{provenance['fingerprint_sha256']}` "
        f"({provenance['row_count']} rows, source: {provenance['source']})\n\n"
        f"Test positive (churn) rate: {fmt(candidate['positive_rate'])}. "
        "Classes are imbalanced, so PR-AUC is reported alongside ROC-AUC; the candidate uses "
        "balanced class weights.\n\n"
        f"Decision threshold for precision/recall/F1: {threshold}.\n\n"
        "| metric | baseline | candidate |\n|---|---|---|\n"
        f"{rows}\n\n"
        f"**Candidate verdict: {verdict}**\n{reasons}\n"
    )


def write_outputs(
    out_dir: Path,
    baseline: Metrics,
    candidate: Metrics,
    failures: list[str],
    threshold: float,
    provenance: dict[str, Any],
) -> None:
    """Write machine-readable metrics.json and human-readable report.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": baseline,
        "candidate": candidate,
        "eligible": not failures,
        "gate_failures": failures,
        "decision_threshold": threshold,
    }
    # newline="\n": Windows would otherwise write CRLF and change the bytes DVC hashes.
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8", newline="\n"
    )
    (out_dir / "report.md").write_text(
        render_report(baseline, candidate, failures, threshold, provenance),
        encoding="utf-8",
        newline="\n",
    )
