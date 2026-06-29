"""JSON report — numbers + arrays needed to redraw figures without retraining."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from sickling.rbc_classification.py_modules.eval.metrics import BinaryMetrics


@dataclass
class EvaluationReport:
    run_name: str
    variant: str
    fold: int
    checkpoint: str
    n_val: int
    metrics: BinaryMetrics
    metrics_ci: dict[str, tuple[float, float, float]]   # name -> (point, lo, hi)
    pr_band: dict[str, np.ndarray]                       # bootstrap_pr_curve output
    y_true: np.ndarray
    y_score: np.ndarray
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, BinaryMetrics):
        d = asdict(obj)
        d["confusion"] = obj.confusion.tolist()
        return d
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_report(report: EvaluationReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": report.run_name,
        "variant": report.variant,
        "fold": report.fold,
        "checkpoint": report.checkpoint,
        "n_val": report.n_val,
        "metrics": _to_jsonable(report.metrics),
        "metrics_ci": _to_jsonable(report.metrics_ci),
        "pr_band": _to_jsonable(report.pr_band),
        "y_true": report.y_true.tolist(),
        "y_score": report.y_score.tolist(),
        "timestamp": report.timestamp,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_report(path: str | Path) -> EvaluationReport:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    metrics = BinaryMetrics(
        pr_auc=d["metrics"]["pr_auc"],
        roc_auc=d["metrics"]["roc_auc"],
        mcc=d["metrics"]["mcc"],
        recall_at_p90=d["metrics"]["recall_at_p90"],
        threshold_at_p90=d["metrics"]["threshold_at_p90"],
        f1_sickle=d["metrics"]["f1_sickle"],
        f1_non_sickle=d["metrics"]["f1_non_sickle"],
        threshold=d["metrics"]["threshold"],
        confusion=np.asarray(d["metrics"]["confusion"], dtype=np.int64),
    )
    pr_band = {k: np.asarray(v) for k, v in d["pr_band"].items()}
    metrics_ci = {k: tuple(v) for k, v in d["metrics_ci"].items()}
    return EvaluationReport(
        run_name=d["run_name"],
        variant=d["variant"],
        fold=d["fold"],
        checkpoint=d["checkpoint"],
        n_val=d["n_val"],
        metrics=metrics,
        metrics_ci=metrics_ci,
        pr_band=pr_band,
        y_true=np.asarray(d["y_true"]),
        y_score=np.asarray(d["y_score"]),
        timestamp=d["timestamp"],
    )
