"""Smoke tests for figure builders + ``EvaluationReport`` round-trip."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use the non-GUI backend so tests don't try to open a window.
matplotlib.use("Agg")

from sickling.rbc_classification.py_modules.eval.bootstrap import bootstrap_pr_curve  # noqa: E402
from sickling.rbc_classification.py_modules.eval.figures import (  # noqa: E402
    calibration_plot,
    confusion_matrix_heatmap,
    pr_curve_with_band,
    render_all_figures,
)
from sickling.rbc_classification.py_modules.eval.metrics import compute_binary_metrics  # noqa: E402
from sickling.rbc_classification.py_modules.eval.report import EvaluationReport, read_report, write_report  # noqa: E402


def _synth_report() -> EvaluationReport:
    rng = np.random.default_rng(0)
    n = 200
    y_true = (rng.random(n) < 0.2).astype(np.int64)
    y_score = y_true + 0.15 * rng.normal(size=n)
    metrics = compute_binary_metrics(y_true, y_score)
    pr_band = bootstrap_pr_curve(y_true, y_score, n_resamples=50, n_thresholds=64, seed=0)
    return EvaluationReport(
        run_name="test_run",
        variant="multimodal",
        fold=0,
        checkpoint="ignored.ckpt",
        n_val=n,
        metrics=metrics,
        metrics_ci={"pr_auc": (metrics.pr_auc, metrics.pr_auc - 0.05, metrics.pr_auc + 0.05)},
        pr_band=pr_band,
        y_true=y_true,
        y_score=y_score,
        timestamp=datetime.now(UTC).isoformat(),
    )


def test_pr_curve_with_band_returns_figure():
    report = _synth_report()
    fig = pr_curve_with_band(report)
    assert fig is not None
    assert fig.axes
    plt.close(fig)


def test_confusion_matrix_returns_figure():
    fig = confusion_matrix_heatmap(_synth_report())
    assert fig.axes
    plt.close(fig)


def test_calibration_plot_returns_figure():
    fig = calibration_plot(_synth_report(), n_bins=8)
    assert fig.axes
    plt.close(fig)


def test_render_all_figures_writes_three_svgs(tmp_path):
    out = render_all_figures(_synth_report(), tmp_path)
    assert set(out.keys()) == {"pr_curve", "confusion_matrix", "calibration"}
    for path in out.values():
        assert path.exists()
        assert path.suffix == ".svg"
        # Sanity: SVG fonttype=none means font name strings appear in the file.
        contents = Path(path).read_text(encoding="utf-8")
        assert "<svg" in contents


def test_report_round_trip(tmp_path):
    report = _synth_report()
    path = tmp_path / "report.json"
    write_report(report, path)
    loaded = read_report(path)
    assert loaded.run_name == report.run_name
    assert loaded.variant == report.variant
    np.testing.assert_array_equal(loaded.y_true, report.y_true)
    np.testing.assert_allclose(loaded.y_score, report.y_score)
    assert loaded.metrics.pr_auc == report.metrics.pr_auc
    assert loaded.metrics_ci["pr_auc"][0] == report.metrics_ci["pr_auc"][0]
