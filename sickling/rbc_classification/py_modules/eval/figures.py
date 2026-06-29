"""Matplotlib figures rendered to SVG with fonts kept as text (not paths).

Setting ``svg.fonttype = 'none'`` keeps font references as ``<text>`` elements
in the SVG so downstream tools (Illustrator, Inkscape) can edit type without
having to re-vectorize. Files are also smaller.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sickling.rbc_classification.py_modules.eval.report import EvaluationReport

# Apply once at import; fonts-as-text is the project default for all figures.
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42  # TrueType in PDFs too — keeps text editable.


def pr_curve_with_band(report: EvaluationReport) -> Figure:
    """Precision-recall curve with bootstrap CI band."""
    band = report.pr_band
    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    ax.fill_between(
        band["recall_grid"], band["precision_low"], band["precision_high"],
        alpha=0.25, color="#3a7ca5", label="95% bootstrap CI",
    )
    ax.plot(
        band["recall_grid"], band["precision_point"],
        color="#1f4e79", linewidth=2.0, label="Point estimate",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    pr_auc_pt, pr_auc_lo, pr_auc_hi = report.metrics_ci.get(
        "pr_auc", (report.metrics.pr_auc, float("nan"), float("nan"))
    )
    ax.set_title(
        f"PR curve · {report.run_name}\n"
        f"PR-AUC = {pr_auc_pt:.3f} [{pr_auc_lo:.3f}, {pr_auc_hi:.3f}]"
    )
    ax.legend(loc="lower left")
    return fig


def confusion_matrix_heatmap(report: EvaluationReport) -> Figure:
    cm = np.asarray(report.metrics.confusion)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums

    labels = ("non_sickle", "sickle")
    fig, ax = plt.subplots(figsize=(5.0, 4.8), constrained_layout=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{int(cm[i, j])}\n({cm_norm[i, j] * 100:.1f}%)",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of true class")
    ax.set_title(
        f"Confusion · {report.run_name}\n"
        f"threshold={report.metrics.threshold:.3f} · MCC={report.metrics.mcc:.3f}"
    )
    return fig


def calibration_plot(report: EvaluationReport, n_bins: int = 10) -> Figure:
    """Reliability diagram. Bin scores into equal-width bins; plot mean
    predicted score vs observed positive rate per bin."""
    y_true = report.y_true.astype(np.int64)
    y_score = report.y_score.astype(np.float64)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_score, bins) - 1, 0, n_bins - 1)

    mean_pred = np.zeros(n_bins)
    obs_rate = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = bin_idx == b
        counts[b] = int(m.sum())
        if counts[b] > 0:
            mean_pred[b] = float(y_score[m].mean())
            obs_rate[b] = float(y_true[m].mean())

    nonempty = counts > 0
    fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Perfect")
    ax.plot(
        mean_pred[nonempty], obs_rate[nonempty],
        marker="o", color="#1f4e79", linewidth=1.8, markersize=6,
        label="Model",
    )
    # Bin support as bar at the bottom.
    width = 1.0 / n_bins
    ax2 = ax.twinx()
    ax2.bar(
        bins[:-1] + width / 2, counts, width=width * 0.85,
        alpha=0.15, color="gray", label="N",
    )
    ax2.set_ylabel("# samples per bin", color="gray")
    ax.set_xlabel("Mean predicted P(sickle) per bin")
    ax.set_ylabel("Observed sickle rate per bin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Calibration · {report.run_name}")
    ax.legend(loc="upper left")
    return fig


def render_all_figures(report: EvaluationReport, output_dir: Path) -> dict[str, Path]:
    """Save all standard evaluation figures to ``output_dir`` as SVG (fonts
    preserved). Returns ``{name: path}``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, builder in (
        ("pr_curve", pr_curve_with_band),
        ("confusion_matrix", confusion_matrix_heatmap),
        ("calibration", calibration_plot),
    ):
        fig = builder(report)
        path = output_dir / f"{name}.svg"
        fig.savefig(path, format="svg")
        plt.close(fig)
        paths[name] = path
    return paths
