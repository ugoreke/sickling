"""Group-stratified k-fold splitter, metrics, bootstrap CIs, figures."""
from sickling.rbc_classification.py_modules.eval.bootstrap import bootstrap_metric, bootstrap_pr_curve
from sickling.rbc_classification.py_modules.eval.figures import (
    calibration_plot,
    confusion_matrix_heatmap,
    pr_curve_with_band,
    render_all_figures,
)
from sickling.rbc_classification.py_modules.eval.metrics import (
    BinaryMetrics,
    compute_binary_metrics,
    pick_threshold_max_mcc,
    recall_at_precision,
)
from sickling.rbc_classification.py_modules.eval.report import EvaluationReport, read_report, write_report
from sickling.rbc_classification.py_modules.eval.splits import (
    balanced_group_kfold,
    fold_diagnostics,
    group_stratified_kfold,
    make_kfold_splits,
)

__all__ = [
    "BinaryMetrics",
    "EvaluationReport",
    "balanced_group_kfold",
    "bootstrap_metric",
    "bootstrap_pr_curve",
    "calibration_plot",
    "compute_binary_metrics",
    "confusion_matrix_heatmap",
    "fold_diagnostics",
    "group_stratified_kfold",
    "make_kfold_splits",
    "pick_threshold_max_mcc",
    "pr_curve_with_band",
    "read_report",
    "recall_at_precision",
    "render_all_figures",
    "write_report",
]
