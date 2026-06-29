"""Ablation runner + table renderer (PIPELINE_PLAN §4)."""
from sickling.rbc_classification.py_modules.ablation.render import (
    render_latex_table,
    render_markdown_table,
    write_tables,
)
from sickling.rbc_classification.py_modules.ablation.runner import (
    DEFAULT_ABLATION,
    AblationResult,
    AblationRow,
    aggregate_results,
    load_results,
    run_ablation_table,
)

__all__ = [
    "AblationResult",
    "AblationRow",
    "DEFAULT_ABLATION",
    "aggregate_results",
    "load_results",
    "render_latex_table",
    "render_markdown_table",
    "run_ablation_table",
    "write_tables",
]
