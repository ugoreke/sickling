"""Tests for the ablation aggregation + renderers (no live training)."""
from __future__ import annotations

from sickling.rbc_classification.py_modules.ablation import (
    AblationResult,
    aggregate_results,
    render_latex_table,
    render_markdown_table,
)


def _synth_results() -> list[AblationResult]:
    rows: list[AblationResult] = []
    for row_name, base in [("Full multimodal", 0.75), ("- morphology", 0.70), ("- image", 0.55)]:
        for seed in (42, 43):
            for fold in (0, 1):
                rows.append(AblationResult(
                    row_name=row_name,
                    variant="multimodal",
                    seed=seed,
                    fold=fold,
                    pr_auc=base + 0.01 * fold,
                    pr_auc_ci=(base - 0.05, base + 0.05),
                    mcc=base - 0.2,
                    mcc_ci=(base - 0.25, base - 0.15),
                    recall_at_p90=base - 0.3,
                    f1_sickle=base - 0.1,
                    f1_non_sickle=base + 0.05,
                    threshold=0.5,
                    n_val=400,
                    checkpoint="ck",
                    eval_dir="eval",
                    duration_seconds=12.0,
                    notes="t",
                ))
    return rows


def test_aggregate_results_groups_by_row_name():
    df = aggregate_results(_synth_results())
    assert set(df["row_name"]) == {"Full multimodal", "- morphology", "- image"}
    full = df[df["row_name"] == "Full multimodal"].iloc[0]
    assert full["n_runs"] == 4
    assert 0.74 < full["pr_auc_mean"] < 0.77
    # std across 4 cells (each pair of folds differs by 0.01, two seeds).
    assert full["pr_auc_std"] >= 0.0


def test_render_markdown_table_contains_every_row():
    df = aggregate_results(_synth_results())
    md = render_markdown_table(df, title="Test")
    assert "# Test" in md
    for name in df["row_name"]:
        assert name in md
    # Header columns.
    for col in ("PR-AUC", "MCC", "recall@p=0.9", "F1 (sickle)", "runs"):
        assert col in md


def test_render_latex_table_is_well_formed():
    df = aggregate_results(_synth_results())
    tex = render_latex_table(df, caption="x", label="tab:y")
    assert r"\begin{table}" in tex
    assert r"\toprule" in tex
    assert r"\bottomrule" in tex
    assert r"\end{table}" in tex
    # Underscores in row names are escaped.
    for name in df["row_name"]:
        if "_" in name:
            assert r"\_" in tex


def test_aggregate_handles_single_run_no_std_nans():
    rows = [AblationResult(
        row_name="solo",
        variant="multimodal", seed=42, fold=0,
        pr_auc=0.5, pr_auc_ci=(0.4, 0.6), mcc=0.1, mcc_ci=(0.0, 0.2),
        recall_at_p90=0.3, f1_sickle=0.4, f1_non_sickle=0.5, threshold=0.5,
        n_val=100, checkpoint="c", eval_dir="e", duration_seconds=1.0,
    )]
    df = aggregate_results(rows)
    assert len(df) == 1
    assert df.iloc[0]["pr_auc_std"] == 0.0
