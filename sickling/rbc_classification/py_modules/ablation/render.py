"""Render ablation aggregates into Markdown / LaTeX tables.

Both formats include mean ± std across (seeds × folds). Bootstrap CIs from
each (seed, fold) cell aren't aggregated into a single number here — milestone
6's per-run report.json contains the raw CIs if needed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt_mean_std(mean: float, std: float, ci_lo: float | None = None, ci_hi: float | None = None) -> str:
    base = f"{mean:.3f} ± {std:.3f}"
    if ci_lo is not None and ci_hi is not None:
        base += f"  [{ci_lo:.3f}, {ci_hi:.3f}]"
    return base


def render_markdown_table(agg_df: pd.DataFrame, title: str = "Ablation results") -> str:
    """Render the aggregate DataFrame as a Markdown table.

    Tolerant of empty / partial frames: if the table has zero rows (e.g. every
    ablation run failed) emits a header + ``_(no rows)_`` placeholder so the
    file is still useful as a "we tried" marker.
    """
    lines = [
        f"# {title}",
        "",
        "| Row | PR-AUC | MCC | recall@p=0.9 | F1 (sickle) | F1 (non-sickle) | runs |",
        "|-----|--------|-----|--------------|-------------|------------------|------|",
    ]
    if agg_df.empty:
        lines += ["| _(no rows)_ |  |  |  |  |  |  |"]
        return "\n".join(lines) + "\n"

    for _, r in agg_df.iterrows():
        lines.append(
            "| "
            + " | ".join([
                str(r["row_name"]),
                _fmt_mean_std(r["pr_auc_mean"], r["pr_auc_std"]),
                _fmt_mean_std(r["mcc_mean"], r["mcc_std"]),
                _fmt_mean_std(r["recall_at_p90_mean"], r["recall_at_p90_std"]),
                f"{r['f1_sickle_mean']:.3f}",
                f"{r['f1_non_sickle_mean']:.3f}",
                str(int(r["n_runs"])),
            ])
            + " |"
        )

    if "notes" in agg_df.columns:
        notes = agg_df["notes"].fillna("")
        if (notes != "").any():
            lines += ["", "## Notes", ""]
            for _, r in agg_df.iterrows():
                if r.get("notes"):
                    lines.append(f"- **{r['row_name']}** — {r['notes']}")
    return "\n".join(lines) + "\n"


def render_latex_table(agg_df: pd.DataFrame, caption: str = "Ablation results", label: str = "tab:ablation") -> str:
    """Render the aggregate DataFrame as a booktabs LaTeX table.

    Designed to drop into a paper with ``\\usepackage{booktabs}`` already loaded.
    """
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{lccccc}",
        r"    \toprule",
        r"    Variant & PR-AUC & MCC & R@P=0.9 & F1$_{\mathrm{sickle}}$ & runs \\",
        r"    \midrule",
    ]
    if agg_df.empty:
        lines.append(r"    \multicolumn{6}{l}{\emph{(no rows)}} \\")
    for _, r in agg_df.iterrows():
        lines.append(
            f"    {_latex_escape(r['row_name'])} & "
            f"${r['pr_auc_mean']:.3f} \\pm {r['pr_auc_std']:.3f}$ & "
            f"${r['mcc_mean']:.3f} \\pm {r['mcc_std']:.3f}$ & "
            f"${r['recall_at_p90_mean']:.3f}$ & "
            f"${r['f1_sickle_mean']:.3f}$ & "
            f"{int(r['n_runs'])} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def write_tables(agg_df: pd.DataFrame, output_dir: Path, title: str = "Ablation results") -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md = output_dir / "table_markdown.md"
    tex = output_dir / "table_latex.tex"
    md.write_text(render_markdown_table(agg_df, title=title), encoding="utf-8")
    tex.write_text(render_latex_table(agg_df, caption=title), encoding="utf-8")
    return {"markdown": md, "latex": tex}
