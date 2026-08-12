from __future__ import annotations

import html
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from smith_agent.reporting.plots import plot_dataset_umap, plot_evaluation_summary
from smith_agent.utils import ensure_dir, write_json


def _latest_epoch_csv(manifest_path: str | Path) -> Path:
    manifest_dir = Path(manifest_path).resolve().parent
    epoch_files = sorted(manifest_dir.glob("epoch_*.csv"))
    if not epoch_files:
        raise FileNotFoundError(f"No epoch_*.csv files found next to manifest: {manifest_path}")
    return epoch_files[-1]


def _load_panel_genes(epoch_csv: str | Path, panel_size: int = 64) -> list[str]:
    df = pd.read_csv(epoch_csv)
    return df.iloc[:panel_size, 0].astype(str).tolist()


def _summarize_evaluation(evaluation_csv: str | Path) -> list[dict[str, Any]]:
    df = pd.read_csv(evaluation_csv)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        metric = str(row["metric"])
        if metric == "explained_variance":
            continue
        rows.append(
            {
                "evaluation": str(row["evaluation"]),
                "metric": metric,
                "label": str(row.get("label", "")),
                "value": float(row["value"]),
            }
        )
    return rows


def render_markdown_pdf(markdown_path: str | Path, output_path: str | Path) -> str | None:
    markdown_path = Path(markdown_path)
    output_path = Path(output_path)
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        return None
    cmd = [
        pandoc,
        str(markdown_path.name),
        "-o",
        str(output_path.name),
        "--resource-path",
        str(markdown_path.parent),
    ]
    pdf_engine = shutil.which("xelatex") or shutil.which("pdflatex")
    if pdf_engine:
        cmd.extend(["--pdf-engine", pdf_engine])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(markdown_path.parent))
    except Exception:  # noqa: BLE001
        return None
    return str(output_path)


def build_run_report(
    output_dir: str | Path,
    title: str,
    manifest_path: str | Path,
    panel_size: int = 64,
    evaluation_csv: str | Path | None = None,
    feasibility_summary: dict[str, Any] | None = None,
    train_adata_file: str | Path | None = None,
    test_adata_file: str | Path | None = None,
    train_color: str = "pathology",
    test_color: str = "pathology",
) -> dict[str, str]:
    report_dir = ensure_dir(output_dir)
    epoch_csv = _latest_epoch_csv(manifest_path)
    panel_genes = _load_panel_genes(epoch_csv, panel_size=panel_size)

    figures: dict[str, str] = {}
    if train_adata_file:
        figures["train_umap"] = plot_dataset_umap(
            train_adata_file,
            report_dir / "train_umap.png",
            color=train_color,
            title="Training Dataset UMAP",
        )
    if test_adata_file:
        figures["test_umap"] = plot_dataset_umap(
            test_adata_file,
            report_dir / "test_umap.png",
            color=test_color,
            title="Test Dataset UMAP",
        )
    metrics: list[dict[str, Any]] = []
    if evaluation_csv:
        metrics = _summarize_evaluation(evaluation_csv)
        figures["evaluation_summary"] = plot_evaluation_summary(
            evaluation_csv,
            report_dir / "evaluation_summary.png",
        )

    summary_payload = {
        "title": title,
        "manifest_path": str(manifest_path),
        "epoch_csv": str(epoch_csv),
        "panel_size": panel_size,
        "panel_genes": panel_genes,
        "evaluation_metrics": metrics,
        "feasibility_summary": feasibility_summary or {},
        "figures": figures,
    }
    summary_json = write_json(report_dir / "report_summary.json", summary_payload)

    markdown_lines = [
        f"# {title}",
        "",
        "## Summary",
        f"- SMITH manifest: `{manifest_path}`",
        f"- Ranked panel file: `{epoch_csv}`",
        f"- Panel size: `{panel_size}`",
        "",
        "## Selected Panel Genes",
        ", ".join(panel_genes),
        "",
    ]
    if feasibility_summary:
        markdown_lines.extend(
            [
                "## Feasibility Summary",
                f"- Passing targets: `{feasibility_summary.get('passing_count', 'NA')}` / `{feasibility_summary.get('total_count', 'NA')}`",
                f"- Integration summary: `{feasibility_summary.get('integration_summary_tsv', '')}`",
                "",
            ]
        )
    if metrics:
        markdown_lines.extend(["## Cross-Dataset Evaluation", ""])
        for row in metrics:
            label_suffix = f" ({row['label']})" if row["label"] else ""
            markdown_lines.append(f"- {row['evaluation']} / {row['metric']}{label_suffix}: `{row['value']:.4f}`")
        markdown_lines.append("")
    if figures:
        markdown_lines.extend(["## Figures", ""])
        for label, path in figures.items():
            rel = Path(path).name
            markdown_lines.append(f"### {label.replace('_', ' ').title()}")
            markdown_lines.append(f"![{label}]({rel})")
            markdown_lines.append("")

    markdown_path = report_dir / "report.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    pdf_path = report_dir / "report.pdf"
    rendered_pdf = render_markdown_pdf(markdown_path, pdf_path)

    html_lines = [
        "<html><head><meta charset='utf-8'><title>{}</title>".format(html.escape(title)),
        "<style>body{font-family:Arial,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.5} img{max-width:100%;border:1px solid #ddd} code{background:#f5f5f5;padding:0.15rem 0.3rem}</style>",
        "</head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<h2>Summary</h2>",
        f"<p>SMITH manifest: <code>{html.escape(str(manifest_path))}</code><br>",
        f"Ranked panel file: <code>{html.escape(str(epoch_csv))}</code><br>",
        f"Panel size: <code>{panel_size}</code></p>",
        "<h2>Selected Panel Genes</h2>",
        f"<p>{html.escape(', '.join(panel_genes))}</p>",
    ]
    if feasibility_summary:
        html_lines.extend(
            [
                "<h2>Feasibility Summary</h2>",
                "<p>"
                f"Passing targets: <code>{html.escape(str(feasibility_summary.get('passing_count', 'NA')))}</code> / "
                f"<code>{html.escape(str(feasibility_summary.get('total_count', 'NA')))}</code><br>"
                f"Integration summary: <code>{html.escape(str(feasibility_summary.get('integration_summary_tsv', '')))}</code>"
                "</p>",
            ]
        )
    if metrics:
        html_lines.append("<h2>Cross-Dataset Evaluation</h2><ul>")
        for row in metrics:
            label_suffix = f" ({html.escape(row['label'])})" if row["label"] else ""
            html_lines.append(
                f"<li>{html.escape(row['evaluation'])} / {html.escape(row['metric'])}{label_suffix}: <code>{row['value']:.4f}</code></li>"
            )
        html_lines.append("</ul>")
    if figures:
        html_lines.append("<h2>Figures</h2>")
        for label, path in figures.items():
            html_lines.append(f"<h3>{html.escape(label.replace('_', ' ').title())}</h3>")
            html_lines.append(f"<img src='{html.escape(Path(path).name)}' alt='{html.escape(label)}'>")
    html_lines.append("</body></html>")
    html_path = report_dir / "report.html"
    html_path.write_text("\n".join(html_lines), encoding="utf-8")

    return {
        "report_dir": str(report_dir),
        "report_markdown": str(markdown_path),
        "report_html": str(html_path),
        "report_pdf": rendered_pdf or "",
        "report_summary_json": str(summary_json),
        "evaluation_metrics": metrics,
        "feasibility_summary": feasibility_summary or {},
        "panel_genes": panel_genes,
        **figures,
    }
