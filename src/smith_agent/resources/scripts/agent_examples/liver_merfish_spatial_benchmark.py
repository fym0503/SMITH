from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smith_agent.panel_rank_aggregation import aggregate_reference_panel_ranks
from smith_agent.reporting.builder import render_markdown_pdf
from smith_agent.utils import ensure_dir


def _repo_root() -> Path:
    return REPO_ROOT


def _clean_gene_symbol(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text.upper()


def _default_source_rank_file() -> Path:
    return _repo_root() / "outputs/liver_merfish_benchmark/liver_source_gene_rank.tsv"


def _default_reference_rank_files() -> list[Path]:
    base = _repo_root() / "outputs/liver_merfish_benchmark/visium5_aggregation"
    return [
        base / "reference_1_PSC011_C1_gene_rank.tsv",
        base / "reference_2_C73_B1_gene_rank.tsv",
        base / "reference_3_H35_sample1_gene_rank.tsv",
        base / "reference_4_WSSS_F_IMMsp9838712_gene_rank.tsv",
        base / "reference_5_H35_sample2_gene_rank.tsv",
    ]


def _default_merfish_adata() -> Path:
    return _repo_root() / "data/liver_merfish/adata_healthy_merfish.h5ad"


def _load_panel_genes(path: str | Path, panel_size: int | None = None) -> list[str]:
    panel_path = Path(path)
    suffix = panel_path.suffix.lower()
    if suffix in {".csv", ".tsv", ".tab"}:
        sep = "\t" if suffix in {".tsv", ".tab"} else ","
        df = pd.read_csv(panel_path, sep=sep)
        gene_column = next((col for col in df.columns if str(col).lower() in {"gene_symbol", "gene", "target"}), df.columns[0])
        genes = df[gene_column].astype(str).tolist()
    else:
        genes = [line.strip().split(",")[0] for line in panel_path.read_text().splitlines() if line.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        cleaned = _clean_gene_symbol(gene)
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    if panel_size is not None:
        out = out[:panel_size]
    return out


def _gene_index(adata: ad.AnnData) -> pd.Index:
    for column in ("feature_name", "gene_symbol", "gene_symbols", "gene_name", "gene_short_name", "symbol"):
        if column in adata.var.columns:
            return pd.Index(adata.var[column].astype(str).map(_clean_gene_symbol))
    return pd.Index(adata.var_names.astype(str).map(_clean_gene_symbol))


def _panel_positions(var_names: pd.Index, genes: Iterable[str]) -> tuple[list[str], list[int], list[str]]:
    genes = [str(g).upper() for g in genes]
    positions = var_names.str.upper().get_indexer(genes)
    shared: list[str] = []
    shared_positions: list[int] = []
    missing: list[str] = []
    for gene, pos in zip(genes, positions):
        if pos >= 0:
            shared.append(gene)
            shared_positions.append(int(pos))
        else:
            missing.append(gene)
    return shared, shared_positions, missing


def _spatial_coordinates(adata: ad.AnnData) -> np.ndarray:
    if "X_spatial" in adata.obsm:
        return np.asarray(adata.obsm["X_spatial"], dtype=np.float32)
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if {"x", "y"}.issubset(adata.obs.columns):
        return adata.obs[["x", "y"]].astype(float).to_numpy(dtype=np.float32)
    raise KeyError("MERFISH dataset missing spatial coordinates.")


def evaluate_space_ridge(
    adata: ad.AnnData,
    genes: list[str],
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, float | int | list[str]]:
    index = _gene_index(adata)
    shared, positions, missing = _panel_positions(index, genes)
    if len(shared) < 2:
        raise ValueError("Need at least 2 shared genes for coordinate regression")

    x = adata.X[:, positions]
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    coords = _spatial_coordinates(adata)

    train_idx, test_idx = train_test_split(np.arange(x.shape[0]), test_size=test_size, random_state=seed)
    model = Ridge(alpha=1.0, random_state=seed)
    model.fit(x[train_idx], coords[train_idx])
    pred = model.predict(x[test_idx])

    residual = coords[test_idx] - pred
    euclidean = np.sqrt(np.sum(np.square(residual), axis=1))
    truth_flat = coords[test_idx].reshape(-1)
    pred_flat = pred.reshape(-1)
    corr = float(np.corrcoef(truth_flat, pred_flat)[0, 1]) if np.std(truth_flat) > 0 and np.std(pred_flat) > 0 else float("nan")
    return {
        "n_shared_genes": len(shared),
        "shared_genes": shared,
        "missing_genes": missing,
        "train_cells": int(train_idx.size),
        "test_cells": int(test_idx.size),
        "spatial_mse": float(np.mean(np.square(residual))),
        "spatial_mae": float(np.mean(np.abs(residual))),
        "spatial_distance_mae": float(np.mean(euclidean)),
        "spatial_distance_median": float(np.median(euclidean)),
        "spatial_pearson": corr,
    }


def evaluate_cell_type_logistic(
    adata: ad.AnnData,
    genes: list[str],
    *,
    label_column: str = "Cell_Type",
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, float | int | list[str] | dict[str, Any]]:
    if label_column not in adata.obs.columns:
        raise KeyError(f"MERFISH dataset missing `{label_column}` cell-type labels.")
    index = _gene_index(adata)
    shared, positions, missing = _panel_positions(index, genes)
    if len(shared) < 2:
        raise ValueError("Need at least 2 shared genes for cell-type classification")

    x = adata.X[:, positions]
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    labels = adata.obs[label_column].astype(str).to_numpy()
    valid = pd.notna(labels) & (labels != "") & (labels != "nan")
    x = x[valid]
    labels = labels[valid]
    if len(np.unique(labels)) < 2:
        raise ValueError(f"Need at least 2 classes in `{label_column}` for cell-type classification")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    indices = np.arange(x.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=y)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            n_jobs=1,
            random_state=seed,
            solver="lbfgs",
        ),
    )
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    report = classification_report(
        y[test_idx],
        pred,
        target_names=[str(item) for item in encoder.classes_],
        output_dict=True,
        zero_division=0,
    )
    return {
        "n_shared_genes": len(shared),
        "shared_genes": shared,
        "missing_genes": missing,
        "train_cells": int(train_idx.size),
        "test_cells": int(test_idx.size),
        "label_column": label_column,
        "n_classes": int(len(encoder.classes_)),
        "classes": [str(item) for item in encoder.classes_],
        "cell_type_accuracy": float(accuracy_score(y[test_idx], pred)),
        "cell_type_balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
        "cell_type_macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
        "cell_type_weighted_f1": float(f1_score(y[test_idx], pred, average="weighted")),
        "classification_report": report,
    }


def _clean_reference_id(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"^reference_\d+_", "", stem)
    stem = re.sub(r"_(gene_rank|visium_gene_rank|source_gene_rank)$", "", stem)
    return stem


def _plot_comparison(summary_df: pd.DataFrame, out_path: Path) -> str:
    metrics = [
        "spatial_mse",
        "spatial_distance_mae",
        "spatial_pearson",
        "cell_type_accuracy",
        "cell_type_balanced_accuracy",
        "cell_type_macro_f1",
    ]
    labels = {
        "spatial_mse": "MSE",
        "spatial_distance_mae": "Coord MAE",
        "spatial_pearson": "Pearson",
        "cell_type_accuracy": "Cell Type Acc.",
        "cell_type_balanced_accuracy": "Balanced Acc.",
        "cell_type_macro_f1": "Macro F1",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8))
    axes = np.asarray(axes).reshape(-1)
    palette = {"source": "#1f6f78", "integrated": "#d98e04"}
    for ax, metric in zip(axes, metrics):
        subset = summary_df[summary_df["metric"] == metric].copy()
        order = ["source", "integrated"]
        values = [float(subset.loc[subset["panel"] == panel, "value"].iloc[0]) for panel in order]
        bars = ax.bar(order, values, color=[palette[p] for p in order], alpha=0.9)
        ax.set_title(labels[metric])
        ax.set_xlabel("panel")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = df.astype(str).to_dict(orient="records")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row[col] for col in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _write_report(
    output_dir: Path,
    title: str,
    comparison: dict[str, Any],
    source_metrics: dict[str, Any],
    integrated_metrics: dict[str, Any],
    source_cell_type_metrics: dict[str, Any],
    integrated_cell_type_metrics: dict[str, Any],
    summary_df: pd.DataFrame,
    figure_path: str,
) -> dict[str, str]:
    report_md = output_dir / "benchmark_report.md"
    report_pdf = output_dir / "benchmark_report.pdf"
    lines = [
        f"# {title}",
        "",
        "## Setup",
        f"- MERFISH target: `{comparison['merfish_adata']}`",
        f"- Source rank file: `{comparison['source_rank_file']}`",
        f"- Integrated rank file: `{comparison['integrated_rank_tsv']}`",
        f"- Restricted gene universe: `{comparison['n_restricted_gene_symbols']}` genes",
        "",
        "## Panels",
        f"- Source panel: {', '.join(comparison['source_top_panel'])}",
        f"- Integrated panel: {', '.join(comparison['integrated_top_panel'])}",
        "",
        "## Metrics",
        "",
        _markdown_table(summary_df),
        "",
        "## Interpretation",
        f"- Source spatial Pearson: `{source_metrics['spatial_pearson']:.4f}`",
        f"- Integrated spatial Pearson: `{integrated_metrics['spatial_pearson']:.4f}`",
        f"- Source coordinate MAE: `{source_metrics['spatial_distance_mae']:.4f}`",
        f"- Integrated coordinate MAE: `{integrated_metrics['spatial_distance_mae']:.4f}`",
        f"- Source cell-type balanced accuracy: `{source_cell_type_metrics['cell_type_balanced_accuracy']:.4f}`",
        f"- Integrated cell-type balanced accuracy: `{integrated_cell_type_metrics['cell_type_balanced_accuracy']:.4f}`",
        f"- Source cell-type macro F1: `{source_cell_type_metrics['cell_type_macro_f1']:.4f}`",
        f"- Integrated cell-type macro F1: `{integrated_cell_type_metrics['cell_type_macro_f1']:.4f}`",
        "",
        f"![comparison]({Path(figure_path).name})",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    rendered_pdf = render_markdown_pdf(report_md, report_pdf)
    return {
        "report_markdown": str(report_md),
        "report_pdf": rendered_pdf or "",
    }


def _write_classification_report(metrics: dict[str, Any], output_path: Path) -> str:
    pd.DataFrame(metrics["classification_report"]).transpose().to_csv(output_path, sep="\t")
    return str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rank-file", default=str(_default_source_rank_file()))
    parser.add_argument(
        "--reference-rank-files",
        nargs="*",
        default=[str(path) for path in _default_reference_rank_files()],
    )
    parser.add_argument("--merfish-adata", default=str(_default_merfish_adata()))
    parser.add_argument("--output-dir", default=str(_repo_root() / "outputs/liver_merfish_benchmark/merfish_target_benchmark"))
    parser.add_argument("--panel-size", type=int, default=64)
    parser.add_argument("--source-weight", type=float, default=0.55)
    parser.add_argument("--reference-weight", type=float, default=0.45)
    parser.add_argument("--min-reference-support", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cell-type-label", default="Cell_Type")
    parser.add_argument("--title", default="Liver MERFISH Source-vs-Integrated Panel Benchmark")
    args = parser.parse_args()

    merfish_adata = ad.read_h5ad(args.merfish_adata, backed="r")
    merfish_genes = [_clean_gene_symbol(gene) for gene in merfish_adata.var_names.astype(str).tolist()]
    merfish_genes = [gene for gene in merfish_genes if gene]
    merfish_adata.file.close()

    reference_paths = [Path(path) for path in args.reference_rank_files]
    reference_ids = [_clean_reference_id(path) for path in reference_paths]
    output_dir = ensure_dir(args.output_dir)
    aggregation_dir = output_dir / "rank_aggregation_restricted_merfish"
    aggregation = aggregate_reference_panel_ranks(
        output_dir=aggregation_dir,
        source_rank_file=args.source_rank_file,
        reference_rank_files=[str(path) for path in reference_paths],
        reference_ids=reference_ids,
        panel_size=int(args.panel_size),
        source_weight=float(args.source_weight),
        reference_weight=float(args.reference_weight),
        min_reference_support=int(args.min_reference_support),
        gene_universe="source",
        restrict_gene_symbols=merfish_genes,
    )

    source_panel_tsv = aggregation["source_top_panel_tsv"]
    integrated_panel_tsv = aggregation["integrated_top_panel_tsv"]

    source_metrics = evaluate_space_ridge(
        ad.read_h5ad(args.merfish_adata),
        _load_panel_genes(source_panel_tsv, panel_size=args.panel_size),
        test_size=float(args.test_size),
        seed=int(args.seed),
    )
    integrated_metrics = evaluate_space_ridge(
        ad.read_h5ad(args.merfish_adata),
        _load_panel_genes(integrated_panel_tsv, panel_size=args.panel_size),
        test_size=float(args.test_size),
        seed=int(args.seed),
    )
    source_cell_type_metrics = evaluate_cell_type_logistic(
        ad.read_h5ad(args.merfish_adata),
        _load_panel_genes(source_panel_tsv, panel_size=args.panel_size),
        label_column=str(args.cell_type_label),
        test_size=float(args.test_size),
        seed=int(args.seed),
    )
    integrated_cell_type_metrics = evaluate_cell_type_logistic(
        ad.read_h5ad(args.merfish_adata),
        _load_panel_genes(integrated_panel_tsv, panel_size=args.panel_size),
        label_column=str(args.cell_type_label),
        test_size=float(args.test_size),
        seed=int(args.seed),
    )

    summary_df = pd.DataFrame(
        [
            {"panel": "source", "metric": key, "value": value}
            for key, value in source_metrics.items()
            if key not in {"shared_genes", "missing_genes"}
        ]
        + [
            {"panel": "integrated", "metric": key, "value": value}
            for key, value in integrated_metrics.items()
            if key not in {"shared_genes", "missing_genes"}
        ]
        + [
            {"panel": "source", "metric": key, "value": value}
            for key, value in source_cell_type_metrics.items()
            if key
            not in {
                "n_shared_genes",
                "shared_genes",
                "missing_genes",
                "train_cells",
                "test_cells",
                "classes",
                "classification_report",
                "label_column",
            }
        ]
        + [
            {"panel": "integrated", "metric": key, "value": value}
            for key, value in integrated_cell_type_metrics.items()
            if key
            not in {
                "n_shared_genes",
                "shared_genes",
                "missing_genes",
                "train_cells",
                "test_cells",
                "classes",
                "classification_report",
                "label_column",
            }
        ]
    )
    summary_tsv = output_dir / "benchmark_summary.tsv"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    source_classification_report_tsv = _write_classification_report(
        source_cell_type_metrics,
        output_dir / "source_cell_type_classification_report.tsv",
    )
    integrated_classification_report_tsv = _write_classification_report(
        integrated_cell_type_metrics,
        output_dir / "integrated_cell_type_classification_report.tsv",
    )
    comparison_payload = {
        "merfish_adata": str(args.merfish_adata),
        "source_rank_file": str(args.source_rank_file),
        "reference_rank_files": [str(path) for path in reference_paths],
        "source_rank_tsv": aggregation["source_rank_tsv"],
        "integrated_rank_tsv": aggregation["integrated_rank_tsv"],
        "source_top_panel_tsv": source_panel_tsv,
        "integrated_top_panel_tsv": integrated_panel_tsv,
        "source_top_panel": aggregation["source_top_panel"],
        "integrated_top_panel": aggregation["integrated_top_panel"],
        "overlap_count": aggregation["overlap_count"],
        "jaccard": aggregation["jaccard"],
        "n_restricted_gene_symbols": aggregation["n_restricted_gene_symbols"],
        "source_metrics": source_metrics,
        "integrated_metrics": integrated_metrics,
        "source_cell_type_metrics": source_cell_type_metrics,
        "integrated_cell_type_metrics": integrated_cell_type_metrics,
        "artifacts": {
            **aggregation["artifacts"],
            "benchmark_summary_tsv": str(summary_tsv),
            "source_cell_type_classification_report_tsv": source_classification_report_tsv,
            "integrated_cell_type_classification_report_tsv": integrated_classification_report_tsv,
        },
    }
    comparison_json = output_dir / "benchmark_result.json"
    comparison_json.write_text(json.dumps(comparison_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    figure_path = _plot_comparison(summary_df, output_dir / "benchmark_comparison.png")
    report_paths = _write_report(
        output_dir=output_dir,
        title=args.title,
        comparison=comparison_payload,
        source_metrics=source_metrics,
        integrated_metrics=integrated_metrics,
        source_cell_type_metrics=source_cell_type_metrics,
        integrated_cell_type_metrics=integrated_cell_type_metrics,
        summary_df=summary_df,
        figure_path=figure_path,
    )
    final_payload = {
        **comparison_payload,
        "benchmark_summary_tsv": str(summary_tsv),
        "benchmark_comparison_png": figure_path,
        **report_paths,
    }
    print(json.dumps(final_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
