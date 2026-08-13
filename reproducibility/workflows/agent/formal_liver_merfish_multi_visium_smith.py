from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SMITH_ROOT = Path("/workspace/fanyimin/SMITH_unified/code/SMITH_tool-main")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smith_agent.benchmarking import (  # noqa: E402
    evaluate_panel_cell_type_classification,
    evaluate_panel_coordinate_regression,
)
from smith_agent.panel_rank_aggregation import aggregate_reference_panel_ranks  # noqa: E402
from smith_agent.reporting.builder import render_markdown_pdf  # noqa: E402
from smith_agent.utils import ensure_dir, write_json  # noqa: E402


DEFAULT_SOURCE = REPO_ROOT / "data/liver_merfish/adata_healthy_nucseq.h5ad"
DEFAULT_MERFISH = REPO_ROOT / "data/liver_merfish/adata_healthy_merfish.h5ad"
DEFAULT_VISIUM_REFERENCES = [
    (
        "PSC011_C1_visium",
        REPO_ROOT
        / "outputs/visium5_pilot/materialized/0cc59004-2b35-4767-8278-83e097ef32d1/0cc59004-2b35-4767-8278-83e097ef32d1.h5ad",
    ),
    (
        "C73_B1_healthy_visium",
        REPO_ROOT
        / "outputs/visium5_pilot/materialized/1fa9f77e-b79e-4ca6-b294-dcb9f47e5825/1fa9f77e-b79e-4ca6-b294-dcb9f47e5825.h5ad",
    ),
    (
        "H35_sample1_visium",
        REPO_ROOT
        / "outputs/visium5_pilot/materialized/266885f6-34bb-4ace-971d-fe18a2de27d4/266885f6-34bb-4ace-971d-fe18a2de27d4.h5ad",
    ),
    (
        "WSSS_F_IMMsp9838712_visium",
        REPO_ROOT
        / "outputs/visium5_pilot/materialized/2d7de988-0325-4c19-9eb8-808bcd1f594d/2d7de988-0325-4c19-9eb8-808bcd1f594d.h5ad",
    ),
    (
        "H35_sample2_visium",
        REPO_ROOT
        / "outputs/visium5_pilot/materialized/31dd9980-8c9b-4b9a-83ea-114e5fbdddee/31dd9980-8c9b-4b9a-83ea-114e5fbdddee.h5ad",
    ),
]

GENE_SYMBOL_COLUMNS = (
    "feature_name",
    "gene_symbol",
    "gene_symbols",
    "gene_name",
    "gene_short_name",
    "symbol",
)

UNKNOWN_LABELS = {"", "nan", "none", "unknown", "unassigned", "na"}


def _clean_gene_symbol(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text.upper()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _load_merfish_gene_universe(path: str | Path) -> list[str]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        genes = [_clean_gene_symbol(gene) for gene in adata.var_names.astype(str).tolist()]
    finally:
        adata.file.close()
    out: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        if gene and gene not in seen:
            out.append(gene)
            seen.add(gene)
    return out


def _gene_symbols(adata: ad.AnnData) -> pd.Series:
    for column in GENE_SYMBOL_COLUMNS:
        if column in adata.var.columns:
            return pd.Series(adata.var[column].astype(str).map(_clean_gene_symbol).to_numpy(), index=adata.var_names)
    return pd.Series(pd.Index(adata.var_names).astype(str).map(_clean_gene_symbol).to_numpy(), index=adata.var_names)


def _select_gene_positions(adata: ad.AnnData, gene_universe: list[str]) -> tuple[list[str], list[int]]:
    symbols = _gene_symbols(adata)
    first_position: dict[str, int] = {}
    for idx, gene in enumerate(symbols.tolist()):
        if gene and gene not in first_position:
            first_position[gene] = idx
    selected_genes: list[str] = []
    selected_positions: list[int] = []
    for gene in gene_universe:
        clean = _clean_gene_symbol(gene)
        if clean in first_position:
            selected_genes.append(clean)
            selected_positions.append(first_position[clean])
    if not selected_genes:
        raise ValueError("No genes overlap with the requested gene universe.")
    return selected_genes, selected_positions


def _resolve_spatial(adata: ad.AnnData) -> np.ndarray | None:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2]
    if "X_spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["X_spatial"], dtype=np.float32)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2]
    lowered = {str(col).lower(): col for col in adata.obs.columns}
    for x_col, y_col in (("array_col", "array_row"), ("x", "y"), ("center_x", "center_y")):
        if x_col in lowered and y_col in lowered:
            return adata.obs[[lowered[x_col], lowered[y_col]]].astype(float).to_numpy(dtype=np.float32)
    return None


def _label_mask(obs: pd.DataFrame, label_column: str, min_label_count: int, drop_unknown: bool) -> tuple[np.ndarray, dict[str, int]]:
    if label_column not in obs.columns:
        raise KeyError(f"Missing required label column `{label_column}`.")
    labels = obs[label_column].astype(str)
    valid = labels.notna()
    if drop_unknown:
        valid &= ~labels.str.strip().str.lower().isin(UNKNOWN_LABELS)
    counts = labels[valid].value_counts()
    keep_labels = set(counts[counts >= int(min_label_count)].index.astype(str))
    valid &= labels.isin(keep_labels)
    final_counts = labels[valid].value_counts().to_dict()
    if len(final_counts) < 2:
        raise ValueError(
            f"Need at least two retained classes in `{label_column}` after filtering; retained={final_counts}"
        )
    return valid.to_numpy(dtype=bool), {str(k): int(v) for k, v in final_counts.items()}


def prepare_smith_input(
    input_h5ad: str | Path,
    output_h5ad: str | Path,
    gene_universe: list[str],
    *,
    label_column: str = "cell_type",
    require_spatial: bool = False,
    drop_unknown_labels: bool = True,
    min_label_count: int = 5,
    max_cells: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    input_h5ad = Path(input_h5ad)
    output_h5ad = Path(output_h5ad)
    ensure_dir(output_h5ad.parent)
    source = ad.read_h5ad(input_h5ad)
    try:
        selected_genes, selected_positions = _select_gene_positions(source, gene_universe)
        obs_mask, label_counts = _label_mask(source.obs, label_column, min_label_count, drop_unknown_labels)
        obs_indices = np.flatnonzero(obs_mask)
        if max_cells is not None and max_cells > 0 and obs_indices.size > int(max_cells):
            rng = np.random.default_rng(seed)
            labels = source.obs.iloc[obs_indices][label_column].astype(str).to_numpy()
            sampled: list[np.ndarray] = []
            counts = pd.Series(labels).value_counts()
            for label, count in counts.items():
                label_positions = obs_indices[np.flatnonzero(labels == label)]
                n_take = max(1, int(round(int(max_cells) * int(count) / max(1, obs_indices.size))))
                n_take = min(n_take, label_positions.size)
                sampled.append(rng.choice(label_positions, size=n_take, replace=False))
            obs_indices = np.sort(np.concatenate(sampled))
        prepared = source[obs_indices, selected_positions].copy()
    finally:
        del source

    if sparse.issparse(prepared.X):
        prepared.X = prepared.X.tocsr().astype(np.float32)
    else:
        prepared.X = np.asarray(prepared.X, dtype=np.float32)
    prepared.var_names = selected_genes
    prepared.var = pd.DataFrame(index=pd.Index(selected_genes, name=None))
    prepared.var["feature_name"] = selected_genes
    prepared.var_names_make_unique()
    if label_column != "cell_type":
        prepared.obs["cell_type"] = prepared.obs[label_column].astype(str)
    else:
        prepared.obs["cell_type"] = prepared.obs["cell_type"].astype(str)

    coords = _resolve_spatial(prepared)
    if require_spatial and coords is None:
        raise KeyError(f"{input_h5ad} does not expose spatial coordinates.")
    if coords is not None:
        prepared.obsm["spatial"] = coords.astype(np.float32)

    prepared.write_h5ad(output_h5ad)
    return {
        "input_h5ad": str(input_h5ad),
        "output_h5ad": str(output_h5ad),
        "n_obs": int(prepared.n_obs),
        "n_vars": int(prepared.n_vars),
        "label_column": label_column,
        "label_counts": label_counts,
        "has_spatial": coords is not None,
    }


def _smith_command(
    *,
    adata_file: Path,
    saving_dir: Path,
    log_dir: Path,
    tasks: str,
    task_name: str,
    panel_size: int,
    epoch: int,
    record: int,
    device: str,
    seed: int,
    batch_size: int,
    balance_mode: str,
    balance_cap: int,
    max_cells: int | None,
    sampling_strategy: str,
) -> list[str]:
    command = [
        sys.executable,
        str(SMITH_ROOT / "main.py"),
        "--adata_file",
        str(adata_file.resolve()),
        "--saving_dir",
        str(saving_dir.resolve()),
        "--log_dir",
        str(log_dir.resolve()),
        "--tasks",
        tasks,
        "--task_name",
        task_name,
        "--panel_size",
        str(panel_size),
        "--epoch",
        str(epoch),
        "--record",
        str(record),
        "--device",
        device,
        "--seed",
        str(seed),
        "--batch_size",
        str(batch_size),
        "--dim",
        "32",
        "--rep_dim",
        "32",
        "--rep_hidden_dims",
        "32",
        "--head_hidden_dims",
        "",
        "--dropout_rate",
        "0.2",
        "--lam",
        "0.5",
        "--sigma",
        "0.5",
        "--activation",
        "tanh",
        "--optimizer",
        "Adam",
        "--balance_mode",
        balance_mode,
        "--balance_cap",
        str(balance_cap),
        "--sampling_strategy",
        sampling_strategy,
    ]
    if max_cells is not None and max_cells > 0:
        command.extend(["--max_cells", str(max_cells)])
    return command


def _latest_epoch_csv(saving_dir: Path) -> Path | None:
    epoch_files = list(saving_dir.glob("epoch_*.csv"))
    if not epoch_files:
        return None

    def epoch_number(path: Path) -> int:
        match = re.search(r"epoch_(\d+)\.csv$", path.name)
        return int(match.group(1)) if match else -1

    return max(epoch_files, key=epoch_number)


def run_smith(
    *,
    name: str,
    adata_file: Path,
    output_dir: Path,
    tasks: str,
    panel_size: int,
    epoch: int,
    record: int,
    device: str,
    seed: int,
    batch_size: int,
    balance_mode: str,
    balance_cap: int,
    max_cells: int | None,
    sampling_strategy: str,
    force: bool,
) -> dict[str, Any]:
    run_dir = ensure_dir(output_dir / _safe_id(name)).resolve()
    saving_dir = ensure_dir(run_dir / "saving")
    log_dir = ensure_dir(run_dir / "logs")
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    rank_csv = _latest_epoch_csv(saving_dir)
    if rank_csv and not force:
        return {
            "name": name,
            "status": "skipped_existing",
            "rank_csv": str(rank_csv),
            "run_dir": str(run_dir),
        }

    command = _smith_command(
        adata_file=adata_file,
        saving_dir=saving_dir,
        log_dir=log_dir,
        tasks=tasks,
        task_name=_safe_id(name),
        panel_size=panel_size,
        epoch=epoch,
        record=record,
        device=device,
        seed=seed,
        batch_size=batch_size,
        balance_mode=balance_mode,
        balance_cap=balance_cap,
        max_cells=max_cells,
        sampling_strategy=sampling_strategy,
    )
    manifest_path = write_json(
        saving_dir / "smith_run_manifest.json",
        {
            "name": name,
            "command": command,
            "tasks": tasks,
            "epoch": epoch,
            "record": record,
            "panel_size": panel_size,
        },
    )
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=str(SMITH_ROOT), stdout=stdout, stderr=stderr, text=True, check=True)
    rank_csv = _latest_epoch_csv(saving_dir)
    if rank_csv is None:
        raise FileNotFoundError(f"SMITH run did not produce an epoch CSV in {saving_dir}.")
    return {
        "name": name,
        "status": "completed",
        "rank_csv": str(rank_csv),
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def _copy_rank_csv_as_tsv(rank_csv: str | Path, output_tsv: str | Path, dataset_id: str) -> str:
    df = pd.read_csv(rank_csv)
    gene_col = next((col for col in df.columns if str(col).lower() in {"marker", "gene", "gene_symbol"}), df.columns[0])
    out = pd.DataFrame({"gene_symbol": df[gene_col].astype(str).map(_clean_gene_symbol)})
    out = out[out["gene_symbol"].astype(bool)].drop_duplicates("gene_symbol").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, out.shape[0] + 1))
    out["dataset_id"] = dataset_id
    out["dataset_path"] = str(rank_csv)
    out["data_role"] = "smith_rank"
    output_tsv = Path(output_tsv)
    ensure_dir(output_tsv.parent)
    out.to_csv(output_tsv, sep="\t", index=False)
    return str(output_tsv)


def _metric_rows(panel: str, coordinate: dict[str, Any], cell_type: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in coordinate["metrics"].items():
        rows.append({"panel": panel, "metric": key, "value": value})
    for key, value in cell_type["metrics"].items():
        rows.append({"panel": panel, "metric": key, "value": value})
    rows.append({"panel": panel, "metric": "n_shared_genes", "value": len(coordinate["shared_genes"])})
    rows.append({"panel": panel, "metric": "n_cell_type_classes", "value": len(cell_type["classes"])})
    return rows


def _plot_summary(summary_df: pd.DataFrame, output_png: Path) -> str:
    metrics = [
        "spatial_distance_mae",
        "spatial_pearson",
        "cell_type_accuracy",
        "cell_type_balanced_accuracy",
        "cell_type_macro_f1",
        "cell_type_weighted_f1",
    ]
    labels = {
        "spatial_distance_mae": "Coord MAE",
        "spatial_pearson": "Spatial Pearson",
        "cell_type_accuracy": "Cell Type Acc.",
        "cell_type_balanced_accuracy": "Balanced Acc.",
        "cell_type_macro_f1": "Macro F1",
        "cell_type_weighted_f1": "Weighted F1",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8))
    axes = np.asarray(axes).reshape(-1)
    palette = {"source_smith": "#1f6f78", "multi_visium_smith": "#d98e04"}
    for ax, metric in zip(axes, metrics):
        subset = summary_df[summary_df["metric"] == metric]
        order = ["source_smith", "multi_visium_smith"]
        values = [float(subset.loc[subset["panel"] == panel, "value"].iloc[0]) for panel in order]
        bars = ax.bar(order, values, color=[palette[p] for p in order], alpha=0.92)
        ax.set_title(labels[metric])
        ax.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return str(output_png)


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.astype(str).to_dict(orient="records"):
        lines.append("| " + " | ".join(row[col] for col in cols) + " |")
    return "\n".join(lines)


def _write_report(output_dir: Path, summary_df: pd.DataFrame, result: dict[str, Any], figure_png: str) -> dict[str, str]:
    report_md = output_dir / "formal_benchmark_report.md"
    report_pdf = output_dir / "formal_benchmark_report.pdf"
    selected = summary_df[
        summary_df["metric"].isin(
            [
                "spatial_distance_mae",
                "spatial_pearson",
                "cell_type_accuracy",
                "cell_type_balanced_accuracy",
                "cell_type_macro_f1",
                "cell_type_weighted_f1",
            ]
        )
    ].copy()
    lines = [
        "# Formal Liver MERFISH Multi-Visium SMITH Benchmark",
        "",
        "## Definition",
        "- Source panel: SMITH trained on the paired liver snRNA-seq reference with `recon,cls`.",
        "- Multi-Visium panel: five independent Visium SMITH runs with `recon,cls,standard_coordination`, then rank aggregation.",
        "- MERFISH target is held out from SMITH training; its 317-gene universe is used only to constrain assay-measurable candidates.",
        "",
        "## Metrics",
        _markdown_table(selected),
        "",
        "## Artifacts",
        f"- Source SMITH panel: `{result['source_panel_tsv']}`",
        f"- Multi-Visium SMITH panel: `{result['integrated_panel_tsv']}`",
        f"- Summary TSV: `{result['summary_tsv']}`",
        "",
        f"![formal benchmark]({Path(figure_png).name})",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    rendered_pdf = render_markdown_pdf(report_md, report_pdf)
    return {"report_markdown": str(report_md), "report_pdf": rendered_pdf or ""}


def parse_reference_args(items: list[str] | None) -> list[tuple[str, Path]]:
    if not items:
        return [(name, path) for name, path in DEFAULT_VISIUM_REFERENCES]
    refs: list[tuple[str, Path]] = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).stem
        refs.append((_safe_id(name), Path(path)))
    return refs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-adata", default=str(DEFAULT_SOURCE))
    parser.add_argument("--source-label-column", default="Cell_Type_final")
    parser.add_argument("--merfish-adata", default=str(DEFAULT_MERFISH))
    parser.add_argument("--visium-reference", action="append", help="Reference as name=/path/to/file.h5ad.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/liver_merfish_benchmark/formal_multi_visium_smith"))
    parser.add_argument("--panel-size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--record", type=int, default=50)
    parser.add_argument("--source-max-cells", type=int, default=30000)
    parser.add_argument("--visium-max-cells", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-label-count", type=int, default=5)
    parser.add_argument("--keep-unknown-labels", action="store_true")
    parser.add_argument("--source-weight", type=float, default=0.50)
    parser.add_argument("--reference-weight", type=float, default=0.50)
    parser.add_argument("--min-reference-support", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    gene_universe = _load_merfish_gene_universe(args.merfish_adata)
    prepared_dir = ensure_dir(output_dir / "prepared")
    ranks_dir = ensure_dir(output_dir / "smith_ranks")

    source_prepared = prepare_smith_input(
        args.source_adata,
        prepared_dir / "source_scrna_smith_input.h5ad",
        gene_universe,
        label_column=str(args.source_label_column),
        require_spatial=False,
        drop_unknown_labels=not args.keep_unknown_labels,
        min_label_count=args.min_label_count,
        max_cells=args.source_max_cells,
        seed=args.seed,
    )
    source_run = run_smith(
        name="source_scrna_smith",
        adata_file=Path(source_prepared["output_h5ad"]),
        output_dir=output_dir / "smith_runs",
        tasks="recon,cls",
        panel_size=args.panel_size,
        epoch=args.epoch,
        record=args.record,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        balance_mode="capped",
        balance_cap=500,
        max_cells=None,
        sampling_strategy="celltype",
        force=args.force,
    )
    source_rank_tsv = _copy_rank_csv_as_tsv(source_run["rank_csv"], ranks_dir / "source_scrna_smith_rank.tsv", "source_scrna_smith")

    reference_runs: list[dict[str, Any]] = []
    reference_rank_tsvs: list[str] = []
    reference_ids: list[str] = []
    for name, path in parse_reference_args(args.visium_reference):
        prepared = prepare_smith_input(
            path,
            prepared_dir / f"{_safe_id(name)}_smith_input.h5ad",
            gene_universe,
            label_column="cell_type",
            require_spatial=True,
            drop_unknown_labels=not args.keep_unknown_labels,
            min_label_count=args.min_label_count,
            max_cells=args.visium_max_cells if args.visium_max_cells > 0 else None,
            seed=args.seed,
        )
        run = run_smith(
            name=name,
            adata_file=Path(prepared["output_h5ad"]),
            output_dir=output_dir / "smith_runs",
            tasks="recon,cls,standard_coordination",
            panel_size=args.panel_size,
            epoch=args.epoch,
            record=args.record,
            device=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
            balance_mode="capped",
            balance_cap=500,
            max_cells=None,
            sampling_strategy="celltype_spatial",
            force=args.force,
        )
        rank_tsv = _copy_rank_csv_as_tsv(run["rank_csv"], ranks_dir / f"{_safe_id(name)}_smith_rank.tsv", name)
        reference_runs.append({**run, "prepared": prepared, "rank_tsv": rank_tsv})
        reference_rank_tsvs.append(rank_tsv)
        reference_ids.append(name)

    aggregation = aggregate_reference_panel_ranks(
        output_dir=output_dir / "formal_rank_aggregation",
        source_rank_file=source_rank_tsv,
        reference_rank_files=reference_rank_tsvs,
        reference_ids=reference_ids,
        panel_size=args.panel_size,
        source_weight=args.source_weight,
        reference_weight=args.reference_weight,
        min_reference_support=args.min_reference_support,
        gene_universe="source",
        restrict_gene_symbols=gene_universe,
    )
    source_panel = aggregation["source_top_panel_tsv"]
    integrated_panel = aggregation["integrated_top_panel_tsv"]

    source_coord = evaluate_panel_coordinate_regression(
        adata_file=args.merfish_adata,
        panel_path=source_panel,
        output_dir=output_dir / "evaluation/source_coordinate",
        panel_size=args.panel_size,
        seed=args.seed,
    ).to_dict()
    integrated_coord = evaluate_panel_coordinate_regression(
        adata_file=args.merfish_adata,
        panel_path=integrated_panel,
        output_dir=output_dir / "evaluation/integrated_coordinate",
        panel_size=args.panel_size,
        seed=args.seed,
    ).to_dict()
    source_cell = evaluate_panel_cell_type_classification(
        adata_file=args.merfish_adata,
        panel_path=source_panel,
        output_dir=output_dir / "evaluation/source_cell_type",
        panel_size=args.panel_size,
        label_column="Cell_Type",
        seed=args.seed,
    ).to_dict()
    integrated_cell = evaluate_panel_cell_type_classification(
        adata_file=args.merfish_adata,
        panel_path=integrated_panel,
        output_dir=output_dir / "evaluation/integrated_cell_type",
        panel_size=args.panel_size,
        label_column="Cell_Type",
        seed=args.seed,
    ).to_dict()

    summary_df = pd.DataFrame(
        _metric_rows("source_smith", source_coord, source_cell)
        + _metric_rows("multi_visium_smith", integrated_coord, integrated_cell)
    )
    summary_tsv = output_dir / "formal_benchmark_summary.tsv"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    figure_png = _plot_summary(summary_df, output_dir / "formal_benchmark_comparison.png")

    result = {
        "source_adata": str(args.source_adata),
        "merfish_adata": str(args.merfish_adata),
        "n_merfish_gene_universe": len(gene_universe),
        "source_prepared": source_prepared,
        "source_run": source_run,
        "reference_runs": reference_runs,
        "source_rank_tsv": source_rank_tsv,
        "reference_rank_tsvs": reference_rank_tsvs,
        "aggregation": aggregation,
        "source_panel_tsv": source_panel,
        "integrated_panel_tsv": integrated_panel,
        "source_coordinate": source_coord,
        "integrated_coordinate": integrated_coord,
        "source_cell_type": source_cell,
        "integrated_cell_type": integrated_cell,
        "summary_tsv": str(summary_tsv),
        "figure_png": figure_png,
    }
    result_json = output_dir / "formal_benchmark_result.json"
    result["result_json"] = str(result_json)
    report_paths = _write_report(output_dir, summary_df, result, figure_png)
    result.update(report_paths)
    write_json(result_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
