#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd


WORKSPACE_ROOT = Path("/workspace/fanyimin/SMITH_unified")
SMITH_ROOT = WORKSPACE_ROOT / "code" / "SMITH_tool-main"

import sys

sys.path.insert(0, str(SMITH_ROOT))

from Smith.eval import (  # noqa: E402
    evaluate_knn_overlap,
    evaluate_split_knn_classification,
    evaluate_split_obsm_regression,
    evaluate_split_time_knn_regression,
    infer_time_label,
    load_panel_genes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate completed elegans baseline panels on fixed train/test splits "
            "using split-aware metrics."
        )
    )
    parser.add_argument(
        "--datasets",
        default="elegans_tf,elegans_mirna",
        help="Comma-separated dataset output roots under outputs/baselines/transfer.",
    )
    parser.add_argument("--celltype-knn-k", type=int, default=5)
    parser.add_argument("--time-knn-k", type=int, default=5)
    parser.add_argument("--overlap-ks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--max-cells", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def dataset_root(name: str) -> Path:
    return WORKSPACE_ROOT / "outputs" / "baselines" / "transfer" / name


def split_dir_from_metadata(metadata: dict[str, object]) -> Path:
    train_h5ad = Path(str(metadata["train_h5ad"]))
    return train_h5ad.parent


def evaluate_one_run(run_dir: Path, metadata: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    panel_csv = Path(str(metadata["panel_csv"]))
    train_h5ad = Path(str(metadata["train_h5ad"]))
    test_h5ad = Path(str(metadata["test_h5ad"]))
    num_markers = int(metadata["num_markers"])
    method = str(metadata["method"])
    split = str(metadata["split"])

    panel_genes = load_panel_genes(panel_csv, panel_size=num_markers)
    train_adata = ad.read_h5ad(train_h5ad)
    test_adata = ad.read_h5ad(test_h5ad)
    train_adata.var_names = train_adata.var_names.str.upper()
    test_adata.var_names = test_adata.var_names.str.upper()

    label_name = "cell_type" if "cell_type" in train_adata.obs.columns and "cell_type" in test_adata.obs.columns else None
    time_label = infer_time_label(test_adata)

    celltype_knn_accuracy = None
    if label_name is not None:
        celltype_knn_accuracy, _ = evaluate_split_knn_classification(
            train_adata,
            test_adata,
            panel_genes,
            label_name=label_name,
            n_neighbors=args.celltype_knn_k,
        )

    spatial_pearson = None
    spatial_label = None
    if "spatial" in train_adata.obsm and "spatial" in test_adata.obsm:
        _, spatial_pearson = evaluate_split_obsm_regression(
            train_adata,
            test_adata,
            panel_genes,
            obsm_key="spatial",
        )
        spatial_label = "spatial"

    time_pearson = None
    if time_label is not None:
        time_pearson = evaluate_split_time_knn_regression(
            train_adata,
            test_adata,
            panel_genes,
            time_label=time_label,
            n_neighbors=args.time_knn_k,
        )

    overlap_df = evaluate_knn_overlap(
        test_adata,
        panel_genes,
        ks=tuple(args.overlap_ks),
        max_cells=args.max_cells,
        random_state=args.seed,
    )
    overlap_avg = None
    if not overlap_df.empty:
        avg_row = overlap_df.loc[overlap_df["k"] < 0]
        if not avg_row.empty:
            overlap_avg = float(avg_row["mean_overlap_fraction"].iloc[0])

    result = {
        "dataset": run_dir.parents[1].name,
        "split": split,
        "method": method,
        "num_markers": num_markers,
        "run_dir": str(run_dir),
        "panel_csv": str(panel_csv),
        "train_h5ad": str(train_h5ad),
        "test_h5ad": str(test_h5ad),
        "celltype_label": label_name,
        "celltype_knn_k": args.celltype_knn_k,
        "celltype_knn_accuracy": celltype_knn_accuracy,
        "time_label": time_label,
        "time_knn_k": args.time_knn_k,
        "time_knn_pearson": time_pearson,
        "spatial_obsm_key": spatial_label,
        "spatial_pearson": spatial_pearson,
        "knn_overlap_mean_avg": overlap_avg,
        "knn_overlap_ks": ",".join(map(str, args.overlap_ks)),
        "knn_overlap_n_cells": None if overlap_df.empty else int(overlap_df["n_cells"].iloc[0]),
    }
    return result


def run_dataset(name: str, args: argparse.Namespace) -> None:
    root = dataset_root(name)
    run_dirs = sorted(path for path in root.glob("split_*/*") if path.is_dir())
    records = []
    for run_dir in run_dirs:
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != "completed":
            continue

        out_csv = run_dir / "fixed_split_eval.csv"
        if out_csv.exists() and not args.force:
            record = pd.read_csv(out_csv).iloc[0].to_dict()
            records.append(record)
            continue

        print(f"[eval] {name} :: {metadata['split']} :: {metadata['method']} :: top{metadata['num_markers']}", flush=True)
        record = evaluate_one_run(run_dir, metadata, args)
        pd.DataFrame([record]).to_csv(out_csv, index=False)
        records.append(record)

    per_run = pd.DataFrame(records)
    per_run_path = root / "fixed_split_metrics_per_run.csv"
    per_run.to_csv(per_run_path, index=False)

    summary_rows = []
    if not per_run.empty:
        for (method, num_markers), group in per_run.groupby(["method", "num_markers"], dropna=False):
            row = {
                "method": method,
                "num_markers": num_markers,
                "n_completed_runs": int(len(group)),
            }
            for metric in [
                "celltype_knn_accuracy",
                "time_knn_pearson",
                "spatial_pearson",
                "knn_overlap_mean_avg",
            ]:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else None
                row[f"{metric}_std"] = float(values.std()) if values.notna().sum() > 1 else None
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = root / "fixed_split_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[done] {name} :: per-run -> {per_run_path} :: summary -> {summary_path}", flush=True)


def main() -> int:
    args = parse_args()
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    for name in datasets:
        run_dataset(name, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
