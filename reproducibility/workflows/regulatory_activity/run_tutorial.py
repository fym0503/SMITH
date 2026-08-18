#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from reproducibility.workflows.common import (
    parse_int_list,
    parse_key_value_list,
    run_smith,
    sha256,
    write_json,
    write_top_panel,
)
from reproducibility.workflows.external_baselines import run_baseline
from reproducibility.workflows.regulatory_activity.evaluate_outputs import evaluate
from reproducibility.workflows.regulatory_activity.analysis import lineage_overlap, write_statistical_analysis


PAPER_METHODS = ("PERSIST-class", "PERSIST", "ActiveSVM", "scGIST", "scGeneFit", "Spapros")
PAPER_SIZES = {"elegans_tf": (32, 64, 128), "elegans_mirna": (16, 24, 32)}


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run(args: argparse.Namespace) -> dict:
    data_base = Path(args.data_root).resolve() / "regulatory_activity" / "elegans" / "splits"
    output_dir = Path(args.output_dir).resolve()
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = _csv_list(args.datasets)
    splits = _csv_list(args.splits)
    seeds = parse_int_list(args.seeds)
    requested_methods = _csv_list(args.methods)
    if "SMITH" not in requested_methods:
        raise ValueError("This biological workflow requires SMITH in --methods.")
    external_methods = [method for method in requested_methods if method != "SMITH"]
    baseline_pythons = parse_key_value_list(args.baseline_python)
    if external_methods and not args.baseline_root:
        raise RuntimeError(
            "Manuscript baselines were requested but --baseline-root was not supplied. "
            "Point it to GPS_tools-main/baselines; no substitute baseline is used."
        )

    inputs = []
    rows = []
    runs = []
    for dataset in datasets:
        if dataset not in PAPER_SIZES:
            raise ValueError(f"Unsupported regulatory dataset: {dataset}")
        panel_sizes = PAPER_SIZES[dataset]
        max_panel = max(panel_sizes)
        for split in splits:
            split_dir = data_base / dataset / split
            train_file, test_file = split_dir / "train.h5ad", split_dir / "test.h5ad"
            for path in (train_file, test_file):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing Figure 3 input: {path}")
                inputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
            overlap = lineage_overlap(train_file, test_file)
            if overlap:
                raise ValueError(f"Lineage-aware split {split} leaks {overlap} cell identifiers")

            for seed in seeds:
                run_dir = output_dir / "runs" / dataset / split / f"seed_{seed}"
                training = run_smith(
                    adata_file=train_file,
                    output_dir=run_dir / "SMITH",
                    tasks="recon,cls,standard_coordination,time" if dataset == "elegans_tf" else "recon,cls,time",
                    task_name=f"{dataset}_{split}_seed{seed}",
                    panel_size=max_panel,
                    epochs=args.epochs,
                    device=args.device,
                    seed=seed,
                    batch_size=args.batch_size,
                    time_label=args.time_column,
                    max_cells=args.max_cells,
                    sampling_strategy="celltype",
                    force=args.force,
                )
                runs.append(training)
                for panel_size in panel_sizes:
                    panel = write_top_panel(
                        training["ranking_csv"], run_dir / "panels" / f"SMITH_top{panel_size}.tsv", panel_size
                    )
                    result = evaluate(
                        train_file, test_file, panel, run_dir / "evaluation" / f"SMITH_top{panel_size}",
                        panel_size, args.time_column, args.neighbors,
                    )
                    rows.append({
                        "dataset": dataset, "split": split, "training_seed": seed,
                        "panel_size": panel_size, "method": "SMITH", **result["metrics"],
                        "panel_file": str(panel),
                    })

            for method in external_methods:
                for panel_size in panel_sizes:
                    method_dir = output_dir / "runs" / dataset / split / "baselines" / method / f"panel_{panel_size}"
                    panel = run_baseline(
                        method, train_file, method_dir, panel_size, "cell_type", args.baseline_root,
                        args.baseline_epochs, args.force, baseline_pythons.get(method),
                    )
                    result = evaluate(
                        train_file, test_file, panel, method_dir / "evaluation",
                        panel_size, args.time_column, args.neighbors,
                    )
                    rows.append({
                        "dataset": dataset, "split": split, "training_seed": 0,
                        "panel_size": panel_size, "method": method, **result["metrics"],
                        "panel_file": str(panel),
                    })

    values = pd.DataFrame(rows)
    figure_dir = output_dir / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    values_path = figure_dir / "figure3_c_f_values.tsv"
    values.to_csv(values_path, sep="\t", index=False)
    summary = (
        values.groupby(["dataset", "panel_size", "method"], as_index=False)
        .agg(
            n=("split", "size"),
            cell_type_accuracy_mean=("cell_type_accuracy", "mean"),
            cell_type_accuracy_std=("cell_type_accuracy", "std"),
            developmental_time_pearson_mean=("developmental_time_pearson", "mean"),
            developmental_time_pearson_std=("developmental_time_pearson", "std"),
        )
    )
    summary_path = figure_dir / "figure3_c_f_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    stats_path = write_statistical_analysis(values_path, figure_dir)
    manifest = {
        "workflow": "02_regulatory_activity",
        "manuscript_figure": "Figure 3c-f",
        "configuration": vars(args),
        "inputs": list({item["path"]: item for item in inputs}.values()),
        "training_runs": runs,
        "outputs": {"figure_values": str(values_path), "figure_summary": str(summary_path),
                    "paired_tests": str(stats_path),
                    "prediction_files": sorted(str(path) for path in output_dir.glob("runs/**/evaluation/**/*predictions.tsv"))},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SMITH to identify regulatory features that preserve C. elegans identity and developmental time."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default="elegans_tf,elegans_mirna")
    parser.add_argument("--splits", default="split_1,split_2,split_3,split_4,split_5")
    parser.add_argument("--methods", default="SMITH,PERSIST-class,PERSIST,ActiveSVM,scGIST,scGeneFit,Spapros")
    parser.add_argument("--baseline-root", default=None)
    parser.add_argument(
        "--baseline-python", action="append", default=[], metavar="METHOD=PATH",
        help="Optional per-method interpreter, for example scGIST=/opt/envs/scgist/bin/python.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--baseline-epochs", type=int, default=200)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--panel-size", type=int, default=128, help="Compatibility option; paper sizes are fixed per dataset.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--time-column", default="absolute_time")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
