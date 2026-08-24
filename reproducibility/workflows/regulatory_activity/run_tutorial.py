#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import anndata as ad

from reproducibility.workflows.common import (
    parse_int_list,
    parse_key_value_list,
    run_smith,
    sha256,
    gene_symbols,
    write_json,
    write_top_panel,
)
from reproducibility.workflows.external_baselines import run_baseline
from reproducibility.workflows.regulatory_activity.evaluate_outputs import evaluate
from reproducibility.workflows.regulatory_activity.analysis import lineage_overlap, write_statistical_analysis
from reproducibility.workflows.regulatory_activity.paper_analysis import (
    coactivity_reconstruction,
    tf_scrna_correlation,
    write_module_coverage,
)


PAPER_METHODS = ("PERSIST-class", "PERSIST", "ActiveSVM", "scGIST", "scGeneFit", "Spapros")
PAPER_SIZES = {"elegans_tf": (32, 64, 128), "elegans_mirna": (16, 24, 32)}


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _data_path(data_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (data_root / candidate).resolve()


def _prepare_shared_scrna(scrna_file: Path, tf_file: Path, output_dir: Path) -> tuple[Path, list[str]]:
    """Restrict the scRNA source to the TF activity shared-gene universe."""
    scrna, tf = ad.read_h5ad(scrna_file), ad.read_h5ad(tf_file)
    try:
        scrna_genes = gene_symbols(scrna)
        tf_genes = gene_symbols(tf)
        shared = list(dict.fromkeys(gene for gene in scrna_genes if gene in set(tf_genes)))
        if len(shared) < 3:
            raise ValueError("scRNA and TF activity sources have fewer than three shared TF features")
        scrna_index = {gene: index for index, gene in enumerate(scrna_genes)}
        prepared = scrna[:, [scrna_index[gene] for gene in shared]].copy()
        prepared.var_names = pd.Index(shared)
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared_file = output_dir / "source_scrna_shared_tf.h5ad"
        prepared.write_h5ad(prepared_file)
    finally:
        scrna.file.close()
        tf.file.close()
    return prepared_file, shared


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

    module_file = _data_path(Path(args.data_root).resolve(), args.module_file)
    pair_file = _data_path(Path(args.data_root).resolve(), args.regulatory_pair_file)
    scrna_file = _data_path(Path(args.data_root).resolve(), args.scrna_file)
    if args.paper_analyses:
        required = (
            ("module annotations", "module-file", module_file),
            ("TF-pair annotations", "regulatory-pair-file", pair_file),
            ("C. elegans scRNA reference", "scrna-file", scrna_file),
        )
        missing = [f"{label} (--{option})" for label, option, path in required if path is None or not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Figure 3g-k requires versioned biological inputs; missing " + ", ".join(missing) + ". "
                "Download the supplementary Zenodo bundle before using --paper-analyses."
            )

    inputs = []
    if args.paper_analyses:
        for path in (module_file, pair_file, scrna_file):
            inputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
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
                    save_model=args.paper_analyses,
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
                        "panel_file": str(panel), "checkpoint_file": training.get("checkpoint_file"),
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
    paper_outputs = {}
    if args.paper_analyses:
        tf_panels = values[values["dataset"] == "elegans_tf"].copy()
        paper_outputs["module_coverage"] = str(write_module_coverage(
            tf_panels, module_file, figure_dir / "figure3_h_module_miss_rate.tsv"
        ))
        coactivity_parts = []
        for row in tf_panels.to_dict("records"):
            safe_method = str(row["method"]).replace("/", "_").replace(" ", "_")
            part = figure_dir / "coactivity_parts" / f"{safe_method}_{row['split']}_seed{row['training_seed']}_panel{row['panel_size']}.tsv"
            coactivity_reconstruction(
                data_base / "elegans_tf" / row["split"] / "train.h5ad",
                data_base / "elegans_tf" / row["split"] / "test.h5ad",
                row["panel_file"], part, pair_file=pair_file, lineage_column=args.lineage_column,
                seed=int(row["training_seed"]), method=str(row["method"]), checkpoint_file=row.get("checkpoint_file"),
            )
            coactivity_parts.append(pd.read_csv(part, sep="\t"))
        if not coactivity_parts:
            raise ValueError("No generated SMITH TF panels available for coactivity analysis")
        coactivity_path = figure_dir / "figure3_i_coactivity.tsv"
        pd.concat(coactivity_parts, ignore_index=True).to_csv(coactivity_path, sep="\t", index=False)
        paper_outputs["coactivity"] = str(coactivity_path)
        tf_file_for_corr = data_base / "elegans_tf" / splits[0] / "train.h5ad"
        corr_path = figure_dir / "figure3_j_tf_scrna_correlation.tsv"
        tf_scrna_correlation(scrna_file, tf_file_for_corr, corr_path)
        paper_outputs["tf_scrna_correlation"] = str(corr_path)

        prepared_scrna, shared_tf = _prepare_shared_scrna(scrna_file, tf_file_for_corr, output_dir / "prepared")
        paper_outputs["prepared_shared_scrna"] = str(prepared_scrna)
        transfer_parts = []
        for split in splits:
            for seed in seeds:
                transfer_dir = output_dir / "runs" / "elegans_tf" / split / f"seed_{seed}" / "transfer_scRNA_SMITH"
                transfer_training = run_smith(
                    adata_file=prepared_scrna, output_dir=transfer_dir, tasks="recon,cls",
                    task_name=f"elegans_scrna_to_tf_{split}_seed{seed}", panel_size=max(PAPER_SIZES["elegans_tf"]),
                    epochs=args.epochs, device=args.device, seed=seed, batch_size=args.batch_size,
                    max_cells=args.max_cells, sampling_strategy="celltype", force=args.force,
                    save_model=True,
                )
                for panel_size in PAPER_SIZES["elegans_tf"]:
                    transfer_panel = write_top_panel(
                        transfer_training["ranking_csv"], transfer_dir / "panels" / f"RNA_to_TF_top{panel_size}.tsv", panel_size
                    )
                    target_train = data_base / "elegans_tf" / split / "train.h5ad"
                    target_test = data_base / "elegans_tf" / split / "test.h5ad"
                    result = evaluate(target_train, target_test, transfer_panel, transfer_dir / "evaluation" / f"top{panel_size}", panel_size, args.time_column, args.neighbors)
                    transfer_parts.append({
                        "split": split, "training_seed": seed, "panel_size": panel_size,
                        "source_modality": "scRNA-to-TF", "method": "SMITH", **result["metrics"],
                        "panel_file": str(transfer_panel), "checkpoint_file": transfer_training.get("checkpoint_file"),
                    })
        transfer_path = figure_dir / "figure3_k_transfer.tsv"
        direct = values[values["dataset"] == "elegans_tf"].copy()
        direct["source_modality"] = "TF-to-TF"
        direct = direct.rename(columns={"training_seed": "training_seed"})
        combined = pd.concat([direct, pd.DataFrame(transfer_parts)], ignore_index=True, sort=False)
        combined.to_csv(transfer_path, sep="\t", index=False)
        paper_outputs["transfer"] = str(transfer_path)
    manifest = {
        "workflow": "02_regulatory_activity",
        "manuscript_figure": "Figure 3c-k" if args.paper_analyses else "Figure 3c-f",
        "configuration": vars(args),
        "inputs": list({item["path"]: item for item in inputs}.values()),
        "training_runs": runs,
        "outputs": {"figure_values": str(values_path), "figure_summary": str(summary_path),
                    "paired_tests": str(stats_path), **paper_outputs,
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
    parser.add_argument("--paper-analyses", action="store_true", help="Run the manuscript-specific Figure 3g-k analyses.")
    parser.add_argument("--module-file", default=None, help="Versioned TF spatiotemporal module annotation TSV, relative to --data-root.")
    parser.add_argument("--regulatory-pair-file", default=None, help="Versioned annotated TF-pair TSV, relative to --data-root.")
    parser.add_argument("--scrna-file", default=None, help="C. elegans scRNA reference H5AD, relative to --data-root.")
    parser.add_argument("--lineage-column", default=None, help="Observation column containing lineage names for coactivity analysis.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
