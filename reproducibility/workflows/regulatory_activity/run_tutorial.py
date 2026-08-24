#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from reproducibility.workflows.common import (
    gene_symbols,
    parse_int_list,
    parse_key_value_list,
    run_smith,
    sha256,
    write_json,
    write_top_panel,
)
from reproducibility.workflows.external_baselines import run_baseline, run_persist_reconstruction
from reproducibility.workflows.regulatory_activity.analysis import lineage_overlap, write_statistical_analysis
from reproducibility.workflows.regulatory_activity.evaluate_outputs import evaluate
from reproducibility.workflows.regulatory_activity.paper_analysis import (
    coactivity_reconstruction,
    tf_scrna_correlation,
    write_module_coverage,
)


PAPER_METHODS = ("PERSIST-class", "PERSIST", "ActiveSVM", "scGIST", "scGeneFit", "Spapros")
BENCHMARK_SIZES = {"elegans_tf": (32, 64, 128), "elegans_mirna": (16, 24, 32)}
MODULE_SIZES = (16, 24, 32)
COACTIVITY_PANEL_SIZE = 32


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _data_path(data_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (data_root / candidate).resolve()


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _normalize_scrna_like_tf(adata: ad.AnnData) -> ad.AnnData:
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    matrix = _dense(matrix).astype(np.float32)
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    matrix = np.log1p(matrix * (1e4 / totals))
    minimum = matrix.min(axis=0)
    denominator = matrix.max(axis=0) - minimum
    denominator[denominator == 0] = 1.0
    adata.X = ((matrix - minimum) / denominator).astype(np.float32)
    adata.uns["normalization"] = "counts->normalize_total_1e4->log1p->per_gene_minmax_0_1"
    return adata


def _prepare_scrna_source(
    scrna_file: Path,
    tf_test_files: list[Path],
    output_file: Path,
) -> tuple[Path, list[str]]:
    shared_tf: set[str] | None = None
    for path in tf_test_files:
        tf = ad.read_h5ad(path)
        try:
            genes = set(tf.var_names.astype(str).str.upper())
            shared_tf = genes if shared_tf is None else shared_tf & genes
        finally:
            tf.file.close()
    if not shared_tf:
        raise ValueError("No TF genes are shared across requested activity test splits")

    scrna = ad.read_h5ad(scrna_file)
    try:
        source_names = (
            scrna.var["gene_short_name"].astype(str).str.upper()
            if "gene_short_name" in scrna.var
            else scrna.var_names.astype(str).str.upper()
        )
        scrna.var_names = pd.Index(source_names)
        scrna.var_names_make_unique()
        if "gene_short_name" in scrna.var:
            scrna.var = scrna.var.rename(columns={"gene_short_name": "gene_short_name_orig"})
        if "cell_type" not in scrna.obs and "cell.type" in scrna.obs:
            scrna.obs["cell_type"] = scrna.obs["cell.type"].astype(str)
        labels = scrna.obs["cell_type"].astype(str).str.strip()
        label_mask = labels.ne("") & ~labels.str.lower().isin({"nan", "none"})
        scrna = scrna[label_mask.to_numpy()].copy()
        scrna.obs["cell_type"] = labels.loc[label_mask].to_numpy()
        if "absolute_time" not in scrna.obs and "embryo.time" in scrna.obs:
            scrna.obs["absolute_time"] = pd.to_numeric(scrna.obs["embryo.time"], errors="coerce")
        time = pd.to_numeric(scrna.obs["absolute_time"], errors="coerce")
        scrna = scrna[time.notna().to_numpy()].copy()
        scrna.obs["absolute_time"] = time.loc[time.notna()].to_numpy()
        keep = scrna.var_names.isin(sorted(shared_tf))
        scrna = scrna[:, keep].copy()
        if scrna.n_vars < 3:
            raise ValueError("Fewer than three shared TF genes remain in the scRNA source")
        shared = scrna.var_names.astype(str).tolist()
        _normalize_scrna_like_tf(scrna)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        scrna.write_h5ad(output_file)
    finally:
        if getattr(scrna, "file", None) is not None:
            scrna.file.close()
    return output_file, shared


def _combine_tf_split(train_file: Path, test_file: Path, output_file: Path) -> Path:
    train, test = ad.read_h5ad(train_file), ad.read_h5ad(test_file)
    try:
        combined = ad.concat([train, test], axis=0, join="inner", merge="same", index_unique=None)
        combined.var_names = train.var_names.copy()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        combined.write_h5ad(output_file)
    finally:
        train.file.close()
        test.file.close()
    return output_file


def _record_input(inputs: list[dict], path: Path) -> None:
    inputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})


def run(args: argparse.Namespace) -> dict:
    data_root = Path(args.data_root).resolve()
    data_base = data_root / "regulatory_activity" / "elegans" / "splits"
    output_dir = Path(args.output_dir).resolve()
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = _csv_list(args.datasets)
    splits = _csv_list(args.splits)
    seeds = parse_int_list(args.seeds)
    requested_methods = _csv_list(args.methods)
    if "SMITH" not in requested_methods:
        raise ValueError("This biological workflow requires SMITH in --methods")
    external_methods = [method for method in requested_methods if method != "SMITH"]
    baseline_pythons = parse_key_value_list(args.baseline_python)
    if external_methods and not args.baseline_root:
        raise RuntimeError(
            "Manuscript baselines were requested but --baseline-root was not supplied. "
            "Point it to GPS_tools-main/baselines; no substitute selector is used."
        )

    module_file = _data_path(data_root, args.module_file)
    pair_file = _data_path(data_root, args.regulatory_pair_file)
    scrna_file = _data_path(data_root, args.scrna_file)
    if args.paper_analyses:
        required = (
            ("module annotations", "module-file", module_file),
            ("TF-pair annotations", "regulatory-pair-file", pair_file),
            ("C. elegans scRNA reference", "scrna-file", scrna_file),
        )
        missing = [f"{label} (--{option})" for label, option, path in required if path is None or not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Figure 3g-k requires versioned biological inputs; missing " + ", ".join(missing)
            )

    inputs: list[dict] = []
    if args.paper_analyses:
        for path in (module_file, pair_file, scrna_file):
            _record_input(inputs, path)
    benchmark_rows: list[dict] = []
    panel_rows: list[dict] = []
    runs: list[dict] = []

    for dataset in datasets:
        if dataset not in BENCHMARK_SIZES:
            raise ValueError(f"Unsupported regulatory dataset: {dataset}")
        benchmark_sizes = BENCHMARK_SIZES[dataset]
        training_sizes = sorted(
            set(benchmark_sizes) | (set(MODULE_SIZES) if args.paper_analyses and dataset == "elegans_tf" else set())
        )
        for split in splits:
            split_dir = data_base / dataset / split
            train_file, test_file = split_dir / "train.h5ad", split_dir / "test.h5ad"
            for path in (train_file, test_file):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing Figure 3 input: {path}")
                _record_input(inputs, path)
            overlap = lineage_overlap(train_file, test_file)
            if overlap:
                raise ValueError(f"Lineage-aware split {split} leaks {overlap} cell identifiers")

            for seed in seeds:
                for panel_size in training_sizes:
                    run_dir = output_dir / "runs" / dataset / split / f"seed_{seed}" / f"panel_{panel_size}" / "SMITH"
                    training = run_smith(
                        adata_file=train_file,
                        output_dir=run_dir,
                        tasks="recon,cls,standard_coordination,time" if dataset == "elegans_tf" else "recon,cls,time",
                        task_name=f"{dataset}_{split}_panel{panel_size}_seed{seed}",
                        panel_size=panel_size,
                        epochs=args.epochs,
                        device=args.device,
                        seed=seed,
                        batch_size=args.batch_size,
                        time_label=args.time_column,
                        max_cells=args.max_cells,
                        sampling_strategy="celltype",
                        save_model=args.paper_analyses and dataset == "elegans_tf" and panel_size == COACTIVITY_PANEL_SIZE,
                        force=args.force,
                    )
                    runs.append(training)
                    panel = write_top_panel(
                        training["ranking_csv"], run_dir.parent / "panels" / f"SMITH_top{panel_size}.tsv", panel_size
                    )
                    panel_row = {
                        "dataset": dataset,
                        "split": split,
                        "training_seed": seed,
                        "panel_size": panel_size,
                        "method": "SMITH",
                        "panel_file": str(panel),
                        "checkpoint_file": training.get("checkpoint_file"),
                        "reconstruction_file": None,
                    }
                    panel_rows.append(panel_row)
                    if panel_size in benchmark_sizes:
                        result = evaluate(
                            train_file,
                            test_file,
                            panel,
                            run_dir.parent / "evaluation",
                            panel_size,
                            args.time_column,
                            args.neighbors,
                        )
                        benchmark_rows.append({**panel_row, **result["metrics"]})

            for method in external_methods:
                for panel_size in training_sizes:
                    method_dir = output_dir / "runs" / dataset / split / "baselines" / method / f"panel_{panel_size}"
                    reconstruction_file = None
                    if (
                        args.paper_analyses
                        and dataset == "elegans_tf"
                        and method == "PERSIST"
                        and panel_size == COACTIVITY_PANEL_SIZE
                    ):
                        panel, reconstruction_file = run_persist_reconstruction(
                            train_file,
                            test_file,
                            method_dir,
                            panel_size,
                            args.baseline_root,
                            args.baseline_epochs,
                            args.force,
                            python_executable=baseline_pythons.get(method),
                            device=args.persist_device,
                            seed=1,
                        )
                    else:
                        panel = run_baseline(
                            method,
                            train_file,
                            method_dir,
                            panel_size,
                            "cell_type",
                            args.baseline_root,
                            args.baseline_epochs,
                            args.force,
                            baseline_pythons.get(method),
                        )
                    panel_row = {
                        "dataset": dataset,
                        "split": split,
                        "training_seed": 1,
                        "panel_size": panel_size,
                        "method": method,
                        "panel_file": str(panel),
                        "checkpoint_file": None,
                        "reconstruction_file": str(reconstruction_file) if reconstruction_file else None,
                    }
                    panel_rows.append(panel_row)
                    if panel_size in benchmark_sizes:
                        result = evaluate(
                            train_file,
                            test_file,
                            panel,
                            method_dir / "evaluation",
                            panel_size,
                            args.time_column,
                            args.neighbors,
                        )
                        benchmark_rows.append({**panel_row, **result["metrics"]})

    values = pd.DataFrame(benchmark_rows)
    panels = pd.DataFrame(panel_rows)
    figure_dir = output_dir / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    values_path = figure_dir / "figure3_c_f_values.tsv"
    panels_path = figure_dir / "generated_panels.tsv"
    values.to_csv(values_path, sep="\t", index=False)
    panels.to_csv(panels_path, sep="\t", index=False)
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

    paper_outputs: dict[str, str] = {}
    if args.paper_analyses:
        coverage_panels = panels[
            (panels["dataset"] == "elegans_tf") & panels["panel_size"].isin(MODULE_SIZES)
        ]
        paper_outputs["module_coverage"] = str(
            write_module_coverage(coverage_panels, module_file, figure_dir / "figure3_h_module_miss_rate.tsv")
        )

        coactivity_panels = panels[
            (panels["dataset"] == "elegans_tf")
            & (panels["panel_size"] == COACTIVITY_PANEL_SIZE)
            & panels["method"].isin(["SMITH", "PERSIST"])
        ]
        coactivity_parts = []
        for row in coactivity_panels.to_dict("records"):
            safe_method = str(row["method"]).replace("/", "_").replace(" ", "_")
            part = figure_dir / "coactivity_parts" / f"{safe_method}_{row['split']}_seed{row['training_seed']}.tsv"
            coactivity_reconstruction(
                data_base / "elegans_tf" / row["split"] / "train.h5ad",
                data_base / "elegans_tf" / row["split"] / "test.h5ad",
                row["panel_file"],
                part,
                pair_file=pair_file,
                lineage_column=args.lineage_column,
                seed=int(row["training_seed"]),
                method=str(row["method"]),
                checkpoint_file=row.get("checkpoint_file"),
                reconstruction_file=row.get("reconstruction_file"),
            )
            part_frame = pd.read_csv(part, sep="\t")
            part_frame["split"] = row["split"]
            part_frame["training_seed"] = row["training_seed"]
            coactivity_parts.append(part_frame)
        if not coactivity_parts:
            raise ValueError("No generated 32-TF SMITH/PERSIST panels are available for co-activity analysis")
        coactivity_path = figure_dir / "figure3_i_coactivity.tsv"
        pd.concat(coactivity_parts, ignore_index=True).to_csv(coactivity_path, sep="\t", index=False)
        paper_outputs["coactivity"] = str(coactivity_path)

        first_tf_split = data_base / "elegans_tf" / splits[0]
        combined_tf = _combine_tf_split(
            first_tf_split / "train.h5ad",
            first_tf_split / "test.h5ad",
            output_dir / "prepared" / "elegans_tf_full_from_split.h5ad",
        )
        corr_path = figure_dir / "figure3_j_tf_scrna_correlation.tsv"
        tf_scrna_correlation(scrna_file, combined_tf, corr_path)
        paper_outputs["tf_scrna_correlation"] = str(corr_path)

        tf_test_files = [data_base / "elegans_tf" / split / "test.h5ad" for split in splits]
        prepared_scrna, shared_tf = _prepare_scrna_source(
            scrna_file,
            tf_test_files,
            output_dir / "prepared" / "elegans_scrna_shared_tf.h5ad",
        )
        paper_outputs["prepared_shared_scrna"] = str(prepared_scrna)
        transfer_rows: list[dict] = []
        for seed in seeds:
            transfer_dir = output_dir / "runs" / "transfer_scRNA" / f"seed_{seed}" / "SMITH"
            transfer_training = run_smith(
                adata_file=prepared_scrna,
                output_dir=transfer_dir,
                tasks="recon,cls,time",
                task_name=f"elegans_scrna_to_tf_seed{seed}",
                panel_size=max(BENCHMARK_SIZES["elegans_tf"]),
                epochs=args.epochs,
                device=args.device,
                seed=seed,
                batch_size=args.batch_size,
                time_label=args.time_column,
                max_cells=args.max_cells,
                sampling_strategy="celltype",
                force=args.force,
            )
            runs.append(transfer_training)
            for panel_size in BENCHMARK_SIZES["elegans_tf"]:
                panel = write_top_panel(
                    transfer_training["ranking_csv"],
                    transfer_dir / "panels" / f"RNA_TF_top{panel_size}.tsv",
                    panel_size,
                )
                for split in splits:
                    target_train = data_base / "elegans_tf" / split / "train.h5ad"
                    target_test = data_base / "elegans_tf" / split / "test.h5ad"
                    result = evaluate(
                        target_train,
                        target_test,
                        panel,
                        transfer_dir / "evaluation" / split / f"top{panel_size}",
                        panel_size,
                        args.time_column,
                        args.neighbors,
                    )
                    transfer_rows.append(
                        {
                            "split": split,
                            "training_seed": seed,
                            "panel_size": panel_size,
                            "source_modality": "RNA-TF",
                            "method": "SMITH",
                            **result["metrics"],
                            "panel_file": str(panel),
                        }
                    )

        if "PERSIST-class" in requested_methods:
            for panel_size in BENCHMARK_SIZES["elegans_tf"]:
                source_dir = output_dir / "runs" / "transfer_scRNA" / "baselines" / "PERSIST-class" / f"panel_{panel_size}"
                panel = run_baseline(
                    "PERSIST-class",
                    prepared_scrna,
                    source_dir,
                    panel_size,
                    "cell_type",
                    args.baseline_root,
                    args.baseline_epochs,
                    args.force,
                    baseline_pythons.get("PERSIST-class"),
                )
                for split in splits:
                    result = evaluate(
                        data_base / "elegans_tf" / split / "train.h5ad",
                        data_base / "elegans_tf" / split / "test.h5ad",
                        panel,
                        source_dir / "evaluation" / split,
                        panel_size,
                        args.time_column,
                        args.neighbors,
                    )
                    transfer_rows.append(
                        {
                            "split": split,
                            "training_seed": 1,
                            "panel_size": panel_size,
                            "source_modality": "RNA-TF",
                            "method": "PERSIST-class",
                            **result["metrics"],
                            "panel_file": str(panel),
                        }
                    )

        direct = values[
            (values["dataset"] == "elegans_tf") & values["method"].isin(["SMITH", "PERSIST-class"])
        ].copy()
        direct["source_modality"] = "TF-TF"
        transfer_path = figure_dir / "figure3_k_transfer.tsv"
        pd.concat([direct, pd.DataFrame(transfer_rows)], ignore_index=True, sort=False).to_csv(
            transfer_path, sep="\t", index=False
        )
        paper_outputs["transfer"] = str(transfer_path)

    manifest = {
        "workflow": "02_regulatory_activity",
        "manuscript_figure": "Figure 3c-k" if args.paper_analyses else "Figure 3c-f",
        "configuration": vars(args),
        "inputs": list({item["path"]: item for item in inputs}.values()),
        "training_runs": runs,
        "outputs": {
            "figure_values": str(values_path),
            "generated_panels": str(panels_path),
            "figure_summary": str(summary_path),
            "paired_tests": str(stats_path),
            **paper_outputs,
            "prediction_files": sorted(
                str(path) for path in output_dir.glob("runs/**/evaluation/**/*predictions.tsv")
            ),
        },
        "transfer_shared_tf_count": len(shared_tf) if args.paper_analyses else None,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SMITH and reproduce the C. elegans regulatory analyses in Figure 3c-k."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default="elegans_tf,elegans_mirna")
    parser.add_argument("--splits", default="split_1,split_2,split_3,split_4,split_5")
    parser.add_argument("--methods", default="SMITH,PERSIST-class,PERSIST,ActiveSVM,scGIST,scGeneFit,Spapros")
    parser.add_argument("--baseline-root", default=None)
    parser.add_argument(
        "--baseline-python",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Optional per-method interpreter, for example PERSIST=/opt/envs/persist/bin/python.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--persist-device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--baseline-epochs", type=int, default=200)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--panel-size", type=int, default=128, help="Compatibility option; manuscript sizes are fixed.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--time-column", default="absolute_time")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--paper-analyses", action="store_true")
    parser.add_argument("--module-file", default=None)
    parser.add_argument("--regulatory-pair-file", default=None)
    parser.add_argument("--scrna-file", default=None)
    parser.add_argument("--lineage-column", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
