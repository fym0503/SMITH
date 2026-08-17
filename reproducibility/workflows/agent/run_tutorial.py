#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from smith_agent.panel_rank_aggregation import aggregate_reference_panel_ranks
from reproducibility.workflows.agent.evaluate_outputs import evaluate
from reproducibility.workflows.common import (
    gene_symbols,
    parse_int_list,
    run_smith,
    sha256,
    write_json,
    write_top_panel,
)


DEFAULT_REFERENCES = (
    "references/PSC011_C1_visium.h5ad",
    "references/C73_B1_healthy_visium.h5ad",
    "references/H35_sample1_visium.h5ad",
    "references/WSSS_F_IMMsp9838712_visium.h5ad",
    "references/H35_sample2_visium.h5ad",
)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _label_column(adata: ad.AnnData, requested: str | None) -> str:
    for column in ([requested] if requested else []) + ["cell_type", "Cell_Type", "Cell_Type_final", "celltype", "subclass"]:
        if column and column in adata.obs:
            return column
    raise KeyError("No usable cell-type label column found.")


def _spatial(adata: ad.AnnData) -> np.ndarray | None:
    for key in ("spatial", "X_spatial"):
        if key in adata.obsm:
            return np.asarray(adata.obsm[key], dtype=np.float32)[:, :2]
    lowered = {str(column).lower(): column for column in adata.obs.columns}
    for x_name, y_name in (("array_col", "array_row"), ("x", "y"), ("center_x", "center_y")):
        if x_name in lowered and y_name in lowered:
            return adata.obs[[lowered[x_name], lowered[y_name]]].astype(float).to_numpy(dtype=np.float32)
    return None


def _prepare(input_file: Path, output_file: Path, universe: list[str], label_request: str | None, require_spatial: bool, max_cells: int | None, seed: int) -> dict:
    adata = ad.read_h5ad(input_file)
    label = _label_column(adata, label_request)
    first = {}
    for index, gene in enumerate(gene_symbols(adata)):
        if gene and gene not in first:
            first[gene] = index
    genes = [gene for gene in universe if gene in first]
    labels = adata.obs[label].astype(str)
    valid = ~labels.str.strip().str.lower().isin({"", "nan", "none", "unknown", "unassigned", "na"})
    rows = np.flatnonzero(valid.to_numpy())
    if max_cells and len(rows) > max_cells:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(rows, max_cells, replace=False))
    prepared = adata[rows, [first[gene] for gene in genes]].copy()
    prepared.var_names = genes
    prepared.var = pd.DataFrame(index=pd.Index(genes))
    prepared.obs["celltype"] = prepared.obs[label].astype(str)
    coords = _spatial(prepared)
    if require_spatial and coords is None:
        raise KeyError(f"Spatial reference {input_file} has no coordinates.")
    if coords is not None:
        prepared.obsm["spatial"] = coords
    prepared.X = prepared.X.tocsr().astype(np.float32) if sparse.issparse(prepared.X) else np.asarray(prepared.X, dtype=np.float32)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(output_file)
    return {"input": str(input_file), "output": str(output_file), "n_obs": prepared.n_obs, "n_vars": prepared.n_vars}


def _merfish_mean_expression(path: Path) -> dict[str, float]:
    adata = ad.read_h5ad(path)
    means = np.asarray(adata.X.mean(axis=0)).ravel() if sparse.issparse(adata.X) else np.asarray(adata.X).mean(axis=0)
    return {gene: float(value) for gene, value in zip(gene_symbols(adata), means) if gene}


def run(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_root).resolve() / "agent"
    source_file, merfish_file = data_dir / args.source, data_dir / args.merfish
    reference_files = [data_dir / path for path in args.reference]
    missing = [str(path) for path in [source_file, merfish_file, *reference_files] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Figure 6 input files: " + ", ".join(missing))
    panel_sizes = parse_int_list(args.panel_sizes)
    training_seeds = parse_int_list(args.training_seeds)
    output_dir = Path(args.output_dir).resolve()
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merfish = ad.read_h5ad(merfish_file, backed="r")
    try:
        universe = list(dict.fromkeys(gene for gene in gene_symbols(merfish) if gene))
    finally:
        merfish.file.close()
    prepared_dir = output_dir / "prepared_data"
    prepared_source = prepared_dir / "source_scrna.h5ad"
    preparation = [_prepare(source_file, prepared_source, universe, args.source_label_column, False, args.max_cells, training_seeds[0])]
    prepared_references = []
    for path in reference_files:
        prepared = prepared_dir / f"reference_{_safe_id(path.stem)}.h5ad"
        preparation.append(_prepare(path, prepared, universe, args.reference_label_column, True, args.max_cells, training_seeds[0]))
        prepared_references.append(prepared)
    source_panel_name = "snRNA-seq"
    integrated_panel_name = f"snRNA-seq + {len(prepared_references)} ST"

    metric_rows, support_rows, seed_runs = [], [], []
    expression = _merfish_mean_expression(merfish_file)
    for seed in training_seeds:
        seed_dir = output_dir / "runs" / f"seed_{seed}"
        source_run = run_smith(
            adata_file=prepared_source, output_dir=seed_dir / "source_smith", tasks="recon,cls",
            task_name=f"liver_source_seed{seed}", panel_size=max(panel_sizes), epochs=args.epochs,
            device=args.device, seed=seed, batch_size=args.batch_size, sampling_strategy="celltype", force=args.force,
        )
        reference_runs = [
            run_smith(
                adata_file=prepared, output_dir=seed_dir / "reference_smith" / prepared.stem,
                tasks="recon,cls,standard_coordination", task_name=f"{prepared.stem}_seed{seed}",
                panel_size=max(panel_sizes), epochs=args.epochs, device=args.device, seed=seed,
                batch_size=args.batch_size, sampling_strategy="celltype_spatial", force=args.force,
            ) for prepared in prepared_references
        ]
        aggregation = aggregate_reference_panel_ranks(
            output_dir=seed_dir / "aggregation", source_rank_file=source_run["ranking_csv"],
            reference_rank_files=[item["ranking_csv"] for item in reference_runs],
            reference_ids=[path.stem for path in prepared_references], panel_size=max(panel_sizes),
            source_weight=args.source_weight, reference_weight=1.0 - args.source_weight,
            min_reference_support=args.min_reference_support, restrict_gene_symbols=universe, gene_universe="source",
        )
        panels = []
        for size in panel_sizes:
            source_panel = write_top_panel(aggregation["source_rank_tsv"], seed_dir / "panels" / f"source_top{size}.tsv", size)
            integrated_panel = write_top_panel(aggregation["integrated_rank_tsv"], seed_dir / "panels" / f"multi_reference_top{size}.tsv", size)
            panels.extend([(source_panel_name, source_panel, size), (integrated_panel_name, integrated_panel, size)])
        for panel_name, panel_file, size in panels:
            evaluation = evaluate(
                merfish_file, [(panel_name, panel_file)], seed_dir / "evaluation" / f"{_safe_id(panel_name)}_{size}",
                size, args.merfish_label_column, args.evaluation_seed,
            )
            details = evaluation["panels"][panel_name]["classification"]["metrics"]
            metric_rows.append({
                "training_seed": seed, "panel_size": size, "panel": panel_name,
                "cell_type_accuracy": details["cell_type_accuracy"], "panel_file": str(panel_file),
            })
            genes = pd.read_csv(panel_file, sep="\t")["gene_symbol"].astype(str)
            support_rows.append({
                "training_seed": seed, "panel_size": size, "panel": panel_name,
                "mean_merfish_expression": float(np.mean([expression.get(gene, 0.0) for gene in genes])),
                "panel_file": str(panel_file),
            })
        seed_runs.append({"seed": seed, "source": source_run, "references": reference_runs, "aggregation": aggregation})

    figure_dir = output_dir / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    accuracy_path = figure_dir / "figure6_c_cell_type_accuracy.tsv"
    expression_path = figure_dir / "figure6_d_merfish_expression.tsv"
    pd.DataFrame(metric_rows).to_csv(accuracy_path, sep="\t", index=False)
    pd.DataFrame(support_rows).to_csv(expression_path, sep="\t", index=False)
    manifest = {
        "workflow": "05_agent", "manuscript_figure": "Figure 6c-d", "configuration": vars(args),
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in [source_file, merfish_file, *reference_files]],
        "preparation": preparation, "training_runs": seed_runs,
        "outputs": {"figure6_c": str(accuracy_path), "figure6_d": str(expression_path)},
        "probe_backend": {"status": "not_run", "reason": "Figure 6e-g requires separately installed ODT/OligoMiner/ProbeDealer backends and indexes."},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SMITH-Agent panels and evaluate which genes preserve liver cell identity in MERFISH."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", default="liver_merfish/adata_healthy_nucseq.h5ad")
    parser.add_argument("--merfish", default="liver_merfish/adata_healthy_merfish.h5ad")
    parser.add_argument("--reference", action="append", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--training-seeds", default="1,2,3,4,5")
    parser.add_argument("--evaluation-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1, help="Compatibility option; use --training-seeds.")
    parser.add_argument("--panel-sizes", default="32,64,128")
    parser.add_argument("--panel-size", type=int, default=128, help="Compatibility option; use --panel-sizes.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells", type=int, default=30000)
    parser.add_argument("--source-label-column", default=None)
    parser.add_argument("--reference-label-column", default=None)
    parser.add_argument("--merfish-label-column", default="Cell_Type")
    parser.add_argument("--source-weight", type=float, default=0.50)
    parser.add_argument("--min-reference-support", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.reference = args.reference or list(DEFAULT_REFERENCES)
    if len(args.reference) < 2:
        parser.error("Figure 6 requires at least two references; the manuscript run uses five.")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
