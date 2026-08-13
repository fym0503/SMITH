#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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
    read_panel,
    run_smith,
    sha256,
    write_json,
    write_top_panel,
)
from reproducibility.workflows.external_baselines import run_baseline
from reproducibility.workflows.ribomap_transfer.evaluate_outputs import evaluate_panel


SOURCE_FILES = {
    "Deep-RIBOmap": "deep_brain_ribomap.h5ad",
    "STARmap": "mouse_brain_starmap_rep2.h5ad",
}
PAPER_METHODS = ("PERSIST-class", "PERSIST", "ActiveSVM", "scGIST", "scGeneFit", "Spapros")


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _label(adata: ad.AnnData, candidates: tuple[str, ...]) -> str:
    column = next((name for name in candidates if name in adata.obs), None)
    if not column:
        raise KeyError(f"No label from {candidates} exists in {list(adata.obs.columns)}")
    return column


def _prepare_shared(source_file: Path, target_file: Path, output_path: Path) -> Path:
    source = ad.read_h5ad(source_file)
    target = ad.read_h5ad(target_file, backed="r")
    try:
        target_genes = set(gene_symbols(target))
    finally:
        target.file.close()
    positions = {gene: i for i, gene in enumerate(gene_symbols(source)) if gene}
    shared = [gene for gene in positions if gene in target_genes]
    prepared = source[:, [positions[gene] for gene in shared]].copy()
    prepared.var_names = shared
    prepared.var = pd.DataFrame(index=pd.Index(shared))
    prepared.obs["celltype"] = prepared.obs[_label(prepared, ("celltype", "cell_type", "Cell_Type", "subclass"))].astype(str)
    region = _label(prepared, ("region", "Region", "spatial_region", "cluster"))
    prepared.obs["region"] = prepared.obs[region].astype(str)
    has_spatial = any(key in prepared.obsm for key in ("spatial", "X_spatial"))
    prepared.uns["smith_has_spatial"] = bool(has_spatial)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(output_path)
    return output_path


def _mean_expression(path: Path) -> tuple[list[str], np.ndarray]:
    adata = ad.read_h5ad(path)
    x = adata.X
    means = np.asarray(x.mean(axis=0)).ravel() if sparse.issparse(x) else np.asarray(x).mean(axis=0)
    return gene_symbols(adata), means.astype(float)


def _bias_table(ribomap_file: Path, starmap_file: Path) -> pd.DataFrame:
    ribo_genes, ribo_mean = _mean_expression(ribomap_file)
    star_genes, star_mean = _mean_expression(starmap_file)
    ribo = {g: v for g, v in zip(ribo_genes, ribo_mean) if g}
    star = {g: v for g, v in zip(star_genes, star_mean) if g}
    shared = sorted(set(ribo) & set(star))
    r = np.log1p([ribo[g] for g in shared])
    s = np.log1p([star[g] for g in shared])
    rz = (r - r.mean()) / (r.std() or 1.0)
    sz = (s - s.mean()) / (s.std() or 1.0)
    return pd.DataFrame({"gene_symbol": shared, "ribomap_bias": rz - sz})


def run(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_root).resolve() / "ribomap_transfer" / "ribomap"
    target_file = data_dir / args.target
    sources = {name: data_dir / SOURCE_FILES[name] for name in _csv_list(args.sources)}
    missing = [str(path) for path in [target_file, *sources.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Figure 4 input files: " + ", ".join(missing))
    panel_sizes = parse_int_list(args.panel_sizes)
    train_seeds = parse_int_list(args.training_seeds)
    eval_seeds = parse_int_list(args.evaluation_seeds)
    methods = _csv_list(args.methods)
    external_methods = [method for method in methods if method != "SMITH"]
    baseline_pythons = parse_key_value_list(args.baseline_python)
    if external_methods and not args.baseline_root:
        raise RuntimeError("Manuscript baselines require --baseline-root; no variance proxy is used.")

    output_dir = Path(args.output_dir).resolve()
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = {
        name: _prepare_shared(path, target_file, output_dir / "prepared_data" / f"{name.lower()}_shared.h5ad")
        for name, path in sources.items()
    }

    panel_records = []
    metric_rows = []
    run_records = []
    for source_name, source_file in prepared.items():
        source_adata = ad.read_h5ad(source_file, backed="r")
        try:
            has_spatial = bool(source_adata.uns.get("smith_has_spatial", False))
        finally:
            source_adata.file.close()
        for train_seed in train_seeds:
            base = output_dir / "runs" / source_name / f"seed_{train_seed}"
            if "SMITH" in methods:
                training = run_smith(
                    adata_file=source_file, output_dir=base / "SMITH",
                    tasks="recon,cls,standard_coordination" if has_spatial else "recon,cls",
                    task_name=f"{source_name}_to_RIBOMap", panel_size=max(panel_sizes), epochs=args.epochs,
                    device=args.device, seed=train_seed, batch_size=args.batch_size, max_cells=args.max_cells,
                    sampling_strategy="celltype_spatial", force=args.force,
                )
                run_records.append(training)
                for size in panel_sizes:
                    panel = write_top_panel(training["ranking_csv"], base / "panels" / f"SMITH_top{size}.tsv", size)
                    panel_records.append({"source": source_name, "method": "SMITH", "training_seed": train_seed, "panel_size": size, "panel_file": str(panel)})
        for method in external_methods:
            for size in panel_sizes:
                panel = run_baseline(
                    method, source_file, output_dir / "runs" / source_name / "baselines" / method / f"panel_{size}",
                    size, "celltype", args.baseline_root, args.baseline_epochs, args.force,
                    baseline_pythons.get(method),
                )
                panel_records.append({"source": source_name, "method": method, "training_seed": 0, "panel_size": size, "panel_file": str(panel)})

    target = ad.read_h5ad(target_file, backed="r")
    try:
        celltype_column = _label(target, ("celltype", "cell_type", "Cell_Type", "subclass"))
        region_column = _label(target, ("region", "Region", "spatial_region", "cluster"))
    finally:
        target.file.close()
    for record in panel_records:
        for eval_seed in eval_seeds:
            for label, column in (("celltype", celltype_column), ("region", region_column)):
                result = evaluate_panel(target_file, record["panel_file"], record["panel_size"], eval_seed, column)
                metric_rows.append({
                    **record, "evaluation_seed": eval_seed, "label": label,
                    "accuracy": result["metrics"]["accuracy"],
                    "balanced_accuracy": result["metrics"]["balanced_accuracy"],
                    "macro_f1": result["metrics"]["macro_f1"],
                })

    figure_dir = output_dir / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    metrics_path = figure_dir / "figure4_c_f_values.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)

    overlaps = []
    for size in panel_sizes:
        current = [record for record in panel_records if record["panel_size"] == size]
        for left, right in itertools.combinations(current, 2):
            a, b = set(read_panel(left["panel_file"], size)), set(read_panel(right["panel_file"], size))
            overlaps.append({
                "panel_size": size,
                "modality_group": "Same modality" if left["source"] == right["source"] else "Cross modality",
                "source_a": left["source"], "source_b": right["source"],
                "method_a": left["method"], "method_b": right["method"],
                "jaccard": len(a & b) / len(a | b) if a | b else np.nan,
            })
    overlap_path = figure_dir / "figure4_g_jaccard.tsv"
    pd.DataFrame(overlaps).to_csv(overlap_path, sep="\t", index=False)

    bias = _bias_table(target_file, sources["STARmap"])
    bias_rows = []
    for size in panel_sizes:
        current = [record for record in panel_records if record["panel_size"] == size]
        by_key = {(record["source"], record["method"], record["training_seed"]): record for record in current}
        pair_keys = sorted(set((method, seed) for source, method, seed in by_key if source == "Deep-RIBOmap") &
                           set((method, seed) for source, method, seed in by_key if source == "STARmap"))
        for method, seed in pair_keys:
            deep = set(read_panel(by_key[("Deep-RIBOmap", method, seed)]["panel_file"], size))
            star = set(read_panel(by_key[("STARmap", method, seed)]["panel_file"], size))
            group_map = {
                gene: "Deep-RIBOmap" if gene in deep - star else
                "Shared" if gene in deep & star else
                "STARmap" if gene in star - deep else "Background"
                for gene in bias["gene_symbol"]
            }
            part = bias.copy()
            part["panel_size"] = size
            part["method"] = method
            part["training_seed"] = seed
            part["group"] = part["gene_symbol"].map(group_map)
            bias_rows.append(part)
    bias_path = figure_dir / "figure4_h_ribomap_bias.tsv"
    pd.concat(bias_rows, ignore_index=True).to_csv(bias_path, sep="\t", index=False)

    manifest = {
        "workflow": "03_ribomap_transfer", "manuscript_figure": "Figure 4c-h",
        "configuration": vars(args),
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in [target_file, *sources.values()]],
        "training_runs": run_records,
        "outputs": {"figure4_c_f": str(metrics_path), "figure4_g": str(overlap_path), "figure4_h": str(bias_path)},
        "not_reproduced": {"Figure 4i": "Requires a versioned Reactome/GO gene-set snapshot; pass it in a separate enrichment analysis."},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce SMITH Figure 4c-h from RIBOMap/STARmap H5AD inputs.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sources", default="Deep-RIBOmap,STARmap")
    parser.add_argument("--target", default="mouse_brain_ribomap_rep2.h5ad")
    parser.add_argument("--methods", default="SMITH,PERSIST-class,PERSIST,ActiveSVM,scGIST,scGeneFit,Spapros")
    parser.add_argument("--baseline-root", default=None)
    parser.add_argument(
        "--baseline-python", action="append", default=[], metavar="METHOD=PATH",
        help="Optional per-method interpreter, for example PERSIST=/opt/envs/persist/bin/python.",
    )
    parser.add_argument("--panel-sizes", default="32,64,128")
    parser.add_argument("--training-seeds", default="1")
    parser.add_argument("--evaluation-seeds", default="1,2,3,4,5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--baseline-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1, help="Compatibility option; use --training-seeds.")
    parser.add_argument("--panel-size", type=int, default=128, help="Compatibility option; use --panel-sizes.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
