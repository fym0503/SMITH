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
from reproducibility.workflows.common import clean_gene, gene_symbols, run_smith, sha256, write_json


DEFAULT_REFERENCES = ("references/PSC011_C1_visium.h5ad", "references/WSSS_F_IMMsp9838712_visium.h5ad")


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _label_column(adata: ad.AnnData, requested: str | None, candidates: tuple[str, ...]) -> str:
    for column in ([requested] if requested else []) + list(candidates):
        if column and column in adata.obs:
            return column
    raise KeyError(f"No usable label column found; tried {candidates}")


def _spatial(adata: ad.AnnData) -> np.ndarray | None:
    for key in ("spatial", "X_spatial"):
        if key in adata.obsm:
            values = np.asarray(adata.obsm[key], dtype=np.float32)
            if values.ndim == 2 and values.shape[1] >= 2:
                return values[:, :2]
    lowered = {str(column).lower(): column for column in adata.obs.columns}
    for x_name, y_name in (("array_col", "array_row"), ("x", "y"), ("center_x", "center_y")):
        if x_name in lowered and y_name in lowered:
            return adata.obs[[lowered[x_name], lowered[y_name]]].astype(float).to_numpy(dtype=np.float32)
    return None


def _prepare(
    input_file: Path,
    output_file: Path,
    gene_universe: list[str],
    requested_label: str | None,
    require_spatial: bool,
    max_cells: int | None,
    seed: int,
) -> dict:
    adata = ad.read_h5ad(input_file)
    label = _label_column(adata, requested_label, ("cell_type", "Cell_Type", "Cell_Type_final", "celltype", "subclass"))
    first = {}
    for index, gene in enumerate(gene_symbols(adata)):
        if gene and gene not in first:
            first[gene] = index
    selected = [gene for gene in gene_universe if gene in first]
    if not selected:
        raise ValueError(f"No genes in {input_file} overlap the MERFISH gene universe.")
    labels = adata.obs[label].astype(str)
    valid = ~labels.str.strip().str.lower().isin({"", "nan", "none", "unknown", "unassigned", "na"})
    positions = np.flatnonzero(valid.to_numpy())
    if max_cells and len(positions) > max_cells:
        rng = np.random.default_rng(seed)
        selected_positions = []
        counts = labels.iloc[positions].value_counts()
        for value, count in counts.items():
            group = positions[np.flatnonzero(labels.iloc[positions].to_numpy() == value)]
            take = max(1, round(max_cells * count / len(positions)))
            selected_positions.append(rng.choice(group, min(len(group), take), replace=False))
        positions = np.sort(np.concatenate(selected_positions))[:max_cells]
    prepared = adata[positions, [first[gene] for gene in selected]].copy()
    prepared.var_names = selected
    prepared.var = pd.DataFrame(index=pd.Index(selected))
    prepared.obs["celltype"] = prepared.obs[label].astype(str)
    coordinates = _spatial(prepared)
    if require_spatial and coordinates is None:
        raise KeyError(f"Spatial reference {input_file} has no coordinates.")
    if coordinates is not None:
        prepared.obsm["spatial"] = coordinates
    if sparse.issparse(prepared.X):
        prepared.X = prepared.X.tocsr().astype(np.float32)
    else:
        prepared.X = np.asarray(prepared.X, dtype=np.float32)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(output_file)
    return {
        "input": str(input_file), "output": str(output_file), "n_obs": prepared.n_obs,
        "n_vars": prepared.n_vars, "label_column": label, "has_spatial": coordinates is not None,
    }


def run(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_root).resolve() / "agent"
    source_file = data_dir / args.source
    merfish_file = data_dir / args.merfish
    reference_files = [data_dir / path for path in args.reference]
    missing = [str(path) for path in [source_file, merfish_file, *reference_files] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing tutorial input files: " + ", ".join(missing))
    output_dir = Path(args.output_dir).resolve()
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merfish = ad.read_h5ad(merfish_file, backed="r")
    try:
        gene_universe = list(dict.fromkeys(gene for gene in gene_symbols(merfish) if gene))
    finally:
        merfish.file.close()
    prepared_dir = output_dir / "prepared_data"
    preparation = []
    prepared_source = prepared_dir / "source_scrna.h5ad"
    preparation.append(_prepare(source_file, prepared_source, gene_universe, args.source_label_column, False, args.max_cells, args.seed))
    prepared_references = []
    for reference_file in reference_files:
        prepared = prepared_dir / f"reference_{_safe_id(reference_file.stem)}.h5ad"
        preparation.append(_prepare(reference_file, prepared, gene_universe, args.reference_label_column, True, args.max_cells, args.seed))
        prepared_references.append(prepared)

    source_run = run_smith(
        adata_file=prepared_source, output_dir=output_dir / "source_smith", tasks="recon,cls",
        task_name="liver_source", panel_size=args.panel_size, epochs=args.epochs, device=args.device,
        seed=args.seed, batch_size=args.batch_size, sampling_strategy="celltype", force=args.force,
    )
    reference_runs = []
    for prepared in prepared_references:
        reference_runs.append(run_smith(
            adata_file=prepared, output_dir=output_dir / "reference_smith" / prepared.stem,
            tasks="recon,cls,standard_coordination", task_name=prepared.stem, panel_size=args.panel_size,
            epochs=args.epochs, device=args.device, seed=args.seed, batch_size=args.batch_size,
            sampling_strategy="celltype_spatial", force=args.force,
        ))
    source_panel = Path(source_run.get("panel_csv") or (output_dir / "source_smith" / f"panel_top{args.panel_size}.csv"))
    aggregation = aggregate_reference_panel_ranks(
        output_dir=output_dir / "aggregation",
        source_rank_file=source_run["ranking_csv"],
        reference_rank_files=[item["ranking_csv"] for item in reference_runs],
        reference_ids=[path.stem for path in prepared_references],
        panel_size=args.panel_size,
        source_weight=args.source_weight,
        reference_weight=1.0 - args.source_weight,
        restrict_gene_symbols=gene_universe,
        gene_universe="source",
    )
    integrated_panel = Path(aggregation["integrated_top_panel_tsv"])
    evaluation = evaluate(
        merfish_file,
        [("source", source_panel), ("multi_reference", integrated_panel)],
        output_dir / "evaluation",
        args.panel_size,
        args.merfish_label_column,
        args.seed,
    )
    manifest = {
        "workflow": "05_agent",
        "configuration": vars(args),
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in [source_file, merfish_file, *reference_files]],
        "preparation": preparation,
        "source_training": source_run,
        "reference_training": reference_runs,
        "aggregation": aggregation,
        "evaluation": evaluation,
        "probe_backend": {
            "status": "not_run",
            "reason": "ODT/OligoMiner/ProbeDealer require separately installed external backends and reference indexes.",
            "next_step": "Run the probe-feasibility scripts only after configuring those backends; no pass rate is inferred here.",
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run source and multi-reference SMITH-Agent panel evaluation.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", default="liver_merfish/adata_healthy_nucseq.h5ad")
    parser.add_argument("--merfish", default="liver_merfish/adata_healthy_merfish.h5ad")
    parser.add_argument("--reference", action="append", default=None, help="Repeat for two or more spatial references.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells", type=int, default=10000)
    parser.add_argument("--source-label-column", default=None)
    parser.add_argument("--reference-label-column", default=None)
    parser.add_argument("--merfish-label-column", default="Cell_Type")
    parser.add_argument("--source-weight", type=float, default=0.55)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.reference = args.reference or list(DEFAULT_REFERENCES)
    if len(args.reference) < 2:
        parser.error("Agent tutorial requires at least two --reference inputs for multi-reference selection.")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
