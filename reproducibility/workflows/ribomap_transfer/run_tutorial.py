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

from reproducibility.workflows.common import clean_gene, gene_symbols, run_smith, sha256, write_json
from reproducibility.workflows.ribomap_transfer.evaluate_outputs import evaluate_panel


DEFAULT_SOURCE = "deep_brain_ribomap.h5ad"
DEFAULT_TARGETS = ("mouse_brain_ribomap_rep1.h5ad", "mouse_brain_ribomap_rep2.h5ad")


def _first_labels(adata: ad.AnnData) -> str:
    for column in ("celltype", "cell_type", "Cell_Type", "subclass", "region"):
        if column in adata.obs:
            return column
    raise KeyError("RIBOMap source does not contain a supported cell-type or region label.")


def _prepare_shared(source_file: Path, target_files: list[Path], output_dir: Path) -> tuple[Path, list[Path], list[str]]:
    source = ad.read_h5ad(source_file)
    targets = [ad.read_h5ad(path) for path in target_files]
    source_symbols = gene_symbols(source)
    target_sets = [set(gene_symbols(target)) for target in targets]
    shared = [gene for gene in source_symbols if gene and all(gene in values for values in target_sets)]
    shared = list(dict.fromkeys(shared))
    if not shared:
        raise ValueError("No genes are shared by the source and target RIBOMap datasets.")
    output_dir.mkdir(parents=True, exist_ok=True)

    def subset(adata: ad.AnnData, path: Path, source_labels: bool = False) -> Path:
        positions = {gene: index for index, gene in enumerate(gene_symbols(adata)) if gene}
        prepared = adata[:, [positions[gene] for gene in shared]].copy()
        prepared.var_names = shared
        prepared.var = pd.DataFrame(index=pd.Index(shared))
        if source_labels:
            label = _first_labels(prepared)
            prepared.obs["celltype"] = prepared.obs[label].astype(str)
        prepared.write_h5ad(path)
        return path

    prepared_source = subset(source, output_dir / "source_shared_genes.h5ad", source_labels=True)
    prepared_targets = [
        subset(target, output_dir / f"target_{index + 1}_{target_files[index].stem}.h5ad")
        for index, target in enumerate(targets)
    ]
    return prepared_source, prepared_targets, shared


def _variance_panel(source_file: Path, panel_size: int, output_path: Path) -> Path:
    adata = ad.read_h5ad(source_file)
    matrix = adata.X
    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_square = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        scores = mean_square - np.square(mean)
    else:
        scores = np.var(np.asarray(matrix), axis=0)
    order = np.argsort(scores)[::-1]
    frame = pd.DataFrame({"marker": adata.var_names[order].astype(str), "variance": scores[order]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.head(panel_size).to_csv(output_path, index=False)
    return output_path


def run(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_root).resolve() / "ribomap_transfer" / "ribomap"
    source_file = data_dir / args.source
    target_files = [data_dir / name for name in args.target]
    missing = [str(path) for path in [source_file, *target_files] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing tutorial input files: " + ", ".join(missing))
    output_dir = Path(args.output_dir).resolve()
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_source, prepared_targets, shared = _prepare_shared(
        source_file, target_files, output_dir / "prepared_data"
    )
    training = run_smith(
        adata_file=prepared_source,
        output_dir=output_dir / "smith",
        tasks="recon,cls",
        task_name="ribomap_deep_to_mouse",
        panel_size=args.panel_size,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        max_cells=args.max_cells,
        sampling_strategy="celltype",
        force=args.force,
    )
    smith_panel = Path(training.get("panel_csv") or (output_dir / "smith" / f"panel_top{args.panel_size}.csv"))
    variance_panel = _variance_panel(prepared_source, args.panel_size, output_dir / "variance" / f"panel_top{args.panel_size}.csv")

    rows = []
    detailed = []
    for target in prepared_targets:
        for method, panel in (("SMITH", smith_panel), ("variance", variance_panel)):
            result = evaluate_panel(target, panel, args.panel_size, args.seed, args.label_column)
            detailed.append({"method": method, **result})
            rows.extend(
                {"dataset": result["dataset"], "method": method, "metric": metric, "value": value}
                for metric, value in result["metrics"].items()
            )
    metrics_path = output_dir / "evaluation" / "metrics.tsv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_path, sep="\t", index=False)
    write_json(output_dir / "evaluation" / "metrics.json", {"results": detailed})
    manifest = {
        "workflow": "03_ribomap_transfer",
        "configuration": vars(args),
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [source_file, *target_files]
        ],
        "shared_gene_count": len(shared),
        "training": training,
        "outputs": {
            "smith_panel": str(smith_panel),
            "variance_panel": str(variance_panel),
            "metrics_tsv": str(metrics_path),
            "metrics_json": str(output_dir / "evaluation" / "metrics.json"),
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RIBOMap transfer from shared-gene preparation through evaluation.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--target", action="append", default=None, help="Repeat for multiple target H5AD files.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.target = args.target or list(DEFAULT_TARGETS)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
