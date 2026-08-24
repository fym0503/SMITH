#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from reproducibility.workflows.common import gene_symbols, read_panel, write_json


LABEL_COLUMNS = ("celltype", "cell_type", "Cell_Type", "subclass", "region")


def prepare_shared_adata(source: ad.AnnData, target: ad.AnnData) -> ad.AnnData:
    """Return a shared-gene source view using the target's measured gene universe."""
    target_genes = set(gene_symbols(target))
    positions = {gene: index for index, gene in enumerate(gene_symbols(source)) if gene}
    shared = [gene for gene in positions if gene in target_genes]
    if not shared:
        raise ValueError("Source and target have no shared gene symbols")
    prepared = source[:, [positions[gene] for gene in shared]].copy()
    prepared.var_names = shared
    prepared.var = pd.DataFrame(index=pd.Index(shared))
    cell_column = next((name for name in ("celltype", "cell_type", "Cell_Type", "subclass") if name in prepared.obs), None)
    region_column = next((name for name in ("region", "Region", "spatial_region", "cluster") if name in prepared.obs), None)
    if cell_column:
        prepared.obs["celltype"] = prepared.obs[cell_column].astype(str)
    if region_column:
        prepared.obs["region"] = prepared.obs[region_column].astype(str)
    return prepared


def evaluate_panel_loaded(
    adata: ad.AnnData,
    panel_genes: list[str],
    panel_size: int,
    seed: int,
    label_column: str | None = None,
    output_dir: str | Path | None = None,
    neighbors: int = 5,
) -> tuple[dict, pd.DataFrame]:
    column = label_column or next((name for name in LABEL_COLUMNS if name in adata.obs), None)
    if not column:
        raise KeyError(f"No evaluation label found; tried {LABEL_COLUMNS}")
    panel = list(panel_genes[:panel_size])
    index = {gene: position for position, gene in enumerate(gene_symbols(adata)) if gene}
    shared = [gene for gene in panel if gene in index]
    if not shared:
        raise ValueError("No panel genes overlap the evaluation dataset")
    matrix = adata.X[:, [index[gene] for gene in shared]]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    x = StandardScaler().fit_transform(np.asarray(matrix, dtype=np.float32))
    labels = adata.obs[column].astype(str).to_numpy()
    n_classes = len(np.unique(labels))
    if n_classes < 2 or n_classes >= len(labels):
        raise ValueError("Need at least two classes and more observations than classes for evaluation.")
    # A stratified holdout must contain at least one observation per class.
    test_fraction = max(0.2, n_classes / len(labels))
    train, test = train_test_split(
        np.arange(adata.n_obs), test_size=test_fraction, random_state=seed, stratify=labels
    )
    n_neighbors = max(1, min(neighbors, len(train)))
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    classifier.fit(x[train], labels[train])
    prediction = classifier.predict(x[test])
    result = {
        "label_column": column,
        "panel_size_requested": int(panel_size),
        "panel_size_evaluated": len(shared),
        "shared_genes": shared,
        "evaluation_seed": int(seed),
        "knn_neighbors": n_neighbors,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "metrics": {
            "accuracy": float(accuracy_score(labels[test], prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(labels[test], prediction)),
            "macro_f1": float(f1_score(labels[test], prediction, average="macro", zero_division=0)),
        },
    }
    predictions = pd.DataFrame({"cell_index": np.asarray(adata.obs_names)[test], "truth": labels[test], "prediction": prediction})
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "metrics.json", result)
        pd.DataFrame([{"metric": key, "value": value} for key, value in result["metrics"].items()]).to_csv(
            output_dir / "metrics.tsv", sep="\t", index=False
        )
        predictions.to_csv(output_dir / "predictions.tsv", sep="\t", index=False)
    return result, predictions


def evaluate_panel(
    adata_file: str | Path,
    panel_file: str | Path,
    panel_size: int,
    seed: int,
    label_column: str | None = None,
    output_dir: str | Path | None = None,
    neighbors: int = 5,
) -> dict:
    """File-based wrapper around :func:`evaluate_panel_loaded`."""
    adata = ad.read_h5ad(adata_file)
    result, _ = evaluate_panel_loaded(
        adata, read_panel(panel_file, panel_size), panel_size, seed,
        label_column=label_column, output_dir=output_dir, neighbors=neighbors,
    )
    result.update({"dataset": Path(adata_file).stem, "panel": Path(panel_file).stem})
    if output_dir is not None:
        write_json(Path(output_dir) / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a newly selected panel on a RIBOMap target dataset.")
    parser.add_argument("--adata-file", required=True)
    parser.add_argument("--panel-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--panel-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()
    result = evaluate_panel(args.adata_file, args.panel_file, args.panel_size, args.seed,
                            args.label_column, args.output_dir, args.neighbors)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
