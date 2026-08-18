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


def evaluate_panel(
    adata_file: str | Path,
    panel_file: str | Path,
    panel_size: int,
    seed: int,
    label_column: str | None = None,
    output_dir: str | Path | None = None,
    neighbors: int = 5,
) -> dict:
    adata = ad.read_h5ad(adata_file)
    column = label_column or next((name for name in LABEL_COLUMNS if name in adata.obs), None)
    if not column:
        raise KeyError(f"No evaluation label found in {adata_file}; tried {LABEL_COLUMNS}")
    panel = read_panel(panel_file, panel_size)
    index = {gene: position for position, gene in enumerate(gene_symbols(adata)) if gene}
    shared = [gene for gene in panel if gene in index]
    if not shared:
        raise ValueError(f"No panel genes overlap {adata_file}")
    matrix = adata.X[:, [index[gene] for gene in shared]]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    x = StandardScaler().fit_transform(np.asarray(matrix, dtype=np.float32))
    labels = adata.obs[column].astype(str).to_numpy()
    train, test = train_test_split(
        np.arange(adata.n_obs), test_size=0.2, random_state=seed, stratify=labels
    )
    n_neighbors = max(1, min(neighbors, len(train)))
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    classifier.fit(x[train], labels[train])
    prediction = classifier.predict(x[test])
    result = {
        "dataset": Path(adata_file).stem,
        "panel": Path(panel_file).stem,
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
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "metrics.json", result)
        pd.DataFrame([{"metric": key, "value": value} for key, value in result["metrics"].items()]).to_csv(
            output_dir / "metrics.tsv", sep="\t", index=False
        )
        pd.DataFrame({"cell_index": np.asarray(adata.obs_names)[test], "truth": labels[test],
                      "prediction": prediction}).to_csv(
            output_dir / "predictions.tsv", sep="\t", index=False
        )
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
