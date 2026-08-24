#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from reproducibility.workflows.common import gene_symbols, read_panel, write_json


CELLTYPE_COLUMNS = ("celltype", "cell_type", "Cell_Type", "subclass")
TIME_COLUMNS = ("absolute_time", "consensus_time", "time_label", "time")


def _column(adata: ad.AnnData, candidates: tuple[str, ...], requested: str | None = None) -> str:
    choices = ([requested] if requested else []) + list(candidates)
    for column in choices:
        if column and column in adata.obs:
            return column
    raise KeyError(f"None of the required obs columns exists: {choices}")


def _shared_matrices(train: ad.AnnData, test: ad.AnnData, panel: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_map = {gene: index for index, gene in enumerate(gene_symbols(train)) if gene}
    test_map = {gene: index for index, gene in enumerate(gene_symbols(test)) if gene}
    shared = [gene for gene in panel if gene in train_map and gene in test_map]
    if not shared:
        raise ValueError("No panel genes are shared by train and test data.")
    x_train = train.X[:, [train_map[gene] for gene in shared]]
    x_test = test.X[:, [test_map[gene] for gene in shared]]
    if sparse.issparse(x_train):
        x_train = x_train.toarray()
    if sparse.issparse(x_test):
        x_test = x_test.toarray()
    return np.asarray(x_train, dtype=np.float32), np.asarray(x_test, dtype=np.float32), shared


def evaluate_loaded(
    train: ad.AnnData,
    test: ad.AnnData,
    panel_genes: list[str],
    panel_size: int,
    time_column: str | None = None,
    neighbors: int = 5,
    output_dir: str | Path | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate an in-memory panel without re-reading any generated artifact."""
    panel = list(panel_genes[:panel_size])
    x_train, x_test, shared = _shared_matrices(train, test, panel)
    n_neighbors = max(1, min(neighbors, len(x_train)))

    train_celltype = _column(train, CELLTYPE_COLUMNS)
    test_celltype = _column(test, CELLTYPE_COLUMNS)
    y_train = train.obs[train_celltype].astype(str).to_numpy()
    y_test = test.obs[test_celltype].astype(str).to_numpy()
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(x_train, y_train)
    predicted_celltype = classifier.predict(x_test)

    train_time = _column(train, TIME_COLUMNS, time_column)
    test_time = _column(test, TIME_COLUMNS, time_column)
    t_train = pd.to_numeric(train.obs[train_time], errors="raise").to_numpy(dtype=np.float32)
    t_test = pd.to_numeric(test.obs[test_time], errors="raise").to_numpy(dtype=np.float32)
    regressor = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    regressor.fit(x_train, t_train)
    predicted_time = regressor.predict(x_test)
    pearson = float(np.corrcoef(t_test, predicted_time)[0, 1]) if np.std(t_test) and np.std(predicted_time) else float("nan")

    metrics = {
        "cell_type_accuracy": float(accuracy_score(y_test, predicted_celltype)),
        "cell_type_balanced_accuracy": float(balanced_accuracy_score(y_test, predicted_celltype)),
        "cell_type_macro_f1": float(f1_score(y_test, predicted_celltype, average="macro", zero_division=0)),
        "developmental_time_pearson": pearson,
        "developmental_time_mae": float(mean_absolute_error(t_test, predicted_time)),
    }
    payload = {
        "panel_size_requested": int(panel_size),
        "panel_size_evaluated": len(shared),
        "shared_genes": shared,
        "celltype_columns": [train_celltype, test_celltype],
        "time_columns": [train_time, test_time],
        "metrics": metrics,
        "knn_neighbors": n_neighbors,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
    time_predictions = pd.DataFrame({"truth": t_test, "prediction": predicted_time})
    celltype_predictions = pd.DataFrame({"truth": y_test, "prediction": predicted_celltype})
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "metrics.json", payload)
        pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()]).to_csv(
            output_dir / "metrics.tsv", sep="\t", index=False
        )
        time_predictions.to_csv(output_dir / "developmental_time_predictions.tsv", sep="\t", index=False)
        celltype_predictions.to_csv(output_dir / "cell_type_predictions.tsv", sep="\t", index=False)
    return payload, celltype_predictions, time_predictions


def evaluate(
    train_file: str | Path,
    test_file: str | Path,
    panel_file: str | Path,
    output_dir: str | Path,
    panel_size: int,
    time_column: str | None = None,
    neighbors: int = 5,
) -> dict:
    """File-based CLI wrapper around :func:`evaluate_loaded`."""
    train = ad.read_h5ad(train_file)
    test = ad.read_h5ad(test_file)
    panel = read_panel(panel_file, panel_size)
    payload, _, _ = evaluate_loaded(
        train, test, panel, panel_size, time_column=time_column, neighbors=neighbors, output_dir=output_dir
    )
    payload.update(
        {
            "train_file": str(Path(train_file).resolve()),
            "test_file": str(Path(test_file).resolve()),
            "panel_file": str(Path(panel_file).resolve()),
        }
    )
    write_json(Path(output_dir) / "metrics.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a newly generated C. elegans panel on a held-out split.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--panel-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--panel-size", type=int, default=32)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--neighbors", type=int, default=15)
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
