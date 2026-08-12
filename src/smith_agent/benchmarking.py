from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from smith_agent.utils import ensure_dir, write_json


GENE_SYMBOL_COLUMNS = (
    "gene_symbol",
    "gene_symbols",
    "gene_name",
    "gene_names",
    "feature_name",
    "gene_short_name",
    "symbol",
)


def _clean_gene_symbol(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text.upper()


def _gene_symbols(adata: ad.AnnData) -> pd.Index:
    for column in GENE_SYMBOL_COLUMNS:
        if column in adata.var.columns:
            return pd.Index(adata.var[column].astype(str).map(_clean_gene_symbol))
    return pd.Index(adata.var_names.astype(str).map(_clean_gene_symbol))


def _matrix(adata: ad.AnnData, gene_symbols: list[str]) -> tuple[np.ndarray, list[str]]:
    index = _gene_symbols(adata)
    positions: list[int] = []
    shared: list[str] = []
    for gene in gene_symbols:
        matches = np.where(index == gene)[0]
        if matches.size:
            positions.append(int(matches[0]))
            shared.append(gene)
    if not shared:
        raise ValueError("No shared genes between panel and dataset.")
    x = adata.X[:, positions]
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32), shared


def _coordinates(adata: ad.AnnData) -> np.ndarray:
    if "X_spatial" in adata.obsm:
        return np.asarray(adata.obsm["X_spatial"], dtype=np.float32)
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], dtype=np.float32)
    lowered = {str(col).lower(): col for col in adata.obs.columns}
    for x_col, y_col in (("x", "y"), ("array_col", "array_row")):
        if x_col in lowered and y_col in lowered:
            coords = adata.obs[[lowered[x_col], lowered[y_col]]].astype(float).to_numpy(dtype=np.float32)
            return coords
    raise KeyError("Dataset does not expose spatial coordinates in obsm['X_spatial'], obsm['spatial'], or obs x/y columns.")


def _load_panel_genes(path: str | Path, panel_size: int | None = None) -> list[str]:
    panel_path = Path(path)
    suffix = panel_path.suffix.lower()
    if suffix in {".csv", ".tsv", ".tab"}:
        sep = "\t" if suffix in {".tsv", ".tab"} else ","
        df = pd.read_csv(panel_path, sep=sep)
        gene_column = next((col for col in df.columns if str(col).lower() in {"gene_symbol", "gene", "target"}), df.columns[0])
        genes = df[gene_column].astype(str).tolist()
    else:
        genes = [line.strip().split(",")[0] for line in panel_path.read_text().splitlines() if line.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        cleaned = _clean_gene_symbol(gene)
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    if panel_size is not None:
        out = out[:panel_size]
    return out


@dataclass
class CoordinateRegressionResult:
    panel_path: str
    panel_size: int
    train_cells: int
    test_cells: int
    shared_genes: list[str]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_path": self.panel_path,
            "panel_size": self.panel_size,
            "train_cells": self.train_cells,
            "test_cells": self.test_cells,
            "shared_genes": self.shared_genes,
            "metrics": self.metrics,
        }


@dataclass
class CellTypeClassificationResult:
    panel_path: str
    panel_size: int
    label_column: str
    train_cells: int
    test_cells: int
    shared_genes: list[str]
    classes: list[str]
    metrics: dict[str, float]
    classification_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_path": self.panel_path,
            "panel_size": self.panel_size,
            "label_column": self.label_column,
            "train_cells": self.train_cells,
            "test_cells": self.test_cells,
            "shared_genes": self.shared_genes,
            "classes": self.classes,
            "metrics": self.metrics,
            "classification_report": self.classification_report,
        }


def evaluate_panel_coordinate_regression(
    adata_file: str | Path,
    panel_path: str | Path,
    output_dir: str | Path,
    panel_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
    alpha: float = 1.0,
) -> CoordinateRegressionResult:
    adata = ad.read_h5ad(adata_file)
    panel_genes = _load_panel_genes(panel_path, panel_size=panel_size)
    x, shared = _matrix(adata, panel_genes)
    coords = _coordinates(adata)
    if x.shape[0] != coords.shape[0]:
        raise ValueError("Expression matrix and coordinate array have incompatible shapes.")

    indices = np.arange(x.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(x[train_idx], coords[train_idx])
    pred = model.predict(x[test_idx])

    x_residual = coords[test_idx] - pred
    mse = float(mean_squared_error(coords[test_idx], pred))
    mae = float(mean_absolute_error(coords[test_idx], pred))
    euclidean = np.sqrt(np.sum(np.square(x_residual), axis=1))
    dist_mae = float(np.mean(euclidean))
    dist_median = float(np.median(euclidean))
    pred_flat = pred.reshape(-1)
    truth_flat = coords[test_idx].reshape(-1)
    if np.std(pred_flat) > 0 and np.std(truth_flat) > 0:
        corr = float(np.corrcoef(pred_flat, truth_flat)[0, 1])
    else:
        corr = float("nan")

    metrics = {
        "spatial_mse": mse,
        "spatial_mae": mae,
        "spatial_distance_mae": dist_mae,
        "spatial_distance_median": dist_median,
        "spatial_pearson": corr,
    }

    out_dir = ensure_dir(output_dir)
    payload = CoordinateRegressionResult(
        panel_path=str(panel_path),
        panel_size=int(panel_size),
        train_cells=int(train_idx.size),
        test_cells=int(test_idx.size),
        shared_genes=shared,
        metrics=metrics,
    )
    write_json(out_dir / "coordinate_regression_result.json", payload.to_dict())
    summary = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in metrics.items()
        ]
    )
    summary.to_csv(out_dir / "coordinate_regression_summary.tsv", sep="\t", index=False)
    return payload


def evaluate_panel_cell_type_classification(
    adata_file: str | Path,
    panel_path: str | Path,
    output_dir: str | Path,
    panel_size: int = 64,
    label_column: str = "Cell_Type",
    test_size: float = 0.2,
    seed: int = 42,
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
) -> CellTypeClassificationResult:
    adata = ad.read_h5ad(adata_file)
    if label_column not in adata.obs.columns:
        raise KeyError(f"Dataset does not contain label column `{label_column}`.")

    panel_genes = _load_panel_genes(panel_path, panel_size=panel_size)
    x, shared = _matrix(adata, panel_genes)
    labels = adata.obs[label_column].astype(str).to_numpy()
    valid = pd.notna(labels) & (labels != "") & (labels != "nan")
    x = x[valid]
    labels = labels[valid]
    if len(np.unique(labels)) < 2:
        raise ValueError(f"Need at least two classes in `{label_column}` for classification.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    indices = np.arange(x.shape[0])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            n_jobs=1,
            random_state=seed,
            solver="lbfgs",
        ),
    )
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])

    metrics = {
        "cell_type_accuracy": float(accuracy_score(y[test_idx], pred)),
        "cell_type_balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
        "cell_type_macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
        "cell_type_weighted_f1": float(f1_score(y[test_idx], pred, average="weighted")),
    }
    report = classification_report(
        y[test_idx],
        pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=[str(item) for item in encoder.classes_],
        output_dict=True,
        zero_division=0,
    )

    out_dir = ensure_dir(output_dir)
    payload = CellTypeClassificationResult(
        panel_path=str(panel_path),
        panel_size=int(panel_size),
        label_column=label_column,
        train_cells=int(train_idx.size),
        test_cells=int(test_idx.size),
        shared_genes=shared,
        classes=[str(item) for item in encoder.classes_],
        metrics=metrics,
        classification_report=report,
    )
    write_json(out_dir / "cell_type_classification_result.json", payload.to_dict())
    summary = pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])
    summary.to_csv(out_dir / "cell_type_classification_summary.tsv", sep="\t", index=False)
    pd.DataFrame(report).transpose().to_csv(out_dir / "cell_type_classification_report.tsv", sep="\t")
    return payload
