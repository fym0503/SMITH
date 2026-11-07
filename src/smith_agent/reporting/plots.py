from __future__ import annotations

from pathlib import Path
import json

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

REPORT_EMBEDDING_MAX_CELLS = 2000
REPORT_EMBEDDING_MAX_FEATURES = 2000


def _report_embedding(adata_file, basis, max_cells, seed):
    """Read/sample before computing a report-only embedding; never edit input."""
    backed = ad.read_h5ad(adata_file, backed="r")
    try:
        existing = basis in backed.obsm
        embedding_cell_limit = max_cells if existing else min(max_cells or REPORT_EMBEDDING_MAX_CELLS, REPORT_EMBEDDING_MAX_CELLS)
        indices = _sample_indices(backed.n_obs, embedding_cell_limit, seed)
        obs = backed.obs.iloc[indices].copy()
        metadata = {
            "source_file": str(Path(adata_file).resolve()), "seed": seed,
            "n_source_cells": backed.n_obs, "n_plotted_cells": len(indices),
            "requested_basis": basis, "generated": not existing,
            "scope": "report visualization only; independent per dataset",
            "embedding_cell_limit": embedding_cell_limit,
        }
        if existing:
            return np.asarray(backed.obsm[basis])[indices], obs, metadata
        if basis != "X_umap":
            raise KeyError(f"Embedding `{basis}` not found in obsm.")
        if len(indices) == 0 or backed.n_vars == 0:
            raise ValueError("Cannot visualize an empty expression matrix.")
        # Materialize only the cells to plot. Labels and target outcomes are
        # never used to calculate the embedding or alter the trained panel.
        n_features = min(backed.n_vars, REPORT_EMBEDDING_MAX_FEATURES)
        if "highly_variable" in backed.var:
            variable = np.flatnonzero(backed.var["highly_variable"].to_numpy(dtype=bool))
            feature_indices = variable[:n_features] if len(variable) else np.array([], dtype=int)
        else:
            feature_indices = np.array([], dtype=int)
        feature_selection = "highly_variable" if len(feature_indices) else "uniform_index_spacing"
        if not len(feature_indices):
            feature_indices = np.linspace(0, backed.n_vars - 1, n_features, dtype=int)
        sampled = backed[indices.tolist(), :].to_memory()
        values = sampled[:, feature_indices.tolist()].X
        metadata["n_features"] = len(feature_indices)
        metadata["feature_selection"] = feature_selection
    finally:
        backed.file.close()

    values = values.tocsr().astype(np.float32) if sparse.issparse(values) else np.asarray(values, dtype=np.float32)
    data_values = values.data if sparse.issparse(values) else values
    if not np.isfinite(data_values).all():
        raise ValueError("Report expression matrix contains non-finite values.")
    components = min(30, values.shape[0] - 1, values.shape[1] - 1)
    if components > 1:
        reducer = TruncatedSVD(n_components=components, random_state=seed) if sparse.issparse(values) else PCA(n_components=components, random_state=seed)
        reduced = reducer.fit_transform(values)
        metadata["preprocessing"] = type(reducer).__name__
    else:
        reduced = values.toarray() if sparse.issparse(values) else values
        metadata["preprocessing"] = "expression"
    if len(indices) >= 4:
        from umap import UMAP
        neighbors = min(15, len(indices) - 1)
        coords = UMAP(n_neighbors=neighbors, n_components=2, random_state=seed,
                      init="random", n_jobs=1).fit_transform(reduced)
        metadata.update(method="UMAP", n_neighbors=neighbors)
    else:
        # UMAP is undefined for these tiny inputs. Label this diagnostic
        # projection explicitly instead of calling padded values UMAP.
        coords = np.zeros((len(indices), 2))
        coords[:, :min(2, reduced.shape[1])] = reduced[:, :2]
        metadata["method"] = "expression projection (fewer than four cells)"
    metadata["n_features"] = values.shape[1]
    return coords, obs, metadata


def _sample_indices(n_obs: int, max_cells: int | None, seed: int = 42) -> np.ndarray:
    if max_cells is None or n_obs <= max_cells:
        return np.arange(n_obs)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_obs, size=max_cells, replace=False))


def plot_dataset_umap(
    adata_file: str | Path,
    out_path: str | Path,
    color: str | None = None,
    basis: str = "X_umap",
    max_cells: int = 50000,
    seed: int = 42,
    title: str | None = None,
) -> str:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    coords, obs, metadata = _report_embedding(adata_file, basis, max_cells, seed)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Embedding `{basis}` must have at least 2 columns.")
    x = coords[:, 0]
    y = coords[:, 1]
    pd.DataFrame({"cell_id": obs.index.astype(str), "embedding_1": x, "embedding_2": y}).to_csv(
        output.with_suffix(".embedding.csv"), index=False,
    )
    output.with_suffix(".embedding.json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if color and color in obs.columns:
        series = obs[color]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == len(series):
            scatter = ax.scatter(x, y, c=numeric.to_numpy(), s=5, alpha=0.7, cmap="viridis", linewidths=0)
            fig.colorbar(scatter, ax=ax, label=color)
        else:
            categories = series.astype(str).fillna("NA")
            unique = list(pd.Index(categories).unique())
            cmap = plt.get_cmap("tab20")
            for idx, category in enumerate(unique):
                mask = categories == category
                ax.scatter(
                    x[mask.to_numpy()],
                    y[mask.to_numpy()],
                    s=5,
                    alpha=0.7,
                    color=cmap(idx % 20),
                    linewidths=0,
                    label=category,
                )
            if len(unique) <= 12:
                ax.legend(loc="best", frameon=False, fontsize=8)
    else:
        ax.scatter(x, y, s=5, alpha=0.65, color="#4c6a92", linewidths=0)

    axis_prefix = "Projection" if str(metadata.get("method", "")).startswith("expression projection") else ("UMAP" if basis == "X_umap" else basis)
    ax.set_xlabel(f"{axis_prefix}1")
    ax.set_ylabel(f"{axis_prefix}2")
    ax.set_title(title or f"{Path(adata_file).stem} {basis}")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return str(output)


def plot_evaluation_summary(
    evaluation_csv: str | Path,
    out_path: str | Path,
    title: str = "Cross-Dataset Evaluation Summary",
) -> str:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(evaluation_csv)
    if df.empty:
        raise ValueError("Evaluation table is empty.")

    labels = [f"{row['evaluation']}\n{row['metric']}" for _, row in df.iterrows()]
    values = df["value"].astype(float).to_numpy()
    colors = ["#1f6f78", "#d98e04", "#8f3b76", "#4f772d", "#6c757d"][: len(values)]

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(values)), 4.5))
    bars = ax.bar(labels, values, color=colors[: len(values)], alpha=0.9)
    ax.axhline(0.0, color="#7f8c8d", linewidth=1)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        y = value if value >= 0 else 0
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.3f}", ha="center", va=va, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return str(output)
