from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    adata = ad.read_h5ad(adata_file, backed="r")
    if basis not in adata.obsm:
        raise KeyError(f"Embedding `{basis}` not found in obsm.")
    coords = np.asarray(adata.obsm[basis])
    if coords.shape[1] < 2:
        raise ValueError(f"Embedding `{basis}` must have at least 2 columns.")
    indices = _sample_indices(adata.n_obs, max_cells=max_cells, seed=seed)
    x = coords[indices, 0]
    y = coords[indices, 1]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if color and color in adata.obs.columns:
        series = adata.obs[color].iloc[indices]
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

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
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
