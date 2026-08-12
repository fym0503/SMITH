from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from umap import UMAP


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPARED_DIR = REPO_ROOT / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_5seed_panels/prepared"
DEFAULT_MERFISH = REPO_ROOT / "data/liver_merfish/adata_healthy_merfish.h5ad"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_5seed_panels/diagnostics"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures/agent_visium_retrieval_refined"
ARIAL_FONT_FILES = [
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
]

DATASETS = [
    {
        "dataset_id": "snRNA",
        "display_name": "snRNA",
        "modality": "snRNA",
        "path": DEFAULT_PREPARED_DIR / "source_scrna_smith_input.h5ad",
        "max_obs": 6000,
        "color": "#BDBDBD",
        "zorder": 1,
    },
    {
        "dataset_id": "PSC011_C1_visium",
        "display_name": "Visium PSC011",
        "modality": "Visium",
        "path": DEFAULT_PREPARED_DIR / "PSC011_C1_visium_smith_input.h5ad",
        "max_obs": 2500,
        "color": "#7E9F5D",
        "zorder": 3,
    },
    {
        "dataset_id": "C73_B1_healthy_visium",
        "display_name": "Visium C73",
        "modality": "Visium",
        "path": DEFAULT_PREPARED_DIR / "C73_B1_healthy_visium_smith_input.h5ad",
        "max_obs": 2500,
        "color": "#D8A34A",
        "zorder": 3,
    },
    {
        "dataset_id": "H35_sample1_visium",
        "display_name": "Visium H35-1",
        "modality": "Visium",
        "path": DEFAULT_PREPARED_DIR / "H35_sample1_visium_smith_input.h5ad",
        "max_obs": 2500,
        "color": "#65A7C2",
        "zorder": 3,
    },
    {
        "dataset_id": "WSSS_F_IMMsp9838712_visium",
        "display_name": "Visium WSSS",
        "modality": "Visium",
        "path": DEFAULT_PREPARED_DIR / "WSSS_F_IMMsp9838712_visium_smith_input.h5ad",
        "max_obs": 2500,
        "color": "#B77AA8",
        "zorder": 3,
    },
    {
        "dataset_id": "H35_sample2_visium",
        "display_name": "Visium H35-2",
        "modality": "Visium",
        "path": DEFAULT_PREPARED_DIR / "H35_sample2_visium_smith_input.h5ad",
        "max_obs": 2500,
        "color": "#8D7CC0",
        "zorder": 3,
    },
    {
        "dataset_id": "MERFISH",
        "display_name": "MERFISH",
        "modality": "MERFISH",
        "path": DEFAULT_MERFISH,
        "max_obs": 6000,
        "color": "#222222",
        "zorder": 4,
    },
]


def _configure_matplotlib() -> None:
    for font_file in ARIAL_FONT_FILES:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    mpl.rcParams.update(
        {
            "font.family": "Arial" if "Arial" in available_fonts else "DejaVu Sans",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
            "legend.frameon": False,
        }
    )


def _gene_index(path: Path) -> dict[str, int]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        index: dict[str, int] = {}
        for idx, gene in enumerate(adata.var_names.astype(str).tolist()):
            clean = str(gene).strip().upper()
            if clean and clean not in index:
                index[clean] = idx
        return index
    finally:
        adata.file.close()


def _shared_genes(dataset_specs: list[dict[str, Any]]) -> list[str]:
    gene_sets = []
    for spec in dataset_specs:
        index = _gene_index(Path(spec["path"]))
        gene_sets.append(set(index))
    shared = sorted(set.intersection(*gene_sets))
    if len(shared) < 20:
        raise ValueError(f"Only {len(shared)} shared genes were found; cannot build a stable shared-gene PCA.")
    return shared


def _sample_indices(n_obs: int, max_obs: int, seed: int) -> np.ndarray:
    if max_obs <= 0 or n_obs <= max_obs:
        return np.arange(n_obs, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_obs, size=int(max_obs), replace=False))


def _matrix_subset(adata: ad.AnnData, obs_indices: np.ndarray, var_positions: list[int]) -> np.ndarray:
    x = adata[obs_indices, var_positions].X
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _normalise_expression_matrix(x: np.ndarray, target_sum: float = 1e4) -> tuple[np.ndarray, str]:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.nanmin(x)) < 0:
        # Several public Visium h5ad files expose transformed residual-like matrices.
        # Re-normalizing those as counts would create invalid log values.
        return x, "already_transformed"
    totals = x.sum(axis=1, keepdims=True)
    x = np.divide(x, np.maximum(totals, 1e-8), out=np.zeros_like(x, dtype=np.float32), where=totals > 0)
    x = x * float(target_sum)
    return np.log1p(x, dtype=np.float32), "library_size_log1p"


def load_sampled_matrix(
    dataset_specs: list[dict[str, Any]],
    shared_genes: list[str],
    *,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    matrices = []
    obs_rows = []
    for spec_idx, spec in enumerate(dataset_specs):
        path = Path(spec["path"])
        gene_index = _gene_index(path)
        var_positions = [gene_index[gene] for gene in shared_genes]
        adata = ad.read_h5ad(path)
        try:
            obs_indices = _sample_indices(adata.n_obs, int(spec["max_obs"]), seed + spec_idx * 1009)
            x = _matrix_subset(adata, obs_indices, var_positions)
            x, transform = _normalise_expression_matrix(x)
            labels = pd.Series(index=adata.obs_names[obs_indices], dtype=str)
            for column in ("cell_type", "Cell_Type", "Cell_Type_final"):
                if column in adata.obs.columns:
                    labels = adata.obs.iloc[obs_indices][column].astype(str)
                    break
            rows = pd.DataFrame(
                {
                    "dataset_id": str(spec["dataset_id"]),
                    "display_name": str(spec["display_name"]),
                    "modality": str(spec["modality"]),
                    "obs_name": adata.obs_names[obs_indices].astype(str),
                    "cell_type": labels.to_numpy(dtype=str),
                    "n_obs_full_dataset": int(adata.n_obs),
                    "n_obs_sampled_dataset": int(len(obs_indices)),
                    "expression_transform": transform,
                }
            )
        finally:
            del adata
        matrices.append(x)
        obs_rows.append(rows)
    return np.vstack(matrices), pd.concat(obs_rows, ignore_index=True)


def compute_embedding(
    x: np.ndarray,
    *,
    n_pcs: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x)
    n_components = min(int(n_pcs), x_scaled.shape[1] - 1, x_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
    pcs = pca.fit_transform(x_scaled)
    reducer = UMAP(
        n_neighbors=30,
        min_dist=0.35,
        metric="euclidean",
        random_state=seed,
        init="spectral",
    )
    umap = reducer.fit_transform(pcs[:, : min(30, pcs.shape[1])])
    return pcs, umap, pca.explained_variance_ratio_


def compute_distance_tables(obs: pd.DataFrame, pcs: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = obs.copy()
    centroids = []
    for dataset_id, group in obs.groupby("dataset_id", sort=False):
        idx = group.index.to_numpy()
        centroid = pcs[idx].mean(axis=0)
        centroids.append(
            {
                "dataset_id": dataset_id,
                "display_name": group["display_name"].iloc[0],
                "modality": group["modality"].iloc[0],
                "n_obs": int(group.shape[0]),
                **{f"PC{i + 1}": float(value) for i, value in enumerate(centroid)},
            }
        )
    centroid_df = pd.DataFrame(centroids)
    merfish = centroid_df[centroid_df["dataset_id"] == "MERFISH"]
    if merfish.empty:
        raise ValueError("MERFISH dataset is required for distance-to-target analysis.")
    pc_cols = [col for col in centroid_df.columns if col.startswith("PC")]
    merfish_centroid = merfish[pc_cols].to_numpy(dtype=float)
    centroid_df["pca_centroid_distance_to_merfish"] = pairwise_distances(
        centroid_df[pc_cols].to_numpy(dtype=float),
        merfish_centroid,
        metric="euclidean",
    ).ravel()
    cell_distance_rows = []
    for dataset_id, group in obs.groupby("dataset_id", sort=False):
        idx = group.index.to_numpy()
        distances = pairwise_distances(pcs[idx], merfish_centroid, metric="euclidean").ravel()
        cell_distance_rows.append(
            {
                "dataset_id": dataset_id,
                "display_name": group["display_name"].iloc[0],
                "modality": group["modality"].iloc[0],
                "n_obs": int(group.shape[0]),
                "median_cell_distance_to_merfish_centroid": float(np.median(distances)),
                "mean_cell_distance_to_merfish_centroid": float(np.mean(distances)),
                "q25_cell_distance_to_merfish_centroid": float(np.quantile(distances, 0.25)),
                "q75_cell_distance_to_merfish_centroid": float(np.quantile(distances, 0.75)),
            }
        )
    return centroid_df, pd.DataFrame(cell_distance_rows)


def compute_umap_neighbor_tables(obs: pd.DataFrame, *, n_neighbors: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords = obs[["UMAP1", "UMAP2"]].to_numpy(dtype=float)
    labels = obs["dataset_id"].to_numpy(dtype=str)
    merfish_mask = labels == "MERFISH"
    non_merfish_mask = ~merfish_mask
    if not merfish_mask.any() or not non_merfish_mask.any():
        raise ValueError("UMAP neighbor diagnostics require MERFISH and at least one non-MERFISH dataset.")

    k = min(int(n_neighbors), int(non_merfish_mask.sum()))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(coords[non_merfish_mask])
    neighbor_idx = nn.kneighbors(coords[merfish_mask], return_distance=False)
    non_merfish_labels = labels[non_merfish_mask]
    neighbor_labels = non_merfish_labels[neighbor_idx].ravel()
    neighbor_counts = pd.Series(neighbor_labels).value_counts()
    display_map = obs.drop_duplicates("dataset_id").set_index("dataset_id")["display_name"].to_dict()
    modality_map = obs.drop_duplicates("dataset_id").set_index("dataset_id")["modality"].to_dict()
    neighbor_fraction = pd.DataFrame(
        {
            "dataset_id": neighbor_counts.index.astype(str),
            "display_name": [display_map.get(dataset, dataset) for dataset in neighbor_counts.index.astype(str)],
            "modality": [modality_map.get(dataset, "") for dataset in neighbor_counts.index.astype(str)],
            "neighbor_count": neighbor_counts.to_numpy(dtype=int),
            "neighbor_fraction": neighbor_counts.to_numpy(dtype=float) / float(neighbor_counts.sum()),
            "query_dataset": "MERFISH",
            "n_query_cells": int(merfish_mask.sum()),
            "k_non_merfish_neighbors": int(k),
        }
    )

    nn_merfish = NearestNeighbors(n_neighbors=1)
    nn_merfish.fit(coords[merfish_mask])
    rows = []
    for dataset_id, group in obs.groupby("dataset_id", sort=False):
        if dataset_id == "MERFISH":
            continue
        distances = nn_merfish.kneighbors(group[["UMAP1", "UMAP2"]].to_numpy(dtype=float), return_distance=True)[0].ravel()
        rows.append(
            {
                "dataset_id": dataset_id,
                "display_name": group["display_name"].iloc[0],
                "modality": group["modality"].iloc[0],
                "n_obs": int(group.shape[0]),
                "median_umap_distance_to_nearest_merfish": float(np.median(distances)),
                "mean_umap_distance_to_nearest_merfish": float(np.mean(distances)),
                "q25_umap_distance_to_nearest_merfish": float(np.quantile(distances, 0.25)),
                "q75_umap_distance_to_nearest_merfish": float(np.quantile(distances, 0.75)),
            }
        )
    nearest_distance = pd.DataFrame(rows)
    return neighbor_fraction, nearest_distance


def plot_umap(obs: pd.DataFrame, output_prefix: Path, dataset_specs: list[dict[str, Any]]) -> dict[str, str]:
    _configure_matplotlib()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    color_map = {str(spec["dataset_id"]): str(spec["color"]) for spec in dataset_specs}
    zorder_map = {str(spec["dataset_id"]): int(spec["zorder"]) for spec in dataset_specs}
    label_map = {str(spec["dataset_id"]): str(spec["display_name"]) for spec in dataset_specs}
    order = [str(spec["dataset_id"]) for spec in dataset_specs]

    fig, ax = plt.subplots(figsize=(3.0, 2.85), facecolor="white")
    for dataset_id in order:
        group = obs[obs["dataset_id"] == dataset_id]
        if group.empty:
            continue
        ax.scatter(
            group["UMAP1"],
            group["UMAP2"],
            s=2.1 if dataset_id != "MERFISH" else 1.8,
            color=color_map[dataset_id],
            alpha=0.45 if dataset_id in {"snRNA", "MERFISH"} else 0.62,
            linewidth=0,
            rasterized=True,
            zorder=zorder_map[dataset_id],
            label=label_map[dataset_id],
        )
    ax.set_xlabel("UMAP 1", fontsize=8.8)
    ax.set_ylabel("UMAP 2", fontsize=8.8)
    ax.tick_params(axis="both", labelsize=7.6, length=2.5, width=0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=6.3,
        markerscale=3.0,
        handletextpad=0.3,
        borderaxespad=0.0,
        labelspacing=0.45,
    )
    paths = {
        "pdf": str(output_prefix.with_suffix(".pdf")),
        "svg": str(output_prefix.with_suffix(".svg")),
        "png": str(output_prefix.with_suffix(".png")),
    }
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=450, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_umap_neighbor_fraction(neighbor_df: pd.DataFrame, output_prefix: Path, dataset_specs: list[dict[str, Any]]) -> dict[str, str]:
    _configure_matplotlib()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    color_map = {str(spec["dataset_id"]): str(spec["color"]) for spec in dataset_specs}
    plot_df = neighbor_df.copy().sort_values("neighbor_fraction", ascending=True)

    fig, ax = plt.subplots(figsize=(2.4, 2.35), facecolor="white")
    y = np.arange(plot_df.shape[0])
    ax.barh(
        y,
        plot_df["neighbor_fraction"],
        color=[color_map.get(dataset, "#BDBDBD") for dataset in plot_df["dataset_id"]],
        edgecolor="#222222",
        linewidth=0.35,
        height=0.62,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_name"], fontsize=6.8)
    ax.set_xlabel("Fraction of MERFISH nearest\nnon-MERFISH neighbors", fontsize=8.0)
    ax.tick_params(axis="x", labelsize=7.4, length=2.5, width=0.55)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(0.05, float(plot_df["neighbor_fraction"].max()) * 1.08))
    paths = {
        "pdf": str(output_prefix.with_suffix(".pdf")),
        "svg": str(output_prefix.with_suffix(".svg")),
        "png": str(output_prefix.with_suffix(".png")),
    }
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=450, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pcs", type=int, default=50)
    args = parser.parse_args()

    dataset_specs = DATASETS
    for spec in dataset_specs:
        if not Path(spec["path"]).exists():
            raise FileNotFoundError(f"Missing dataset: {spec['path']}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    shared_genes = _shared_genes(dataset_specs)
    x, obs = load_sampled_matrix(dataset_specs, shared_genes, seed=int(args.seed))
    pcs, umap, explained = compute_embedding(x, n_pcs=int(args.n_pcs), seed=int(args.seed))
    obs["UMAP1"] = umap[:, 0]
    obs["UMAP2"] = umap[:, 1]
    obs["PC1"] = pcs[:, 0]
    obs["PC2"] = pcs[:, 1]

    centroid_distance, cell_distance = compute_distance_tables(obs, pcs)
    umap_neighbor_fraction, umap_nearest_distance = compute_umap_neighbor_tables(obs, n_neighbors=30)
    shared_gene_tsv = output_dir / "shared_gene_pca_umap_genes.tsv"
    embedding_tsv = output_dir / "shared_gene_pca_umap_embedding.tsv.gz"
    centroid_tsv = output_dir / "shared_gene_pca_centroid_distance_to_merfish.tsv"
    cell_distance_tsv = output_dir / "shared_gene_pca_cell_distance_to_merfish.tsv"
    umap_neighbor_tsv = output_dir / "shared_gene_umap_merfish_neighbor_fraction.tsv"
    umap_distance_tsv = output_dir / "shared_gene_umap_nearest_merfish_distance.tsv"
    pd.DataFrame({"gene_symbol": shared_genes}).to_csv(shared_gene_tsv, sep="\t", index=False)
    obs.to_csv(embedding_tsv, sep="\t", index=False)
    centroid_distance.to_csv(centroid_tsv, sep="\t", index=False)
    cell_distance.to_csv(cell_distance_tsv, sep="\t", index=False)
    umap_neighbor_fraction.to_csv(umap_neighbor_tsv, sep="\t", index=False)
    umap_nearest_distance.to_csv(umap_distance_tsv, sep="\t", index=False)
    figure_paths = {
        "umap": plot_umap(obs, figure_dir / "07_shared_gene_pca_umap", dataset_specs),
        "umap_neighbor_fraction": plot_umap_neighbor_fraction(
            umap_neighbor_fraction,
            figure_dir / "08_umap_merfish_neighbor_fraction",
            dataset_specs,
        ),
    }

    payload = {
        "n_shared_genes": int(len(shared_genes)),
        "n_observations": int(obs.shape[0]),
        "pca_explained_variance_first_10": [float(value) for value in explained[:10]],
        "shared_gene_tsv": str(shared_gene_tsv),
        "embedding_tsv": str(embedding_tsv),
        "centroid_distance_tsv": str(centroid_tsv),
        "cell_distance_tsv": str(cell_distance_tsv),
        "umap_neighbor_tsv": str(umap_neighbor_tsv),
        "umap_distance_tsv": str(umap_distance_tsv),
        "figure_paths": figure_paths,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
