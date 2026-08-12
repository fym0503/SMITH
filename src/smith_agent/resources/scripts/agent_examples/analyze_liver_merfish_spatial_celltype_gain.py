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
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERFISH = REPO_ROOT / "data/liver_merfish/adata_healthy_merfish.h5ad"
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_5seed_panels"
DEFAULT_OUTPUT_DIR = DEFAULT_BENCHMARK_DIR / "diagnostics"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures/agent_visium_retrieval_refined"
ARIAL_FONT_FILES = [
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
]

PALETTE = {
    "source": "#BDBDBD",
    "integrated": "#9DBCDC",
    "integrated_dark": "#2F7FB9",
    "hepatocyte": "#D9B36C",
    "nonparenchymal": "#2F7FB9",
    "ink": "#222222",
}


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
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "legend.frameon": False,
        }
    )


def _coordinates(adata: ad.AnnData) -> np.ndarray:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2]
    if "X_spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["X_spatial"], dtype=np.float64)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2]
    lowered = {str(col).lower(): col for col in adata.obs.columns}
    for x_col, y_col in (("x", "y"), ("array_col", "array_row"), ("center_x", "center_y")):
        if x_col in lowered and y_col in lowered:
            return adata.obs[[lowered[x_col], lowered[y_col]]].astype(float).to_numpy(dtype=np.float64)
    raise KeyError("Could not resolve spatial coordinates from obsm or obs x/y columns.")


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    if not values:
        return float("nan")
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compute_spatial_localization(
    merfish_file: Path,
    *,
    label_column: str = "Cell_Type",
    sample_column: str = "sample_id",
    n_neighbors: int = 30,
) -> pd.DataFrame:
    adata = ad.read_h5ad(merfish_file)
    try:
        if label_column not in adata.obs.columns:
            raise KeyError(f"Missing label column `{label_column}`.")
        labels = adata.obs[label_column].astype(str).to_numpy()
        valid = pd.notna(labels) & (labels != "") & (labels != "nan")
        labels = labels[valid]
        coords = _coordinates(adata)[valid]
        if sample_column in adata.obs.columns:
            samples = adata.obs[sample_column].astype(str).to_numpy()[valid]
        else:
            samples = np.repeat("all", labels.shape[0])
    finally:
        del adata

    classes = sorted(pd.unique(labels).astype(str))
    per_sample_rows: list[dict[str, Any]] = []
    for sample in sorted(pd.unique(samples)):
        mask = samples == sample
        sample_labels = labels[mask]
        sample_coords = coords[mask]
        n_cells = int(sample_labels.shape[0])
        if n_cells <= n_neighbors + 1:
            continue
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto")
        nn.fit(sample_coords)
        neighbor_idx = nn.kneighbors(sample_coords, return_distance=False)[:, 1:]
        neighbor_labels = sample_labels[neighbor_idx]
        sample_counts = pd.Series(sample_labels).value_counts()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        global_distribution = np.zeros(len(classes), dtype=float)
        for cls, count in sample_counts.items():
            global_distribution[class_to_idx[str(cls)]] = float(count)
        global_distribution /= max(float(global_distribution.sum()), 1e-12)
        for cls in classes:
            cls_mask = sample_labels == cls
            n_cls = int(cls_mask.sum())
            if n_cls == 0:
                continue
            global_fraction = float(n_cls / n_cells)
            same_neighbor_fraction = float((neighbor_labels[cls_mask] == cls).mean())
            local_neighbor_counts = pd.Series(neighbor_labels[cls_mask].ravel()).value_counts()
            local_neighbor_distribution = np.zeros(len(classes), dtype=float)
            for neighbor_cls, count in local_neighbor_counts.items():
                local_neighbor_distribution[class_to_idx[str(neighbor_cls)]] = float(count)
            local_neighbor_distribution /= max(float(local_neighbor_distribution.sum()), 1e-12)
            neighborhood_js_divergence = _js_divergence(local_neighbor_distribution, global_distribution)
            enrichment = same_neighbor_fraction / max(global_fraction, 1e-12)
            x = cls_mask.astype(float)
            mean = global_fraction
            dev = x - mean
            denominator = float(np.sum(np.square(dev)))
            if denominator > 0:
                numerator = float(np.sum(dev[:, None] * dev[neighbor_idx]))
                moran_i = float((n_cells / (n_cells * n_neighbors)) * (numerator / denominator))
            else:
                moran_i = float("nan")
            per_sample_rows.append(
                {
                    "sample_id": sample,
                    "class": cls,
                    "sample_cells": n_cells,
                    "cell_type_cells": n_cls,
                    "global_fraction": global_fraction,
                    "same_neighbor_fraction": same_neighbor_fraction,
                    "same_neighbor_enrichment": enrichment,
                    "log2_same_neighbor_enrichment": float(np.log2(max(enrichment, 1e-12))),
                    "neighborhood_js_divergence": neighborhood_js_divergence,
                    "moran_i": moran_i,
                }
            )

    sample_df = pd.DataFrame(per_sample_rows)
    rows = []
    for cls, group in sample_df.groupby("class"):
        weights = group["cell_type_cells"].astype(int).tolist()
        rows.append(
            {
                "class": cls,
                "total_cells": int(group["cell_type_cells"].sum()),
                "mean_global_fraction": _weighted_mean(group["global_fraction"].tolist(), weights),
                "same_neighbor_fraction": _weighted_mean(group["same_neighbor_fraction"].tolist(), weights),
                "same_neighbor_enrichment": _weighted_mean(group["same_neighbor_enrichment"].tolist(), weights),
                "log2_same_neighbor_enrichment": _weighted_mean(group["log2_same_neighbor_enrichment"].tolist(), weights),
                "neighborhood_js_divergence": _weighted_mean(group["neighborhood_js_divergence"].tolist(), weights),
                "moran_i": _weighted_mean(group["moran_i"].tolist(), weights),
                "n_samples_present": int(group.shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("log2_same_neighbor_enrichment", ascending=False).reset_index(drop=True)


def load_f1_delta(benchmark_dir: Path) -> pd.DataFrame:
    by_seed_path = benchmark_dir / "diagnostics/per_class_f1_delta_by_seed_panel_size.tsv"
    if not by_seed_path.exists():
        raise FileNotFoundError(
            f"Missing {by_seed_path}. Generate per-class delta diagnostics before running this analysis."
        )
    return pd.read_csv(by_seed_path, sep="\t")


def merge_spatial_and_delta(spatial_df: pd.DataFrame, delta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        delta_df.groupby(["panel_size", "class"], as_index=False)
        .agg(
            support=("support", "mean"),
            source_f1_mean=("source_f1", "mean"),
            multi_f1_mean=("multi_f1", "mean"),
            delta_f1_mean=("delta_f1", "mean"),
            delta_f1_std=("delta_f1", "std"),
        )
        .merge(spatial_df, on="class", how="left")
    )
    all_size = (
        delta_df.groupby("class", as_index=False)
        .agg(
            support=("support", "mean"),
            source_f1_mean=("source_f1", "mean"),
            multi_f1_mean=("multi_f1", "mean"),
            delta_f1_mean=("delta_f1", "mean"),
            delta_f1_std=("delta_f1", "std"),
        )
        .merge(spatial_df, on="class", how="left")
    )
    all_size.insert(0, "panel_size", "all")
    return summary, all_size


def compute_correlations(comparison: pd.DataFrame, all_size: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["log2_same_neighbor_enrichment", "neighborhood_js_divergence", "moran_i"]
    for metric in metrics:
        for panel_size, group in comparison.groupby("panel_size"):
            rows.append(_correlation_row(group, panel_size, metric))
        rows.append(_correlation_row(all_size, "all", metric))
    return pd.DataFrame(rows)


def _correlation_row(group: pd.DataFrame, panel_size: str | int, spatial_metric: str = "log2_same_neighbor_enrichment") -> dict[str, Any]:
    group = group.dropna(subset=[spatial_metric, "delta_f1_mean"])
    x = group[spatial_metric].astype(float).to_numpy()
    y = group["delta_f1_mean"].astype(float).to_numpy()
    if len(group) >= 3:
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        pearson_r = float(pearson.statistic)
        pearson_p = float(pearson.pvalue)
        spearman_r = float(spearman.statistic)
        spearman_p = float(spearman.pvalue)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")
    return {
        "panel_size": panel_size,
        "spatial_metric": spatial_metric,
        "n_cell_types": int(len(group)),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def _cell_type_family(cell_type: str) -> str:
    return "Hepatocyte" if str(cell_type).startswith("Hep_") else "Non-parenchymal"


def plot_spatial_vs_delta(
    panel64: pd.DataFrame,
    output_prefix: Path,
    *,
    spatial_metric: str = "moran_i",
) -> dict[str, str]:
    _configure_matplotlib()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2.35, 2.35), facecolor="white")
    plot_df = panel64.dropna(subset=[spatial_metric, "delta_f1_mean"]).copy()
    plot_df["cell_type_family"] = plot_df["class"].map(_cell_type_family)
    ax.axhline(0, color="#888888", lw=0.55, zorder=0)
    for family, color in (
        ("Hepatocyte", PALETTE["hepatocyte"]),
        ("Non-parenchymal", PALETTE["nonparenchymal"]),
    ):
        family_df = plot_df[plot_df["cell_type_family"] == family]
        if family_df.empty:
            continue
        ax.scatter(
            family_df[spatial_metric],
            family_df["delta_f1_mean"],
            s=34,
            color=color,
            edgecolor=PALETTE["ink"],
            linewidth=0.35,
            alpha=0.95,
            label=family,
            zorder=3,
        )
    label_offsets = {
        "Cholangiocyte": (0.008, -0.004),
        "HSC_1": (0.008, 0.003),
        "HSC_2": (0.008, -0.003),
        "Hep_1": (0.008, 0.001),
        "Hep_2": (0.008, 0.000),
        "Hep_3": (0.008, 0.000),
        "LSEC": (0.008, 0.006),
        "Macrophage_1": (0.008, 0.003),
        "Macrophage_2": (0.008, -0.003),
    }
    for _, row in plot_df.iterrows():
        dx, dy = label_offsets.get(str(row["class"]), (0.008, 0.0))
        ax.text(
            row[spatial_metric] + dx,
            row["delta_f1_mean"] + dy,
            str(row["class"]),
            fontsize=6.4,
            ha="left",
            va="center",
        )
    corr = _correlation_row(plot_df, 64, spatial_metric)
    ax.text(
        0.04,
        0.06,
        f"Spearman r={corr['spearman_r']:.2f}, p={corr['spearman_p']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
    )
    if spatial_metric == "moran_i":
        xlabel = "MERFISH cell-type spatial autocorrelation\n(Moran's I)"
    elif spatial_metric == "log2_same_neighbor_enrichment":
        xlabel = "MERFISH same-type neighbor enrichment\n(log2)"
    else:
        xlabel = "MERFISH neighborhood distinctiveness\n(JS divergence)"
    ax.set_xlabel(xlabel, fontsize=8.8)
    ax.set_ylabel("Cell type F1 gain\n(multi-Visium - snRNA)", fontsize=7.6)
    ax.tick_params(axis="both", labelsize=7.8)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 1.18),
        ncol=2,
        fontsize=6.6,
        handletextpad=0.35,
        borderaxespad=0.0,
        columnspacing=0.8,
    )
    paths = {
        "pdf": str(output_prefix.with_suffix(".pdf")),
        "svg": str(output_prefix.with_suffix(".svg")),
        "png": str(output_prefix.with_suffix(".png")),
    }
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merfish-file", default=str(DEFAULT_MERFISH))
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--label-column", default="Cell_Type")
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--n-neighbors", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_df = compute_spatial_localization(
        Path(args.merfish_file),
        label_column=str(args.label_column),
        sample_column=str(args.sample_column),
        n_neighbors=int(args.n_neighbors),
    )
    spatial_tsv = output_dir / "merfish_cell_type_spatial_localization.tsv"
    spatial_df.to_csv(spatial_tsv, sep="\t", index=False)

    delta_df = load_f1_delta(Path(args.benchmark_dir))
    comparison, all_size = merge_spatial_and_delta(spatial_df, delta_df)
    comparison_tsv = output_dir / "spatial_localization_vs_f1_delta_by_panel_size.tsv"
    all_size_tsv = output_dir / "spatial_localization_vs_f1_delta_all_panel_sizes.tsv"
    comparison.to_csv(comparison_tsv, sep="\t", index=False)
    all_size.to_csv(all_size_tsv, sep="\t", index=False)

    correlations = compute_correlations(comparison, all_size)
    correlation_tsv = output_dir / "spatial_localization_delta_f1_correlations.tsv"
    correlations.to_csv(correlation_tsv, sep="\t", index=False)

    panel64 = comparison[comparison["panel_size"].astype(str) == "64"].copy()
    figure_paths = plot_spatial_vs_delta(
        panel64,
        Path(args.figure_dir) / "05_spatial_autocorrelation_vs_delta_f1",
        spatial_metric="moran_i",
    )

    payload = {
        "spatial_tsv": str(spatial_tsv),
        "comparison_tsv": str(comparison_tsv),
        "all_size_tsv": str(all_size_tsv),
        "correlation_tsv": str(correlation_tsv),
        "figure_paths": figure_paths,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
