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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERFISH = REPO_ROOT / "data/liver_merfish/adata_healthy_merfish.h5ad"
DEFAULT_GENE_COMPATIBILITY = (
    REPO_ROOT
    / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_5seed_panels/diagnostics/visium_mechanism/visium_added_gene_compatibility.tsv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_5seed_panels/diagnostics/visium_mechanism"
)
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures/agent_visium_mechanism"
ARIAL_FONT_FILES = [
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
]
PALETTE = {
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
            "axes.linewidth": 0.55,
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "legend.frameon": False,
        }
    )


def _save_figure(fig: plt.Figure, output_prefix: Path) -> dict[str, str]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
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


def _normalise_expression_matrix(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    x = np.maximum(np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    totals = x.sum(axis=1, keepdims=True)
    x = np.divide(x, np.maximum(totals, 1e-8), out=np.zeros_like(x, dtype=np.float32), where=totals > 0)
    return np.log1p(x * 1e4, dtype=np.float32)


def select_visium_added_genes(
    gene_compatibility_file: Path,
    *,
    n_genes: int,
    min_detection: float,
    min_expression: float,
    strict_added: bool,
) -> pd.DataFrame:
    df = pd.read_csv(gene_compatibility_file, sep="\t")
    selected = df[(df["gene_class"] == "Visium-added")].copy()
    if strict_added:
        selected = selected[
            (selected["Visium-added"].astype(float) > 0)
            & (selected["Shared"].astype(float) == 0)
            & (selected["snRNA-only"].astype(float) == 0)
        ].copy()
    selected = selected[
        (selected["merfish_detection_rate"].astype(float) >= float(min_detection))
        & (selected["merfish_mean_expression"].astype(float) >= float(min_expression))
    ].copy()
    selected = selected.sort_values(
        ["merfish_moran_i_mean", "merfish_coord_score_mean", "visium_coord_score_mean", "merfish_detection_rate"],
        ascending=False,
    )
    return selected.head(int(n_genes)).copy()


def load_merfish_gene_expression(merfish_file: Path, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    adata = ad.read_h5ad(merfish_file)
    try:
        var_lookup = {str(gene).strip().upper(): idx for idx, gene in enumerate(adata.var_names.astype(str).tolist())}
        present = [gene for gene in genes if gene in var_lookup]
        if not present:
            raise ValueError("None of the requested genes are present in MERFISH var_names.")
        positions = [var_lookup[gene] for gene in present]
        expr = _normalise_expression_matrix(adata[:, positions].X)
        obs = adata.obs[["sample_id", "x", "y", "Cell_Type"]].copy()
        obs["cell_id"] = obs.index.astype(str)
    finally:
        del adata
    expr_df = pd.DataFrame(expr, columns=present)
    return obs.reset_index(drop=True), expr_df


def _sample_cells(obs: pd.DataFrame, expr: pd.DataFrame, *, max_cells_per_sample: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_cells_per_sample <= 0:
        return obs, expr
    rng = np.random.default_rng(seed)
    keep_indices = []
    for sample, sample_obs in obs.groupby("sample_id", sort=True, observed=True):
        indices = sample_obs.index.to_numpy()
        if indices.size > max_cells_per_sample:
            indices = rng.choice(indices, size=max_cells_per_sample, replace=False)
        keep_indices.extend(indices.tolist())
    keep_indices = sorted(keep_indices)
    return obs.loc[keep_indices].reset_index(drop=True), expr.loc[keep_indices].reset_index(drop=True)


def _bin_sample_expression(sample_df: pd.DataFrame, gene: str, *, n_bins: int, min_cells_per_bin: int) -> pd.DataFrame:
    xs = sample_df["x"].to_numpy(dtype=float)
    ys = sample_df["y"].to_numpy(dtype=float)
    x_edges = np.linspace(float(xs.min()), float(xs.max()) + 1e-6, int(n_bins) + 1)
    y_edges = np.linspace(float(ys.min()), float(ys.max()) + 1e-6, int(n_bins) + 1)
    x_bin = np.clip(np.searchsorted(x_edges, xs, side="right") - 1, 0, int(n_bins) - 1)
    y_bin = np.clip(np.searchsorted(y_edges, ys, side="right") - 1, 0, int(n_bins) - 1)
    bins = (
        sample_df.assign(x_bin=x_bin, y_bin=y_bin)
        .groupby(["x_bin", "y_bin"], as_index=False, observed=True)
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            expression=(gene, "mean"),
            n_cells=(gene, "size"),
        )
    )
    return bins[bins["n_cells"] >= int(min_cells_per_bin)].copy()


def plot_binned_spatial_gene_patterns(
    obs: pd.DataFrame,
    expr: pd.DataFrame,
    gene_summary: pd.DataFrame,
    output_prefix: Path,
    *,
    n_bins: int = 65,
    min_cells_per_bin: int = 3,
    point_size: float = 5.0,
) -> dict[str, str]:
    _configure_matplotlib()
    genes = [gene for gene in gene_summary["gene_symbol"].astype(str).tolist() if gene in expr.columns]
    samples = sorted(obs["sample_id"].astype(str).unique())
    plot_df = pd.concat([obs[["sample_id", "x", "y"]].reset_index(drop=True), expr[genes].reset_index(drop=True)], axis=1)

    fig, axes = plt.subplots(
        len(genes),
        len(samples),
        figsize=(1.5 * len(samples), 1.08 * len(genes)),
        facecolor="white",
        squeeze=False,
    )
    scatter = None
    for row_idx, gene in enumerate(genes):
        binned_by_sample = {
            sample: _bin_sample_expression(
                plot_df[plot_df["sample_id"].astype(str) == sample],
                gene,
                n_bins=int(n_bins),
                min_cells_per_bin=int(min_cells_per_bin),
            )
            for sample in samples
        }
        all_bin_values = pd.concat([df["expression"] for df in binned_by_sample.values()], ignore_index=True)
        vmax = float(np.quantile(all_bin_values[all_bin_values > 0], 0.985)) if np.any(all_bin_values > 0) else 1.0
        vmax = max(vmax, 1e-6)
        for sample in samples:
            binned_by_sample[sample]["scaled_expression"] = np.clip(
                binned_by_sample[sample]["expression"].to_numpy(dtype=float) / vmax,
                0.0,
                1.0,
            )
        meta = gene_summary[gene_summary["gene_symbol"].astype(str) == gene].iloc[0]
        spatial_label = (
            f"I={meta['merfish_moran_i_mean']:.2f}, V={meta['visium_coord_score_mean']:.2f}"
            if "merfish_moran_i_mean" in meta.index and pd.notna(meta["merfish_moran_i_mean"])
            else f"M={meta['merfish_coord_score_mean']:.2f}, V={meta['visium_coord_score_mean']:.2f}"
        )
        row_label = (
            f"{gene}\n"
            f"{spatial_label}"
        )
        for col_idx, sample in enumerate(samples):
            ax = axes[row_idx, col_idx]
            bin_df = binned_by_sample[sample]
            scatter = ax.scatter(
                bin_df["x"],
                bin_df["y"],
                c=bin_df["scaled_expression"],
                s=point_size,
                cmap="magma",
                vmin=0,
                vmax=1,
                linewidth=0,
                rasterized=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            if row_idx == 0:
                ax.set_title(sample, fontsize=7.2)
            if col_idx == 0:
                ax.set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=6.2, linespacing=0.9)
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.018, pad=0.012)
    cbar.ax.tick_params(labelsize=6.2, length=2, width=0.5)
    cbar.set_label("Scaled expression\n(per gene)", fontsize=6.8)
    fig.subplots_adjust(wspace=0.04, hspace=0.07)
    return _save_figure(fig, output_prefix)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merfish-file", default=str(DEFAULT_MERFISH))
    parser.add_argument("--gene-compatibility-file", default=str(DEFAULT_GENE_COMPATIBILITY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--genes", default="", help="Optional comma-separated gene list. Overrides automatic selection.")
    parser.add_argument("--n-genes", type=int, default=5)
    parser.add_argument("--min-detection", type=float, default=0.20)
    parser.add_argument("--min-expression", type=float, default=0.75)
    parser.add_argument("--allow-shared", action="store_true")
    parser.add_argument("--max-cells-per-sample", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=65)
    parser.add_argument("--min-cells-per-bin", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.genes.strip():
        requested_genes = [gene.strip().upper() for gene in args.genes.split(",") if gene.strip()]
        compatibility = pd.read_csv(args.gene_compatibility_file, sep="\t")
        gene_summary = compatibility[compatibility["gene_symbol"].astype(str).str.upper().isin(requested_genes)].copy()
        gene_summary["gene_symbol"] = gene_summary["gene_symbol"].astype(str).str.upper()
        gene_summary = gene_summary.set_index("gene_symbol").loc[requested_genes].reset_index()
    else:
        gene_summary = select_visium_added_genes(
            Path(args.gene_compatibility_file),
            n_genes=int(args.n_genes),
            min_detection=float(args.min_detection),
            min_expression=float(args.min_expression),
            strict_added=not bool(args.allow_shared),
        )
    genes = gene_summary["gene_symbol"].astype(str).str.upper().tolist()
    obs, expr = load_merfish_gene_expression(Path(args.merfish_file), genes)
    obs, expr = _sample_cells(obs, expr, max_cells_per_sample=int(args.max_cells_per_sample), seed=int(args.seed))

    selected_tsv = output_dir / "visium_added_merfish_spatial_pattern_genes.tsv"
    gene_summary.to_csv(selected_tsv, sep="\t", index=False)
    figure_paths = {
        "spatial_patterns": plot_binned_spatial_gene_patterns(
            obs,
            expr,
            gene_summary,
            figure_dir / "30_visium_added_merfish_spatial_gene_patterns",
            n_bins=int(args.n_bins),
            min_cells_per_bin=int(args.min_cells_per_bin),
        )
    }
    payload = {
        "selected_genes": genes,
        "selected_gene_table": str(selected_tsv),
        "figure_paths": figure_paths,
        "n_cells_plotted": int(obs.shape[0]),
        "samples": sorted(obs["sample_id"].astype(str).unique()),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
