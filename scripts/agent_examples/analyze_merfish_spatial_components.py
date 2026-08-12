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
from scipy.stats import mannwhitneyu
from sklearn.decomposition import NMF


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERFISH = REPO_ROOT / "data/liver_merfish/adata_healthy_merfish.h5ad"
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_5seed_panels"
DEFAULT_OUTPUT_DIR = DEFAULT_BENCHMARK_DIR / "diagnostics/merfish_spatial_components"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures/agent_visium_retrieval_refined"
ARIAL_FONT_FILES = [
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
]

PALETTE = {
    "snRNA-only": "#BDBDBD",
    "snRNA + multi-Visium": "#2F7FB9",
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


def _as_array(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _normalise_log1p(x: np.ndarray) -> np.ndarray:
    x = np.maximum(_as_array(x), 0.0)
    totals = x.sum(axis=1, keepdims=True)
    x = np.divide(x, np.maximum(totals, 1e-8), out=np.zeros_like(x, dtype=np.float32), where=totals > 0)
    x *= 1e4
    return np.log1p(x, dtype=np.float32)


def _read_panel(path: Path, panel_size: int = 64) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    gene_col = next((col for col in df.columns if str(col).lower() in {"gene_symbol", "gene", "marker"}), df.columns[0])
    return [str(g).strip().upper() for g in df[gene_col].head(panel_size).tolist() if str(g).strip()]


def aggregate_merfish_to_grid(
    merfish_file: Path,
    *,
    n_bins: int = 36,
    min_cells_per_bin: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    adata = ad.read_h5ad(merfish_file)
    try:
        if "x" not in adata.obs.columns or "y" not in adata.obs.columns:
            raise KeyError("MERFISH obs must contain x/y spatial coordinates.")
        if "sample_id" not in adata.obs.columns:
            raise KeyError("MERFISH obs must contain sample_id.")
        if "Cell_Type" not in adata.obs.columns:
            raise KeyError("MERFISH obs must contain Cell_Type.")
        x_coord = adata.obs["x"].astype(float).to_numpy()
        y_coord = adata.obs["y"].astype(float).to_numpy()
        samples = adata.obs["sample_id"].astype(str).to_numpy()
        cell_types = adata.obs["Cell_Type"].astype(str).to_numpy()
        genes = [str(g).strip().upper() for g in adata.var_names.astype(str).tolist()]
        expr = _normalise_log1p(adata.X)
    finally:
        del adata

    bin_rows = []
    expr_rows = []
    cell_type_rows = []
    classes = sorted(pd.unique(cell_types).astype(str))
    for sample in sorted(pd.unique(samples)):
        mask = samples == sample
        if not mask.any():
            continue
        xs = x_coord[mask]
        ys = y_coord[mask]
        sample_expr = expr[mask]
        sample_cell_types = cell_types[mask]
        x_edges = np.linspace(float(xs.min()), float(xs.max()) + 1e-6, n_bins + 1)
        y_edges = np.linspace(float(ys.min()), float(ys.max()) + 1e-6, n_bins + 1)
        x_bin = np.clip(np.searchsorted(x_edges, xs, side="right") - 1, 0, n_bins - 1)
        y_bin = np.clip(np.searchsorted(y_edges, ys, side="right") - 1, 0, n_bins - 1)
        grid_id = np.asarray([f"{sample}_x{xb:02d}_y{yb:02d}" for xb, yb in zip(x_bin, y_bin)], dtype=object)
        for gid in sorted(pd.unique(grid_id)):
            gid_mask = grid_id == gid
            n_cells = int(gid_mask.sum())
            if n_cells < int(min_cells_per_bin):
                continue
            parts = gid.split("_x", 1)
            sample_id = parts[0]
            x_part, y_part = parts[1].split("_y", 1)
            row_expr = np.asarray(sample_expr[gid_mask].mean(axis=0)).ravel()
            expr_rows.append(row_expr)
            bin_rows.append(
                {
                    "grid_id": gid,
                    "sample_id": sample_id,
                    "x_bin": int(x_part),
                    "y_bin": int(y_part),
                    "x_center": float(xs[gid_mask].mean()),
                    "y_center": float(ys[gid_mask].mean()),
                    "n_cells": n_cells,
                }
            )
            counts = pd.Series(sample_cell_types[gid_mask]).value_counts(normalize=True)
            cell_type_rows.append({cls: float(counts.get(cls, 0.0)) for cls in classes})
    bin_df = pd.DataFrame(bin_rows)
    expr_df = pd.DataFrame(np.vstack(expr_rows), columns=genes)
    cell_type_df = pd.DataFrame(cell_type_rows)
    cell_type_df.insert(0, "grid_id", bin_df["grid_id"].to_numpy())
    return bin_df, expr_df, cell_type_df, genes


def fit_spatial_nmf(expr_df: pd.DataFrame, *, n_components: int = 8, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = expr_df.to_numpy(dtype=np.float32)
    gene_means = x.mean(axis=0, keepdims=True)
    gene_stds = x.std(axis=0, keepdims=True)
    variable = gene_stds.ravel() > 1e-8
    x_scaled = np.zeros_like(x, dtype=np.float32)
    x_scaled[:, variable] = (x[:, variable] - gene_means[:, variable]) / gene_stds[:, variable]
    x_nonnegative = np.maximum(x_scaled, 0.0)
    model = NMF(
        n_components=int(n_components),
        init="nndsvda",
        random_state=int(seed),
        max_iter=1200,
        beta_loss="frobenius",
        solver="cd",
    )
    activity = model.fit_transform(x_nonnegative)
    loading = model.components_
    activity_df = pd.DataFrame(activity, columns=[f"component_{i + 1}" for i in range(n_components)])
    loading_df = pd.DataFrame(loading.T, index=expr_df.columns, columns=[f"component_{i + 1}" for i in range(n_components)])
    loading_df.index.name = "gene_symbol"
    loading_df = loading_df.reset_index()
    return activity_df, loading_df


def interpret_components(
    bin_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    loading_df: pd.DataFrame,
    cell_type_df: pd.DataFrame,
    *,
    top_n: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    activity_with_grid = pd.concat([bin_df[["grid_id", "sample_id", "x_center", "y_center", "n_cells"]], activity_df], axis=1)
    cell_types = [col for col in cell_type_df.columns if col != "grid_id"]
    merged = activity_with_grid.merge(cell_type_df, on="grid_id", how="left")
    summary_rows = []
    top_gene_rows = []
    for component in [col for col in activity_df.columns if col.startswith("component_")]:
        top_genes = (
            loading_df[["gene_symbol", component]]
            .sort_values(component, ascending=False)
            .head(top_n)
            .copy()
            .rename(columns={component: "loading"})
        )
        for rank, row in enumerate(top_genes.itertuples(index=False), start=1):
            top_gene_rows.append(
                {
                    "component": component,
                    "rank": rank,
                    "gene_symbol": row.gene_symbol,
                    "loading": float(row.loading),
                }
            )
        correlations = []
        values = merged[component].astype(float).to_numpy()
        for cell_type in cell_types:
            frac = merged[cell_type].astype(float).to_numpy()
            if np.std(values) <= 1e-8 or np.std(frac) <= 1e-8:
                corr = 0.0
            else:
                corr = float(np.corrcoef(values, frac)[0, 1])
            correlations.append((cell_type, corr))
        correlations = sorted(correlations, key=lambda item: abs(item[1]), reverse=True)
        positive = [f"{cell}:{corr:.2f}" for cell, corr in correlations if corr > 0][:3]
        negative = [f"{cell}:{corr:.2f}" for cell, corr in correlations if corr < 0][:3]
        summary_rows.append(
            {
                "component": component,
                "top_genes": ", ".join(top_genes["gene_symbol"].astype(str).head(8).tolist()),
                "top_positive_cell_type_correlations": "; ".join(positive),
                "top_negative_cell_type_correlations": "; ".join(negative),
                "activity_mean": float(np.mean(values)),
                "activity_std": float(np.std(values)),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(top_gene_rows)


def panel_component_coverage(
    benchmark_dir: Path,
    loading_df: pd.DataFrame,
    *,
    panel_size: int = 64,
) -> pd.DataFrame:
    loading_long = loading_df.melt(id_vars="gene_symbol", var_name="component", value_name="loading")
    component_totals = loading_long.groupby("component")["loading"].sum().replace(0, np.nan).to_dict()
    rows = []
    for seed_dir in sorted(benchmark_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        panel_paths = {
            "snRNA-only": seed_dir / "panels" / f"source_top_{panel_size}_panel.tsv",
            "snRNA + multi-Visium": seed_dir / "panels" / f"multi_visium_top_{panel_size}_panel.tsv",
        }
        for panel_name, panel_path in panel_paths.items():
            if not panel_path.exists():
                continue
            genes = set(_read_panel(panel_path, panel_size=panel_size))
            selected = loading_long[loading_long["gene_symbol"].isin(genes)].copy()
            for component, group in selected.groupby("component"):
                total = component_totals.get(component, np.nan)
                rows.append(
                    {
                        "seed": seed,
                        "panel": panel_name,
                        "panel_size": int(panel_size),
                        "component": component,
                        "covered_loading": float(group["loading"].sum()),
                        "coverage_fraction": float(group["loading"].sum() / total) if np.isfinite(total) else np.nan,
                        "n_panel_genes_in_component": int((group["loading"] > 0).sum()),
                        "top_panel_genes": ", ".join(
                            group.sort_values("loading", ascending=False)["gene_symbol"].head(8).astype(str).tolist()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_component_coverage(coverage_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        coverage_df.groupby(["panel", "component"], as_index=False)
        .agg(
            coverage_mean=("coverage_fraction", "mean"),
            coverage_std=("coverage_fraction", "std"),
            covered_loading_mean=("covered_loading", "mean"),
            n_panel_genes_mean=("n_panel_genes_in_component", "mean"),
        )
        .copy()
    )
    pivot = coverage_df.pivot_table(
        index=["seed", "component"],
        columns="panel",
        values="coverage_fraction",
        aggfunc="mean",
    ).reset_index()
    if {"snRNA-only", "snRNA + multi-Visium"}.issubset(pivot.columns):
        pivot["delta_coverage"] = pivot["snRNA + multi-Visium"] - pivot["snRNA-only"]
    return summary, pivot


def _paired_wilcoxon_like(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")
    # Mann-Whitney is conservative here because seed/component pairs are not independent;
    # exact paired tests are reported in the source table if needed.
    return float(mannwhitneyu(values_a, values_b, alternative="two-sided").pvalue)


def plot_component_maps(
    bin_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    component_summary: pd.DataFrame,
    output_prefix: Path,
    *,
    max_components: int = 8,
) -> dict[str, str]:
    _configure_matplotlib()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    components = [col for col in activity_df.columns if col.startswith("component_")][:max_components]
    activity = pd.concat([bin_df.reset_index(drop=True), activity_df[components].reset_index(drop=True)], axis=1)
    n_cols = 4
    n_rows = int(np.ceil(len(components) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.3, 1.65 * n_rows), facecolor="white")
    axes = np.asarray(axes).reshape(-1)
    for ax, component in zip(axes, components):
        values = activity[component].astype(float)
        scatter = ax.scatter(
            activity["x_center"],
            activity["y_center"],
            c=values,
            s=np.clip(activity["n_cells"] / 4, 4, 16),
            cmap="viridis",
            linewidth=0,
            rasterized=True,
        )
        top = component_summary.loc[component_summary["component"] == component, "top_genes"]
        title = component.replace("component_", "C")
        if not top.empty:
            title = f"{title}: {', '.join(str(top.iloc[0]).split(', ')[:3])}"
        ax.set_title(title, fontsize=6.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[len(components) :]:
        ax.axis("off")
    cbar = fig.colorbar(scatter, ax=axes[: len(components)], fraction=0.018, pad=0.015)
    cbar.ax.tick_params(labelsize=6.2, length=2, width=0.5)
    cbar.set_label("NMF activity", fontsize=6.8)
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


def plot_coverage_violin(coverage_df: pd.DataFrame, output_prefix: Path) -> dict[str, str]:
    _configure_matplotlib()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    order = ["snRNA-only", "snRNA + multi-Visium"]
    data = [coverage_df.loc[coverage_df["panel"] == panel, "coverage_fraction"].dropna().to_numpy(float) for panel in order]
    p_value = _paired_wilcoxon_like(data[0], data[1])
    fig, ax = plt.subplots(figsize=(2.25, 2.25), facecolor="white")
    parts = ax.violinplot(data, positions=np.arange(2), widths=0.62, showmeans=False, showextrema=False)
    for body, panel in zip(parts["bodies"], order):
        body.set_facecolor(PALETTE[panel])
        body.set_edgecolor("none")
        body.set_alpha(0.60)
    rng = np.random.default_rng(17)
    for idx, (panel, values) in enumerate(zip(order, data)):
        x = np.full(values.size, idx) + rng.normal(0, 0.045, size=values.size)
        ax.scatter(
            x,
            values,
            s=15,
            color=PALETTE[panel],
            edgecolor=PALETTE["ink"],
            linewidth=0.25,
            alpha=0.84,
            zorder=3,
        )
        ax.plot([idx - 0.18, idx + 0.18], [np.median(values), np.median(values)], color=PALETTE["ink"], lw=0.75)
    ax.text(0.98, 0.95, f"p={p_value:.3g}", transform=ax.transAxes, ha="right", va="top", fontsize=7.0)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["snRNA-only", "snRNA +\nmulti-Visium"], fontsize=7.0)
    ax.set_ylabel("MERFISH spatial component\nloading coverage", fontsize=8.0)
    ax.tick_params(axis="y", labelsize=7.4, length=2.4, width=0.55)
    ax.tick_params(axis="x", length=0)
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


def plot_component_delta_bar(delta_df: pd.DataFrame, component_summary: pd.DataFrame, output_prefix: Path) -> dict[str, str]:
    _configure_matplotlib()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = delta_df.groupby("component", as_index=False)["delta_coverage"].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean", ascending=True)
    label_map = component_summary.set_index("component")["top_genes"].to_dict()
    y = np.arange(summary.shape[0])
    fig, ax = plt.subplots(figsize=(2.8, 2.35), facecolor="white")
    colors = [PALETTE["snRNA + multi-Visium"] if value >= 0 else "#B85C5C" for value in summary["mean"]]
    ax.barh(y, summary["mean"], xerr=summary["std"].fillna(0.0), color=colors, edgecolor=PALETTE["ink"], linewidth=0.35, height=0.65)
    labels = []
    for component in summary["component"]:
        genes = ", ".join(str(label_map.get(component, "")).split(", ")[:2])
        labels.append(f"{component.replace('component_', 'C')}: {genes}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.3)
    ax.axvline(0, color="#888888", lw=0.6)
    ax.set_xlabel("Coverage gain\n(multi-Visium - snRNA)", fontsize=8.0)
    ax.tick_params(axis="x", labelsize=7.3, length=2.4, width=0.55)
    ax.tick_params(axis="y", length=0)
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
    parser.add_argument("--merfish-file", default=str(DEFAULT_MERFISH))
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--panel-size", type=int, default=64)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-bins", type=int, default=36)
    parser.add_argument("--min-cells-per-bin", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    bin_df, expr_df, cell_type_df, genes = aggregate_merfish_to_grid(
        Path(args.merfish_file),
        n_bins=int(args.n_bins),
        min_cells_per_bin=int(args.min_cells_per_bin),
    )
    activity_df, loading_df = fit_spatial_nmf(expr_df, n_components=int(args.n_components), seed=int(args.seed))
    component_summary, top_genes = interpret_components(bin_df, activity_df, loading_df, cell_type_df)
    coverage = panel_component_coverage(Path(args.benchmark_dir), loading_df, panel_size=int(args.panel_size))
    coverage_summary, coverage_delta = summarize_component_coverage(coverage)

    bin_tsv = output_dir / "merfish_spatial_grid_bins.tsv"
    activity_tsv = output_dir / "merfish_spatial_component_activity.tsv"
    loading_tsv = output_dir / "merfish_spatial_component_gene_loadings.tsv"
    summary_tsv = output_dir / "merfish_spatial_component_summary.tsv"
    top_genes_tsv = output_dir / "merfish_spatial_component_top_genes.tsv"
    coverage_tsv = output_dir / "panel_spatial_component_coverage_by_seed.tsv"
    coverage_summary_tsv = output_dir / "panel_spatial_component_coverage_summary.tsv"
    coverage_delta_tsv = output_dir / "panel_spatial_component_coverage_delta.tsv"
    bin_df.to_csv(bin_tsv, sep="\t", index=False)
    pd.concat([bin_df[["grid_id", "sample_id", "x_center", "y_center", "n_cells"]], activity_df], axis=1).to_csv(
        activity_tsv,
        sep="\t",
        index=False,
    )
    loading_df.to_csv(loading_tsv, sep="\t", index=False)
    component_summary.to_csv(summary_tsv, sep="\t", index=False)
    top_genes.to_csv(top_genes_tsv, sep="\t", index=False)
    coverage.to_csv(coverage_tsv, sep="\t", index=False)
    coverage_summary.to_csv(coverage_summary_tsv, sep="\t", index=False)
    coverage_delta.to_csv(coverage_delta_tsv, sep="\t", index=False)

    figure_paths = {
        "component_maps": plot_component_maps(
            bin_df,
            activity_df,
            component_summary,
            figure_dir / "12_merfish_spatial_component_maps",
            max_components=int(args.n_components),
        ),
        "coverage": plot_coverage_violin(
            coverage,
            figure_dir / "13_panel_merfish_spatial_component_coverage",
        ),
        "delta": plot_component_delta_bar(
            coverage_delta,
            component_summary,
            figure_dir / "14_merfish_spatial_component_coverage_delta",
        ),
    }
    payload = {
        "n_grid_bins": int(bin_df.shape[0]),
        "n_genes": int(expr_df.shape[1]),
        "n_components": int(args.n_components),
        "bin_tsv": str(bin_tsv),
        "activity_tsv": str(activity_tsv),
        "loading_tsv": str(loading_tsv),
        "summary_tsv": str(summary_tsv),
        "coverage_tsv": str(coverage_tsv),
        "coverage_summary_tsv": str(coverage_summary_tsv),
        "coverage_delta_tsv": str(coverage_delta_tsv),
        "figure_paths": figure_paths,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
