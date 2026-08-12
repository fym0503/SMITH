from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_merfish_spatial_components import (  # noqa: E402
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_FIGURE_DIR,
    DEFAULT_MERFISH,
    PALETTE,
    aggregate_merfish_to_grid,
    fit_spatial_nmf,
    interpret_components,
    panel_component_coverage,
    summarize_component_coverage,
)


DEFAULT_OUTPUT_DIR = DEFAULT_BENCHMARK_DIR / "diagnostics/merfish_spatial_components_by_sample"
ARIAL_FONT_FILES = [
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
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


def plot_global_components_by_sample(
    bin_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    component_summary: pd.DataFrame,
    output_prefix: Path,
) -> dict[str, str]:
    _configure_matplotlib()
    samples = sorted(bin_df["sample_id"].astype(str).unique())
    components = [col for col in activity_df.columns if col.startswith("component_")]
    activity = pd.concat([bin_df.reset_index(drop=True), activity_df[components].reset_index(drop=True)], axis=1)
    title_map = component_summary.set_index("component")["top_genes"].to_dict()

    fig, axes = plt.subplots(
        len(components),
        len(samples),
        figsize=(1.55 * len(samples), 1.08 * len(components)),
        facecolor="white",
        squeeze=False,
    )
    for row_idx, component in enumerate(components):
        values = activity[component].astype(float)
        vmax = float(np.quantile(values, 0.985))
        for col_idx, sample in enumerate(samples):
            ax = axes[row_idx, col_idx]
            sample_df = activity[activity["sample_id"].astype(str) == sample]
            ax.scatter(
                sample_df["x_center"],
                sample_df["y_center"],
                c=sample_df[component],
                s=np.clip(sample_df["n_cells"] / 5, 3, 11),
                cmap="viridis",
                vmin=0,
                vmax=vmax,
                linewidth=0,
                rasterized=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            if row_idx == 0:
                ax.set_title(sample, fontsize=7.0)
            if col_idx == 0:
                genes = ", ".join(str(title_map.get(component, "")).split(", ")[:2])
                ax.set_ylabel(f"{component.replace('component_', 'C')}\n{genes}", fontsize=6.2, rotation=0, ha="right", va="center")
    return _save_figure(fig, output_prefix)


def fit_sample_specific_nmf(
    bin_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    cell_type_df: pd.DataFrame,
    *,
    n_components: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_activity = []
    all_loading = []
    all_summary = []
    all_top_genes = []
    for sample_idx, sample in enumerate(sorted(bin_df["sample_id"].astype(str).unique())):
        mask = bin_df["sample_id"].astype(str) == sample
        sample_bin = bin_df.loc[mask].reset_index(drop=True)
        sample_expr = expr_df.loc[mask].reset_index(drop=True)
        sample_cell = cell_type_df.loc[mask].reset_index(drop=True)
        activity, loading = fit_spatial_nmf(sample_expr, n_components=n_components, seed=seed + sample_idx * 101)
        summary, top_genes = interpret_components(sample_bin, activity, loading, sample_cell)
        for df in (activity, loading, summary, top_genes):
            df.insert(0, "sample_id", sample)
        activity = pd.concat([sample_bin[["grid_id", "sample_id", "x_center", "y_center", "n_cells"]], activity.drop(columns=["sample_id"])], axis=1)
        summary["sample_component"] = summary["sample_id"] + ":" + summary["component"].astype(str)
        top_genes["sample_component"] = top_genes["sample_id"] + ":" + top_genes["component"].astype(str)
        all_activity.append(activity)
        all_loading.append(loading)
        all_summary.append(summary)
        all_top_genes.append(top_genes)
    return (
        pd.concat(all_activity, ignore_index=True),
        pd.concat(all_loading, ignore_index=True),
        pd.concat(all_summary, ignore_index=True),
        pd.concat(all_top_genes, ignore_index=True),
    )


def panel_sample_component_coverage(
    benchmark_dir: Path,
    sample_loading_df: pd.DataFrame,
    *,
    panel_size: int,
) -> pd.DataFrame:
    rows = []
    for sample, loading in sample_loading_df.groupby("sample_id"):
        loading = loading.drop(columns=["sample_id"], errors="ignore")
        coverage = panel_component_coverage(benchmark_dir, loading, panel_size=panel_size)
        coverage.insert(0, "sample_id", sample)
        coverage["sample_component"] = coverage["sample_id"].astype(str) + ":" + coverage["component"].astype(str)
        rows.append(coverage)
    return pd.concat(rows, ignore_index=True)


def summarize_sample_coverage_delta(coverage_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        coverage_df.groupby(["sample_id", "panel", "component"], as_index=False)
        .agg(
            coverage_mean=("coverage_fraction", "mean"),
            coverage_std=("coverage_fraction", "std"),
            covered_loading_mean=("covered_loading", "mean"),
            n_panel_genes_mean=("n_panel_genes_in_component", "mean"),
        )
        .copy()
    )
    delta_by_sample = (
        coverage_df.pivot_table(
            index=["sample_id", "seed", "component"],
            columns="panel",
            values="coverage_fraction",
            aggfunc="mean",
        )
        .reset_index()
        .copy()
    )
    if {"snRNA-only", "snRNA + multi-Visium"}.issubset(delta_by_sample.columns):
        delta_by_sample["delta_coverage"] = delta_by_sample["snRNA + multi-Visium"] - delta_by_sample["snRNA-only"]
    return summary, delta_by_sample


def component_morans_i(activity_df: pd.DataFrame, *, n_neighbors: int = 8) -> pd.DataFrame:
    rows = []
    components = [col for col in activity_df.columns if col.startswith("component_")]
    for sample, sample_df in activity_df.groupby("sample_id"):
        coords = sample_df[["x_center", "y_center"]].to_numpy(dtype=float)
        n_obs = coords.shape[0]
        if n_obs < 4:
            for component in components:
                rows.append({"sample_id": sample, "component": component, "morans_i": np.nan})
            continue
        k = min(int(n_neighbors), n_obs - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(coords)
        indices = nn.kneighbors(coords, return_distance=False)[:, 1:]
        weight_sum = float(indices.size)
        for component in components:
            values = sample_df[component].to_numpy(dtype=float)
            centered = values - float(np.mean(values))
            denom = float(np.sum(centered**2))
            if denom <= 1e-12 or weight_sum <= 0:
                morans_i = np.nan
            else:
                numerator = float(np.sum(centered[:, None] * centered[indices]))
                morans_i = (float(n_obs) / weight_sum) * numerator / denom
            rows.append({"sample_id": sample, "component": component, "morans_i": morans_i})
    return pd.DataFrame(rows)


def plot_sample_specific_component_maps(
    sample_activity_df: pd.DataFrame,
    sample_summary_df: pd.DataFrame,
    output_prefix: Path,
    *,
    sample_id: str,
    max_components: int = 6,
) -> dict[str, str]:
    _configure_matplotlib()
    sample_df = sample_activity_df[sample_activity_df["sample_id"].astype(str) == str(sample_id)].copy()
    components = [col for col in sample_df.columns if col.startswith("component_")][:max_components]
    summary = sample_summary_df[sample_summary_df["sample_id"].astype(str) == str(sample_id)].set_index("component")
    n_cols = 3
    n_rows = int(np.ceil(len(components) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8, 1.65 * n_rows), facecolor="white")
    axes = np.asarray(axes).reshape(-1)
    for ax, component in zip(axes, components):
        values = sample_df[component].astype(float)
        vmax = float(np.quantile(values, 0.985))
        ax.scatter(
            sample_df["x_center"],
            sample_df["y_center"],
            c=values,
            s=np.clip(sample_df["n_cells"] / 5, 3, 12),
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            linewidth=0,
            rasterized=True,
        )
        top_genes = ""
        if component in summary.index:
            top_genes = ", ".join(str(summary.loc[component, "top_genes"]).split(", ")[:3])
        ax.set_title(f"{component.replace('component_', 'C')}\n{top_genes}", fontsize=5.9, linespacing=0.95)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[len(components) :]:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.18, hspace=0.48)
    return _save_figure(fig, output_prefix)


def plot_sample_delta_single(
    delta_summary_df: pd.DataFrame,
    output_prefix: Path,
    *,
    sample_id: str,
) -> dict[str, str]:
    _configure_matplotlib()
    sample_df = delta_summary_df[delta_summary_df["sample_id"].astype(str) == str(sample_id)].copy()
    sample_df["component_num"] = sample_df["component"].str.extract(r"(\d+)").astype(int)
    sample_df = sample_df.sort_values("delta_mean", ascending=True)
    y = np.arange(sample_df.shape[0])
    colors = ["#2F7FB9" if value >= 0 else "#B85C5C" for value in sample_df["delta_mean"]]
    fig, ax = plt.subplots(figsize=(2.35, 2.35), facecolor="white")
    ax.barh(
        y,
        sample_df["delta_mean"],
        xerr=sample_df["delta_std"].fillna(0.0),
        color=colors,
        edgecolor=PALETTE["ink"],
        linewidth=0.35,
        height=0.64,
    )
    labels = []
    for _, row in sample_df.iterrows():
        genes = ", ".join(str(row["top_genes"]).split(", ")[:2])
        labels.append(f"{row['component'].replace('component_', 'C')}: {genes}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.axvline(0, color="#777777", lw=0.6)
    ax.set_xlabel("Coverage gain\n(multi-Visium - snRNA)", fontsize=7.8)
    ax.tick_params(axis="x", labelsize=7.0, length=2.3, width=0.55)
    ax.tick_params(axis="y", length=0)
    ax.set_title(str(sample_id), fontsize=7.5)
    return _save_figure(fig, output_prefix)


def plot_top_gain_programs_by_sample(
    sample_activity_df: pd.DataFrame,
    delta_summary_df: pd.DataFrame,
    output_prefix: Path,
) -> dict[str, str]:
    _configure_matplotlib()
    top_rows = (
        delta_summary_df.sort_values("delta_mean", ascending=False)
        .groupby("sample_id", as_index=False)
        .head(1)
        .sort_values("sample_id")
    )
    samples = top_rows["sample_id"].astype(str).tolist()
    fig, axes = plt.subplots(1, len(samples), figsize=(1.95 * len(samples), 1.85), facecolor="white", squeeze=False)
    axes = axes.ravel()
    for ax, row in zip(axes, top_rows.itertuples(index=False)):
        sample = str(row.sample_id)
        component = str(row.component)
        sample_df = sample_activity_df[sample_activity_df["sample_id"].astype(str) == sample].copy()
        values = sample_df[component].astype(float)
        vmax = float(np.quantile(values, 0.985))
        ax.scatter(
            sample_df["x_center"],
            sample_df["y_center"],
            c=values,
            s=np.clip(sample_df["n_cells"] / 5, 3, 12),
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            linewidth=0,
            rasterized=True,
        )
        ax.set_title(f"{sample} {component.replace('component_', 'C')}", fontsize=7.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(wspace=0.18)
    return _save_figure(fig, output_prefix)


def plot_delta_vs_morans_i(
    delta_summary_df: pd.DataFrame,
    output_prefix: Path,
) -> dict[str, str]:
    _configure_matplotlib()
    df = delta_summary_df.dropna(subset=["delta_mean", "morans_i"]).copy()
    samples = sorted(df["sample_id"].astype(str).unique())
    colors = dict(zip(samples, ["#2F7FB9", "#E19A3E", "#5B8E5A", "#8D6BB8", "#B85C5C"]))
    fig, ax = plt.subplots(figsize=(2.65, 2.35), facecolor="white")
    for sample in samples:
        sample_df = df[df["sample_id"].astype(str) == sample]
        ax.scatter(
            sample_df["morans_i"],
            sample_df["delta_mean"],
            s=24,
            color=colors[sample],
            edgecolor=PALETTE["ink"],
            linewidth=0.35,
            alpha=0.86,
            label=sample,
            zorder=3,
        )
    ax.axhline(0, color="#777777", lw=0.6)
    ax.set_xlabel("Spatial autocorrelation\n(Moran's I)", fontsize=8.0)
    ax.set_ylabel("Coverage gain\n(multi-Visium - snRNA)", fontsize=8.0)
    ax.tick_params(axis="both", labelsize=7.2, length=2.4, width=0.55)
    ax.legend(loc="best", fontsize=6.2, handlelength=0.9, borderaxespad=0.2)
    return _save_figure(fig, output_prefix)


def plot_sample_delta_heatmap(
    delta_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_prefix: Path,
) -> dict[str, str]:
    _configure_matplotlib()
    mean_delta = (
        delta_df.groupby(["sample_id", "component"], as_index=False)["delta_coverage"]
        .mean()
        .merge(summary_df[["sample_id", "component", "top_genes", "top_positive_cell_type_correlations"]], on=["sample_id", "component"], how="left")
    )
    samples = sorted(mean_delta["sample_id"].astype(str).unique())
    n_components = int(mean_delta["component"].nunique())
    fig, axes = plt.subplots(1, len(samples), figsize=(2.05 * len(samples), 2.75), facecolor="white", sharex=True)
    axes = np.atleast_1d(axes)
    vmax = float(np.nanmax(np.abs(mean_delta["delta_coverage"])))
    vmax = max(vmax, 0.01)
    for ax, sample in zip(axes, samples):
        sample_df = mean_delta[mean_delta["sample_id"].astype(str) == sample].copy()
        sample_df["component_num"] = sample_df["component"].str.extract(r"(\d+)").astype(int)
        sample_df = sample_df.sort_values("component_num", ascending=True)
        y = np.arange(sample_df.shape[0])
        colors = ["#2F7FB9" if value >= 0 else "#B85C5C" for value in sample_df["delta_coverage"]]
        ax.barh(y, sample_df["delta_coverage"], color=colors, edgecolor=PALETTE["ink"], linewidth=0.35, height=0.68)
        labels = []
        for _, row in sample_df.iterrows():
            genes = ", ".join(str(row["top_genes"]).split(", ")[:2])
            labels.append(f"{row['component'].replace('component_', 'C')}: {genes}")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=5.8)
        ax.axvline(0, color="#777777", lw=0.6)
        ax.set_title(sample, fontsize=7.2)
        ax.set_xlim(-vmax * 1.12, vmax * 1.12)
        ax.tick_params(axis="x", labelsize=6.5, length=2.2, width=0.55)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlabel("")
    fig.supxlabel("Coverage gain (multi-Visium - snRNA)", fontsize=7.4, y=0.02)
    return _save_figure(fig, output_prefix)


def plot_sample_mean_delta(delta_df: pd.DataFrame, output_prefix: Path) -> dict[str, str]:
    _configure_matplotlib()
    mean_delta = (
        delta_df.groupby(["sample_id", "seed"], as_index=False)["delta_coverage"]
        .mean()
        .rename(columns={"delta_coverage": "mean_delta_coverage"})
    )
    samples = sorted(mean_delta["sample_id"].astype(str).unique())
    fig, ax = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
    rng = np.random.default_rng(31)
    for idx, sample in enumerate(samples):
        values = mean_delta.loc[mean_delta["sample_id"].astype(str) == sample, "mean_delta_coverage"].to_numpy(float)
        ax.scatter(
            np.full(values.shape[0], idx) + rng.normal(0, 0.035, size=values.shape[0]),
            values,
            s=22,
            color="#2F7FB9",
            edgecolor=PALETTE["ink"],
            linewidth=0.3,
            alpha=0.86,
            zorder=3,
        )
        ax.plot([idx - 0.18, idx + 0.18], [np.median(values), np.median(values)], color=PALETTE["ink"], lw=0.75)
    ax.axhline(0, color="#777777", lw=0.6)
    ax.set_xticks(np.arange(len(samples)))
    ax.set_xticklabels(samples, fontsize=7.2)
    ax.set_ylabel("Mean spatial component\ncoverage gain", fontsize=7.8)
    ax.tick_params(axis="y", labelsize=7.2, length=2.4, width=0.55)
    ax.tick_params(axis="x", length=0)
    return _save_figure(fig, output_prefix)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merfish-file", default=str(DEFAULT_MERFISH))
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--panel-size", type=int, default=64)
    parser.add_argument("--global-components", type=int, default=8)
    parser.add_argument("--sample-components", type=int, default=6)
    parser.add_argument("--n-bins", type=int, default=36)
    parser.add_argument("--min-cells-per-bin", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    bin_df, expr_df, cell_type_df, _ = aggregate_merfish_to_grid(
        Path(args.merfish_file),
        n_bins=int(args.n_bins),
        min_cells_per_bin=int(args.min_cells_per_bin),
    )
    global_activity, global_loading = fit_spatial_nmf(expr_df, n_components=int(args.global_components), seed=int(args.seed))
    global_summary, _ = interpret_components(bin_df, global_activity, global_loading, cell_type_df)
    sample_activity, sample_loading, sample_summary, sample_top_genes = fit_sample_specific_nmf(
        bin_df,
        expr_df,
        cell_type_df,
        n_components=int(args.sample_components),
        seed=int(args.seed),
    )
    sample_coverage = panel_sample_component_coverage(
        Path(args.benchmark_dir),
        sample_loading,
        panel_size=int(args.panel_size),
    )
    sample_coverage_summary, sample_delta = summarize_sample_coverage_delta(sample_coverage)
    sample_morans_i = component_morans_i(sample_activity)
    sample_delta_summary = (
        sample_delta.groupby(["sample_id", "component"], as_index=False)
        .agg(delta_mean=("delta_coverage", "mean"), delta_std=("delta_coverage", "std"))
        .merge(sample_summary, on=["sample_id", "component"], how="left")
        .merge(sample_morans_i, on=["sample_id", "component"], how="left")
    )
    top_gain_summary = (
        sample_delta_summary.sort_values("delta_mean", ascending=False)
        .groupby("sample_id", as_index=False)
        .head(1)
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    paths = {
        "global_component_activity_by_grid": output_dir / "global_merfish_spatial_component_activity_by_sample.tsv",
        "global_component_summary": output_dir / "global_merfish_spatial_component_summary.tsv",
        "sample_component_activity": output_dir / "sample_specific_spatial_component_activity.tsv",
        "sample_component_loadings": output_dir / "sample_specific_spatial_component_loadings.tsv",
        "sample_component_summary": output_dir / "sample_specific_spatial_component_summary.tsv",
        "sample_component_top_genes": output_dir / "sample_specific_spatial_component_top_genes.tsv",
        "sample_component_coverage": output_dir / "sample_specific_panel_component_coverage_by_seed.tsv",
        "sample_component_delta": output_dir / "sample_specific_panel_component_coverage_delta.tsv",
        "sample_component_delta_summary": output_dir / "sample_specific_panel_component_coverage_delta_summary.tsv",
        "sample_component_morans_i": output_dir / "sample_specific_spatial_component_morans_i.tsv",
        "top_gain_program_summary": output_dir / "top_gain_merfish_spatial_programs_by_sample.tsv",
    }
    pd.concat([bin_df[["grid_id", "sample_id", "x_center", "y_center", "n_cells"]], global_activity], axis=1).to_csv(
        paths["global_component_activity_by_grid"],
        sep="\t",
        index=False,
    )
    global_summary.to_csv(paths["global_component_summary"], sep="\t", index=False)
    sample_activity.to_csv(paths["sample_component_activity"], sep="\t", index=False)
    sample_loading.to_csv(paths["sample_component_loadings"], sep="\t", index=False)
    sample_summary.to_csv(paths["sample_component_summary"], sep="\t", index=False)
    sample_top_genes.to_csv(paths["sample_component_top_genes"], sep="\t", index=False)
    sample_coverage.to_csv(paths["sample_component_coverage"], sep="\t", index=False)
    sample_delta.to_csv(paths["sample_component_delta"], sep="\t", index=False)
    sample_delta_summary.to_csv(paths["sample_component_delta_summary"], sep="\t", index=False)
    sample_morans_i.to_csv(paths["sample_component_morans_i"], sep="\t", index=False)
    top_gain_summary.to_csv(paths["top_gain_program_summary"], sep="\t", index=False)

    figure_paths = {
        "global_components_by_sample": plot_global_components_by_sample(
            bin_df,
            global_activity,
            global_summary,
            figure_dir / "15_merfish_global_spatial_components_by_sample",
        ),
        "sample_component_delta": plot_sample_delta_heatmap(
            sample_delta,
            sample_summary,
            figure_dir / "16_sample_specific_spatial_component_delta",
        ),
        "sample_mean_delta": plot_sample_mean_delta(
            sample_delta,
            figure_dir / "17_sample_mean_spatial_component_coverage_gain",
        ),
        "delta_vs_morans_i": plot_delta_vs_morans_i(
            sample_delta_summary,
            figure_dir / "18_sample_component_delta_vs_morans_i",
        ),
        "top_gain_programs_by_sample": plot_top_gain_programs_by_sample(
            sample_activity,
            sample_delta_summary,
            figure_dir / "20_top_gain_merfish_spatial_programs_by_sample",
        ),
    }
    for sample in sorted(sample_activity["sample_id"].astype(str).unique()):
        figure_paths[f"sample_component_maps_{sample}"] = plot_sample_specific_component_maps(
            sample_activity,
            sample_summary,
            figure_dir / f"19_{sample}_sample_specific_spatial_component_maps",
            sample_id=sample,
            max_components=int(args.sample_components),
        )
        figure_paths[f"sample_component_delta_{sample}"] = plot_sample_delta_single(
            sample_delta_summary,
            figure_dir / f"21_{sample}_sample_specific_spatial_component_delta",
            sample_id=sample,
        )
    payload = {
        "n_grid_bins": int(bin_df.shape[0]),
        "samples": sorted(bin_df["sample_id"].astype(str).unique()),
        "output_tables": {key: str(value) for key, value in paths.items()},
        "figure_paths": figure_paths,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
