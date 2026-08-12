#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCREEN_DIR = PROJECT_ROOT / "outputs/liver_pareto_hpo_screen8"
DEFAULT_FULL_DIR = PROJECT_ROOT / "outputs/liver_pareto_hpo_screen8_full"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures/liver_pareto_hpo"

COLORS = {
    "ink": "#1F2528",
    "muted": "#6B7280",
    "grid": "#D9DEE3",
    "visium": "#2F7F73",
    "merfish": "#6E4AA5",
    "cell": "#2F7F73",
    "spatial": "#6E4AA5",
    "screen": "#B8C7C2",
    "full": "#2F7F73",
    "pareto": "#D68C45",
    "box": "#F4F6F5",
    "edge": "#9AA6A3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot liver Pareto HPO manuscript panels.")
    parser.add_argument("--screen-dir", default=str(DEFAULT_SCREEN_DIR))
    parser.add_argument("--full-dir", default=str(DEFAULT_FULL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stem", default="liver_pareto_hpo")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png", "tiff"):
        path = output_dir / f"{stem}.{suffix}"
        if suffix in {"png", "tiff"}:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")


def _best_so_far_screen(screen_trajectory: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for metric, group in screen_trajectory.groupby("objective_metric", sort=False):
        group = group.sort_values("iteration").copy()
        group["visium_best_so_far"] = group["visium_value"].cummax()
        group["merfish_best_so_far"] = group["merfish_value"].cummax()
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def write_source_data(screen_dir: Path, full_dir: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    screen_trajectory = pd.read_csv(screen_dir / "hpo_objective_trajectory.tsv", sep="\t")
    screen_frontier = pd.read_csv(screen_dir / "frontier_progress.tsv", sep="\t")
    full_trajectory = pd.read_csv(full_dir / "full_rerun_objective_trajectory_best_so_far.tsv", sep="\t")
    full_summary = pd.read_csv(full_dir / "full_rerun_config_summary.tsv", sep="\t")
    full_pareto = pd.read_csv(full_dir / "full_rerun_visium_pareto.tsv", sep="\t")

    screen_best = _best_so_far_screen(screen_trajectory)
    screen_best_path = output_dir / "source_data_screen_best_so_far.tsv"
    frontier_path = output_dir / "source_data_screen_frontier.tsv"
    full_best_path = output_dir / "source_data_full_best_so_far.tsv"
    full_summary_path = output_dir / "source_data_full_config_summary.tsv"
    full_pareto_path = output_dir / "source_data_full_visium_pareto.tsv"
    screen_best.to_csv(screen_best_path, sep="\t", index=False)
    screen_frontier.to_csv(frontier_path, sep="\t", index=False)
    full_trajectory.to_csv(full_best_path, sep="\t", index=False)
    full_summary.to_csv(full_summary_path, sep="\t", index=False)
    full_pareto.to_csv(full_pareto_path, sep="\t", index=False)
    return {
        "screen_best": screen_best_path,
        "screen_frontier": frontier_path,
        "full_best": full_best_path,
        "full_summary": full_summary_path,
        "full_pareto": full_pareto_path,
    }


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.12, 1.07, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def _draw_pipeline(ax: mpl.axes.Axes) -> None:
    ax.set_axis_off()
    boxes = [
        (0.11, 0.77, 0.76, 0.13, "1", "Train", "liver snRNA-seq"),
        (0.11, 0.58, 0.76, 0.13, "2", "Search", "Visium Pareto objectives"),
        (0.11, 0.39, 0.76, 0.13, "3", "Freeze", "stable frontier trajectory"),
        (0.11, 0.20, 0.76, 0.13, "4", "Full rerun", "selected configs only"),
        (0.11, 0.01, 0.76, 0.13, "5", "Test", "locked MERFISH metrics"),
    ]
    for x, y, w, h, step, title, body in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            facecolor=COLORS["box"],
            edgecolor=COLORS["edge"],
            linewidth=0.8,
        )
        ax.add_patch(patch)
        ax.text(x + 0.055, y + h / 2, step, fontsize=6.8, fontweight="bold", va="center", ha="center")
        ax.text(x + 0.15, y + h * 0.66, title, fontsize=6.2, fontweight="bold", va="center", ha="left")
        ax.text(x + 0.15, y + h * 0.30, body, fontsize=5.1, va="center", ha="left", color=COLORS["muted"])

    arrows = [
        ((0.49, 0.765), (0.49, 0.715)),
        ((0.49, 0.575), (0.49, 0.525)),
        ((0.49, 0.385), (0.49, 0.335)),
        ((0.49, 0.195), (0.49, 0.145)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.9,
                color=COLORS["ink"],
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0.0",
            )
        )


def _plot_screen_frontier(ax: mpl.axes.Axes, frontier: pd.DataFrame) -> None:
    x = frontier["iteration"].astype(int).to_numpy()
    y_cell = frontier["best_visium_mean_cell_type_accuracy"].astype(float).to_numpy()
    y_spatial = frontier["best_visium_mean_spatial_pearson"].astype(float).to_numpy()
    ax.plot(x, y_cell, color=COLORS["cell"], marker="o", markersize=3.2, linewidth=1.5, label="Cell-type accuracy")
    ax.plot(
        x,
        y_spatial,
        color=COLORS["spatial"],
        marker="s",
        markersize=3.0,
        linewidth=1.5,
        label="Spatial Pearson",
    )
    stop_rows = frontier[frontier["stop_reason"].fillna("").astype(str) != ""]
    if not stop_rows.empty:
        stop_x = int(stop_rows["iteration"].iloc[-1])
        ax.axvline(stop_x, color=COLORS["muted"], linewidth=0.7, linestyle=(0, (2, 2)))
        ax.text(stop_x - 0.1, ax.get_ylim()[0], "frontier\nstable", fontsize=5.9, ha="right", va="bottom", color=COLORS["muted"])
    ax.set_xlabel("HPO iteration")
    ax.set_ylabel("Best Visium validation value")
    ax.set_xticks(x)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax.legend(loc="lower right", fontsize=6.1, handlelength=1.6)


def _plot_cell_accuracy(ax: mpl.axes.Axes, screen_best: pd.DataFrame, full_best: pd.DataFrame) -> None:
    screen_cell = screen_best[screen_best["objective_metric"] == "cell_type_accuracy"].sort_values("iteration")
    full_cell = full_best[full_best["objective_metric"] == "cell_type_accuracy"].sort_values("iteration")
    x = screen_cell["iteration"].astype(int).to_numpy()
    ax.plot(
        x,
        screen_cell["visium_best_so_far"].astype(float),
        color=COLORS["visium"],
        linestyle=(0, (2.5, 2)),
        linewidth=1.1,
        marker="o",
        markersize=2.7,
        alpha=0.75,
        label="Visium screen",
    )
    ax.plot(
        x,
        screen_cell["merfish_best_so_far"].astype(float),
        color=COLORS["merfish"],
        linestyle=(0, (2.5, 2)),
        linewidth=1.1,
        marker="o",
        markersize=2.7,
        alpha=0.75,
        label="MERFISH screen panel",
    )
    ax.plot(
        full_cell["iteration"].astype(int),
        full_cell["full_visium_best_so_far"].astype(float),
        color=COLORS["visium"],
        linewidth=1.8,
        marker="o",
        markersize=3.4,
        label="Visium full rerun",
    )
    ax.plot(
        full_cell["iteration"].astype(int),
        full_cell["full_merfish_best_so_far"].astype(float),
        color=COLORS["merfish"],
        linewidth=1.8,
        marker="o",
        markersize=3.4,
        label="MERFISH locked test",
    )
    ax.set_xlabel("HPO iteration")
    ax.set_ylabel("Best-so-far cell-type accuracy")
    ax.set_xticks(x)
    ax.set_ylim(0.64, 0.86)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax.legend(loc="lower right", fontsize=5.7, handlelength=1.5, ncol=1)
    ax.text(
        0.02,
        0.97,
        "config direction selected by Visium only",
        transform=ax.transAxes,
        fontsize=6.0,
        va="top",
        color=COLORS["muted"],
    )


def _plot_pareto(ax: mpl.axes.Axes, summary: pd.DataFrame, pareto: pd.DataFrame) -> None:
    ax.scatter(
        summary["visium_mean_spatial_pearson"],
        summary["visium_mean_cell_type_accuracy"],
        s=24,
        facecolor="#CBD5D1",
        edgecolor="white",
        linewidth=0.4,
        label="Full rerun configs",
        zorder=2,
    )
    pareto_sorted = pareto.sort_values("visium_mean_spatial_pearson")
    ax.plot(
        pareto_sorted["visium_mean_spatial_pearson"],
        pareto_sorted["visium_mean_cell_type_accuracy"],
        color=COLORS["pareto"],
        linewidth=1.2,
        zorder=3,
        label="Visium Pareto frontier",
    )
    ax.scatter(
        pareto["visium_mean_spatial_pearson"],
        pareto["visium_mean_cell_type_accuracy"],
        s=34,
        facecolor=COLORS["pareto"],
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    best_cell = summary.sort_values("visium_mean_cell_type_accuracy", ascending=False).iloc[0]
    ax.scatter(
        [best_cell["visium_mean_spatial_pearson"]],
        [best_cell["visium_mean_cell_type_accuracy"]],
        s=48,
        marker="*",
        facecolor=COLORS["merfish"],
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    ax.annotate(
        "selected\ncell-best",
        xy=(best_cell["visium_mean_spatial_pearson"], best_cell["visium_mean_cell_type_accuracy"]),
        xytext=(10, -6),
        textcoords="offset points",
        fontsize=5.9,
        color=COLORS["muted"],
        va="top",
    )
    ax.set_xlabel("Visium spatial Pearson")
    ax.set_ylabel("Visium cell-type accuracy")
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    ax.legend(loc="lower left", fontsize=5.9, handlelength=1.4)


def _plot_merfish_transfer(ax: mpl.axes.Axes, summary: pd.DataFrame) -> None:
    scatter = ax.scatter(
        summary["merfish_spatial_pearson"],
        summary["merfish_cell_type_accuracy"],
        c=summary["visium_mean_cell_type_accuracy"],
        cmap="viridis",
        s=28,
        edgecolor="white",
        linewidth=0.4,
    )
    best = summary.sort_values("visium_mean_cell_type_accuracy", ascending=False).iloc[0]
    ax.scatter(
        [best["merfish_spatial_pearson"]],
        [best["merfish_cell_type_accuracy"]],
        s=48,
        marker="*",
        facecolor=COLORS["pareto"],
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    ax.set_xlabel("MERFISH spatial Pearson")
    ax.set_ylabel("MERFISH cell-type accuracy")
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    cb = plt.colorbar(scatter, ax=ax, fraction=0.08, pad=0.02)
    cb.set_label("Visium cell accuracy", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.8, length=2)


def _plot_single_merfish_metric(
    *,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> None:
    metric_data = data[data["objective_metric"] == metric].sort_values("iteration").copy()
    fig, ax = plt.subplots(figsize=(3.0, 2.25))
    x = metric_data["iteration"].astype(int)
    ax.plot(
        x,
        metric_data["full_merfish_value"].astype(float),
        color="#9B8BC2",
        linewidth=1.0,
        marker="o",
        markersize=2.7,
        label="Selected config",
        zorder=2,
    )
    ax.plot(
        x,
        metric_data["full_merfish_best_so_far"].astype(float),
        color=COLORS["merfish"],
        linewidth=1.5,
        marker="o",
        markersize=3.0,
        label="Best so far",
        zorder=3,
    )
    best_idx = metric_data["full_merfish_best_so_far"].astype(float).idxmax()
    best_row = metric_data.loc[best_idx]
    ax.scatter(
        [int(best_row["iteration"])],
        [float(best_row["full_merfish_best_so_far"])],
        s=42,
        marker="*",
        facecolor=COLORS["pareto"],
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    ax.set_title(title, fontsize=7.0, fontweight="bold", pad=4)
    ax.set_xlabel("HPO iteration", fontsize=6.8)
    ax.set_ylabel(ylabel, fontsize=6.8)
    ax.set_xticks(x)
    ax.tick_params(labelsize=6.4)
    ymin = float(metric_data[["full_merfish_value", "full_merfish_best_so_far"]].min().min())
    ymax = float(metric_data[["full_merfish_value", "full_merfish_best_so_far"]].max().max())
    pad = max((ymax - ymin) * 0.22, 0.004)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.0), fontsize=5.8, handlelength=1.4, ncol=2)
    save_figure(fig, output_dir, stem, dpi)
    plt.close(fig)


def plot_figure(screen_dir: Path, full_dir: Path, output_dir: Path, stem: str, dpi: int) -> None:
    configure_matplotlib()
    source_paths = write_source_data(screen_dir, full_dir, output_dir)
    screen_best = pd.read_csv(source_paths["screen_best"], sep="\t")
    frontier = pd.read_csv(source_paths["screen_frontier"], sep="\t")
    full_best = pd.read_csv(source_paths["full_best"], sep="\t")
    full_summary = pd.read_csv(source_paths["full_summary"], sep="\t")
    full_pareto = pd.read_csv(source_paths["full_pareto"], sep="\t")

    fig = plt.figure(figsize=(7.6, 5.1), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.92, 1.25, 1.03], height_ratios=[0.95, 1.05], wspace=0.54, hspace=0.58)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])

    _draw_pipeline(ax_a)
    _plot_screen_frontier(ax_b, frontier)
    _plot_pareto(ax_c, full_summary, full_pareto)
    _plot_cell_accuracy(ax_d, screen_best, full_best)
    _plot_merfish_transfer(ax_e, full_summary)

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e], "abcde", strict=True):
        _panel_label(ax, label)

    fig.text(
        0.02,
        0.985,
        "Pareto-guided hyperparameter tuning for liver panel design",
        fontsize=8.8,
        fontweight="bold",
        va="top",
        ha="left",
    )
    legend_handles = [
        Line2D([0], [0], color=COLORS["visium"], lw=1.8, marker="o", markersize=3.4, label="Visium validation"),
        Line2D([0], [0], color=COLORS["merfish"], lw=1.8, marker="o", markersize=3.4, label="MERFISH locked test"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.985, 0.995), fontsize=6.4, ncol=2)
    save_figure(fig, output_dir, stem, dpi)
    plt.close(fig)

    _plot_single_merfish_metric(
        data=full_best,
        metric="cell_type_accuracy",
        ylabel="MERFISH cell-type accuracy",
        title="Locked MERFISH cell-type accuracy",
        output_dir=output_dir,
        stem="merfish_cell_type_accuracy_by_iteration",
        dpi=dpi,
    )
    _plot_single_merfish_metric(
        data=full_best,
        metric="spatial_pearson",
        ylabel="MERFISH spatial Pearson",
        title="Locked MERFISH spatial Pearson",
        output_dir=output_dir,
        stem="merfish_spatial_pearson_by_iteration",
        dpi=dpi,
    )


def main() -> int:
    args = parse_args()
    plot_figure(Path(args.screen_dir), Path(args.full_dir), Path(args.output_dir), args.stem, args.dpi)
    print(Path(args.output_dir) / f"{args.stem}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
