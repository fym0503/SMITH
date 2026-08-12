#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCREEN_DIR = PROJECT_ROOT / "outputs/liver_pareto_hpo_screen8_epoch15"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures/liver_pareto_hpo_epoch15"

COLORS = {
    "ink": "#1F2528",
    "grid": "#D9DEE3",
    "selected": "#9B8BC2",
    "best": "#6E4AA5",
    "star": "#D68C45",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot locked MERFISH screen HPO trajectories.")
    parser.add_argument("--screen-dir", default=str(DEFAULT_SCREEN_DIR))
    parser.add_argument("--trajectory-file", default="", help="Optional trajectory TSV; overrides --screen-dir.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--title-prefix", default="Epoch-15 screen")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def setup_style() -> None:
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


def source_data(trajectory: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for metric, group in trajectory.groupby("objective_metric", sort=False):
        group = group.sort_values("iteration").copy()
        if "merfish_value" not in group and "full_merfish_value" in group:
            group["merfish_value"] = group["full_merfish_value"]
        if "visium_value" not in group and "full_visium_value" in group:
            group["visium_value"] = group["full_visium_value"]
        group["visium_best_so_far"] = group["visium_value"].cummax()
        group["merfish_best_so_far"] = group["merfish_value"].cummax()
        rows.append(group)
    out = pd.concat(rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "source_data_screen_merfish_best_so_far.tsv", sep="\t", index=False)
    return out


def plot_metric(data: pd.DataFrame, metric: str, ylabel: str, title: str, output_dir: Path, stem: str, dpi: int) -> None:
    df = data[data["objective_metric"] == metric].sort_values("iteration").copy()
    fig, ax = plt.subplots(figsize=(3.0, 2.25))
    ax.plot(
        df["iteration"],
        df["merfish_value"],
        color=COLORS["selected"],
        linewidth=1.0,
        marker="o",
        markersize=2.7,
        label="Selected config",
        zorder=2,
    )
    ax.plot(
        df["iteration"],
        df["merfish_best_so_far"],
        color=COLORS["best"],
        linewidth=1.5,
        marker="o",
        markersize=3.0,
        label="Best so far",
        zorder=3,
    )
    best_row = df.loc[df["merfish_best_so_far"].astype(float).idxmax()]
    ax.scatter(
        [best_row["iteration"]],
        [best_row["merfish_best_so_far"]],
        s=42,
        marker="*",
        facecolor=COLORS["star"],
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    ymin = float(df[["merfish_value", "merfish_best_so_far"]].min().min())
    ymax = float(df[["merfish_value", "merfish_best_so_far"]].max().max())
    pad = max((ymax - ymin) * 0.22, 0.004)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xticks(df["iteration"].astype(int))
    ax.set_xlabel("HPO iteration", fontsize=6.8)
    ax.set_ylabel(ylabel, fontsize=6.8)
    ax.set_title(title, fontsize=7.0, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.4)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.0), fontsize=5.8, handlelength=1.4, ncol=2)
    save_figure(fig, output_dir, stem, dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    setup_style()
    output_dir = Path(args.output_dir)
    trajectory_path = Path(args.trajectory_file) if args.trajectory_file else Path(args.screen_dir) / "hpo_objective_trajectory.tsv"
    trajectory = pd.read_csv(trajectory_path, sep="\t")
    data = source_data(trajectory, output_dir)
    plot_metric(
        data,
        "cell_type_accuracy",
        "MERFISH cell-type accuracy",
        f"{args.title_prefix}: MERFISH cell-type accuracy",
        output_dir,
        "merfish_cell_type_accuracy_by_iteration",
        args.dpi,
    )
    plot_metric(
        data,
        "spatial_pearson",
        "MERFISH spatial Pearson",
        f"{args.title_prefix}: MERFISH spatial Pearson",
        output_dir,
        "merfish_spatial_pearson_by_iteration",
        args.dpi,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
