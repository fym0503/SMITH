#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORY = (
    PROJECT_ROOT
    / "outputs/liver_pareto_hpo_screen8_epoch15_extended/combined_sum/combined_by_trial_count.tsv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures/liver_pareto_hpo_epoch15_extended_combined"

COLORS = {
    "ink": "#1F2528",
    "muted": "#6B7280",
    "grid": "#D9DEE3",
    "line": "#5B4B8A",
    "point": "#9B8BC2",
    "star": "#D68C45",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot unified combined-sum HPO trajectories by attempted config count.")
    parser.add_argument("--trajectory-file", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--title-prefix", default="Unified Visium-sum HPO")
    parser.add_argument("--max-configs", type=int, default=0, help="Plot only the first N attempted configs; 0 keeps all.")
    parser.add_argument(
        "--curve-mode",
        choices=("selected", "metric-best-so-far"),
        default="selected",
        help=(
            "selected plots the MERFISH score of the Visium-selected config; "
            "metric-best-so-far plots the retrospective locked-test envelope."
        ),
    )
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


def _selected_change_rows(data: pd.DataFrame) -> pd.DataFrame:
    changed = data["selected_trial_id"].astype(str).ne(data["selected_trial_id"].astype(str).shift())
    return data[changed].copy()


def _metric_change_rows(data: pd.DataFrame, value_col: str, min_delta: float = 1e-4) -> pd.DataFrame:
    values = data[value_col].astype(float)
    changed = values.diff().fillna(1.0).gt(min_delta)
    changed.iloc[0] = True
    return data[changed].copy()


def plot_metric(
    *,
    data: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    stem: str,
    dpi: int,
    highlight_mode: str,
) -> None:
    data = data.sort_values("trial_index").copy()
    x = data["trial_index"].astype(int)
    y = data[value_col].astype(float)
    if highlight_mode == "metric":
        change_rows = _metric_change_rows(data, value_col)
    else:
        change_rows = _selected_change_rows(data)

    fig, ax = plt.subplots(figsize=(3.05, 2.25))
    if highlight_mode == "metric":
        ax.step(
            x,
            y,
            where="post",
            color=COLORS["line"],
            linewidth=1.65,
            zorder=2,
        )
    else:
        ax.plot(
            x,
            y,
            color=COLORS["line"],
            linewidth=1.45,
            marker="o",
            markersize=2.4,
            markerfacecolor=COLORS["point"],
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=2,
        )
    ax.scatter(
        change_rows["trial_index"].astype(int),
        change_rows[value_col].astype(float),
        s=26 if highlight_mode == "metric" else 22,
        facecolor=COLORS["line"],
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    best_row = data.loc[y.idxmax()]
    ax.scatter(
        [int(best_row["trial_index"])],
        [float(best_row[value_col])],
        s=34,
        marker="*",
        facecolor=COLORS["star"],
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )

    ymin = float(y.min())
    ymax = float(y.max())
    pad = max((ymax - ymin) * 0.22, 0.004)
    ax.set_ylim(ymin - pad, ymax + pad)
    max_x = int(x.max())
    step = 10 if max_x <= 80 else 40
    ticks = [1] + [value for value in range(step, max_x, step)]
    if max_x - ticks[-1] < step * 0.5:
        ticks[-1] = max_x
    elif ticks[-1] != max_x:
        ticks.append(max_x)
    ax.set_xticks(ticks)
    ax.set_xlabel("Attempted configs", fontsize=7.8)
    ax.set_ylabel(ylabel, fontsize=7.8)
    ax.set_title(title, fontsize=8.0, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7.0)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    save_figure(fig, output_dir, stem, dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    setup_style()
    output_dir = Path(args.output_dir)
    data = pd.read_csv(args.trajectory_file, sep="\t")
    if args.max_configs > 0:
        data = data[data["trial_index"].astype(int) <= int(args.max_configs)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "source_data_combined_by_trial_count.tsv", sep="\t", index=False)

    if args.curve_mode == "metric-best-so-far":
        cell_col = "merfish_cell_type_accuracy_best_so_far"
        spatial_col = "merfish_spatial_pearson_best_so_far"
        cell_ylabel = "Best MERFISH accuracy"
        spatial_ylabel = "Best MERFISH Pearson"
        cell_title = "Cell type"
        spatial_title = "Spatial"
        cell_stem = "merfish_cell_type_accuracy_best_so_far_by_config_count"
        spatial_stem = "merfish_spatial_pearson_best_so_far_by_config_count"
        highlight_mode = "metric"
    else:
        cell_col = "merfish_cell_type_accuracy"
        spatial_col = "merfish_spatial_pearson"
        cell_ylabel = "Locked MERFISH accuracy"
        spatial_ylabel = "Locked MERFISH Pearson"
        cell_title = "Cell type"
        spatial_title = "Spatial"
        cell_stem = "merfish_cell_type_accuracy_by_config_count"
        spatial_stem = "merfish_spatial_pearson_by_config_count"
        highlight_mode = "selected"

    plot_metric(
        data=data,
        value_col=cell_col,
        ylabel=cell_ylabel,
        title=cell_title,
        output_dir=output_dir,
        stem=cell_stem,
        dpi=args.dpi,
        highlight_mode=highlight_mode,
    )
    plot_metric(
        data=data,
        value_col=spatial_col,
        ylabel=spatial_ylabel,
        title=spatial_title,
        output_dir=output_dir,
        stem=spatial_stem,
        dpi=args.dpi,
        highlight_mode=highlight_mode,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
