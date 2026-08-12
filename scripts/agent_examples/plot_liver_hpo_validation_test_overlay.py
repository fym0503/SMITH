#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORY = (
    PROJECT_ROOT
    / "outputs/liver_pareto_hpo_screen8_epoch15_exhaustive/combined_sum/combined_by_trial_count.tsv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures/liver_pareto_hpo_epoch15_exhaustive_validation_test_overlay_first50"

COLORS = {
    "ink": "#1F2528",
    "muted": "#687178",
    "grid": "#D9DEE3",
    "visium": "#AEC7E8",
    "merfish": "#1F77B4",
    "star": "#D68C45",
}

DATASETS = {
    "visium": {
        "label": "Visium (Validation)",
        "color": COLORS["visium"],
    },
    "merfish": {
        "label": "MERFISH (Test)",
        "color": COLORS["merfish"],
    },
}

ARIAL_FONT_FILES = (
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
)

METRICS = {
    "cell_type": {
        "title": "Cell type",
        "ylabel": "Cell-type accuracy",
        "gain_ylabel": "Gain in cell-type accuracy",
        "visium_col": "visium_mean_cell_type_accuracy",
        "merfish_col": "merfish_cell_type_accuracy",
        "merfish_best_col": "merfish_cell_type_accuracy_best_so_far",
    },
    "spatial": {
        "title": "Spatial",
        "ylabel": "Spatial Pearson",
        "gain_ylabel": "Gain in spatial Pearson",
        "visium_col": "visium_mean_spatial_pearson",
        "merfish_col": "merfish_spatial_pearson",
        "merfish_best_col": "merfish_spatial_pearson_best_so_far",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Visium-validation and MERFISH-test HPO trajectories together.")
    parser.add_argument("--trajectory-file", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-configs", type=int, default=50, help="Plot only the first N attempted configs; 0 keeps all.")
    parser.add_argument(
        "--curve-mode",
        choices=("selected", "best-so-far"),
        default="selected",
        help=(
            "selected plots the same Visium-selected config on validation and test; "
            "best-so-far plots retrospective envelopes for display."
        ),
    )
    parser.add_argument(
        "--value-mode",
        choices=("absolute", "gain"),
        default="absolute",
        help="absolute plots raw metric values; gain plots improvement from the first attempted config.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def setup_style() -> None:
    for font_file in ARIAL_FONT_FILES:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 10.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
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


def _visible_change_rows(x: pd.Series, y: pd.Series, *, min_delta: float) -> tuple[list[int], list[float]]:
    values = y.astype(float).reset_index(drop=True)
    xs = x.astype(int).reset_index(drop=True)
    changed = values.diff().abs().fillna(1.0).gt(min_delta)
    changed.iloc[0] = True
    return xs[changed].tolist(), values[changed].tolist()


def _plateau_label_position(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    values = y.astype(float).reset_index(drop=True)
    xs = x.astype(float).reset_index(drop=True)
    end_value = float(values.iloc[-1])
    plateau = values.sub(end_value).abs().le(1e-12)
    start_idx = int(plateau[plateau].index[0])
    x_mid = float((xs.iloc[start_idx] + xs.iloc[-1]) / 2)
    return x_mid, end_value


def _build_overlay_data(data: pd.DataFrame, curve_mode: str, value_mode: str) -> pd.DataFrame:
    rows: list[dict] = []
    for metric_key, spec in METRICS.items():
        if curve_mode == "best-so-far":
            visium_values = data[spec["visium_col"]].astype(float).cummax()
            merfish_values = data[spec["merfish_best_col"]].astype(float)
        else:
            visium_values = data[spec["visium_col"]].astype(float)
            merfish_values = data[spec["merfish_col"]].astype(float)

        for dataset_key, values in (("visium", visium_values), ("merfish", merfish_values)):
            raw_values = values.astype(float).reset_index(drop=True)
            if value_mode == "gain":
                plot_values = raw_values - float(raw_values.iloc[0])
            else:
                plot_values = raw_values
            for trial_index, selected_trial_id, selected_trial_index, config_key, value in zip(
                data["trial_index"],
                data["selected_trial_id"],
                data["selected_trial_index"],
                data["config_key"],
                plot_values,
                strict=True,
            ):
                rows.append(
                    {
                        "trial_index": int(trial_index),
                        "selected_trial_id": selected_trial_id,
                        "selected_trial_index": int(selected_trial_index),
                        "config_key": config_key,
                        "metric": metric_key,
                        "dataset": dataset_key,
                        "dataset_label": DATASETS[dataset_key]["label"],
                        "curve_mode": curve_mode,
                        "value_mode": value_mode,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _plot_metric(ax: plt.Axes, data: pd.DataFrame, metric_key: str, curve_mode: str, value_mode: str) -> None:
    spec = METRICS[metric_key]
    metric_data = data[data["metric"] == metric_key].copy()
    x_max = int(metric_data["trial_index"].max())

    for dataset_key in ("visium", "merfish"):
        subset = metric_data[metric_data["dataset"] == dataset_key].sort_values("trial_index")
        x = subset["trial_index"].astype(int)
        y = subset["value"].astype(float)
        color = DATASETS[dataset_key]["color"]
        label = DATASETS[dataset_key]["label"]
        ax.step(x, y, where="post", color=color, linewidth=1.8, zorder=2)
        point_x, point_y = _visible_change_rows(x, y, min_delta=1e-4)
        ax.scatter(
            point_x,
            point_y,
            s=22,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        label_x, label_y = _plateau_label_position(x, y)
        ax.annotate(
            label,
            xy=(label_x, label_y),
            xytext=(0, 4),
            textcoords="offset points",
            va="bottom",
            ha="center",
            color=color,
            fontsize=9.8,
            clip_on=True,
        )

    y_min = float(metric_data["value"].min())
    y_max = float(metric_data["value"].max())
    pad = max((y_max - y_min) * 0.18, 0.006)
    if value_mode == "gain":
        ax.axhline(0, color=COLORS["grid"], linewidth=0.75, zorder=1)
        ax.set_ylim(min(y_min - pad, -pad), y_max + pad)
    else:
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(1, x_max)
    step = 10 if x_max <= 80 else 40
    ticks = [1] + [value for value in range(step, x_max, step)]
    if ticks[-1] != x_max:
        ticks.append(x_max)
    ax.set_xticks(ticks)
    ax.set_xlabel("Trials", fontsize=11.5)
    ylabel = "Gains" if value_mode == "gain" else spec["ylabel"]
    ax.set_ylabel(ylabel, fontsize=11.5)
    ax.tick_params(labelsize=10.5)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55)
    ax.set_box_aspect(1)


def plot_combined(overlay_data: pd.DataFrame, output_dir: Path, curve_mode: str, value_mode: str, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.05), constrained_layout=True)
    for ax, metric_key in zip(axes, ("cell_type", "spatial"), strict=True):
        _plot_metric(ax, overlay_data, metric_key, curve_mode, value_mode)
    save_figure(fig, output_dir, f"validation_test_overlay_{curve_mode}_{value_mode}_by_config_count", dpi)
    plt.close(fig)


def plot_separate(overlay_data: pd.DataFrame, output_dir: Path, curve_mode: str, value_mode: str, dpi: int) -> None:
    for metric_key in ("cell_type", "spatial"):
        fig, ax = plt.subplots(figsize=(3.05, 3.05))
        _plot_metric(ax, overlay_data, metric_key, curve_mode, value_mode)
        fig.subplots_adjust(left=0.22, bottom=0.18, right=0.98, top=0.98)
        save_figure(fig, output_dir, f"{metric_key}_validation_test_overlay_{curve_mode}_{value_mode}_by_config_count", dpi)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    setup_style()
    output_dir = Path(args.output_dir)
    data = pd.read_csv(args.trajectory_file, sep="\t").sort_values("trial_index").copy()
    if args.max_configs > 0:
        data = data[data["trial_index"].astype(int) <= int(args.max_configs)].copy()
    overlay_data = _build_overlay_data(data, args.curve_mode, args.value_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_data.to_csv(
        output_dir / f"source_data_validation_test_overlay_{args.curve_mode}_{args.value_mode}.tsv",
        sep="\t",
        index=False,
    )
    plot_combined(overlay_data, output_dir, args.curve_mode, args.value_mode, args.dpi)
    plot_separate(overlay_data, output_dir, args.curve_mode, args.value_mode, args.dpi)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
