#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon

from reproducibility.workflows.figure_style import configure, save_figure


SOURCE_COLOR = "#BDBDBD"
INTEGRATED_COLOR = "#9DBCDC"
SOURCE_EDGE = "#666666"
INTEGRATED_EDGE = "#3E7FAF"


def _draw_violin_panel(
    ax,
    frame: pd.DataFrame,
    value_column: str,
    ylabel: str,
    *,
    show_legend: bool,
    rng_seed: int,
) -> None:
    sizes = [32, 64, 128]
    integrated = next(name for name in frame["panel"].unique() if name.startswith("snRNA-seq + "))
    panels = ["snRNA-seq", integrated]
    colors = [SOURCE_COLOR, INTEGRATED_COLOR]
    edge_colors = [SOURCE_EDGE, INTEGRATED_EDGE]
    offsets = [-0.16, 0.16]
    rng = np.random.default_rng(rng_seed)

    for size_index, size in enumerate(sizes):
        for panel_index, panel in enumerate(panels):
            values = frame[(frame["panel_size"] == size) & (frame["panel"] == panel)][value_column].dropna().to_numpy(float)
            if not len(values):
                continue
            position = size_index + offsets[panel_index]
            violin = ax.violinplot(
                [values],
                positions=[position],
                widths=0.26,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            body = violin["bodies"][0]
            body.set_facecolor(colors[panel_index])
            body.set_alpha(0.40)
            body.set_edgecolor("none")
            ax.scatter(
                np.full(len(values), position) + rng.normal(0, 0.018, len(values)),
                values,
                s=10,
                color=colors[panel_index],
                edgecolor="black",
                linewidth=0.25,
                zorder=3,
            )
            median = float(np.median(values))
            ax.plot(
                [position - 0.10, position + 0.10],
                [median, median],
                color=edge_colors[panel_index],
                lw=0.75,
                zorder=4,
            )

    finite = frame[value_column].dropna().to_numpy(float)
    if value_column == "cell_type_accuracy":
        ymin = max(0.0, float(np.min(finite)) - 0.035)
        ymax = min(1.0, float(np.max(finite)) + 0.055)
        if ymax - ymin < 0.12:
            center = (ymin + ymax) / 2
            ymin, ymax = max(0.0, center - 0.06), min(1.0, center + 0.06)
    else:
        ymin = max(0.0, float(np.min(finite)) - 0.01)
        ymax = float(np.max(finite)) + 0.025
    ax.set_ylim(ymin, ymax)
    y_span = ymax - ymin
    stat_base = ymax - 0.11 * y_span
    stat_step = 0.045 * y_span
    for size_index, size in enumerate(sizes):
        paired = frame[frame["panel_size"] == size].pivot_table(
            index="training_seed", columns="panel", values=value_column
        ).dropna(subset=panels)
        if len(paired) < 2:
            continue
        try:
            pvalue = wilcoxon(paired[panels[1]], paired[panels[0]], alternative="greater", zero_method="wilcox").pvalue
        except ValueError:
            pvalue = 1.0
        left, right = size_index + offsets[0], size_index + offsets[1]
        y = stat_base + (size_index % 2) * stat_step
        height = stat_step * 0.25
        ax.plot([left, left, right, right], [y, y + height, y + height, y], color="#333333", lw=0.45)
        ax.text((left + right) / 2, y + stat_step * 0.36, f"p={pvalue:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(3), sizes)
    ax.set_xlabel("Panel Size")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", labelsize=7)
    if show_legend:
        handles = [
            Rectangle((0, 0), 1, 1, facecolor=SOURCE_COLOR, edgecolor="black", linewidth=0.35, alpha=0.65),
            Rectangle((0, 0), 1, 1, facecolor=INTEGRATED_COLOR, edgecolor="black", linewidth=0.35, alpha=0.65),
        ]
        ax.legend(handles, panels, loc="lower right", fontsize=6.0, handlelength=1.1, borderaxespad=0.3)


def plot(
    accuracy_path: str | Path,
    expression_path: str | Path,
    output_dir: str | Path,
) -> dict[str, dict[str, str]]:
    configure()
    accuracy = pd.read_csv(accuracy_path, sep="\t")
    expression = pd.read_csv(expression_path, sep="\t")
    output_dir = Path(output_dir)
    outputs: dict[str, dict[str, str]] = {}

    fig_c, ax_c = plt.subplots(figsize=(2.25, 2.25), facecolor="white")
    _draw_violin_panel(
        ax_c,
        accuracy,
        "cell_type_accuracy",
        "Cell Type Classification Accuracy",
        show_legend=True,
        rng_seed=17,
    )
    fig_c.text(0.015, 0.985, "c", ha="left", va="top", fontsize=10, weight="bold")
    fig_c.subplots_adjust(left=0.27, right=0.97, bottom=0.22, top=0.94)
    outputs["figure6_c"] = save_figure(fig_c, output_dir / "figure6_c")
    plt.close(fig_c)

    fig_d, ax_d = plt.subplots(figsize=(2.25, 2.25), facecolor="white")
    _draw_violin_panel(
        ax_d,
        expression,
        "mean_merfish_expression",
        "Mean MERFISH Expression",
        show_legend=False,
        rng_seed=23,
    )
    fig_d.text(0.015, 0.985, "d", ha="left", va="top", fontsize=10, weight="bold")
    fig_d.subplots_adjust(left=0.27, right=0.97, bottom=0.22, top=0.94)
    outputs["figure6_d"] = save_figure(fig_d, output_dir / "figure6_d")
    plt.close(fig_d)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export manuscript Figure 6c-d as separate panels.")
    parser.add_argument("--accuracy", required=True)
    parser.add_argument("--expression", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.accuracy, args.expression, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
