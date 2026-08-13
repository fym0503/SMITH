#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from reproducibility.workflows.figure_style import configure, save_figure


SOURCE_COLOR = "#BDBDBD"
INTEGRATED_COLOR = "#9DBCDC"


def _violin_panel(ax, df, value_column, ylabel, letter):
    sizes = [32, 64, 128]
    panels = ["snRNA-seq", next(name for name in df["panel"].unique() if name.startswith("snRNA-seq + "))]
    colors = {panels[0]: SOURCE_COLOR, panels[1]: INTEGRATED_COLOR}
    offsets = [-0.16, 0.16]
    rng = np.random.default_rng(17)
    for size_index, size in enumerate(sizes):
        for panel_index, panel in enumerate(panels):
            values = df[(df["panel_size"] == size) & (df["panel"] == panel)][value_column].to_numpy(float)
            if not len(values):
                continue
            pos = size_index + offsets[panel_index]
            violin = ax.violinplot([values], positions=[pos], widths=0.27, showmeans=False, showmedians=False, showextrema=False)
            violin["bodies"][0].set_facecolor(colors[panel]); violin["bodies"][0].set_alpha(0.48); violin["bodies"][0].set_edgecolor("none")
            ax.scatter(np.full(len(values), pos) + rng.normal(0, 0.018, len(values)), values, s=12, color=colors[panel], edgecolor="black", linewidth=0.3, zorder=3)
            ax.plot([pos - 0.09, pos + 0.09], [np.median(values)] * 2, color="#444444", lw=0.8)
        paired = df[df["panel_size"] == size].pivot_table(index="training_seed", columns="panel", values=value_column)
        if set(panels).issubset(paired.columns) and len(paired.dropna()) >= 2:
            try:
                p = wilcoxon(paired[panels[1]], paired[panels[0]], alternative="greater").pvalue
            except ValueError:
                p = 1.0
            ymax = df[value_column].max(); ymin = df[value_column].min(); y = ymax + (0.04 + size_index * 0.005) * max(ymax - ymin, 0.01)
            x0, x1 = size_index + offsets[0], size_index + offsets[1]
            ax.plot([x0, x0, x1, x1], [y, y + 0.003, y + 0.003, y], color="#333333", lw=0.5)
            ax.text(size_index, y + 0.004, f"p={p:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(3), sizes)
    ax.set_xlabel("Panel size")
    ax.set_ylabel(ylabel)
    ax.text(-0.24, 1.04, letter, transform=ax.transAxes, fontsize=15, va="top")


def plot(accuracy_path: str | Path, expression_path: str | Path, output_prefix: str | Path) -> dict[str, str]:
    configure()
    accuracy = pd.read_csv(accuracy_path, sep="\t")
    expression = pd.read_csv(expression_path, sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3))
    _violin_panel(axes[0], accuracy, "cell_type_accuracy", "Cell-type classification accuracy", "c")
    _violin_panel(axes[1], expression, "mean_merfish_expression", "Mean MERFISH expression", "d")
    integrated = next(name for name in accuracy["panel"].unique() if name.startswith("snRNA-seq + "))
    labels = ["snRNA-seq", integrated]
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.65) for color in (SOURCE_COLOR, INTEGRATED_COLOR)]
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.23, wspace=0.42)
    return save_figure(fig, output_prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot manuscript Figure 6c-d from fresh workflow outputs.")
    parser.add_argument("--accuracy", required=True)
    parser.add_argument("--expression", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.accuracy, args.expression, args.output_prefix), indent=2))


if __name__ == "__main__":
    main()
