#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from reproducibility.workflows.figure_style import METHOD_COLORS, METHOD_ORDER, configure, save_figure


def _bar_panel(ax, data: pd.DataFrame, dataset: str, metric: str, sizes: list[int], ylabel: str, letter: str, title: str) -> None:
    subset = data[data["dataset"] == dataset]
    methods = [method for method in METHOD_ORDER if method in set(subset["method"])]
    width = 0.78 / max(1, len(methods))
    rng = np.random.default_rng(31)
    positions: dict[tuple[int, str], float] = {}
    for method_index, method in enumerate(methods):
        values, errors = [], []
        for size in sizes:
            rows = subset[(subset["method"] == method) & (subset["panel_size"] == size)]
            values.append(float(rows[metric].mean()) if len(rows) else np.nan)
            errors.append(float(rows[metric].std(ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else 0.0)
        x = np.arange(len(sizes)) - 0.39 + width / 2 + method_index * width
        positions.update({(size, method): float(position) for size, position in zip(sizes, x)})
        ax.bar(x, values, width, yerr=errors, capsize=1.2, color=METHOD_COLORS[method], edgecolor="black", linewidth=0.35, label=method)
        for position, size in zip(x, sizes):
            points = subset[(subset["method"] == method) & (subset["panel_size"] == size)][metric].dropna().to_numpy(float)
            ax.scatter(
                np.full(len(points), position) + rng.uniform(-width * 0.18, width * 0.18, len(points)),
                points,
                s=5,
                c="black",
                alpha=0.75,
                linewidths=0,
                zorder=4,
            )
    if "PERSIST-class" in methods:
        for size_index, size in enumerate(sizes):
            paired = subset[subset["panel_size"] == size].pivot_table(
                index="split", columns="method", values=metric, aggfunc="mean"
            )
            paired = paired.dropna(subset=["SMITH", "PERSIST-class"])
            if len(paired) >= 2:
                try:
                    pvalue = wilcoxon(paired["SMITH"], paired["PERSIST-class"], alternative="greater").pvalue
                except ValueError:
                    pvalue = 1.0
                label = "*" if pvalue < 0.05 else "ns"
                y0, y1 = ax.get_ylim()
                y = y1 - (0.10 + 0.015 * (size_index % 2)) * (y1 - y0)
                left, right = positions[(size, "SMITH")], positions[(size, "PERSIST-class")]
                h = 0.025 * (y1 - y0)
                ax.plot([left, left, right, right], [y, y + h, y + h, y], color="black", lw=0.5)
                ax.text((left + right) / 2, y + h, label, ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(np.arange(len(sizes)), [str(size) for size in sizes])
    ax.set_xlabel("Panel Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.text(-0.30, 1.04, letter, transform=ax.transAxes, fontsize=15, va="top")


def plot(values_path: str | Path, output_prefix: str | Path) -> dict[str, str]:
    configure()
    data = pd.read_csv(values_path, sep="\t")
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.15))
    _bar_panel(axes[0], data, "elegans_tf", "cell_type_accuracy", [32, 64, 128], "Cell Type Classification Accuracy", "c", "Transcription Factor Activities")
    _bar_panel(axes[1], data, "elegans_tf", "developmental_time_pearson", [32, 64, 128], "Time Correlation", "d", "")
    _bar_panel(axes[2], data, "elegans_mirna", "cell_type_accuracy", [16, 24, 32], "Cell Type Classification Accuracy", "e", "miRNA Activities")
    _bar_panel(axes[3], data, "elegans_mirna", "developmental_time_pearson", [16, 24, 32], "Time Correlation", "f", "")
    for ax, limits in zip(axes, [(0.2, 0.8), (0.6, 1.0), (0.2, 0.5), (0.8, 1.0)]):
        ax.set_ylim(*limits)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
    fig.subplots_adjust(bottom=0.23, wspace=0.48)
    return save_figure(fig, output_prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot manuscript Figure 3c-f from newly generated values.")
    parser.add_argument("--values", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.values, args.output_prefix), indent=2))


if __name__ == "__main__":
    main()
