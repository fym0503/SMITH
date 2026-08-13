#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from reproducibility.workflows.figure_style import METHOD_COLORS, METHOD_ORDER, configure, save_figure


def _performance(ax, df, source, label, letter, title):
    sub = df[(df["source"] == source) & (df["label"] == label)]
    methods = [method for method in METHOD_ORDER if method in set(sub["method"])]
    sizes = [32, 64, 128]
    width = 0.78 / max(1, len(methods))
    rng = np.random.default_rng(41)
    positions = {}
    for index, method in enumerate(methods):
        means, sems = [], []
        for size in sizes:
            values = sub[(sub["method"] == method) & (sub["panel_size"] == size)]["accuracy"]
            means.append(values.mean())
            sems.append(values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0)
        x = np.arange(3) - 0.39 + width / 2 + index * width
        positions.update({(size, method): float(position) for size, position in zip(sizes, x)})
        ax.bar(x, means, width, yerr=sems, capsize=1, color=METHOD_COLORS[method], edgecolor="black", linewidth=0.3, label=method)
        for position, size in zip(x, sizes):
            points = sub[(sub["method"] == method) & (sub["panel_size"] == size)]["accuracy"].dropna().to_numpy(float)
            ax.scatter(np.full(len(points), position) + rng.uniform(-width * 0.18, width * 0.18, len(points)), points,
                       s=5, c="black", alpha=0.72, linewidths=0, zorder=4)
    if "PERSIST-class" in methods:
        for size_index, size in enumerate(sizes):
            paired = sub[sub["panel_size"] == size].pivot_table(
                index="evaluation_seed", columns="method", values="accuracy", aggfunc="mean"
            ).dropna(subset=["SMITH", "PERSIST-class"])
            if len(paired) >= 2:
                try:
                    pvalue = wilcoxon(paired["SMITH"], paired["PERSIST-class"], alternative="greater").pvalue
                except ValueError:
                    pvalue = 1.0
                y0, y1 = ax.get_ylim()
                y = y1 - (0.10 + 0.015 * (size_index % 2)) * (y1 - y0)
                left, right = positions[(size, "SMITH")], positions[(size, "PERSIST-class")]
                h = 0.025 * (y1 - y0)
                ax.plot([left, left, right, right], [y, y + h, y + h, y], color="black", lw=0.5)
                ax.text((left + right) / 2, y + h, "*" if pvalue < 0.05 else "ns", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(3), sizes)
    ax.set_xlabel("Panel Size")
    ax.set_ylabel("Cell Type Classification Accuracy" if label == "celltype" else "Region Classification Accuracy")
    ax.set_title(title, fontsize=8.5)
    ax.text(-0.31, 1.05, letter, transform=ax.transAxes, fontsize=15, va="top")


def plot(metrics_path: str | Path, overlap_path: str | Path, bias_path: str | Path, output_prefix: str | Path) -> dict[str, str]:
    configure()
    metrics = pd.read_csv(metrics_path, sep="\t")
    overlap = pd.read_csv(overlap_path, sep="\t")
    bias = pd.read_csv(bias_path, sep="\t")
    fig = plt.figure(figsize=(10.2, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.55, wspace=0.55)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    _performance(axes[0], metrics, "Deep-RIBOmap", "celltype", "c", "Deep-RIBOmap to RIBOMap")
    _performance(axes[1], metrics, "Deep-RIBOmap", "region", "d", "")
    _performance(axes[2], metrics, "STARmap", "celltype", "e", "STARmap to RIBOMap")
    _performance(axes[3], metrics, "STARmap", "region", "f", "")
    for ax, limits in zip(axes, [(0.0, 0.4), (0.1, 0.6), (0.1, 0.45), (0.1, 0.6)]):
        ax.set_ylim(*limits)

    axg = fig.add_subplot(gs[1, 0])
    groups = ["Same modality", "Cross modality"]
    colors = ["#3E7FAF", "#ACC3E2"]
    overlap_plot = overlap[(overlap["method_a"] != "Spapros") & (overlap["method_b"] != "Spapros")].copy()
    rng = np.random.default_rng(2)
    for gi, group in enumerate(groups):
        summary = overlap_plot[overlap_plot["modality_group"] == group].groupby("panel_size")["jaccard"]
        means = summary.mean().reindex([32, 64, 128])
        sems = summary.sem().reindex([32, 64, 128]).fillna(0)
        positions = np.arange(3) + (gi - 0.5) * 0.32
        axg.bar(positions, means, 0.32, yerr=sems, capsize=1.5, color=colors[gi], label=group)
        for position, size in zip(positions, [32, 64, 128]):
            points = overlap_plot[(overlap_plot["modality_group"] == group) & (overlap_plot["panel_size"] == size)]["jaccard"].dropna().to_numpy(float)
            axg.scatter(np.full(len(points), position) + rng.uniform(-0.045, 0.045, len(points)), points,
                        s=6, c="black", edgecolors="white", linewidths=0.2, alpha=0.82, zorder=4)
    axg.set_xticks(range(3), [32, 64, 128])
    axg.set_xlabel("Panel size")
    axg.set_ylabel("Jaccard similarity")
    axg.legend(fontsize=6.5)
    axg.text(-0.22, 1.05, "g", transform=axg.transAxes, fontsize=15, va="top")

    axh = fig.add_subplot(gs[1, 1:4])
    h = bias[bias["panel_size"] == 128]
    order = ["Deep-RIBOmap", "Shared", "STARmap", "Background"]
    data = [h.loc[h["group"] == group, "ribomap_bias"].dropna().to_numpy() for group in order]
    parts = axh.violinplot(data, positions=range(4), showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], ["#3E7FAF", "#B7A6C9", "#E3A15B", "#BDBDBD"]):
        body.set_facecolor(color); body.set_edgecolor("black"); body.set_linewidth(0.3); body.set_alpha(0.9)
    rng = np.random.default_rng(8)
    for index, (group, values) in enumerate(zip(order, data)):
        shown = values if group != "Background" or len(values) <= 600 else rng.choice(values, 600, replace=False)
        axh.scatter(np.full(len(shown), index) + rng.uniform(-0.16, 0.16, len(shown)), shown,
                    s=4 if group == "Background" else 6, c="black", alpha=0.12 if group == "Background" else 0.5,
                    linewidths=0, zorder=4)
    for left, right, level in ((0, 1, 0), (0, 3, 1)):
        if len(data[left]) and len(data[right]):
            pvalue = mannwhitneyu(data[left], data[right], alternative="greater").pvalue
            top = float(np.nanquantile(h["ribomap_bias"], 0.985)) + level * 1.2
            height = 0.45
            axh.plot([left, left, right, right], [top, top + height, top + height, top], color="black", lw=0.5)
            label = "p<0.001" if pvalue < 0.001 else f"p={pvalue:.3f}"
            axh.text((left + right) / 2, top + height, label, ha="center", va="bottom", fontsize=6.5)
    axh.axhline(0, color="#777777", lw=0.5)
    axh.set_xticks(range(4), order, rotation=28, ha="right")
    axh.set_ylabel("RIBOMap bias score")
    axh.text(-0.12, 1.05, "h", transform=axh.transAxes, fontsize=15, va="top")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.005), fontsize=7.2)
    fig.subplots_adjust(bottom=0.14)
    return save_figure(fig, output_prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot manuscript Figure 4c-h from fresh workflow outputs.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--overlap", required=True)
    parser.add_argument("--bias", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.metrics, args.overlap, args.bias, args.output_prefix), indent=2))


if __name__ == "__main__":
    main()
