#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from reproducibility.workflows.figure_style import (
    METHOD_COLORS,
    METHOD_ORDER,
    configure,
    save_figure,
    save_method_legend,
)


PERFORMANCE_SPECS = {
    "c": ("Deep-RIBOmap", "celltype", "Deep-RIBOmap to RIBOMap", (0.0, 0.4)),
    "d": ("Deep-RIBOmap", "region", "", (0.1, 0.6)),
    "e": ("STARmap", "celltype", "STARmap to RIBOMap", (0.1, 0.45)),
    "f": ("STARmap", "region", "", (0.1, 0.6)),
}


def _draw_performance(ax, frame: pd.DataFrame, source: str, label: str, title: str, ylim: tuple[float, float]) -> list[str]:
    sub = frame[(frame["source"] == source) & (frame["label"] == label)]
    methods = [method for method in METHOD_ORDER if method in set(sub["method"])]
    sizes = [32, 64, 128]
    width = 0.78 / max(1, len(methods))
    rng = np.random.default_rng(41)
    positions: dict[tuple[int, str], float] = {}
    for method_index, method in enumerate(methods):
        means, sems = [], []
        for size in sizes:
            values = sub[(sub["method"] == method) & (sub["panel_size"] == size)]["accuracy"].dropna()
            means.append(float(values.mean()) if len(values) else np.nan)
            sems.append(float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)
        x = np.arange(3) - 0.39 + width / 2 + method_index * width
        positions.update({(size, method): float(position) for size, position in zip(sizes, x)})
        ax.bar(
            x,
            means,
            width,
            yerr=sems,
            capsize=1.5,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.4,
        )
        for position, size in zip(x, sizes):
            points = sub[(sub["method"] == method) & (sub["panel_size"] == size)]["accuracy"].dropna().to_numpy(float)
            ax.scatter(
                np.full(len(points), position) + rng.uniform(-width * 0.18, width * 0.18, len(points)),
                points,
                s=5,
                c="black",
                alpha=0.72,
                linewidths=0,
                zorder=4,
            )

    ax.set_ylim(*ylim)
    if "PERSIST-class" in methods:
        for size_index, size in enumerate(sizes):
            paired = sub[sub["panel_size"] == size].pivot_table(
                index="evaluation_seed", columns="method", values="accuracy", aggfunc="mean"
            ).dropna(subset=["SMITH", "PERSIST-class"])
            if len(paired) < 2:
                continue
            try:
                pvalue = wilcoxon(paired["SMITH"], paired["PERSIST-class"], alternative="greater").pvalue
            except ValueError:
                pvalue = 1.0
            y0, y1 = ylim
            y = y1 - (0.105 + 0.012 * (size_index % 2)) * (y1 - y0)
            left, right = positions[(size, "SMITH")], positions[(size, "PERSIST-class")]
            height = 0.022 * (y1 - y0)
            ax.plot([left, left, right, right], [y, y + height, y + height, y], color="black", lw=0.5)
            ax.text((left + right) / 2, y + height, "*" if pvalue < 0.05 else "ns", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(range(3), sizes)
    ax.set_xlabel("Panel Size")
    ax.set_ylabel("Cell Type Classification Accuracy" if label == "celltype" else "Region Classification Accuracy")
    if title:
        ax.set_title(title, fontsize=8.5, pad=5)
    return methods


def _draw_jaccard(ax, overlap: pd.DataFrame) -> None:
    groups = ["Same modality", "Cross modality"]
    colors = ["#3E7FAF", "#ACC3E2"]
    plot_frame = overlap[(overlap["method_a"] != "Spapros") & (overlap["method_b"] != "Spapros")].copy()
    rng = np.random.default_rng(2)
    for group_index, group in enumerate(groups):
        summary = plot_frame[plot_frame["modality_group"] == group].groupby("panel_size")["jaccard"]
        means = summary.mean().reindex([32, 64, 128])
        sems = summary.sem().reindex([32, 64, 128]).fillna(0)
        positions = np.arange(3) + (group_index - 0.5) * 0.32
        ax.bar(positions, means, 0.32, yerr=sems, capsize=1.5, color=colors[group_index], label=group)
        for position, size in zip(positions, [32, 64, 128]):
            points = plot_frame[(plot_frame["modality_group"] == group) & (plot_frame["panel_size"] == size)]["jaccard"].dropna().to_numpy(float)
            ax.scatter(
                np.full(len(points), position) + rng.uniform(-0.045, 0.045, len(points)),
                points,
                s=6,
                c="black",
                edgecolors="white",
                linewidths=0.2,
                alpha=0.82,
                zorder=4,
            )
    ax.set_xticks(range(3), [32, 64, 128])
    ax.set_xlabel("Panel Size")
    ax.set_ylabel("Jaccard Similarity")
    ax.set_ylim(0, max(0.45, float(plot_frame["jaccard"].max()) + 0.035))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=6.3, handlelength=1.2, borderaxespad=0)


def _draw_bias(ax, bias: pd.DataFrame, tests: pd.DataFrame | None = None) -> None:
    frame = bias[bias["panel_size"] == 128]
    if "method" in frame:
        frame = frame[frame["method"] == "SMITH"]
    order = ["Deep-RIBOmap", "Shared", "STARmap", "Background"]
    values = [frame.loc[frame["group"] == group, "ribomap_bias"].dropna().to_numpy() for group in order]
    parts = ax.violinplot(values, positions=range(4), showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], ["#3E7FAF", "#B7A6C9", "#E3A15B", "#BDBDBD"]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.35)
        body.set_alpha(0.9)
    rng = np.random.default_rng(8)
    for index, (group, group_values) in enumerate(zip(order, values)):
        shown = group_values if group != "Background" or len(group_values) <= 600 else rng.choice(group_values, 600, replace=False)
        ax.scatter(
            np.full(len(shown), index) + rng.uniform(-0.16, 0.16, len(shown)),
            shown,
            s=4 if group == "Background" else 6,
            c="black",
            alpha=0.12 if group == "Background" else 0.5,
            linewidths=0,
            zorder=4,
        )
    comparisons = ((0, 1), (0, 3))
    finite = np.concatenate([value[np.isfinite(value)] for value in values if len(value)])
    y_min, y_max = (float(finite.min()), float(finite.max())) if len(finite) else (-1.0, 1.0)
    span = max(1.0, y_max - y_min)
    for comparison_index, (left, right) in enumerate(comparisons):
        if len(values[left]) and len(values[right]):
            row = None
            if tests is not None:
                match = tests[(tests["panel_size"] == 128) & (tests["group_a"] == order[left]) &
                              (tests["group_b"] == order[right])]
                row = match.iloc[0] if len(match) else None
            pvalue = float(row["pvalue"]) if row is not None else np.nan
            y = y_max + span * (0.10 + 0.14 * comparison_index)
            height = 0.35
            ax.plot([left, left, right, right], [y, y + height, y + height, y], color="black", lw=0.5)
            label = "p<0.001" if pvalue < 0.001 else (f"p={pvalue:.3f}" if np.isfinite(pvalue) else "p=n/a")
            ax.text((left + right) / 2, y + height, label, ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0, color="#777777", lw=0.5)
    ax.set_ylim(y_min - span * 0.08, y_max + span * 0.42)
    ax.set_xticks(range(4), order, rotation=30, ha="right")
    ax.set_ylabel("RIBOMap Bias Score")


def plot(
    metrics_path: str | Path,
    overlap_path: str | Path,
    bias_path: str | Path,
    output_dir: str | Path,
    bias_tests_path: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    configure()
    metrics = pd.read_csv(metrics_path, sep="\t")
    overlap = pd.read_csv(overlap_path, sep="\t")
    bias = pd.read_csv(bias_path, sep="\t")
    bias_tests = pd.read_csv(bias_tests_path, sep="\t") if bias_tests_path else None
    output_dir = Path(output_dir)
    outputs: dict[str, dict[str, str]] = {}
    methods: list[str] = []

    for letter, (source, label, title, ylim) in PERFORMANCE_SPECS.items():
        fig, ax = plt.subplots(figsize=(2.25, 2.25), facecolor="white")
        methods = _draw_performance(ax, metrics, source, label, title, ylim)
        fig.text(0.015, 0.985, letter, ha="left", va="top", fontsize=10, weight="bold")
        fig.subplots_adjust(left=0.26, right=0.97, bottom=0.22, top=0.86)
        outputs[f"figure4_{letter}"] = save_figure(fig, output_dir / f"figure4_{letter}")
        plt.close(fig)

    fig_g, ax_g = plt.subplots(figsize=(2.15, 2.62), facecolor="white")
    _draw_jaccard(ax_g, overlap)
    fig_g.text(0.015, 0.985, "g", ha="left", va="top", fontsize=10, weight="bold")
    fig_g.subplots_adjust(left=0.25, right=0.97, bottom=0.17, top=0.78)
    outputs["figure4_g"] = save_figure(fig_g, output_dir / "figure4_g")
    plt.close(fig_g)

    fig_h, ax_h = plt.subplots(figsize=(2.30, 2.59), facecolor="white")
    _draw_bias(ax_h, bias, bias_tests)
    fig_h.text(0.015, 0.985, "h", ha="left", va="top", fontsize=10, weight="bold")
    fig_h.subplots_adjust(left=0.25, right=0.97, bottom=0.28, top=0.86)
    outputs["figure4_h"] = save_figure(fig_h, output_dir / "figure4_h")
    plt.close(fig_h)

    outputs["method_legend"] = save_method_legend(methods, output_dir / "figure4_method_legend")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export manuscript Figure 4c-h as separate panels.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--overlap", required=True)
    parser.add_argument("--bias", required=True)
    parser.add_argument("--bias-tests", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.metrics, args.overlap, args.bias, args.output_dir, args.bias_tests), indent=2))


if __name__ == "__main__":
    main()
