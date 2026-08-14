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


PANEL_SPECS = {
    "c": {
        "dataset": "elegans_tf",
        "metric": "cell_type_accuracy",
        "sizes": [32, 64, 128],
        "ylabel": "Cell Type Classification Accuracy",
        "title": "Transcription Factor Activities",
        "ylim": (0.2, 0.8),
    },
    "d": {
        "dataset": "elegans_tf",
        "metric": "developmental_time_pearson",
        "sizes": [32, 64, 128],
        "ylabel": "Time Correlation",
        "title": "",
        "ylim": (0.6, 1.0),
    },
    "e": {
        "dataset": "elegans_mirna",
        "metric": "cell_type_accuracy",
        "sizes": [16, 24, 32],
        "ylabel": "Cell Type Classification Accuracy",
        "title": "miRNA Activities",
        "ylim": (0.2, 0.5),
    },
    "f": {
        "dataset": "elegans_mirna",
        "metric": "developmental_time_pearson",
        "sizes": [16, 24, 32],
        "ylabel": "Time Correlation",
        "title": "",
        "ylim": (0.8, 1.0),
    },
}


def _draw_bar_panel(ax, data: pd.DataFrame, spec: dict) -> list[str]:
    subset = data[data["dataset"] == spec["dataset"]]
    methods = [method for method in METHOD_ORDER if method in set(subset["method"])]
    sizes = spec["sizes"]
    width = 0.78 / max(1, len(methods))
    rng = np.random.default_rng(31)
    positions: dict[tuple[int, str], float] = {}
    for method_index, method in enumerate(methods):
        means, errors = [], []
        for size in sizes:
            rows = subset[(subset["method"] == method) & (subset["panel_size"] == size)]
            values = rows[spec["metric"]].dropna()
            means.append(float(values.mean()) if len(values) else np.nan)
            errors.append(float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)
        x = np.arange(len(sizes)) - 0.39 + width / 2 + method_index * width
        positions.update({(size, method): float(position) for size, position in zip(sizes, x)})
        ax.bar(
            x,
            means,
            width,
            yerr=errors,
            capsize=1.5,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.4,
        )
        for position, size in zip(x, sizes):
            points = subset[(subset["method"] == method) & (subset["panel_size"] == size)][spec["metric"]].dropna().to_numpy(float)
            ax.scatter(
                np.full(len(points), position) + rng.uniform(-width * 0.18, width * 0.18, len(points)),
                points,
                s=5,
                c="black",
                alpha=0.75,
                linewidths=0,
                zorder=4,
            )

    ax.set_ylim(*spec["ylim"])
    if "PERSIST-class" in methods:
        for size_index, size in enumerate(sizes):
            paired = subset[subset["panel_size"] == size].pivot_table(
                index="split", columns="method", values=spec["metric"], aggfunc="mean"
            ).dropna(subset=["SMITH", "PERSIST-class"])
            if len(paired) < 2:
                continue
            try:
                pvalue = wilcoxon(paired["SMITH"], paired["PERSIST-class"], alternative="greater").pvalue
            except ValueError:
                pvalue = 1.0
            y0, y1 = spec["ylim"]
            y = y1 - (0.105 + 0.012 * (size_index % 2)) * (y1 - y0)
            left, right = positions[(size, "SMITH")], positions[(size, "PERSIST-class")]
            height = 0.022 * (y1 - y0)
            ax.plot([left, left, right, right], [y, y + height, y + height, y], color="black", lw=0.5)
            ax.text((left + right) / 2, y + height, "*" if pvalue < 0.05 else "ns", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(np.arange(len(sizes)), [str(size) for size in sizes])
    ax.set_xlabel("Panel Size")
    ax.set_ylabel(spec["ylabel"])
    if spec["title"]:
        ax.set_title(spec["title"], fontsize=8.5, pad=5)
    return methods


def plot(values_path: str | Path, output_dir: str | Path) -> dict[str, dict[str, str]]:
    configure()
    data = pd.read_csv(values_path, sep="\t")
    output_dir = Path(output_dir)
    outputs: dict[str, dict[str, str]] = {}
    methods: list[str] = []
    for letter, spec in PANEL_SPECS.items():
        fig, ax = plt.subplots(figsize=(2.35, 2.10), facecolor="white")
        methods = _draw_bar_panel(ax, data, spec)
        fig.text(0.015, 0.985, letter, ha="left", va="top", fontsize=10, weight="bold")
        fig.subplots_adjust(left=0.25, right=0.97, bottom=0.22, top=0.86)
        outputs[f"figure3_{letter}"] = save_figure(fig, output_dir / f"figure3_{letter}")
        plt.close(fig)
    outputs["method_legend"] = save_method_legend(methods, output_dir / "figure3_method_legend")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export manuscript Figure 3c-f as separate panels.")
    parser.add_argument("--values", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.values, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
