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

LINEAGE_NAMES = ("muscle", "neuron", "pharynx", "skin")


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


def _draw_module_schematic(ax, modules: pd.DataFrame) -> None:
    grouped = modules.groupby("module_id")["gene_symbol"].apply(list)
    names = list(grouped.index)
    if not names:
        raise ValueError("No annotated modules available for Figure 3g")
    for index, name in enumerate(names):
        genes = grouped.loc[name]
        y = len(names) - index
        ax.scatter(np.arange(len(genes)), np.full(len(genes), y), s=18, color="#2f75b5", zorder=3)
        ax.plot(np.arange(len(genes)), np.full(len(genes), y), color="#a8b7c7", lw=0.8, zorder=1)
        ax.text(len(genes) + 0.15, y, str(name), va="center", fontsize=6)
    ax.set_title("Annotated spatiotemporal modules", fontsize=8.5, pad=5)
    ax.set_xlabel("TFs within module")
    ax.set_yticks([])
    ax.set_xlim(-0.5, max(len(values) for values in grouped) + 2)
    ax.set_ylim(0.3, len(names) + 0.7)


def _draw_module_coverage(ax, data: pd.DataFrame) -> None:
    for method, frame in data.groupby("method"):
        summary = frame.groupby("panel_size")["module_miss_rate"].mean().sort_index()
        ax.plot(summary.index, summary.values, marker="o", lw=1.2, label=method, color=METHOD_COLORS.get(method, "#777777"))
    ax.set_title("Developmental module coverage", fontsize=8.5, pad=5)
    ax.set_xlabel("Panel size")
    ax.set_ylabel("Module miss rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=5.5)


def _draw_coactivity(ax, data: pd.DataFrame) -> None:
    grouped = data.groupby("lineage")["pearson"].mean().reindex(LINEAGE_NAMES).dropna()
    ax.bar(np.arange(len(grouped)), grouped.values, color="#4f81bd", edgecolor="black", linewidth=0.4)
    ax.set_title("TF co-activity reconstruction", fontsize=8.5, pad=5)
    ax.set_ylabel("Pearson agreement")
    ax.set_xticks(np.arange(len(grouped)), [str(item).title() for item in grouped.index], rotation=25, ha="right")
    ax.set_ylim(-1, 1)


def _draw_tf_correlation(ax, data: pd.DataFrame) -> None:
    if data.empty:
        raise ValueError("TF/scRNA correlation output is empty")
    value = float(data.iloc[0]["pearson"])
    ax.text(0.5, 0.55, f"r = {value:.2f}", ha="center", va="center", fontsize=18, color="#2f75b5")
    ax.text(0.5, 0.28, "TF-TF correlation structure\nscRNA versus activity atlas", ha="center", va="center", fontsize=7)
    ax.set_title("Cross-modality TF structure", fontsize=8.5, pad=5)
    ax.axis("off")


def _draw_transfer(ax, data: pd.DataFrame) -> None:
    metric = "developmental_time_pearson"
    if metric not in data:
        raise ValueError(f"Transfer output lacks {metric}")
    grouped = data.groupby(["panel_size", "source_modality"])[metric].mean().unstack(fill_value=np.nan)
    grouped.plot(kind="bar", ax=ax, color={"TF-to-TF": "#2f75b5", "scRNA-to-TF": "#e07a5f"}, edgecolor="black", linewidth=0.4)
    ax.set_title("scRNA-to-TF panel transfer", fontsize=8.5, pad=5)
    ax.set_xlabel("Panel size")
    ax.set_ylabel("Developmental-time Pearson r")
    ax.legend(frameon=False, fontsize=5.5)


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


def plot(
    values_path: str | Path,
    output_dir: str | Path,
    *,
    modules_path: str | Path | None = None,
    module_coverage_path: str | Path | None = None,
    coactivity_path: str | Path | None = None,
    correlation_path: str | Path | None = None,
    transfer_path: str | Path | None = None,
) -> dict[str, dict[str, str]]:
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
    paper_inputs = (modules_path, module_coverage_path, coactivity_path, correlation_path, transfer_path)
    if any(item is not None for item in paper_inputs) and not all(item is not None for item in paper_inputs):
        raise ValueError("Figure 3g-k plotting requires modules, coverage, coactivity, correlation and transfer outputs")
    if all(item is not None for item in paper_inputs):
        module_table = pd.read_csv(modules_path, sep="\t")
        panels = {
            "g": lambda ax: _draw_module_schematic(ax, module_table),
            "h": lambda ax: _draw_module_coverage(ax, pd.read_csv(module_coverage_path, sep="\t")),
            "i": lambda ax: _draw_coactivity(ax, pd.read_csv(coactivity_path, sep="\t")),
            "j": lambda ax: _draw_tf_correlation(ax, pd.read_csv(correlation_path, sep="\t")),
            "k": lambda ax: _draw_transfer(ax, pd.read_csv(transfer_path, sep="\t")),
        }
        for letter, draw in panels.items():
            fig, ax = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
            draw(ax)
            fig.text(0.015, 0.985, letter, ha="left", va="top", fontsize=10, weight="bold")
            fig.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
            outputs[f"figure3_{letter}"] = save_figure(fig, output_dir / f"figure3_{letter}")
            plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export manuscript Figure 3c-k as separate panels.")
    parser.add_argument("--values", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modules")
    parser.add_argument("--module-coverage")
    parser.add_argument("--coactivity")
    parser.add_argument("--correlation")
    parser.add_argument("--transfer")
    args = parser.parse_args()
    print(json.dumps(plot(args.values, args.output_dir, modules_path=args.modules, module_coverage_path=args.module_coverage,
                          coactivity_path=args.coactivity, correlation_path=args.correlation, transfer_path=args.transfer), indent=2))


if __name__ == "__main__":
    main()
