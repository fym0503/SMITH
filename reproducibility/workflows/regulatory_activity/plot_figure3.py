#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
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
    required = {"tissue", "progenitor_lineage", "temporal_module", "gene_symbol"}
    if not required.issubset(modules.columns):
        raise ValueError(f"Module table lacks {sorted(required - set(modules.columns))}")
    summary = (
        modules.groupby(["tissue", "temporal_module"], observed=False)["gene_symbol"]
        .nunique()
        .unstack(fill_value=0)
    )
    if summary.empty:
        raise ValueError("No annotated modules available for Figure 3g")
    temporal_order = [item for item in ("Early", "Middle", "Late") if item in summary.columns]
    temporal_order.extend(item for item in summary.columns if item not in temporal_order)
    summary = summary.reindex(columns=temporal_order)
    image = ax.imshow(summary.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(np.arange(len(summary.columns)), summary.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(summary.index)), [str(item).title() for item in summary.index])
    ax.set_xlabel("Temporal module")
    ax.set_ylabel("Tissue system")
    ax.set_title("Spatiotemporal TF programs", fontsize=8.5, pad=5)
    for row in range(summary.shape[0]):
        for column in range(summary.shape[1]):
            value = int(summary.iloc[row, column])
            ax.text(column, row, value, ha="center", va="center", fontsize=6,
                    color="white" if value > summary.to_numpy().max() * 0.55 else "black")
    image.set_rasterized(True)


def _draw_module_coverage(ax, data: pd.DataFrame) -> None:
    for method, frame in data.groupby("method"):
        summary = frame.groupby("panel_size")["module_miss_rate"].mean().sort_index()
        ax.scatter(summary.index, summary.values, s=20, label=method,
                   color=METHOD_COLORS.get(method, "#777777"), edgecolor="black", linewidth=0.35)
    ax.set_title("Developmental module coverage", fontsize=8.5, pad=5)
    ax.set_xlabel("Panel size")
    ax.set_ylabel("Module miss rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=5.5)


def _draw_coactivity(ax, data: pd.DataFrame) -> None:
    methods = [method for method in ("SMITH", "PERSIST") if method in set(data["method"])]
    if not methods:
        raise ValueError("Co-activity output contains neither SMITH nor PERSIST")
    width = 0.72 / len(methods)
    x = np.arange(len(LINEAGE_NAMES))
    for method_index, method in enumerate(methods):
        means, errors = [], []
        for lineage in LINEAGE_NAMES:
            values = data[(data["method"] == method) & (data["lineage"] == lineage)]["pearson"].dropna()
            means.append(float(values.mean()) if len(values) else np.nan)
            errors.append(float(values.sem()) if len(values) > 1 else 0.0)
        positions = x - 0.36 + width / 2 + method_index * width
        ax.bar(positions, means, width, yerr=errors, capsize=1.5,
               color=METHOD_COLORS[method], edgecolor="black", linewidth=0.4, label=method)
    ax.set_title("TF co-activity reconstruction", fontsize=8.5, pad=5)
    ax.set_ylabel("Pearson agreement")
    ax.set_xticks(x, [item.title() for item in LINEAGE_NAMES], rotation=25, ha="right")
    ax.set_ylim(-1, 1)
    ax.legend(frameon=False, fontsize=5.5)


def _plot_tf_correlation(data: pd.DataFrame, metrics_path: str | Path):
    if data.empty:
        raise ValueError("TF/scRNA correlation output is empty")
    matrix_file = Path(metrics_path).parent / str(data.iloc[0]["matrix_file"])
    if not matrix_file.is_file():
        raise FileNotFoundError(f"Figure 3j matrix output is missing: {matrix_file}")
    matrices = np.load(matrix_file, allow_pickle=False)
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.25), facecolor="white")
    for ax, key, title in zip(axes, ("scrna", "tf"), ("scRNA-seq", "TF activity")):
        image = ax.imshow(matrices[key], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        image.set_rasterized(True)
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].text(0.98, 0.02, f"corr = {float(data.iloc[0]['mean_rowwise_pearson']):.2f}",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=7,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.86, wspace=0.08)
    return fig


def _draw_transfer(ax, data: pd.DataFrame) -> None:
    metric = "cell_type_accuracy"
    if metric not in data:
        raise ValueError(f"Transfer output lacks {metric}")
    sources = [source for source in ("TF-TF", "RNA-TF") if source in set(data["source_modality"])]
    methods = [method for method in ("SMITH", "PERSIST-class") if method in set(data["method"])]
    combinations = [(method, source) for method in methods for source in sources]
    sizes = sorted(data["panel_size"].dropna().astype(int).unique())
    width = 0.78 / max(len(combinations), 1)
    colors = {("SMITH", "TF-TF"): "#2f75b5", ("SMITH", "RNA-TF"): "#e07a5f",
              ("PERSIST-class", "TF-TF"): "#9ab7d5", ("PERSIST-class", "RNA-TF"): "#edb09e"}
    y = np.arange(len(sizes))
    for index, (method, source) in enumerate(combinations):
        means = []
        for size in sizes:
            values = data[(data["method"] == method) & (data["source_modality"] == source)
                          & (data["panel_size"] == size)][metric].dropna()
            means.append(float(values.mean()) if len(values) else np.nan)
        positions = y - 0.39 + width / 2 + index * width
        ax.barh(positions, means, height=width, color=colors[(method, source)],
                edgecolor="black", linewidth=0.35, label=f"{method}: {source}")
    ax.set_title("Panel transfer into TF activity", fontsize=8.5, pad=5)
    ax.set_xlabel("Cell-type accuracy")
    ax.set_ylabel("Panel size")
    ax.set_yticks(y, [str(size) for size in sizes])
    ax.legend(frameon=False, fontsize=5.2, loc="lower right")


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
    panels: Iterable[str] | None = None,
) -> dict[str, dict[str, str]]:
    configure()
    data = pd.read_csv(values_path, sep="\t")
    output_dir = Path(output_dir)
    paper_paths = {
        "g": modules_path,
        "h": module_coverage_path,
        "i": coactivity_path,
        "j": correlation_path,
        "k": transfer_path,
    }
    if panels is None:
        selected = {*PANEL_SPECS, "legend"}
        selected.update(letter for letter, path in paper_paths.items() if path is not None)
    else:
        selected = set(panels)
    unknown = selected - {*PANEL_SPECS, "g", "h", "i", "j", "k", "legend"}
    if unknown:
        raise ValueError(f"Unknown Figure 3 panels: {sorted(unknown)}")
    outputs: dict[str, dict[str, str]] = {}
    methods: list[str] = []
    for letter, spec in PANEL_SPECS.items():
        if letter not in selected:
            continue
        fig, ax = plt.subplots(figsize=(2.35, 2.10), facecolor="white")
        methods = _draw_bar_panel(ax, data, spec)
        fig.text(0.015, 0.985, letter, ha="left", va="top", fontsize=10, weight="bold")
        fig.subplots_adjust(left=0.25, right=0.97, bottom=0.22, top=0.86)
        outputs[f"figure3_{letter}"] = save_figure(fig, output_dir / f"figure3_{letter}")
        plt.close(fig)
    if "legend" in selected:
        if not methods:
            methods = [method for method in METHOD_ORDER if method in set(data["method"])]
        outputs["method_legend"] = save_method_legend(methods, output_dir / "figure3_method_legend")

    missing = [letter for letter, path in paper_paths.items() if letter in selected and path is None]
    if missing:
        raise ValueError(f"Figure 3 panels {missing} require their corresponding analysis inputs")
    if selected & set(paper_paths):
        panels = {
            "g": lambda ax: _draw_module_schematic(ax, pd.read_csv(modules_path, sep="\t")),
            "h": lambda ax: _draw_module_coverage(ax, pd.read_csv(module_coverage_path, sep="\t")),
            "i": lambda ax: _draw_coactivity(ax, pd.read_csv(coactivity_path, sep="\t")),
            "k": lambda ax: _draw_transfer(ax, pd.read_csv(transfer_path, sep="\t")),
        }
        for letter, draw in panels.items():
            if letter not in selected:
                continue
            fig, ax = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
            draw(ax)
            fig.text(0.015, 0.985, letter, ha="left", va="top", fontsize=10, weight="bold")
            fig.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
            outputs[f"figure3_{letter}"] = save_figure(fig, output_dir / f"figure3_{letter}")
            plt.close(fig)
        if "j" in selected:
            correlation_data = pd.read_csv(correlation_path, sep="\t")
            fig = _plot_tf_correlation(correlation_data, correlation_path)
            fig.text(0.008, 0.985, "j", ha="left", va="top", fontsize=10, weight="bold")
            outputs["figure3_j"] = save_figure(fig, output_dir / "figure3_j")
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
    parser.add_argument(
        "--panels",
        default=None,
        help="Comma-separated panels to render (c,d,e,f,g,h,i,j,k,legend).",
    )
    args = parser.parse_args()
    print(json.dumps(plot(args.values, args.output_dir, modules_path=args.modules, module_coverage_path=args.module_coverage,
                          coactivity_path=args.coactivity, correlation_path=args.correlation, transfer_path=args.transfer,
                          panels=args.panels.split(",") if args.panels else None), indent=2))


if __name__ == "__main__":
    main()
