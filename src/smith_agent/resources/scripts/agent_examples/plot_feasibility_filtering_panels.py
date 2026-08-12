#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEASIBILITY_TABLE = (
    PROJECT_ROOT / "outputs/agent_feasibility_filtering_merfish317/three_tool_feasibility_table.tsv"
)
DEFAULT_TOOL_SUMMARY = PROJECT_ROOT / "outputs/agent_feasibility_filtering_merfish317/tool_pass_summary.tsv"
DEFAULT_OVERLAP = PROJECT_ROOT / "outputs/agent_feasibility_filtering_merfish317/three_tool_overlap_counts.tsv"
DEFAULT_EXAMPLES = PROJECT_ROOT / "outputs/agent_feasibility_probe_examples/manuscript_probe_specificity_examples.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures/agent_feasibility_filtering"


COLORS = {
    "pass": "#2F6F73",
    "fail": "#D7DDD8",
    "odt": "#9FB7B5",
    "oligominer": "#6E9E9A",
    "probedealer": "#2F6F73",
    "cross": "#B75D4A",
    "rescue": "#577590",
    "unknown": "#8A8F93",
    "neutral": "#C9D0CD",
    "text": "#1F2523",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot feasibility filtering result panels.")
    parser.add_argument("--feasibility-table", default=str(DEFAULT_FEASIBILITY_TABLE))
    parser.add_argument("--tool-summary", default=str(DEFAULT_TOOL_SUMMARY))
    parser.add_argument("--overlap", default=str(DEFAULT_OVERLAP))
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        if suffix == "png":
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")


def plot_filtering_funnel(tool_summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    gate_labels = {
        "transcript_resolved": "Transcript\nresolved",
        "ODT_property_ge20": "ODT\nproperty",
        "OligoMiner_geneaware_specific_ge10": "OligoMiner\nspecificity",
        "ProbeDealer_target_final_ge20": "ProbeDealer\nfinal probes",
        "three_tool_feasibility": "Integrated\nfeasible",
    }
    gates = list(gate_labels)
    summary = tool_summary.set_index("gate").loc[gates].copy()
    total = float(summary["total_count"].max())
    pass_counts = summary["pass_count"].astype(float).to_numpy()
    fail_counts = total - pass_counts
    labels = [gate_labels[g] for g in gates]

    fig, ax = plt.subplots(figsize=(3.35, 3.25))
    y = np.arange(len(labels))
    ax.barh(y, pass_counts, color=COLORS["pass"], height=0.58)
    ax.barh(y, fail_counts, left=pass_counts, color=COLORS["fail"], height=0.58)
    for idx, count in enumerate(pass_counts):
        ax.text(count + total * 0.018, idx, f"{int(count)}", ha="left", va="center", fontsize=9)
    ax.set_xlim(0, total * 1.18)
    ax.set_xlabel("Genes retained", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([label.replace("\n", " ") for label in labels], fontsize=8)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.spines["bottom"].set_bounds(0, total)
    save_figure(fig, output_dir, "01_feasibility_filtering_funnel", dpi)
    plt.close(fig)


def plot_tool_overlap(overlap: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    patterns = overlap.copy()
    order = [
        "ODT+OligoMiner+ProbeDealer",
        "ODT+OligoMiner",
        "ODT",
        "none",
    ]
    patterns["tool_pattern"] = pd.Categorical(patterns["tool_pattern"], categories=order, ordered=True)
    patterns = patterns.sort_values("tool_pattern")
    labels = patterns["tool_pattern"].astype(str).str.replace("+", "\n+", regex=False)
    counts = patterns["gene_count"].astype(int).to_numpy()
    colors = [COLORS["probedealer"], COLORS["oligominer"], COLORS["odt"], COLORS["fail"]]

    fig, ax = plt.subplots(figsize=(3.35, 3.25))
    y = np.arange(len(patterns))
    ax.barh(y, counts, color=colors, height=0.62)
    for idx, count in enumerate(counts):
        ax.text(count + 3, idx, str(count), va="center", ha="left", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Genes", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(counts) * 1.18)
    save_figure(fig, output_dir, "02_three_tool_overlap", dpi)
    plt.close(fig)


def _example_subset(examples: pd.DataFrame) -> pd.DataFrame:
    priority = [
        "IGF1",
        "NDUFB2",
        "SRSF9",
        "PYCARD",
        "PTPRCAP",
        "ARHGEF26",
        "CTSB",
        "TCF7L1",
    ]
    existing = examples[examples["gene_symbol"].isin(priority)].copy()
    if existing["gene_symbol"].nunique() < 8:
        clean = examples[examples["example_class"] == "clean_cross_gene_risk"].copy()
        clean = clean.sort_values(
            [
                "clean_different_symbol_probe_count",
                "probes_with_known_different_symbol",
                "unknown_symbol_probe_count",
            ],
            ascending=[False, False, True],
        )
        existing = pd.concat([existing, clean], ignore_index=True)
    existing["priority"] = existing["gene_symbol"].map({gene: i for i, gene in enumerate(priority)}).fillna(99)
    existing = existing.sort_values(["priority", "clean_different_symbol_probe_count"], ascending=[True, False])
    return existing.drop_duplicates("gene_symbol").head(8)


def plot_probe_specificity_examples(examples: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    data = _example_subset(examples).copy()
    data = data.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    target = data["symbol_target_only_probe_count_known"].fillna(0).astype(float)
    off = data["probes_with_known_different_symbol"].fillna(0).astype(float)
    unknown = data["unknown_symbol_probe_count"].fillna(0).astype(float)
    no_hit = data["no_hit_probe_count"].fillna(0).astype(float)

    fig, ax = plt.subplots(figsize=(3.55, 3.25))
    ax.barh(y, target, color=COLORS["pass"], height=0.62, label="Target symbol")
    ax.barh(y, off, left=target, color=COLORS["cross"], height=0.62, label="Known off-target")
    ax.barh(y, unknown, left=target + off, color=COLORS["unknown"], height=0.62, label="Unknown")
    ax.barh(y, no_hit, left=target + off + unknown, color=COLORS["neutral"], height=0.62, label="No BLAST hit")
    ax.axvline(20, color=COLORS["text"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(20, len(data) - 0.15, "20-probe cutoff", ha="center", va="bottom", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["gene_symbol"], fontsize=9)
    ax.set_xlabel("Candidate probes", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.54, -0.18),
        ncol=2,
        fontsize=7.8,
        handlelength=1.1,
        columnspacing=1.1,
    )
    ax.set_xlim(0, max((target + off + unknown + no_hit).max() * 1.05, 80))
    save_figure(fig, output_dir, "03_probe_specificity_examples", dpi)
    plt.close(fig)


def plot_annotation_rescue(examples: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    data = examples[examples["example_class"] == "gene_symbol_annotation_rescue"].copy()
    data = data[data["gene_symbol"].isin(["ARHGEF26", "CTSB", "TCF7L1"])]
    order = ["ARHGEF26", "CTSB", "TCF7L1"]
    data["gene_symbol"] = pd.Categorical(data["gene_symbol"], categories=order, ordered=True)
    data = data.sort_values("gene_symbol")
    x = np.arange(len(data))
    gene_id = data.get("geneid_target_only_probe_count")
    if gene_id is None:
        gene_id = pd.Series([0] * len(data))
    gene_id = gene_id.fillna(0).astype(float)
    symbol = data["symbol_target_only_probe_count_known"].fillna(0).astype(float)

    fig, ax = plt.subplots(figsize=(3.35, 3.25))
    ax.vlines(x, gene_id, symbol, color=COLORS["neutral"], linewidth=2.1, zorder=1)
    ax.scatter(x, gene_id, color=COLORS["neutral"], s=42, label="Gene ID only", zorder=3)
    ax.scatter(x, symbol, color=COLORS["rescue"], s=54, label="Symbol-aware", zorder=4)
    ax.axhline(20, color=COLORS["text"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(len(data) - 0.55, 22, "20-probe cutoff", ha="right", va="bottom", fontsize=8)
    for idx, value in enumerate(symbol):
        ax.text(idx, value + 4, f"{int(value)}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(gene_id):
        ax.text(idx, value + 3.2, f"{int(value)}", ha="center", va="bottom", fontsize=7.5, color=COLORS["unknown"])
    ax.set_xticks(x)
    ax.set_xticklabels(data["gene_symbol"].astype(str), fontsize=9)
    ax.set_ylabel("Target-compatible probes", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper right", fontsize=8, handlelength=1.1)
    ax.set_ylim(0, max(symbol.max() * 1.22, 120))
    ax.set_xlim(-0.45, len(data) - 0.55)
    save_figure(fig, output_dir, "04_annotation_aware_rescue", dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    output_dir = Path(args.output_dir).resolve()
    tool_summary = pd.read_csv(args.tool_summary, sep="\t")
    overlap = pd.read_csv(args.overlap, sep="\t")
    examples = pd.read_csv(args.examples, sep="\t")
    plot_filtering_funnel(tool_summary, output_dir, args.dpi)
    plot_tool_overlap(overlap, output_dir, args.dpi)
    plot_probe_specificity_examples(examples, output_dir, args.dpi)
    plot_annotation_rescue(examples, output_dir, args.dpi)
    print(f"Wrote feasibility filtering figures to {output_dir}")


if __name__ == "__main__":
    main()
