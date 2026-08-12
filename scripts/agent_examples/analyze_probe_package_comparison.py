#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERFISH_TABLE = (
    PROJECT_ROOT / "outputs/agent_feasibility_filtering_merfish317/three_tool_feasibility_table.tsv"
)
DEFAULT_TOP128_TABLE = (
    PROJECT_ROOT / "outputs/agent_feasibility_filtering_top128/three_tool_feasibility_table.tsv"
)
DEFAULT_FULL_SCRNA_TABLE = (
    PROJECT_ROOT / "outputs/scrna_full_three_tool_feasibility/three_tool_feasibility_table.tsv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures/agent_probe_package_comparison"


COLORS = {
    "odt": "#6E9E9A",
    "oligominer": "#8BA7C2",
    "oligominer_strict": "#C8B56A",
    "probedealer": "#B75D4A",
    "integrated": "#2F6F73",
    "neutral": "#C9D0CD",
    "text": "#1F2523",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare probe feasibility package outputs.")
    parser.add_argument("--merfish-table", default=str(DEFAULT_MERFISH_TABLE))
    parser.add_argument("--top128-table", default=str(DEFAULT_TOP128_TABLE))
    parser.add_argument("--full-scrna-table", default=str(DEFAULT_FULL_SCRNA_TABLE))
    parser.add_argument(
        "--mode",
        choices=["full-scrna", "legacy-panels"],
        default="full-scrna",
        help="`full-scrna` uses the full scRNA-wide three-tool scan; `legacy-panels` uses MERFISH 317 and top-128.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def setup_style() -> None:
    arial_fonts = [
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
    ]
    for font_path in arial_fonts:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
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


def load_tables(
    merfish_table: str | Path,
    top128_table: str | Path,
    full_scrna_table: str | Path,
    mode: str,
) -> pd.DataFrame:
    frames = []
    if mode == "full-scrna":
        table_specs = [("Full scRNA gene universe", full_scrna_table)]
    else:
        table_specs = [
            ("MERFISH gene universe", merfish_table),
            ("SMITH top-128 panel", top128_table),
        ]
    for label, path in table_specs:
        df = pd.read_csv(path, sep="\t")
        df["analysis_set"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_pass_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tools = [
        ("Transcript resolved", "transcript_resolved", COLORS["neutral"]),
        ("ODT property", "pass_odt_property_20", COLORS["odt"]),
        ("OligoMiner gene-aware", "pass_oligominer_geneaware_10", COLORS["oligominer"]),
        ("ProbeDealer target", "pass_probedealer_target_20", COLORS["probedealer"]),
        ("Integrated", "pass_three_tool_feasibility", COLORS["integrated"]),
    ]
    for analysis_set, sub in df.groupby("analysis_set", sort=False):
        for tool_name, col, color in tools:
            passed = int(sub[col].fillna(False).astype(bool).sum())
            rows.append(
                {
                    "analysis_set": analysis_set,
                    "tool": tool_name,
                    "column": col,
                    "pass_count": passed,
                    "total_count": int(len(sub)),
                    "pass_fraction": passed / len(sub),
                    "color": color,
                }
            )
    return pd.DataFrame(rows)


def summarize_failure_attribution(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis_set, sub in df.groupby("analysis_set", sort=False):
        for row in sub.itertuples(index=False):
            if bool(getattr(row, "pass_three_tool_feasibility")):
                category = "pass"
            elif not bool(getattr(row, "transcript_resolved")):
                category = "transcript unresolved"
            elif not bool(getattr(row, "pass_odt_property_20")):
                category = "ODT property"
            elif not bool(getattr(row, "pass_oligominer_geneaware_10")):
                category = "OligoMiner gene-aware"
            elif not bool(getattr(row, "pass_probedealer_target_20")):
                category = "ProbeDealer target"
            else:
                category = "other"
            rows.append(
                {
                    "analysis_set": analysis_set,
                    "gene_symbol": getattr(row, "gene_symbol"),
                    "failure_attribution": category,
                }
            )
    return pd.DataFrame(rows)


def summarize_probe_counts(df: pd.DataFrame) -> pd.DataFrame:
    metric_map = {
        "ODT property": "odt_property_probe_count",
        "OligoMiner strict": "oligominer_strict_specific_probe_count",
        "OligoMiner gene-aware": "oligominer_geneaware_specific_probe_count",
        "ProbeDealer target": "probedealer_target_final_probe_count",
    }
    rows = []
    for analysis_set, sub in df.groupby("analysis_set", sort=False):
        for tool_name, col in metric_map.items():
            values = pd.to_numeric(sub[col], errors="coerce").fillna(0)
            rows.append(
                {
                    "analysis_set": analysis_set,
                    "tool": tool_name,
                    "n": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "p10": float(values.quantile(0.10)),
                    "p25": float(values.quantile(0.25)),
                    "p75": float(values.quantile(0.75)),
                    "p90": float(values.quantile(0.90)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def plot_pass_rates(pass_rates: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    data = pass_rates[
        pass_rates["tool"].isin(
            ["ODT property", "OligoMiner gene-aware", "ProbeDealer target", "Integrated"]
        )
    ].copy()
    order = ["ODT property", "OligoMiner gene-aware", "ProbeDealer target", "Integrated"]
    labels = ["ODT", "OligoMiner", "ProbeDealer", "Integrated"]
    tab20 = plt.get_cmap("tab20").colors
    colors = [tab20[15], tab20[15], tab20[15], tab20[0]]
    sub = data[data["analysis_set"] == data["analysis_set"].drop_duplicates().iloc[0]].set_index("tool").loc[order]
    values = sub["pass_fraction"].to_numpy() * 100
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(3.05, 3.0))
    ax.bar(x, values, color=colors, width=0.68, edgecolor="none")
    for xi, value in zip(x, values, strict=False):
        ax.text(
            xi,
            value + 1.6,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.8,
            color=COLORS["text"],
        )

    ax.set_xlabel("")
    ax.set_ylabel("Genes passing (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.8, rotation=28, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=10.8, length=3.5, width=0.8)
    ax.tick_params(axis="x", length=0)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    fig.subplots_adjust(left=0.22, bottom=0.27, right=0.98, top=0.95)
    save_figure(fig, output_dir, "01_package_pass_rate_comparison", dpi)
    plt.close(fig)


def plot_failure_attribution(attribution: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    order = ["pass", "ProbeDealer target", "OligoMiner gene-aware", "ODT property", "transcript unresolved"]
    colors = {
        "pass": COLORS["integrated"],
        "ProbeDealer target": COLORS["probedealer"],
        "OligoMiner gene-aware": COLORS["oligominer"],
        "ODT property": COLORS["odt"],
        "transcript unresolved": COLORS["neutral"],
    }
    counts = (
        attribution.groupby(["analysis_set", "failure_attribution"])
        .size()
        .reset_index(name="gene_count")
    )
    sets = attribution["analysis_set"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(3.35, 3.15))
    y = np.arange(len(sets))
    left = np.zeros(len(sets))
    for category in order:
        vals = []
        for analysis_set in sets:
            match = counts[(counts["analysis_set"] == analysis_set) & (counts["failure_attribution"] == category)]
            vals.append(int(match["gene_count"].iloc[0]) if len(match) else 0)
        ax.barh(y, vals, left=left, color=colors[category], height=0.55, label=category)
        left += np.array(vals)
    ax.set_yticks(y)
    labels = {
        "Full scRNA gene universe": "Full scRNA\ngenes",
        "MERFISH gene universe": "MERFISH\n317",
        "SMITH top-128 panel": "Top-128\npanel",
    }
    ax.set_yticklabels([labels.get(item, item) for item in sets], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Genes", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7.3)
    save_figure(fig, output_dir, "02_failure_attribution_by_package", dpi)
    plt.close(fig)


def plot_failure_only_attribution(attribution: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fail = attribution[attribution["failure_attribution"] != "pass"].copy()
    order = ["ProbeDealer target", "OligoMiner gene-aware", "ODT property", "transcript unresolved"]
    colors = {
        "ProbeDealer target": COLORS["probedealer"],
        "OligoMiner gene-aware": COLORS["oligominer"],
        "ODT property": COLORS["odt"],
        "transcript unresolved": COLORS["neutral"],
    }
    counts = fail.groupby(["analysis_set", "failure_attribution"]).size().reset_index(name="gene_count")
    sets = attribution["analysis_set"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    y = np.arange(len(sets))
    left = np.zeros(len(sets))
    for category in order:
        vals = []
        for analysis_set in sets:
            match = counts[(counts["analysis_set"] == analysis_set) & (counts["failure_attribution"] == category)]
            vals.append(int(match["gene_count"].iloc[0]) if len(match) else 0)
        ax.barh(y, vals, left=left, color=colors[category], height=0.55, label=category)
        for idx, val in enumerate(vals):
            if val:
                ax.text(left[idx] + val / 2, idx, str(val), ha="center", va="center", fontsize=8)
        left += np.array(vals)
    totals = fail.groupby("analysis_set").size().reindex(sets).fillna(0).astype(int)
    for idx, total in enumerate(totals):
        ax.text(total + 0.6, idx, f"n={total}", ha="left", va="center", fontsize=8)
    ax.set_yticks(y)
    labels = {
        "Full scRNA gene universe": "Full scRNA\nfailures",
        "MERFISH gene universe": "MERFISH\nfailures",
        "SMITH top-128 panel": "Top-128\nfailures",
    }
    ax.set_yticklabels([labels.get(item, item) for item in sets], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Failed genes", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(totals.max() * 1.25, 5))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7.3)
    save_figure(fig, output_dir, "02b_failure_only_attribution_by_package", dpi)
    plt.close(fig)


def plot_probe_count_distributions(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    if "Full scRNA gene universe" in set(df["analysis_set"].astype(str)):
        sub = df[df["analysis_set"] == "Full scRNA gene universe"].copy()
    else:
        sub = df[df["analysis_set"] == "MERFISH gene universe"].copy()
    metrics = [
        ("ODT", "odt_property_probe_count", COLORS["odt"]),
        ("OligoMiner\nstrict", "oligominer_strict_specific_probe_count", COLORS["oligominer_strict"]),
        ("OligoMiner\ngene-aware", "oligominer_geneaware_specific_probe_count", COLORS["oligominer"]),
        ("ProbeDealer", "probedealer_target_final_probe_count", COLORS["probedealer"]),
    ]
    values = [np.log10(pd.to_numeric(sub[col], errors="coerce").fillna(0).to_numpy() + 1) for _, col, _ in metrics]
    fig, ax = plt.subplots(figsize=(3.35, 3.15))
    violin = ax.violinplot(values, showmeans=False, showmedians=True, widths=0.78)
    for body, (_, _, color) in zip(violin["bodies"], metrics, strict=False):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.82)
    for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
        violin[key].set_color(COLORS["text"])
        violin[key].set_linewidth(0.8)
    ax.axhline(np.log10(21), color=COLORS["text"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(4.35, np.log10(21) + 0.05, "20-probe cutoff", ha="right", va="bottom", fontsize=7.5)
    ax.set_ylabel("log10(probe count + 1)", fontsize=10)
    ax.set_xticks(np.arange(1, len(metrics) + 1))
    ax.set_xticklabels([item[0] for item in metrics], fontsize=7.4)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", length=0)
    save_figure(fig, output_dir, "03_probe_count_distribution_by_package", dpi)
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    combined: pd.DataFrame,
    pass_rates: pd.DataFrame,
    attribution: pd.DataFrame,
    probe_counts: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pass_rates.drop(columns=["color"]).to_csv(source_dir / "package_pass_rates.tsv", sep="\t", index=False)
    attribution.to_csv(source_dir / "failure_attribution_by_gene.tsv", sep="\t", index=False)
    (
        attribution.groupby(["analysis_set", "failure_attribution"])
        .size()
        .reset_index(name="gene_count")
        .to_csv(source_dir / "failure_attribution_counts.tsv", sep="\t", index=False)
    )
    probe_counts.to_csv(source_dir / "probe_count_distribution_summary.tsv", sep="\t", index=False)

    notes = {
        "inputs": {
            "analysis_sets": combined["analysis_set"].drop_duplicates().tolist(),
            "n_rows": int(len(combined)),
        },
        "interpretation": [
            "ODT, OligoMiner and ProbeDealer are not interchangeable tools; they measure different probe-design constraints.",
            "OligoMiner is summarized with the gene-aware criterion, which is the relevant main criterion for symbol-targeted panels.",
            "Integrated feasibility is defined as transcript resolution plus passing ODT property, OligoMiner gene-aware specificity and ProbeDealer target-compatible probe-count gates.",
            "The default figure summarizes the full ranked scRNA candidate gene universe.",
        ],
    }
    (output_dir / "package_comparison_summary.json").write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    legend = """# Probe Package Comparison Figure Notes

## Suggested interpretation

The package-comparison analysis evaluates three complementary feasibility constraints on the full ranked scRNA candidate gene universe. ODT/SCRINSHOT reports property-filtered candidate probes, OligoMiner reports gene-aware sequence specificity, and ProbeDealer reports target-compatible deployable probes after transcriptome filtering. These outputs should be interpreted as orthogonal gates rather than as a single accuracy benchmark.

In the full scRNA-wide scan, 12,060 of 12,160 genes were resolved to local human transcripts. ODT property filtering retained 12,057 genes, gene-aware OligoMiner retained 11,332 genes, ProbeDealer retained 11,069 genes, and 10,873 genes passed all gates.

Integrated feasibility is defined as passing all three gates after transcript resolution: ODT property filtering, gene-aware OligoMiner specificity and ProbeDealer target-compatible probe count.
"""
    (output_dir / "figure_legend.md").write_text(legend, encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_style()
    output_dir = Path(args.output_dir).resolve()
    combined = load_tables(args.merfish_table, args.top128_table, args.full_scrna_table, args.mode)
    pass_rates = summarize_pass_rates(combined)
    attribution = summarize_failure_attribution(combined)
    probe_counts = summarize_probe_counts(combined)
    write_outputs(output_dir, combined, pass_rates, attribution, probe_counts)
    plot_pass_rates(pass_rates, output_dir, args.dpi)
    plot_failure_attribution(attribution, output_dir, args.dpi)
    plot_failure_only_attribution(attribution, output_dir, args.dpi)
    plot_probe_count_distributions(combined, output_dir, args.dpi)
    print(f"Wrote probe package comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
