#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN_DIR = PROJECT_ROOT / "outputs/scrna_probedealer_full_gene_scan"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures/agent_scrna_full_probe_scan"


COLORS = {
    "feasible": "#2F6F73",
    "known_offtarget": "#B75D4A",
    "annotation_limited": "#8A8F93",
    "low_probe": "#D6A65A",
    "unresolved": "#C9D0CD",
    "rescue": "#577590",
    "target": "#2F6F73",
    "unknown": "#90979B",
    "no_hit": "#CDD3D0",
    "text": "#1F2523",
}

CLASS_LABELS = {
    "feasible_symbol_specific": "Feasible",
    "known_offtarget_risk": "Known off-target risk",
    "annotation_limited_failure": "Annotation-limited",
    "low_probe_count_failure": "Low probe count",
    "unresolved": "Unresolved",
    "annotation_rescue": "Annotation rescue",
}

CLASS_COLORS = {
    "feasible_symbol_specific": COLORS["feasible"],
    "known_offtarget_risk": COLORS["known_offtarget"],
    "annotation_limited_failure": COLORS["annotation_limited"],
    "low_probe_count_failure": COLORS["low_probe"],
    "unresolved": COLORS["unresolved"],
    "annotation_rescue": COLORS["rescue"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full scRNA ProbeDealer feasibility scan results.")
    parser.add_argument("--scan-dir", default=str(DEFAULT_SCAN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-examples", type=int, default=10)
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


def plot_feasibility_classes(summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    order = [
        "feasible_symbol_specific",
        "annotation_limited_failure",
        "known_offtarget_risk",
        "low_probe_count_failure",
        "unresolved",
        "annotation_rescue",
    ]
    data = summary.set_index("probe_feasibility_class").reindex(order).fillna(0).reset_index()
    data["label"] = data["probe_feasibility_class"].map(CLASS_LABELS)
    total = float(data["gene_count"].sum())
    data["percent"] = data["gene_count"] / total * 100

    fig, ax = plt.subplots(figsize=(3.55, 3.2))
    y = np.arange(len(data))
    colors = [CLASS_COLORS[item] for item in data["probe_feasibility_class"]]
    ax.barh(y, data["gene_count"], color=colors, height=0.62)
    for idx, row in data.iterrows():
        ax.text(
            row["gene_count"] + total * 0.012,
            idx,
            f"{int(row['gene_count']):,} ({row['percent']:.1f}%)",
            ha="left",
            va="center",
            fontsize=8.5,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("scRNA ranked genes", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, total * 1.18)
    save_figure(fig, output_dir, "probe_feasibility_class_breakdown", dpi)
    plt.close(fig)


def _is_target_locus_symbol(symbol: str, target_gene: str) -> bool:
    symbol = str(symbol or "").upper()
    target_gene = str(target_gene or "").upper()
    if not symbol or symbol == "?" or symbol.startswith("ENSG"):
        return True
    tokens = [item for item in re.split(r"[^A-Z0-9]+", symbol) if item]
    return target_gene in tokens


def _primary_offtarget(row: pd.Series) -> tuple[str, int]:
    target_gene = str(row.get("gene_symbol", ""))
    for field in ("top_cross_gene_ids", "top_cross_symbols"):
        value = str(row.get(field, "") or "")
        for item in value.split(";"):
            parts = item.split(":")
            if field == "top_cross_gene_ids":
                if len(parts) < 3:
                    continue
                symbol = parts[1]
                count_text = parts[2]
            else:
                if len(parts) < 2:
                    continue
                symbol = parts[0]
                count_text = parts[1]
            if _is_target_locus_symbol(symbol, target_gene):
                continue
            try:
                count = int(float(count_text))
            except ValueError:
                count = 0
            return symbol, count
    return "", 0


def _select_known_offtarget_examples(risk: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    data = risk[risk["clean_known_offtarget_risk"].fillna(False)].copy()
    data["gene_symbol"] = data["gene_symbol"].astype(str)
    data["unknown_fraction"] = (
        data["unknown_symbol_probe_count"].fillna(0) / data["initial_probe_count"].replace(0, np.nan)
    )
    primary = data.apply(_primary_offtarget, axis=1)
    data["primary_offtarget_symbol"] = [item[0] for item in primary]
    data["primary_offtarget_count"] = [item[1] for item in primary]
    data = data[
        ~data["gene_symbol"].str.startswith("ENSG")
        & (data["unknown_fraction"].fillna(1) <= 0.25)
        & (data["primary_offtarget_symbol"] != "")
        & (data["primary_offtarget_count"] >= 20)
    ].copy()
    preferred = [
        "TMEM140",
        "IGF1",
        "SERF2",
        "NDUFB2",
        "SRSF9",
        "PTPRCAP",
        "ZBED6",
        "JPX",
        "RGPD5",
        "PLGLB1",
    ]
    chosen = data[data["gene_symbol"].isin(preferred)].copy()
    chosen["priority"] = chosen["gene_symbol"].map({gene: idx for idx, gene in enumerate(preferred)}).fillna(999)
    chosen = chosen.sort_values("priority")
    fill = data.sort_values(
        ["clean_different_symbol_probe_count", "probes_with_known_different_symbol", "rank"],
        ascending=[False, False, True],
    )
    chosen = pd.concat([chosen, fill], ignore_index=True, sort=False)
    chosen = chosen.drop_duplicates("gene_symbol").head(max_examples)
    chosen = chosen.sort_values(
        ["probes_with_known_different_symbol", "clean_different_symbol_probe_count"],
        ascending=[False, False],
    )
    return chosen


def plot_known_offtarget_examples(risk: pd.DataFrame, output_dir: Path, dpi: int, max_examples: int) -> pd.DataFrame:
    examples = _select_known_offtarget_examples(risk, max_examples=max_examples)
    examples = examples.copy()
    examples.to_csv(output_dir / "known_offtarget_examples_for_plot.tsv", sep="\t", index=False)
    data = examples.reset_index(drop=True)
    y = np.arange(len(data))
    target = data["symbol_target_only_probe_count_known"].fillna(0).astype(float)
    # Mutually exclusive stacked categories. `probes_with_known_different_symbol`
    # can overlap with unknown-symbol hits, so the plot uses clean different-symbol
    # counts for the unambiguous off-target segment.
    off = data["clean_different_symbol_probe_count"].fillna(0).astype(float)
    unknown = (
        data["unknown_symbol_probe_count"].fillna(0).astype(float)
        + data["no_hit_probe_count"].fillna(0).astype(float)
    )

    tab20 = plt.get_cmap("tab20").colors
    target_color = tab20[0]
    off_color = tab20[1]
    unknown_color = tab20[14]

    fig, ax = plt.subplots(figsize=(3.7, 3.55))
    bar_height = 0.82
    ax.barh(y, target, color=target_color, height=bar_height, label="Target")
    ax.barh(
        y,
        off,
        left=target,
        color=off_color,
        height=bar_height,
        label="Known off-target",
    )
    ax.barh(y, unknown, left=target + off, color=unknown_color, height=bar_height, label="Unknown")
    for yi, left, width, symbol in zip(y, target, off, data["primary_offtarget_symbol"].astype(str), strict=False):
        if not symbol:
            continue
        if width >= 42:
            ax.text(
                left + width / 2,
                yi,
                symbol,
                ha="center",
                va="center",
                fontsize=8.8,
                fontstyle="italic",
                color=COLORS["text"],
            )
        else:
            ax.text(
                left + width + 3,
                yi,
                symbol,
                ha="left",
                va="center",
                fontsize=8.8,
                fontstyle="italic",
                color=COLORS["text"],
            )
    ax.set_yticks(y)
    ax.set_yticklabels(data["gene_symbol"], fontsize=10.8, fontstyle="italic")
    ax.invert_yaxis()
    ax.set_xlabel("Candidate probes", fontsize=12.2, labelpad=4)
    ax.tick_params(axis="x", labelsize=10.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max((target + off + unknown).max() * 1.06, 90))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.045),
        ncol=3,
        fontsize=11.0,
        handlelength=1.25,
        handletextpad=0.5,
        columnspacing=1.15,
        frameon=False,
    )
    fig.subplots_adjust(left=0.22, right=0.99, top=0.98, bottom=0.27)
    save_figure(fig, output_dir, "known_offtarget_examples_full_scrna", dpi)
    plt.close(fig)
    return examples


def write_source_data_and_legend(
    scan_dir: Path,
    output_dir: Path,
    summary: pd.DataFrame,
    examples: pd.DataFrame,
) -> None:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(source_dir / "probe_feasibility_class_breakdown_source_data.tsv", sep="\t", index=False)
    examples.to_csv(source_dir / "known_offtarget_examples_source_data.tsv", sep="\t", index=False)

    rank_bin_path = scan_dir / "rank_bin_feasibility_summary.tsv"
    if rank_bin_path.exists():
        rank_bins = pd.read_csv(rank_bin_path, sep="\t")
        rank_bins.to_csv(source_dir / "rank_bin_feasibility_source_data.tsv", sep="\t", index=False)

    counts = summary.set_index("probe_feasibility_class")["gene_count"].to_dict()
    total = int(sum(counts.values()))
    feasible = int(counts.get("feasible_symbol_specific", 0) + counts.get("annotation_rescue", 0))
    known_risk = int(counts.get("known_offtarget_risk", 0))
    annotation_limited = int(counts.get("annotation_limited_failure", 0))
    low_probe = int(counts.get("low_probe_count_failure", 0))
    unresolved = int(counts.get("unresolved", 0))
    rescue = int(counts.get("annotation_rescue", 0))

    legend = f"""# Full scRNA ProbeDealer Scan Figure Legend

## Suggested Legend

ProbeDealer feasibility screen across the ranked scRNA candidate gene universe used for the liver SMITH benchmark. Of {total:,} ranked genes, {feasible:,} genes were target-symbol feasible, while {known_risk:,} genes showed known off-target risk, {annotation_limited:,} were annotation-limited, {low_probe:,} had low target-compatible probe count, and {unresolved:,} could not be resolved to a local transcript. A small set of {rescue:,} genes passed after gene-symbol-aware interpretation despite failing raw Ensembl gene-ID specificity.

The known off-target example panel shows selected clean, interpretable risk genes. Bars decompose designed candidate probes into mutually exclusive target-symbol-compatible probes, unambiguous known different-symbol hits, and unknown probes. The unknown class combines probes with unresolved gene-symbol annotation and probes without a transcriptome BLAST hit under the current reference/search settings.

## Source Data

- `source_data/probe_feasibility_class_breakdown_source_data.tsv`
- `source_data/known_offtarget_examples_source_data.tsv`
- `source_data/rank_bin_feasibility_source_data.tsv`

## Interpretation Notes

- This is ProbeDealer-specific feasibility, not the integrated ODT/OligoMiner/ProbeDealer feasibility result.
- Unknown-symbol-heavy failures are separated from confirmed known off-target risks. In the example panel, the unknown category combines probes with transcriptome BLAST hits whose gene symbols could not be resolved in the local annotation map and probes without a transcriptome BLAST hit under the current reference/search settings.
- Gene-symbol-aware interpretation is required because MERFISH-style assays target RNA sequences/symbols rather than raw Ensembl gene IDs.
"""
    (output_dir / "figure_legend.md").write_text(legend, encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_style()
    scan_dir = Path(args.scan_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(scan_dir / "probe_feasibility_class_summary.tsv", sep="\t")
    risk = pd.read_csv(scan_dir / "probe_risk_summary.tsv", sep="\t")
    plot_feasibility_classes(summary, output_dir, args.dpi)
    examples = plot_known_offtarget_examples(risk, output_dir, args.dpi, args.max_examples)
    write_source_data_and_legend(scan_dir, output_dir, summary, examples)
    print(f"Wrote full scRNA ProbeDealer scan figures to {output_dir}")


if __name__ == "__main__":
    main()
