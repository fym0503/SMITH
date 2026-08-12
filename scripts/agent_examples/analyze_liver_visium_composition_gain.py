from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_5seed_panels"
DEFAULT_PREPARED_DIR = DEFAULT_BENCHMARK_DIR / "prepared"
DEFAULT_OUTPUT_DIR = DEFAULT_BENCHMARK_DIR / "diagnostics"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures/agent_visium_retrieval_refined"
ARIAL_FONT_FILES = [
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
]

PALETTE = {
    "hepatocyte": "#D9B36C",
    "endothelial/LSEC": "#2F7FB9",
    "mesenchymal/HSC": "#5E8C8A",
    "cholangiocyte": "#B98C3E",
    "macrophage": "#8A6AA8",
    "immune/blood": "#8A6AA8",
    "other": "#BDBDBD",
    "ink": "#222222",
}


def _configure_matplotlib() -> None:
    for font_file in ARIAL_FONT_FILES:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    mpl.rcParams.update(
        {
            "font.family": "Arial" if "Arial" in available_fonts else "DejaVu Sans",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "legend.frameon": False,
        }
    )


def _normalise_label(label: Any) -> str:
    return str(label or "").strip().lower().replace("_", " ")


def exact_family(label: Any) -> str:
    text = _normalise_label(label)
    if not text or text in {"nan", "none", "unknown"}:
        return "other"
    if text.startswith("hep") or "hepatocyte" in text:
        return "hepatocyte"
    if "lsec" in text or "endothelial" in text:
        return "endothelial/LSEC"
    if "hsc" in text or "stellate" in text or "smooth muscle" in text or "fibroblast" in text:
        return "mesenchymal/HSC"
    if "cholangiocyte" in text or "bile duct" in text:
        return "cholangiocyte"
    if "macrophage" in text:
        return "macrophage"
    if "monocyte" in text or "lymphocyte" in text or "b cell" in text or "t cell" in text:
        return "other"
    if "blood" in text or "erythroid" in text:
        return "other"
    return "other"


def broad_family(label: Any) -> str:
    family = exact_family(label)
    if family == "macrophage":
        return "immune/blood"
    text = _normalise_label(label)
    if "monocyte" in text or "lymphocyte" in text or "b cell" in text or "t cell" in text:
        return "immune/blood"
    if "blood" in text or "erythroid" in text:
        return "immune/blood"
    return family


def _read_label_counts(path: Path, label_column: str = "cell_type") -> pd.Series:
    adata = ad.read_h5ad(path, backed="r")
    try:
        if label_column not in adata.obs.columns:
            raise KeyError(f"{path} is missing `{label_column}`.")
        labels = adata.obs[label_column].astype(str)
        labels = labels[labels.notna() & ~labels.str.strip().str.lower().isin({"", "nan", "none", "unknown"})]
        return labels.value_counts().sort_index()
    finally:
        adata.file.close()


def _family_rows_for_counts(
    *,
    dataset_id: str,
    role: str,
    counts: pd.Series,
    balance_cap: int,
    mapping_name: str,
    mapper: Any,
) -> list[dict[str, Any]]:
    label_df = pd.DataFrame({"label": counts.index.astype(str), "raw_count": counts.astype(int).to_numpy()})
    label_df["family"] = label_df["label"].map(mapper)
    label_df["effective_count"] = np.minimum(label_df["raw_count"].to_numpy(dtype=int), int(balance_cap))
    total_raw = max(1, int(label_df["raw_count"].sum()))
    total_effective = max(1, int(label_df["effective_count"].sum()))
    rows = []
    for family, group in label_df.groupby("family"):
        rows.append(
            {
                "dataset_id": dataset_id,
                "role": role,
                "mapping": mapping_name,
                "family": family,
                "raw_count": int(group["raw_count"].sum()),
                "raw_fraction": float(group["raw_count"].sum() / total_raw),
                "effective_count": int(group["effective_count"].sum()),
                "effective_fraction": float(group["effective_count"].sum() / total_effective),
                "n_original_labels": int(group.shape[0]),
                "original_labels": ";".join(group["label"].astype(str).tolist()),
            }
        )
    return rows


def load_family_composition(prepared_dir: Path, *, balance_cap: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = prepared_dir / "source_scrna_smith_input.h5ad"
    visium_paths = sorted(prepared_dir.glob("*_visium_smith_input.h5ad"))
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source prepared h5ad: {source_path}")
    if not visium_paths:
        raise FileNotFoundError(f"No prepared Visium h5ad files found in {prepared_dir}")

    label_rows = []
    family_rows = []
    mapping_specs = {"exact": exact_family, "broad": broad_family}
    for role, paths in (("source_snRNA", [source_path]), ("visium_reference", visium_paths)):
        for path in paths:
            dataset_id = path.name.replace("_smith_input.h5ad", "")
            counts = _read_label_counts(path)
            for label, count in counts.items():
                label_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "role": role,
                        "label": str(label),
                        "count": int(count),
                        "exact_family": exact_family(label),
                        "broad_family": broad_family(label),
                    }
                )
            for mapping_name, mapper in mapping_specs.items():
                family_rows.extend(
                    _family_rows_for_counts(
                        dataset_id=dataset_id,
                        role=role,
                        counts=counts,
                        balance_cap=balance_cap,
                        mapping_name=mapping_name,
                        mapper=mapper,
                    )
                )
    return pd.DataFrame(label_rows), pd.DataFrame(family_rows)


def summarise_reference_composition(family_df: pd.DataFrame) -> pd.DataFrame:
    families = sorted(family_df["family"].unique())
    rows = []
    for mapping_name, mapping_df in family_df.groupby("mapping"):
        source = mapping_df[mapping_df["role"] == "source_snRNA"].set_index("family")
        visium = mapping_df[mapping_df["role"] == "visium_reference"].copy()
        n_refs = int(visium["dataset_id"].nunique())
        for family in families:
            source_raw = float(source["raw_fraction"].get(family, 0.0))
            source_effective = float(source["effective_fraction"].get(family, 0.0))
            family_ref = visium[visium["family"] == family]
            if family_ref.empty:
                ref_support = 0
                raw_mean = raw_sum = effective_mean = effective_sum = 0.0
            else:
                ref_support = int(family_ref["dataset_id"].nunique())
                raw_sum = float(family_ref["raw_count"].sum() / max(1, visium["raw_count"].sum()))
                effective_sum = float(family_ref["effective_count"].sum() / max(1, visium["effective_count"].sum()))
                per_ref = (
                    visium.pivot_table(
                        index="dataset_id",
                        columns="family",
                        values=["raw_fraction", "effective_fraction"],
                        aggfunc="sum",
                        fill_value=0.0,
                    )
                    if not visium.empty
                    else pd.DataFrame()
                )
                raw_mean = float(per_ref[("raw_fraction", family)].mean()) if ("raw_fraction", family) in per_ref else 0.0
                effective_mean = (
                    float(per_ref[("effective_fraction", family)].mean()) if ("effective_fraction", family) in per_ref else 0.0
                )
            integrated_effective = 0.5 * source_effective + 0.5 * effective_mean
            rows.append(
                {
                    "mapping": mapping_name,
                    "family": family,
                    "source_raw_fraction": source_raw,
                    "source_effective_fraction": source_effective,
                    "visium_raw_pooled_fraction": raw_sum,
                    "visium_raw_mean_reference_fraction": raw_mean,
                    "visium_effective_pooled_fraction": effective_sum,
                    "visium_effective_mean_reference_fraction": effective_mean,
                    "visium_reference_support": ref_support,
                    "visium_reference_support_fraction": float(ref_support / max(1, n_refs)),
                    "integrated_effective_fraction": integrated_effective,
                    "integrated_minus_source_effective_fraction": integrated_effective - source_effective,
                    "log2_visium_source_effective_ratio": float(
                        np.log2((effective_mean + 1e-4) / (source_effective + 1e-4))
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_delta_by_class(benchmark_dir: Path) -> pd.DataFrame:
    path = benchmark_dir / "diagnostics/per_class_f1_delta_by_seed_panel_size.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-class delta file: {path}")
    df = pd.read_csv(path, sep="\t")
    summary = (
        df.groupby(["panel_size", "class"], as_index=False)
        .agg(
            support=("support", "mean"),
            source_f1_mean=("source_f1", "mean"),
            multi_f1_mean=("multi_f1", "mean"),
            delta_f1_mean=("delta_f1", "mean"),
            delta_f1_std=("delta_f1", "std"),
        )
        .copy()
    )
    summary["exact_family"] = summary["class"].map(exact_family)
    summary["broad_family"] = summary["class"].map(broad_family)
    return summary


def compare_composition_to_delta(delta_df: pd.DataFrame, composition_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_rows = []
    for mapping_name in ("exact", "broad"):
        family_col = f"{mapping_name}_family"
        comp = composition_summary[composition_summary["mapping"] == mapping_name].copy()
        comp = comp.drop(columns=["mapping"])
        comp = comp.rename(columns={"family": family_col})
        merged = delta_df.merge(comp, on=family_col, how="left")
        merged.insert(0, "mapping", mapping_name)
        merged = merged.rename(columns={family_col: "family"})
        merged_rows.append(merged)
    merged_df = pd.concat(merged_rows, ignore_index=True)
    family_df = (
        merged_df.groupby(["mapping", "panel_size", "family"], as_index=False)
        .agg(
            n_merfish_classes=("class", "nunique"),
            support=("support", "sum"),
            source_f1_mean=("source_f1_mean", "mean"),
            multi_f1_mean=("multi_f1_mean", "mean"),
            delta_f1_mean=("delta_f1_mean", "mean"),
            source_effective_fraction=("source_effective_fraction", "first"),
            visium_effective_mean_reference_fraction=("visium_effective_mean_reference_fraction", "first"),
            visium_raw_mean_reference_fraction=("visium_raw_mean_reference_fraction", "first"),
            visium_reference_support_fraction=("visium_reference_support_fraction", "first"),
            integrated_minus_source_effective_fraction=("integrated_minus_source_effective_fraction", "first"),
            log2_visium_source_effective_ratio=("log2_visium_source_effective_ratio", "first"),
        )
        .copy()
    )
    weighted = (
        merged_df.assign(weighted_delta=lambda df: df["delta_f1_mean"] * df["support"])
        .groupby(["mapping", "panel_size", "family"], as_index=False)
        .agg(weighted_delta_sum=("weighted_delta", "sum"), support_sum=("support", "sum"))
    )
    weighted["delta_f1_support_weighted"] = weighted["weighted_delta_sum"] / weighted["support_sum"].clip(lower=1e-12)
    family_df = family_df.merge(
        weighted[["mapping", "panel_size", "family", "delta_f1_support_weighted"]],
        on=["mapping", "panel_size", "family"],
        how="left",
    )
    return merged_df, family_df


def _correlation_row(group: pd.DataFrame, metric: str, level: str) -> dict[str, Any]:
    group = group.dropna(subset=[metric, "delta_f1_mean"]).copy()
    x = group[metric].astype(float).to_numpy()
    y = group["delta_f1_mean"].astype(float).to_numpy()
    if group.shape[0] >= 3 and np.unique(x).shape[0] >= 3:
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        pearson_r = float(pearson.statistic)
        pearson_p = float(pearson.pvalue)
        spearman_r = float(spearman.statistic)
        spearman_p = float(spearman.pvalue)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")
    return {
        "level": level,
        "mapping": str(group["mapping"].iloc[0]) if "mapping" in group and not group.empty else "",
        "panel_size": str(group["panel_size"].iloc[0]) if "panel_size" in group and not group.empty else "",
        "composition_metric": metric,
        "n": int(group.shape[0]),
        "n_unique_x": int(np.unique(x).shape[0]) if x.size else 0,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def compute_correlations(class_df: pd.DataFrame, family_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "source_effective_fraction",
        "visium_raw_mean_reference_fraction",
        "visium_effective_mean_reference_fraction",
        "visium_reference_support_fraction",
        "integrated_minus_source_effective_fraction",
        "log2_visium_source_effective_ratio",
    ]
    rows = []
    for level, df in (("merfish_class", class_df), ("coarse_family", family_df)):
        for _, group in df.groupby(["mapping", "panel_size"]):
            for metric in metrics:
                rows.append(_correlation_row(group, metric, level))
    return pd.DataFrame(rows)


def plot_composition_vs_gain(family_df: pd.DataFrame, output_prefix: Path, *, mapping: str = "broad", panel_size: int = 64) -> dict[str, str]:
    _configure_matplotlib()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_df = family_df[(family_df["mapping"] == mapping) & (family_df["panel_size"].astype(int) == int(panel_size))].copy()
    plot_df = plot_df[plot_df["family"] != "other"].copy()
    x_metric = "visium_effective_mean_reference_fraction"
    corr = _correlation_row(plot_df, x_metric, "coarse_family")

    fig, ax = plt.subplots(figsize=(2.35, 2.35), facecolor="white")
    ax.axhline(0, color="#8A8A8A", lw=0.55, zorder=0)
    for _, row in plot_df.iterrows():
        family = str(row["family"])
        ax.scatter(
            row[x_metric],
            row["delta_f1_mean"],
            s=42,
            color=PALETTE.get(family, PALETTE["other"]),
            edgecolor=PALETTE["ink"],
            linewidth=0.35,
            zorder=3,
        )
        ax.text(
            row[x_metric] + 0.012,
            row["delta_f1_mean"],
            family,
            ha="left",
            va="center",
            fontsize=6.5,
        )
    ax.text(
        0.04,
        0.06,
        f"Spearman r={corr['spearman_r']:.2f}, p={corr['spearman_p']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
    )
    ax.set_xlabel("Mean effective Visium family fraction", fontsize=8.2)
    ax.set_ylabel("Cell type F1 gain\n(multi-Visium - snRNA)", fontsize=7.6)
    ax.tick_params(axis="both", labelsize=7.8)
    ax.set_xlim(left=-0.02)
    paths = {
        "pdf": str(output_prefix.with_suffix(".pdf")),
        "svg": str(output_prefix.with_suffix(".svg")),
        "png": str(output_prefix.with_suffix(".png")),
    }
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--balance-cap", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_df, family_composition = load_family_composition(Path(args.prepared_dir), balance_cap=int(args.balance_cap))
    composition_summary = summarise_reference_composition(family_composition)
    delta_df = load_delta_by_class(Path(args.benchmark_dir))
    class_comparison, family_comparison = compare_composition_to_delta(delta_df, composition_summary)
    correlations = compute_correlations(class_comparison, family_comparison)

    label_tsv = output_dir / "training_reference_label_composition.tsv"
    family_tsv = output_dir / "training_reference_family_composition_by_dataset.tsv"
    composition_summary_tsv = output_dir / "training_reference_family_composition_summary.tsv"
    class_comparison_tsv = output_dir / "visium_composition_vs_merfish_class_delta_f1.tsv"
    family_comparison_tsv = output_dir / "visium_composition_vs_merfish_family_delta_f1.tsv"
    correlations_tsv = output_dir / "visium_composition_delta_f1_correlations.tsv"
    label_df.to_csv(label_tsv, sep="\t", index=False)
    family_composition.to_csv(family_tsv, sep="\t", index=False)
    composition_summary.to_csv(composition_summary_tsv, sep="\t", index=False)
    class_comparison.to_csv(class_comparison_tsv, sep="\t", index=False)
    family_comparison.to_csv(family_comparison_tsv, sep="\t", index=False)
    correlations.to_csv(correlations_tsv, sep="\t", index=False)
    figure_paths = plot_composition_vs_gain(
        family_comparison,
        Path(args.figure_dir) / "06_visium_composition_vs_delta_f1",
        mapping="broad",
        panel_size=64,
    )

    print(
        json.dumps(
            {
                "label_tsv": str(label_tsv),
                "family_tsv": str(family_tsv),
                "composition_summary_tsv": str(composition_summary_tsv),
                "class_comparison_tsv": str(class_comparison_tsv),
                "family_comparison_tsv": str(family_comparison_tsv),
                "correlations_tsv": str(correlations_tsv),
                "figure_paths": figure_paths,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
