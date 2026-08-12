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
from scipy import sparse
from scipy.stats import mannwhitneyu


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_5seed_panels"
DEFAULT_PREPARED_DIR = DEFAULT_BENCHMARK_DIR / "prepared"
DEFAULT_MERFISH = REPO_ROOT / "data/liver_merfish/adata_healthy_merfish.h5ad"
DEFAULT_OUTPUT_DIR = DEFAULT_BENCHMARK_DIR / "diagnostics"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures/agent_visium_retrieval_refined"
ARIAL_FONT_FILES = [
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
]

VISIUM_REFERENCE_NAMES = [
    "PSC011_C1_visium",
    "C73_B1_healthy_visium",
    "H35_sample1_visium",
    "WSSS_F_IMMsp9838712_visium",
    "H35_sample2_visium",
]
HIGH_GAIN_CLASSES = ("LSEC", "HSC_1", "HSC_2", "Macrophage_1", "Macrophage_2")

PALETTE = {
    "Visium-added": "#2F7FB9",
    "snRNA-only": "#BDBDBD",
    "Shared": "#6EA45E",
    "Other": "#E4E4E4",
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


def _read_gene_list(path: Path, panel_size: int | None = None) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    gene_col = next((col for col in df.columns if str(col).lower() in {"gene", "genes", "gene_symbol", "marker"}), df.columns[0])
    genes = [str(gene).strip().upper() for gene in df[gene_col].tolist() if str(gene).strip()]
    return genes if panel_size is None else genes[:panel_size]


def _normalise_expression_matrix(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x
    if float(np.nanmin(x)) < 0:
        return x
    totals = x.sum(axis=1, keepdims=True)
    x = np.divide(x, np.maximum(totals, 1e-8), out=np.zeros_like(x, dtype=np.float32), where=totals > 0)
    x = x * 1e4
    return np.log1p(x, dtype=np.float32)


def _gene_index(path: Path) -> dict[str, int]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        index: dict[str, int] = {}
        for idx, gene in enumerate(adata.var_names.astype(str).tolist()):
            clean = str(gene).strip().upper()
            if clean and clean not in index:
                index[clean] = idx
        return index
    finally:
        adata.file.close()


def load_rank_and_membership(benchmark_dir: Path, panel_sizes: tuple[int, ...] = (32, 64, 128)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_rows = []
    membership_rows = []
    for seed_dir in sorted(benchmark_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        rank_path = seed_dir / "formal_rank_aggregation" / "integrated_panel_rank.tsv"
        if not rank_path.exists():
            continue
        rank = pd.read_csv(rank_path, sep="\t")
        rank["seed"] = seed
        rank["visium_rank_lift"] = rank["mean_reference_score"] - rank["source_score"]
        rank_rows.append(rank)
        for panel_size in panel_sizes:
            source_panel = seed_dir / "panels" / f"source_top_{panel_size}_panel.tsv"
            integrated_panel = seed_dir / "panels" / f"multi_visium_top_{panel_size}_panel.tsv"
            if not source_panel.exists() or not integrated_panel.exists():
                continue
            source_genes = set(_read_gene_list(source_panel, panel_size=panel_size))
            integrated_genes = set(_read_gene_list(integrated_panel, panel_size=panel_size))
            for gene in sorted(source_genes | integrated_genes):
                in_source = gene in source_genes
                in_integrated = gene in integrated_genes
                if in_integrated and not in_source:
                    status = "Visium-added"
                elif in_source and not in_integrated:
                    status = "snRNA-only"
                elif in_source and in_integrated:
                    status = "Shared"
                else:
                    status = "Other"
                membership_rows.append(
                    {
                        "seed": seed,
                        "panel_size": panel_size,
                        "gene_symbol": gene,
                        "in_source_panel": in_source,
                        "in_integrated_panel": in_integrated,
                        "membership_status": status,
                    }
                )
    if not rank_rows:
        raise FileNotFoundError(f"No integrated_panel_rank.tsv files found under {benchmark_dir}")
    return pd.concat(rank_rows, ignore_index=True), pd.DataFrame(membership_rows)


def summarize_gene_status(rank_df: pd.DataFrame, membership_df: pd.DataFrame, panel_size: int = 64) -> pd.DataFrame:
    rank_summary = (
        rank_df.groupby("gene_symbol", as_index=False)
        .agg(
            source_score_mean=("source_score", "mean"),
            source_score_std=("source_score", "std"),
            mean_reference_score_mean=("mean_reference_score", "mean"),
            mean_reference_score_std=("mean_reference_score", "std"),
            integrated_score_mean=("integrated_score", "mean"),
            integrated_rank_mean=("integrated_rank", "mean"),
            reference_support_mean=("reference_support", "mean"),
            visium_rank_lift_mean=("visium_rank_lift", "mean"),
            visium_rank_lift_std=("visium_rank_lift", "std"),
        )
        .copy()
    )
    selected = membership_df[membership_df["panel_size"] == int(panel_size)].copy()
    status_counts = (
        selected.pivot_table(
            index="gene_symbol",
            columns="membership_status",
            values="seed",
            aggfunc="nunique",
            fill_value=0,
        )
        .reset_index()
        .copy()
    )
    for col in ("Visium-added", "snRNA-only", "Shared"):
        if col not in status_counts:
            status_counts[col] = 0
    n_seeds = max(1, int(selected["seed"].nunique()))
    status_counts["visium_added_frequency"] = status_counts["Visium-added"] / n_seeds
    status_counts["snrna_only_frequency"] = status_counts["snRNA-only"] / n_seeds
    status_counts["shared_frequency"] = status_counts["Shared"] / n_seeds
    merged = rank_summary.merge(status_counts, on="gene_symbol", how="left").fillna(
        {
            "Visium-added": 0,
            "snRNA-only": 0,
            "Shared": 0,
            "visium_added_frequency": 0.0,
            "snrna_only_frequency": 0.0,
            "shared_frequency": 0.0,
        }
    )

    def category(row: pd.Series) -> str:
        if row["visium_added_frequency"] >= 0.4:
            return "Visium-added"
        if row["snrna_only_frequency"] >= 0.4:
            return "snRNA-only"
        if row["shared_frequency"] >= 0.4:
            return "Shared"
        return "Other"

    merged["gene_class"] = merged.apply(category, axis=1)
    return merged


def _coordinates(adata: ad.AnnData) -> np.ndarray | None:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2]
    lowered = {str(col).lower(): col for col in adata.obs.columns}
    for x_col, y_col in (("array_col", "array_row"), ("x", "y"), ("center_x", "center_y")):
        if x_col in lowered and y_col in lowered:
            return adata.obs[[lowered[x_col], lowered[y_col]]].astype(float).to_numpy(dtype=np.float32)
    return None


def compute_visium_spatial_scores(prepared_dir: Path, genes: list[str]) -> pd.DataFrame:
    rows = []
    for name in VISIUM_REFERENCE_NAMES:
        path = prepared_dir / f"{name}_smith_input.h5ad"
        if not path.exists():
            continue
        gene_index = _gene_index(path)
        present_genes = [gene for gene in genes if gene in gene_index]
        positions = [gene_index[gene] for gene in present_genes]
        adata = ad.read_h5ad(path)
        try:
            coords = _coordinates(adata)
            if coords is None:
                continue
            x = _normalise_expression_matrix(adata[:, positions].X)
        finally:
            del adata
        coords = np.asarray(coords, dtype=np.float32)
        coord_scores = []
        for axis in range(2):
            coord = coords[:, axis]
            coord = coord - np.nanmean(coord)
            coord_std = float(np.nanstd(coord))
            if coord_std <= 0 or not np.isfinite(coord_std):
                coord_scores.append(np.zeros(x.shape[1], dtype=float))
                continue
            x_centered = x - np.nanmean(x, axis=0, keepdims=True)
            x_std = np.nanstd(x, axis=0)
            cov = np.nanmean(x_centered * coord[:, None], axis=0)
            corr = np.divide(cov, x_std * coord_std, out=np.zeros_like(cov), where=x_std > 0)
            coord_scores.append(np.clip(np.abs(corr), 0, 1))
        spatial_score = np.sqrt(np.square(coord_scores[0]) + np.square(coord_scores[1])) / np.sqrt(2)
        detection = np.mean(x > 0, axis=0)
        mean_expression = np.mean(x, axis=0)
        for gene, score, det, expr in zip(present_genes, spatial_score, detection, mean_expression):
            rows.append(
                {
                    "dataset_id": name,
                    "gene_symbol": gene,
                    "visium_coord_score": float(score),
                    "visium_detection_rate": float(det),
                    "visium_mean_expression": float(expr),
                }
            )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("gene_symbol", as_index=False)
        .agg(
            visium_coord_score_mean=("visium_coord_score", "mean"),
            visium_coord_score_max=("visium_coord_score", "max"),
            visium_detection_rate_mean=("visium_detection_rate", "mean"),
            visium_mean_expression_mean=("visium_mean_expression", "mean"),
            visium_spatial_reference_support=("dataset_id", "nunique"),
        )
        .copy()
    )
    return summary


def compute_merfish_discriminative_scores(merfish_file: Path, genes: list[str], label_column: str = "Cell_Type") -> pd.DataFrame:
    gene_index = _gene_index(merfish_file)
    present_genes = [gene for gene in genes if gene in gene_index]
    positions = [gene_index[gene] for gene in present_genes]
    adata = ad.read_h5ad(merfish_file)
    try:
        if label_column not in adata.obs.columns:
            raise KeyError(f"Missing MERFISH label column `{label_column}`.")
        labels = adata.obs[label_column].astype(str).to_numpy()
        valid = pd.notna(labels) & (labels != "") & (labels != "nan")
        labels = labels[valid]
        x = _normalise_expression_matrix(adata[valid, positions].X)
    finally:
        del adata
    global_var = np.var(x, axis=0) + 1e-8
    classes = sorted(pd.unique(labels).astype(str))
    rows = []
    for gene_idx, gene in enumerate(present_genes):
        values = x[:, gene_idx]
        global_mean = float(np.mean(values))
        between = 0.0
        max_class_effect = 0.0
        max_class = ""
        high_gain_score = 0.0
        high_gain_class = ""
        hepatocyte_score = 0.0
        for cls in classes:
            cls_mask = labels == cls
            rest_mask = ~cls_mask
            cls_values = values[cls_mask]
            rest_values = values[rest_mask]
            if cls_values.size == 0 or rest_values.size == 0:
                continue
            cls_mean = float(np.mean(cls_values))
            rest_mean = float(np.mean(rest_values))
            between += cls_values.size * (cls_mean - global_mean) ** 2
            pooled = np.sqrt(0.5 * (float(np.var(cls_values)) + float(np.var(rest_values))) + 1e-8)
            effect = abs(cls_mean - rest_mean) / pooled
            if effect > max_class_effect:
                max_class_effect = effect
                max_class = cls
            if cls in HIGH_GAIN_CLASSES and effect > high_gain_score:
                high_gain_score = effect
                high_gain_class = cls
            if cls.startswith("Hep_") and effect > hepatocyte_score:
                hepatocyte_score = effect
        eta_squared = between / (values.size * float(global_var[gene_idx]))
        rows.append(
            {
                "gene_symbol": gene,
                "merfish_mean_expression": float(np.mean(values)),
                "merfish_detection_rate": float(np.mean(values > 0)),
                "merfish_cell_type_eta_squared": float(eta_squared),
                "merfish_max_one_vs_rest_effect": float(max_class_effect),
                "merfish_max_effect_class": max_class,
                "merfish_high_gain_effect": float(high_gain_score),
                "merfish_high_gain_effect_class": high_gain_class,
                "merfish_hepatocyte_effect": float(hepatocyte_score),
            }
        )
    return pd.DataFrame(rows)


def _mannwhitney(grouped: pd.DataFrame, metric: str, a: str = "Visium-added", b: str = "snRNA-only") -> dict[str, Any]:
    a_values = grouped.loc[grouped["gene_class"] == a, metric].dropna().astype(float)
    b_values = grouped.loc[grouped["gene_class"] == b, metric].dropna().astype(float)
    if len(a_values) >= 2 and len(b_values) >= 2:
        stat = mannwhitneyu(a_values, b_values, alternative="two-sided")
        p_value = float(stat.pvalue)
        statistic = float(stat.statistic)
    else:
        p_value = statistic = float("nan")
    return {
        "metric": metric,
        "group_a": a,
        "group_b": b,
        "n_a": int(len(a_values)),
        "n_b": int(len(b_values)),
        "median_a": float(np.median(a_values)) if len(a_values) else float("nan"),
        "median_b": float(np.median(b_values)) if len(b_values) else float("nan"),
        "mannwhitney_u": statistic,
        "p_value": p_value,
    }


def compute_statistics(gene_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "visium_rank_lift_mean",
        "reference_support_mean",
        "visium_coord_score_mean",
        "merfish_high_gain_effect",
        "merfish_cell_type_eta_squared",
        "merfish_mean_expression",
    ]
    return pd.DataFrame([_mannwhitney(gene_summary, metric) for metric in metrics])


def _jitter(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.045, size=n)


def plot_rank_lift(gene_summary: pd.DataFrame, output_prefix: Path) -> dict[str, str]:
    _configure_matplotlib()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2.45, 2.35), facecolor="white")
    for gene_class in ("Other", "Shared", "snRNA-only", "Visium-added"):
        group = gene_summary[gene_summary["gene_class"] == gene_class]
        if group.empty:
            continue
        ax.scatter(
            group["source_score_mean"],
            group["mean_reference_score_mean"],
            s=8 if gene_class == "Other" else 16,
            color=PALETTE[gene_class],
            edgecolor=PALETTE["ink"] if gene_class != "Other" else "none",
            linewidth=0.25,
            alpha=0.35 if gene_class == "Other" else 0.82,
            label=gene_class,
            zorder=2 if gene_class == "Other" else 4,
        )
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.55, ls="--", zorder=1)
    labels = (
        gene_summary[gene_summary["gene_class"] == "Visium-added"]
        .sort_values("visium_rank_lift_mean", ascending=False)
        .head(8)
    )
    for _, row in labels.iterrows():
        ax.text(
            row["source_score_mean"] + 0.012,
            row["mean_reference_score_mean"] + 0.004,
            row["gene_symbol"],
            fontsize=5.8,
            ha="left",
            va="center",
        )
    ax.set_xlabel("snRNA SMITH rank score", fontsize=8.1)
    ax.set_ylabel("Mean Visium SMITH rank score", fontsize=8.1)
    ax.tick_params(axis="both", labelsize=7.5, length=2.4, width=0.55)
    ax.legend(loc="lower right", fontsize=5.7, handletextpad=0.2, labelspacing=0.25)
    paths = {
        "pdf": str(output_prefix.with_suffix(".pdf")),
        "svg": str(output_prefix.with_suffix(".svg")),
        "png": str(output_prefix.with_suffix(".png")),
    }
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=450, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_gene_class_metric(
    gene_summary: pd.DataFrame,
    output_prefix: Path,
    *,
    metric: str,
    ylabel: str,
) -> dict[str, str]:
    _configure_matplotlib()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    order = ["snRNA-only", "Shared", "Visium-added"]
    data = [gene_summary.loc[gene_summary["gene_class"] == group, metric].dropna().astype(float).to_numpy() for group in order]
    fig, ax = plt.subplots(figsize=(2.25, 2.25), facecolor="white")
    parts = ax.violinplot(data, positions=np.arange(len(order)), widths=0.62, showmeans=False, showextrema=False)
    for body, group in zip(parts["bodies"], order):
        body.set_facecolor(PALETTE[group])
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    for idx, (group, values) in enumerate(zip(order, data)):
        if values.size == 0:
            continue
        ax.scatter(
            np.full(values.size, idx) + _jitter(values.size, seed=idx + 7),
            values,
            s=13,
            color=PALETTE[group],
            edgecolor=PALETTE["ink"],
            linewidth=0.25,
            alpha=0.85,
            zorder=3,
        )
        ax.plot([idx - 0.19, idx + 0.19], [np.median(values), np.median(values)], color=PALETTE["ink"], lw=0.75, zorder=4)
    stat = _mannwhitney(gene_summary, metric)
    ymax = max([float(np.max(values)) for values in data if values.size] or [1.0])
    ymin = min([float(np.min(values)) for values in data if values.size] or [0.0])
    yrange = max(ymax - ymin, 1e-6)
    ax.text(
        0.98,
        0.95,
        f"p={stat['p_value']:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
    )
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(["snRNA-only", "Shared", "Visium-added"], rotation=28, ha="right", fontsize=7.0)
    ax.set_ylabel(ylabel, fontsize=8.0)
    ax.tick_params(axis="y", labelsize=7.5, length=2.4, width=0.55)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(ymin - 0.06 * yrange, ymax + 0.12 * yrange)
    paths = {
        "pdf": str(output_prefix.with_suffix(".pdf")),
        "svg": str(output_prefix.with_suffix(".svg")),
        "png": str(output_prefix.with_suffix(".png")),
    }
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=450, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--merfish-file", default=str(DEFAULT_MERFISH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--panel-size", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rank_df, membership_df = load_rank_and_membership(Path(args.benchmark_dir))
    gene_summary = summarize_gene_status(rank_df, membership_df, panel_size=int(args.panel_size))
    genes = gene_summary["gene_symbol"].astype(str).tolist()
    visium_spatial = compute_visium_spatial_scores(Path(args.prepared_dir), genes)
    merfish_scores = compute_merfish_discriminative_scores(Path(args.merfish_file), genes)
    gene_summary = gene_summary.merge(visium_spatial, on="gene_symbol", how="left").merge(
        merfish_scores,
        on="gene_symbol",
        how="left",
    )
    stats = compute_statistics(gene_summary)

    rank_tsv = output_dir / "visium_unique_rank_signal_by_seed.tsv"
    membership_tsv = output_dir / "visium_unique_panel_membership_by_seed.tsv"
    gene_summary_tsv = output_dir / "visium_unique_gene_signal_summary.tsv"
    stats_tsv = output_dir / "visium_unique_gene_signal_statistics.tsv"
    rank_df.to_csv(rank_tsv, sep="\t", index=False)
    membership_df.to_csv(membership_tsv, sep="\t", index=False)
    gene_summary.to_csv(gene_summary_tsv, sep="\t", index=False)
    stats.to_csv(stats_tsv, sep="\t", index=False)

    figure_paths = {
        "rank_lift": plot_rank_lift(gene_summary, figure_dir / "09_visium_rank_lift_scatter"),
        "visium_spatial": plot_gene_class_metric(
            gene_summary,
            figure_dir / "10_visium_spatial_score_by_gene_class",
            metric="visium_coord_score_mean",
            ylabel="Mean Visium spatial\ncoordinate score",
        ),
        "merfish_high_gain": plot_gene_class_metric(
            gene_summary,
            figure_dir / "11_merfish_high_gain_marker_score_by_gene_class",
            metric="merfish_high_gain_effect",
            ylabel="MERFISH high-gain\ncell-type marker score",
        ),
    }
    payload = {
        "rank_tsv": str(rank_tsv),
        "membership_tsv": str(membership_tsv),
        "gene_summary_tsv": str(gene_summary_tsv),
        "stats_tsv": str(stats_tsv),
        "figure_paths": figure_paths,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
