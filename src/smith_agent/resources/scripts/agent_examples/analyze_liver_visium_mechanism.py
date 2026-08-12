from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import sparse
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_liver_visium_unique_signal import (  # noqa: E402
    compute_merfish_discriminative_scores,
    compute_visium_spatial_scores,
    load_rank_and_membership,
    summarize_gene_status,
)


DEFAULT_BENCHMARK_DIR = REPO_ROOT / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_5seed_panels"
DEFAULT_MERFISH = REPO_ROOT / "data/liver_merfish/adata_healthy_merfish.h5ad"
DEFAULT_OUTPUT_DIR = DEFAULT_BENCHMARK_DIR / "diagnostics/visium_mechanism"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures/agent_visium_mechanism"
ARIAL_FONT_FILES = [
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
]

PALETTE = {
    "source_smith": "#BDBDBD",
    "multi_visium_smith": "#2F7FB9",
    "Shared": "#6EA45E",
    "Visium-added": "#2F7FB9",
    "snRNA-only": "#BDBDBD",
    "Other": "#E4E4E4",
    "gain": "#2F7FB9",
    "loss": "#B85C5C",
    "hepatocyte": "#D9B36C",
    "niche": "#2F7FB9",
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


def _save_figure(fig: plt.Figure, output_prefix: Path) -> dict[str, str]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
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


def _normalise_expression_matrix(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    x = np.maximum(np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    totals = x.sum(axis=1, keepdims=True)
    x = np.divide(x, np.maximum(totals, 1e-8), out=np.zeros_like(x, dtype=np.float32), where=totals > 0)
    return np.log1p(x * 1e4, dtype=np.float32)


def _gene_index(path: Path) -> dict[str, int]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        out: dict[str, int] = {}
        for idx, gene in enumerate(adata.var_names.astype(str).tolist()):
            clean = str(gene).strip().upper()
            if clean and clean not in out:
                out[clean] = idx
        return out
    finally:
        adata.file.close()


def _family(label: str) -> str:
    text = str(label).lower()
    if text.startswith("hep"):
        return "hepatocyte"
    return "niche"


def compute_merfish_class_gene_effects(
    merfish_file: Path,
    genes: list[str],
    *,
    label_column: str = "Cell_Type",
) -> pd.DataFrame:
    gene_index = _gene_index(merfish_file)
    present_genes = [gene for gene in genes if gene in gene_index]
    positions = [gene_index[gene] for gene in present_genes]
    adata = ad.read_h5ad(merfish_file)
    try:
        if label_column not in adata.obs.columns:
            raise KeyError(f"Missing label column `{label_column}` in {merfish_file}.")
        labels = adata.obs[label_column].astype(str).to_numpy()
        valid = pd.notna(labels) & (labels != "") & (labels != "nan")
        labels = labels[valid]
        x = _normalise_expression_matrix(adata[valid, positions].X)
    finally:
        del adata

    rows = []
    for class_name in sorted(pd.unique(labels).astype(str)):
        class_mask = labels == class_name
        rest_mask = ~class_mask
        class_x = x[class_mask]
        rest_x = x[rest_mask]
        class_mean = class_x.mean(axis=0)
        rest_mean = rest_x.mean(axis=0)
        pooled = np.sqrt(0.5 * (class_x.var(axis=0) + rest_x.var(axis=0)) + 1e-8)
        signed_effect = (class_mean - rest_mean) / pooled
        for gene, signed, mean_expr, detection in zip(
            present_genes,
            signed_effect,
            class_mean,
            (class_x > 0).mean(axis=0),
        ):
            rows.append(
                {
                    "class": class_name,
                    "family": _family(class_name),
                    "gene_symbol": gene,
                    "signed_effect": float(signed),
                    "positive_effect": float(max(signed, 0.0)),
                    "abs_effect": float(abs(signed)),
                    "class_mean_expression": float(mean_expr),
                    "class_detection_rate": float(detection),
                }
            )
    return pd.DataFrame(rows)


def compute_panel_marker_support(
    membership_df: pd.DataFrame,
    class_gene_effects: pd.DataFrame,
    *,
    top_k: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    panel_specs = {
        "source_smith": "in_source_panel",
        "multi_visium_smith": "in_integrated_panel",
    }
    for (seed, panel_size), group in membership_df.groupby(["seed", "panel_size"]):
        for panel, membership_column in panel_specs.items():
            panel_genes = set(group.loc[group[membership_column], "gene_symbol"].astype(str))
            selected = class_gene_effects[class_gene_effects["gene_symbol"].isin(panel_genes)].copy()
            for class_name, class_df in selected.groupby("class"):
                positive = class_df.sort_values("positive_effect", ascending=False).head(top_k)
                absolute = class_df.sort_values("abs_effect", ascending=False).head(top_k)
                rows.append(
                    {
                        "seed": int(seed),
                        "panel_size": int(panel_size),
                        "panel": panel,
                        "class": class_name,
                        "family": _family(class_name),
                        "topk": int(top_k),
                        "top_positive_effect_sum": float(positive["positive_effect"].sum()),
                        "top_positive_effect_mean": float(positive["positive_effect"].mean()),
                        "top_abs_effect_sum": float(absolute["abs_effect"].sum()),
                        "n_positive_effect_gt_0_5": int((class_df["positive_effect"] > 0.5).sum()),
                        "n_positive_effect_gt_1": int((class_df["positive_effect"] > 1.0).sum()),
                        "top_positive_genes": ",".join(positive["gene_symbol"].astype(str).head(top_k).tolist()),
                        "top_abs_genes": ",".join(absolute["gene_symbol"].astype(str).head(top_k).tolist()),
                    }
                )
    support = pd.DataFrame(rows)
    pivot = support.pivot_table(
        index=["seed", "panel_size", "class", "family"],
        columns="panel",
        values=[
            "top_positive_effect_sum",
            "top_positive_effect_mean",
            "top_abs_effect_sum",
            "n_positive_effect_gt_0_5",
            "n_positive_effect_gt_1",
        ],
        aggfunc="mean",
    ).reset_index()
    pivot.columns = ["_".join(str(part) for part in col if part) for col in pivot.columns]
    for metric in (
        "top_positive_effect_sum",
        "top_positive_effect_mean",
        "top_abs_effect_sum",
        "n_positive_effect_gt_0_5",
        "n_positive_effect_gt_1",
    ):
        pivot[f"delta_{metric}"] = pivot[f"{metric}_multi_visium_smith"] - pivot[f"{metric}_source_smith"]
    return support, pivot


def merge_marker_support_with_f1(benchmark_dir: Path, marker_delta: pd.DataFrame) -> pd.DataFrame:
    f1_path = benchmark_dir / "diagnostics/per_class_f1_delta_by_seed_panel_size.tsv"
    if not f1_path.exists():
        raise FileNotFoundError(f"Missing per-class F1 delta file: {f1_path}")
    f1 = pd.read_csv(f1_path, sep="\t")
    return marker_delta.merge(f1, on=["seed", "panel_size", "class"], how="inner")


def compute_marker_support_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "delta_top_positive_effect_sum",
        "delta_top_positive_effect_mean",
        "delta_top_abs_effect_sum",
        "delta_n_positive_effect_gt_0_5",
        "delta_n_positive_effect_gt_1",
    ]
    groups: list[tuple[str | int, pd.DataFrame]] = list(merged.groupby("panel_size"))
    groups.append(("all", merged))
    for panel_size, group in groups:
        for metric in metrics:
            valid = group.dropna(subset=[metric, "delta_f1"])
            if valid.shape[0] < 3 or valid[metric].nunique() < 2 or valid["delta_f1"].nunique() < 2:
                pearson_stat = pearson_p = spearman_stat = spearman_p = float("nan")
            else:
                pearson_stat, pearson_p = pearsonr(valid[metric], valid["delta_f1"])
                spearman_stat, spearman_p = spearmanr(valid[metric], valid["delta_f1"])
            rows.append(
                {
                    "panel_size": panel_size,
                    "metric": metric,
                    "n": int(valid.shape[0]),
                    "pearson_r": float(pearson_stat),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_stat),
                    "spearman_p": float(spearman_p),
                }
            )
    return pd.DataFrame(rows)


def compute_panel_target_compatibility(benchmark_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = benchmark_dir / "diagnostics"
    metric_specs = [
        (diagnostics / "panel_size_merfish_expression_support.tsv", "mean_merfish_expression"),
        (diagnostics / "panel_size_merfish_gene_support.tsv", "mean_merfish_cell_type_mi"),
    ]
    frames = []
    for path, metric in metric_specs:
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        panel_map = {"snRNA only": "source_smith", "snRNA + multi-Visium": "multi_visium_smith"}
        df["panel"] = df["panel"].map(panel_map).fillna(df["panel"])
        df["metric"] = metric
        frames.append(df[["seed", "panel_size", "panel", "metric", "value", "n_genes"]])
    combined = pd.concat(frames, ignore_index=True)
    summary = (
        combined.groupby(["panel_size", "metric", "panel"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), median=("value", "median"))
        .copy()
    )
    rows = []
    for (panel_size, metric), group in combined.groupby(["panel_size", "metric"]):
        pivot = group.pivot_table(index="seed", columns="panel", values="value", aggfunc="mean")
        if {"source_smith", "multi_visium_smith"}.issubset(pivot.columns):
            delta = pivot["multi_visium_smith"] - pivot["source_smith"]
            p_value = float(wilcoxon(pivot["multi_visium_smith"], pivot["source_smith"]).pvalue) if pivot.shape[0] >= 2 else float("nan")
            rows.append(
                {
                    "panel_size": int(panel_size),
                    "metric": metric,
                    "n_seeds": int(pivot.shape[0]),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "wilcoxon_p": p_value,
                }
            )
    return combined, pd.DataFrame(rows).merge(summary, on=["panel_size", "metric"], how="left")


def compute_gene_compatibility(
    benchmark_dir: Path,
    merfish_file: Path,
    *,
    panel_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_df, membership_df = load_rank_and_membership(benchmark_dir)
    gene_summary = summarize_gene_status(rank_df, membership_df, panel_size=panel_size)
    genes = gene_summary["gene_symbol"].astype(str).tolist()
    visium_scores = compute_visium_spatial_scores(benchmark_dir / "prepared", genes)
    merfish_scores = compute_merfish_discriminative_scores(merfish_file, genes)
    merged = gene_summary.merge(visium_scores, on="gene_symbol", how="left").merge(merfish_scores, on="gene_symbol", how="left")
    rows = []
    for metric in [
        "visium_coord_score_mean",
        "merfish_mean_expression",
        "merfish_detection_rate",
        "merfish_cell_type_eta_squared",
        "merfish_high_gain_effect",
        "visium_rank_lift_mean",
    ]:
        a = merged.loc[merged["gene_class"] == "Visium-added", metric].dropna().astype(float)
        b = merged.loc[merged["gene_class"] == "snRNA-only", metric].dropna().astype(float)
        if len(a) >= 2 and len(b) >= 2:
            stat = mannwhitneyu(a, b, alternative="two-sided")
            p_value = float(stat.pvalue)
            u_value = float(stat.statistic)
        else:
            p_value = u_value = float("nan")
        rows.append(
            {
                "metric": metric,
                "group_a": "Visium-added",
                "group_b": "snRNA-only",
                "n_a": int(len(a)),
                "n_b": int(len(b)),
                "median_a": float(a.median()) if len(a) else float("nan"),
                "median_b": float(b.median()) if len(b) else float("nan"),
                "mean_a": float(a.mean()) if len(a) else float("nan"),
                "mean_b": float(b.mean()) if len(b) else float("nan"),
                "mannwhitney_u": u_value,
                "p_value": p_value,
            }
        )
    return merged, pd.DataFrame(rows)


def compute_merfish_spatial_scores(
    merfish_file: Path,
    genes: list[str],
    *,
    sample_column: str = "sample_id",
    n_neighbors: int = 8,
    chunk_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_index = _gene_index(merfish_file)
    present_genes = [gene for gene in genes if gene in gene_index]
    positions = [gene_index[gene] for gene in present_genes]
    adata = ad.read_h5ad(merfish_file)
    try:
        if sample_column not in adata.obs.columns:
            raise KeyError(f"Missing MERFISH sample column `{sample_column}`.")
        if "x" not in adata.obs.columns or "y" not in adata.obs.columns:
            raise KeyError("MERFISH obs must contain x/y coordinates.")
        samples = adata.obs[sample_column].astype(str).to_numpy()
        x_coord = adata.obs["x"].astype(float).to_numpy()
        y_coord = adata.obs["y"].astype(float).to_numpy()
        expr = _normalise_expression_matrix(adata[:, positions].X)
    finally:
        del adata

    rows = []
    for sample in sorted(pd.unique(samples).astype(str)):
        mask = samples == sample
        if int(mask.sum()) < 5:
            continue
        coords = np.column_stack([x_coord[mask], y_coord[mask]]).astype(np.float32)
        sample_expr = expr[mask]
        n_obs = int(sample_expr.shape[0])
        k = min(int(n_neighbors), n_obs - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(coords)
        neighbor_indices = nn.kneighbors(coords, return_distance=False)[:, 1:]
        weight_sum = float(neighbor_indices.size)
        coord_scores = []
        for axis in range(2):
            coord = coords[:, axis]
            coord = coord - np.nanmean(coord)
            coord_std = float(np.nanstd(coord))
            if coord_std <= 0 or not np.isfinite(coord_std):
                coord_scores.append(np.zeros(sample_expr.shape[1], dtype=float))
                continue
            x_centered = sample_expr - np.nanmean(sample_expr, axis=0, keepdims=True)
            x_std = np.nanstd(sample_expr, axis=0)
            cov = np.nanmean(x_centered * coord[:, None], axis=0)
            corr = np.divide(cov, x_std * coord_std, out=np.zeros_like(cov), where=x_std > 0)
            coord_scores.append(np.clip(np.abs(corr), 0, 1))
        score = np.sqrt(np.square(coord_scores[0]) + np.square(coord_scores[1])) / np.sqrt(2)
        moran_i = np.full(sample_expr.shape[1], np.nan, dtype=np.float64)
        for start in range(0, sample_expr.shape[1], int(chunk_size)):
            stop = min(start + int(chunk_size), sample_expr.shape[1])
            chunk = sample_expr[:, start:stop].astype(np.float64, copy=False)
            centered = chunk - np.nanmean(chunk, axis=0, keepdims=True)
            denom = np.sum(centered**2, axis=0)
            neighbor_sum = centered[neighbor_indices].sum(axis=1)
            numerator = np.sum(centered * neighbor_sum, axis=0)
            values = np.divide(
                (float(n_obs) / weight_sum) * numerator,
                denom,
                out=np.full(stop - start, np.nan, dtype=np.float64),
                where=denom > 1e-12,
            )
            moran_i[start:stop] = values
        detection = np.mean(sample_expr > 0, axis=0)
        mean_expression = np.mean(sample_expr, axis=0)
        for gene, gene_score, gene_moran, det, mean_expr in zip(present_genes, score, moran_i, detection, mean_expression):
            rows.append(
                {
                    "sample_id": sample,
                    "gene_symbol": gene,
                    "merfish_coord_score": float(gene_score),
                    "merfish_moran_i": float(gene_moran),
                    "moran_neighbors": int(k),
                    "merfish_sample_detection_rate": float(det),
                    "merfish_sample_mean_expression": float(mean_expr),
                }
            )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("gene_symbol", as_index=False)
        .agg(
            merfish_coord_score_mean=("merfish_coord_score", "mean"),
            merfish_coord_score_max=("merfish_coord_score", "max"),
            merfish_coord_score_std=("merfish_coord_score", "std"),
            merfish_moran_i_mean=("merfish_moran_i", "mean"),
            merfish_moran_i_max=("merfish_moran_i", "max"),
            merfish_moran_i_std=("merfish_moran_i", "std"),
            merfish_spatial_sample_support=("sample_id", "nunique"),
        )
        .copy()
    )
    return raw, summary


def compute_gene_compatibility_statistics(gene_compatibility: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in [
        "visium_coord_score_mean",
        "merfish_moran_i_mean",
        "merfish_moran_i_max",
        "merfish_coord_score_mean",
        "merfish_coord_score_max",
        "merfish_mean_expression",
        "merfish_detection_rate",
        "merfish_cell_type_eta_squared",
        "merfish_high_gain_effect",
        "visium_rank_lift_mean",
    ]:
        if metric not in gene_compatibility.columns:
            continue
        a = gene_compatibility.loc[gene_compatibility["gene_class"] == "Visium-added", metric].dropna().astype(float)
        b = gene_compatibility.loc[gene_compatibility["gene_class"] == "snRNA-only", metric].dropna().astype(float)
        if len(a) >= 2 and len(b) >= 2:
            stat = mannwhitneyu(a, b, alternative="two-sided")
            p_value = float(stat.pvalue)
            u_value = float(stat.statistic)
        else:
            p_value = u_value = float("nan")
        rows.append(
            {
                "metric": metric,
                "group_a": "Visium-added",
                "group_b": "snRNA-only",
                "n_a": int(len(a)),
                "n_b": int(len(b)),
                "median_a": float(a.median()) if len(a) else float("nan"),
                "median_b": float(b.median()) if len(b) else float("nan"),
                "mean_a": float(a.mean()) if len(a) else float("nan"),
                "mean_b": float(b.mean()) if len(b) else float("nan"),
                "mannwhitney_u": u_value,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def plot_per_class_f1_gain(f1_summary: pd.DataFrame, output_prefix: Path, *, panel_size: int = 64) -> dict[str, str]:
    _configure_matplotlib()
    df = f1_summary[f1_summary["panel_size"].astype(int) == int(panel_size)].sort_values("delta_f1_mean", ascending=True)
    y = np.arange(df.shape[0])
    colors = [PALETTE["hepatocyte"] if _family(cls) == "hepatocyte" else PALETTE["niche"] for cls in df["class"]]
    fig, ax = plt.subplots(figsize=(2.55, 2.45), facecolor="white")
    ax.barh(
        y,
        df["delta_f1_mean"],
        xerr=df["delta_f1_std"].fillna(0.0),
        color=colors,
        edgecolor=PALETTE["ink"],
        linewidth=0.35,
        height=0.66,
    )
    ax.axvline(0, color="#777777", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["class"], fontsize=7.2)
    ax.set_xlabel("Cell-type F1 gain\n(multi-Visium - snRNA)", fontsize=8.0)
    ax.tick_params(axis="x", labelsize=7.4, length=2.4, width=0.55)
    ax.tick_params(axis="y", length=0)
    return _save_figure(fig, output_prefix)


def plot_panel_target_metric(
    compatibility: pd.DataFrame,
    output_prefix: Path,
    *,
    metric: str,
    ylabel: str,
) -> dict[str, str]:
    _configure_matplotlib()
    df = compatibility[compatibility["metric"] == metric].copy()
    panel_order = ["source_smith", "multi_visium_smith"]
    x_positions = {32: 0, 64: 1, 128: 2}
    fig, ax = plt.subplots(figsize=(2.45, 2.25), facecolor="white")
    rng = np.random.default_rng(13)
    for panel in panel_order:
        panel_df = df[df["panel"] == panel]
        for panel_size, group in panel_df.groupby("panel_size"):
            x = x_positions[int(panel_size)] + (-0.09 if panel == "source_smith" else 0.09)
            values = group["value"].astype(float).to_numpy()
            ax.scatter(
                np.full(values.shape[0], x) + rng.normal(0, 0.015, values.shape[0]),
                values,
                s=15,
                color=PALETTE[panel],
                edgecolor=PALETTE["ink"],
                linewidth=0.25,
                alpha=0.88,
                zorder=3,
            )
            ax.plot([x - 0.06, x + 0.06], [np.mean(values), np.mean(values)], color=PALETTE["ink"], lw=0.75)
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PALETTE["source_smith"],
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.3,
            markersize=4.5,
            label="snRNA only",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PALETTE["multi_visium_smith"],
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.3,
            markersize=4.5,
            label="snRNA + multi-Visium",
        ),
    ]
    for panel_size in sorted(x_positions):
        sub = df[df["panel_size"].astype(int) == panel_size]
        pivot = sub.pivot_table(index="seed", columns="panel", values="value")
        if {"source_smith", "multi_visium_smith"}.issubset(pivot.columns):
            for _, row in pivot.iterrows():
                ax.plot(
                    [x_positions[panel_size] - 0.09, x_positions[panel_size] + 0.09],
                    [row["source_smith"], row["multi_visium_smith"]],
                    color="#9A9A9A",
                    lw=0.45,
                    alpha=0.55,
                    zorder=1,
                )
    ax.set_xticks([x_positions[size] for size in sorted(x_positions)])
    ax.set_xticklabels([str(size) for size in sorted(x_positions)], fontsize=7.3)
    ax.set_xlabel("Panel size", fontsize=8.0)
    ax.set_ylabel(ylabel, fontsize=8.0)
    ax.tick_params(axis="y", labelsize=7.3, length=2.4, width=0.55)
    ax.tick_params(axis="x", length=0)
    ax.legend(handles=legend_handles, loc="best", fontsize=5.9, handletextpad=0.2, labelspacing=0.25)
    return _save_figure(fig, output_prefix)


def plot_marker_support_vs_f1(
    merged: pd.DataFrame,
    correlations: pd.DataFrame,
    output_prefix: Path,
    *,
    panel_size: int = 64,
    metric: str = "delta_top_abs_effect_sum",
) -> dict[str, str]:
    _configure_matplotlib()
    df = (
        merged[merged["panel_size"].astype(int) == int(panel_size)]
        .groupby(["class", "family"], as_index=False)
        .agg(
            delta_marker_mean=(metric, "mean"),
            delta_marker_std=(metric, "std"),
            delta_f1_mean=("delta_f1", "mean"),
            delta_f1_std=("delta_f1", "std"),
        )
        .copy()
    )
    corr = correlations[
        (correlations["panel_size"].astype(str) == str(panel_size)) & (correlations["metric"] == metric)
    ]
    corr_label = ""
    if not corr.empty:
        row = corr.iloc[0]
        corr_label = f"Spearman r={row['spearman_r']:.2f}, p={row['spearman_p']:.2g}"
    fig, ax = plt.subplots(figsize=(2.75, 2.55), facecolor="white")
    label_offsets = {
        "LSEC": (0.04, 0.006),
        "Macrophage_1": (0.04, -0.006),
        "Macrophage_2": (0.04, -0.005),
        "HSC_1": (0.04, 0.004),
        "HSC_2": (0.04, -0.004),
        "Cholangiocyte": (0.04, -0.002),
        "Hep_1": (0.04, -0.006),
        "Hep_2": (0.04, 0.004),
        "Hep_3": (0.04, 0.010),
    }
    for _, row in df.iterrows():
        color = PALETTE["hepatocyte"] if row["family"] == "hepatocyte" else PALETTE["niche"]
        ax.errorbar(
            row["delta_marker_mean"],
            row["delta_f1_mean"],
            xerr=row["delta_marker_std"] if np.isfinite(row["delta_marker_std"]) else 0.0,
            yerr=row["delta_f1_std"] if np.isfinite(row["delta_f1_std"]) else 0.0,
            fmt="o",
            ms=4.5,
            color=color,
            ecolor="#777777",
            elinewidth=0.55,
            capsize=0,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.3,
            zorder=3,
        )
        dx, dy = label_offsets.get(str(row["class"]), (0.04, 0.0))
        ax.text(row["delta_marker_mean"] + dx, row["delta_f1_mean"] + dy, row["class"], fontsize=5.5, va="center")
    ax.axhline(0, color="#777777", lw=0.55)
    ax.axvline(0, color="#777777", lw=0.55)
    ax.text(0.04, 0.96, corr_label, transform=ax.transAxes, ha="left", va="top", fontsize=6.3)
    ax.set_xlabel("Class marker support gain", fontsize=8.0)
    ax.set_ylabel("Cell-type F1 gain", fontsize=8.0)
    ax.tick_params(axis="both", labelsize=7.2, length=2.4, width=0.55)
    ax.set_xlim(left=min(-0.25, float(df["delta_marker_mean"].min()) - 0.25))
    ax.set_ylim(bottom=min(-0.025, float(df["delta_f1_mean"].min()) - 0.025))
    return _save_figure(fig, output_prefix)


def _jitter(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.04, n)


def plot_gene_class_metric(
    gene_summary: pd.DataFrame,
    output_prefix: Path,
    *,
    metric: str,
    ylabel: str,
) -> dict[str, str]:
    _configure_matplotlib()
    order = ["snRNA-only", "Shared", "Visium-added"]
    fig, ax = plt.subplots(figsize=(2.25, 2.25), facecolor="white")
    for idx, group_name in enumerate(order):
        values = gene_summary.loc[gene_summary["gene_class"] == group_name, metric].dropna().astype(float).to_numpy()
        if values.size == 0:
            continue
        parts = ax.violinplot([values], positions=[idx], widths=0.58, showmeans=False, showextrema=False)
        parts["bodies"][0].set_facecolor(PALETTE[group_name])
        parts["bodies"][0].set_edgecolor("none")
        parts["bodies"][0].set_alpha(0.55)
        ax.scatter(
            np.full(values.shape[0], idx) + _jitter(values.shape[0], idx + 20),
            values,
            s=13,
            color=PALETTE[group_name],
            edgecolor=PALETTE["ink"],
            linewidth=0.25,
            alpha=0.84,
            zorder=3,
        )
        ax.plot([idx - 0.18, idx + 0.18], [np.median(values), np.median(values)], color=PALETTE["ink"], lw=0.75)
    a = gene_summary.loc[gene_summary["gene_class"] == "Visium-added", metric].dropna().astype(float)
    b = gene_summary.loc[gene_summary["gene_class"] == "snRNA-only", metric].dropna().astype(float)
    p_value = mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) >= 2 and len(b) >= 2 else float("nan")
    ax.text(0.98, 0.95, f"p={p_value:.2g}", transform=ax.transAxes, ha="right", va="top", fontsize=7.0)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(["snRNA only", "Shared", "Visium added"], rotation=28, ha="right", fontsize=7.0)
    ax.set_ylabel(ylabel, fontsize=8.0)
    ax.tick_params(axis="y", labelsize=7.3, length=2.4, width=0.55)
    ax.tick_params(axis="x", length=0)
    return _save_figure(fig, output_prefix)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--merfish-file", default=str(DEFAULT_MERFISH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--panel-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rank_df, membership_df = load_rank_and_membership(benchmark_dir)
    gene_universe = sorted(rank_df["gene_symbol"].astype(str).unique())
    class_gene_effects = compute_merfish_class_gene_effects(Path(args.merfish_file), gene_universe)
    marker_support, marker_delta = compute_panel_marker_support(membership_df, class_gene_effects, top_k=int(args.top_k))
    marker_delta_with_f1 = merge_marker_support_with_f1(benchmark_dir, marker_delta)
    marker_correlations = compute_marker_support_correlations(marker_delta_with_f1)
    target_compatibility, target_compatibility_summary = compute_panel_target_compatibility(benchmark_dir)
    gene_compatibility, gene_compatibility_stats = compute_gene_compatibility(
        benchmark_dir,
        Path(args.merfish_file),
        panel_size=int(args.panel_size),
    )
    merfish_spatial_raw, merfish_spatial_summary = compute_merfish_spatial_scores(
        Path(args.merfish_file),
        gene_compatibility["gene_symbol"].astype(str).tolist(),
    )
    gene_compatibility = gene_compatibility.merge(merfish_spatial_summary, on="gene_symbol", how="left")
    gene_compatibility_stats = compute_gene_compatibility_statistics(gene_compatibility)
    f1_summary = pd.read_csv(benchmark_dir / "diagnostics/per_class_f1_delta_summary.tsv", sep="\t")

    paths = {
        "class_gene_effects": output_dir / "merfish_class_gene_effects.tsv",
        "panel_marker_support": output_dir / "panel_marker_support_by_seed.tsv",
        "panel_marker_delta_with_f1": output_dir / "panel_marker_support_delta_with_f1.tsv",
        "panel_marker_delta_correlations": output_dir / "panel_marker_support_delta_correlations.tsv",
        "panel_target_compatibility": output_dir / "panel_target_compatibility_by_seed.tsv",
        "panel_target_compatibility_summary": output_dir / "panel_target_compatibility_summary.tsv",
        "gene_compatibility": output_dir / "visium_added_gene_compatibility.tsv",
        "gene_compatibility_stats": output_dir / "visium_added_gene_compatibility_statistics.tsv",
        "merfish_spatial_scores_by_sample": output_dir / "merfish_gene_spatial_scores_by_sample.tsv",
        "merfish_spatial_score_summary": output_dir / "merfish_gene_spatial_score_summary.tsv",
        "top_visium_added_genes": output_dir / "top_visium_added_genes.tsv",
    }
    class_gene_effects.to_csv(paths["class_gene_effects"], sep="\t", index=False)
    marker_support.to_csv(paths["panel_marker_support"], sep="\t", index=False)
    marker_delta_with_f1.to_csv(paths["panel_marker_delta_with_f1"], sep="\t", index=False)
    marker_correlations.to_csv(paths["panel_marker_delta_correlations"], sep="\t", index=False)
    target_compatibility.to_csv(paths["panel_target_compatibility"], sep="\t", index=False)
    target_compatibility_summary.to_csv(paths["panel_target_compatibility_summary"], sep="\t", index=False)
    gene_compatibility.to_csv(paths["gene_compatibility"], sep="\t", index=False)
    gene_compatibility_stats.to_csv(paths["gene_compatibility_stats"], sep="\t", index=False)
    merfish_spatial_raw.to_csv(paths["merfish_spatial_scores_by_sample"], sep="\t", index=False)
    merfish_spatial_summary.to_csv(paths["merfish_spatial_score_summary"], sep="\t", index=False)
    (
        gene_compatibility[gene_compatibility["gene_class"] == "Visium-added"]
        .sort_values(["visium_rank_lift_mean", "merfish_detection_rate"], ascending=False)
        .head(50)
        .to_csv(paths["top_visium_added_genes"], sep="\t", index=False)
    )

    figure_paths = {
        "per_class_f1_gain": plot_per_class_f1_gain(
            f1_summary,
            figure_dir / "22_panel64_per_class_f1_gain",
            panel_size=int(args.panel_size),
        ),
        "target_expression": plot_panel_target_metric(
            target_compatibility,
            figure_dir / "23_panel_target_merfish_expression",
            metric="mean_merfish_expression",
            ylabel="Mean MERFISH expression",
        ),
        "target_cell_type_mi": plot_panel_target_metric(
            target_compatibility,
            figure_dir / "24_panel_target_cell_type_mi",
            metric="mean_merfish_cell_type_mi",
            ylabel="Mean MERFISH cell-type MI",
        ),
        "marker_support_vs_f1": plot_marker_support_vs_f1(
            marker_delta_with_f1,
            marker_correlations,
            figure_dir / "25_marker_support_gain_vs_f1_gain",
            panel_size=int(args.panel_size),
            metric="delta_top_abs_effect_sum",
        ),
        "visium_spatial_score": plot_gene_class_metric(
            gene_compatibility,
            figure_dir / "26_visium_added_gene_spatial_score",
            metric="visium_coord_score_mean",
            ylabel="Mean Visium spatial\ncoordinate score",
        ),
        "merfish_spatial_score": plot_gene_class_metric(
            gene_compatibility,
            figure_dir / "29_visium_added_gene_merfish_spatial_score",
            metric="merfish_coord_score_mean",
            ylabel="Mean MERFISH spatial\ncoordinate score",
        ),
        "merfish_moran_i": plot_gene_class_metric(
            gene_compatibility,
            figure_dir / "31_visium_added_gene_merfish_morans_i",
            metric="merfish_moran_i_mean",
            ylabel="Mean MERFISH\nMoran's I",
        ),
        "merfish_detection": plot_gene_class_metric(
            gene_compatibility,
            figure_dir / "27_visium_added_gene_merfish_detection",
            metric="merfish_detection_rate",
            ylabel="MERFISH detection rate",
        ),
        "merfish_expression": plot_gene_class_metric(
            gene_compatibility,
            figure_dir / "28_visium_added_gene_merfish_expression",
            metric="merfish_mean_expression",
            ylabel="Mean MERFISH expression",
        ),
    }

    payload = {
        "output_tables": {key: str(value) for key, value in paths.items()},
        "figure_paths": figure_paths,
        "n_rank_genes": int(rank_df["gene_symbol"].nunique()),
        "n_panel_membership_rows": int(membership_df.shape[0]),
        "panel_size": int(args.panel_size),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
