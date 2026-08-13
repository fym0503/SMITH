#!/usr/bin/env python3
"""Assemble ribo transfer figure panels with consistent names and style."""

from __future__ import annotations

import json
import math
import textwrap
import colorsys
import re
import argparse
from pathlib import Path

import anndata as ad
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.spatial import cKDTree

from analyze_aligned_translatome_pathways import (
    DEFAULT_MSIGDB,
    load_terms,
    read_marker_list,
    run_enrichment,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "figures" / "ribo_figure"
BY_PANEL_DIRNAME = "panel_size_versions"

STARMAP_H5AD = ROOT / "data" / "SMITH_data_ribomap" / "data" / "mouse_brain_starmap_rep2.h5ad"
RIBOMAP_H5AD = ROOT / "data" / "SMITH_data_ribomap" / "data" / "mouse_brain_ribomap_rep2.h5ad"
DEEP_RIBOMAP_H5AD = ROOT / "data" / "SMITH_data_ribomap" / "data" / "deep_brain_ribomap.h5ad"
ALIGNED_H5AD = (
    ROOT
    / "outputs"
    / "analysis"
    / "clean_fusion_alignment_hpo_20260509"
    / "stage3_alignment_celltype128"
    / "selected_alignments"
    / "stage3_05__aligned_n5000_a0p1_b0p35_fg0p15_cosine_hvg2000_pc60.h5ad"
)
ALIGNED_EVAL_SUMMARY = ROOT / "outputs" / "deep_fgw_aligned_smith_hpo" / "eval_summary.csv"
CURRENT_ALIGNED_EVAL_SUMMARIES = [
    ROOT
    / "outputs"
    / "analysis"
    / "clean_fusion_alignment_hpo_20260509"
    / "stage2_refine"
    / "eval_summary.csv",
    ROOT
    / "outputs"
    / "analysis"
    / "clean_fusion_alignment_hpo_20260509"
    / "stage3_alignment_celltype128"
    / "hpo"
    / "eval_summary.csv",
    ROOT
    / "outputs"
    / "analysis"
    / "clean_fusion_alignment_hpo_20260509"
    / "stage4_targeted_alpha_beta128"
    / "hpo"
    / "eval_summary.csv",
    ROOT
    / "outputs"
    / "analysis"
    / "clean_fusion_alignment_hpo_20260509"
    / "stage4_targeted_alpha_beta128"
    / "manual_refine_promising"
    / "eval_summary.csv",
]

MODALITY_DIR = ROOT / "outputs" / "figures" / "modality_marker_analysis"
ALIGNED_DIR = ROOT / "outputs" / "figures" / "aligned_translatome_pathway_analysis_clean_fusion_current_20260509"
PANEL_M_PROPORTION_DIRNAME = "panel_m_enriched_proportion_tables"
DEEP_SMITH_MARKER_128 = (
    ROOT
    / "outputs"
    / "smith_hpo_standard_coord_refine"
    / "ribomap_deep_to_mouse_rep2_region_p128"
    / "trial_0000"
    / "SMITH"
    / "deep_to_mouse"
    / "marker_128.csv"
)
STARMAP_SMITH_MARKER_128 = (
    ROOT
    / "outputs"
    / "smith_hpo_region_queue_starmap_gpu01"
    / "starmap_rep2_region"
    / "round_0314"
    / "trial_0003"
    / "SMITH"
    / "mouse_starmap_to_ribomap_rep2"
    / "marker_128.csv"
)

TAB20 = plt.get_cmap("tab20").colors
PANEL_SIZES = (32, 64, 128)


def configure_plotting() -> None:
    arial_path = Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")
    family = "Arial"
    if arial_path.exists():
        font_manager.fontManager.addfont(str(arial_path))
        family = font_manager.FontProperties(fname=str(arial_path)).get_name()
    rc = {
        "font.family": family,
        "font.sans-serif": [family],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
    }
    plt.rcParams.update(rc)
    sns.set_theme(style="whitegrid", rc=rc)


def save_figure(fig: plt.Figure, stem: str) -> None:
    save_figure_multi(fig, [stem])


def save_figure_multi(fig: plt.Figure, stems: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (OUT_DIR / stem).parent.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=350)
    plt.close(fig)


def figure_output_path(stem: str, suffix: str) -> Path:
    path = OUT_DIR / f"{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_prompts() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt_a = """Create a clean scientific schematic for panel A of a Nature-style multi-panel figure. The panel should explain the experimental principle of spatial translatome profiling by RIBOMap, analogous to a technology-principle schematic rather than a benchmarking workflow.

Scene layout: start from a mouse brain tissue section containing spatially organized cells. Zoom into one cell and show mRNAs associated with translating ribosomes/polysomes. Depict ribosome-bound transcripts as orange mRNA strands with multiple ribosome particles, distinct from unbound cytosolic mRNA. Show the RIBOMap idea: in situ probes/readout selectively capture or report ribosome-associated mRNAs, producing a spatial map of actively translated transcripts in tissue. Then show that a compact target gene panel is selected for this translatome readout, enabling spatially resolved measurement of translation-related cell states and tissue regions. Use visual motifs: orange ribosome-bound mRNA for translatome signal, muted gray unbound RNA background, blue cell nuclei, green selected-gene panel cards, and a stylized tissue map on the right with colored cell populations. Keep the panel focused on the principle: tissue -> ribosome-bound mRNA -> targeted RIBOMap readout -> spatial translatome map. Minimal text labels only if needed: "RIBOMap", "ribosome-bound mRNA", "target gene panel", "spatial translatome". No chart axes, no benchmark bars, no captions. White background, flat vector style, thin clean lines, Arial-like typography, colorblind-friendly palette, publication-ready, high resolution."""

    prompt_j = """Create a clean scientific schematic for a Nature-style multi-panel figure.

Scene: PASTE-style fused transfer before SMITH gene-panel selection. On the left, show STARmap transcriptome cells providing the tissue coordinate system, spatial structure, cell-type labels, and region framework. On the right, show RIBOMap translatome cells providing ribosome-associated expression signals. In the middle, show an optimal-transport/alignment layer as curved matching lines or a soft coupling matrix connecting the two modalities. Then show an aligned hybrid training dataset flowing into SMITH: STARmap-like spatial coordinates and labels, but RIBOMap-derived translatome expression. SMITH outputs a gene panel that preserves STARmap-like spatial/cell-type utility and RIBOMap-like translatome-biased signals. Use blue for STARmap space/labels, orange for RIBOMap/translatome expression, teal/green for aligned SMITH output. No captions, no axes, no tiny unreadable text. Minimal vector style, white background, clean arrows, Arial-like typography, publication-ready, high resolution."""

    (OUT_DIR / "panel_a_schematic_prompt.txt").write_text(prompt_a + "\n")
    (OUT_DIR / "panel_j_alignment_schematic_prompt.txt").write_text(prompt_j + "\n")


def sample_adata(adata: ad.AnnData, n: int, seed: int) -> ad.AnnData:
    if adata.n_obs <= n:
        return adata
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(adata.n_obs, size=n, replace=False))
    return adata[idx].copy()


def make_distinct_palette(n: int) -> list[str]:
    base = [
        to_hex(color)
        for cmap_name in ("tab20", "tab20b", "tab20c")
        for color in plt.get_cmap(cmap_name).colors
    ]
    if n <= len(base):
        return base[:n]

    palette = list(base)
    i = 0
    golden_ratio = 0.618033988749895
    while len(palette) < n:
        hue = (0.07 + golden_ratio * i) % 1.0
        sat = (0.62, 0.80, 0.95, 0.70)[i % 4]
        val = (0.88, 0.72, 0.96, 0.64)[(i // 4) % 4]
        color = to_hex(colorsys.hsv_to_rgb(hue, sat, val))
        if color not in palette:
            palette.append(color)
        i += 1
    return palette


def categorical_colors(labels: pd.Series | list[str] | np.ndarray) -> dict[str, str]:
    cats = sorted(pd.Series(labels, dtype="object").astype(str).unique())
    palette = make_distinct_palette(len(cats))
    return {cat: palette[i] for i, cat in enumerate(cats)}


def spatial_scatter(ax: plt.Axes, adata: ad.AnnData, color_col: str, title: str, max_legend: int = 16) -> None:
    coords = np.asarray(adata.obsm["spatial"])
    labels = adata.obs[color_col].astype(str).reset_index(drop=True)
    colors = categorical_colors(labels)
    for label, sub_idx in labels.groupby(labels).groups.items():
        sub = coords[np.asarray(list(sub_idx), dtype=int)]
        ax.scatter(sub[:, 0], sub[:, 1], s=2.0, c=[colors[label]], alpha=0.78, linewidths=0, label=label)
    ax.set_title(title, fontsize=11, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    if labels.nunique() <= max_legend:
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            markerscale=3,
            handletextpad=0.2,
            borderpad=0.1,
        )


def matrix_to_dense(x: object) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def normalize_coords(coords: np.ndarray, ref_coords: np.ndarray | None = None) -> np.ndarray:
    coords = np.asarray(coords)[:, :2].astype(float)
    ref = coords if ref_coords is None else np.asarray(ref_coords)[:, :2].astype(float)
    min_v = np.nanmin(ref, axis=0)
    max_v = np.nanmax(ref, axis=0)
    denom = max_v - min_v
    denom[denom == 0] = 1.0
    return (coords - min_v) / denom


def spatial_knn_transfer_regions(
    query_coords: np.ndarray,
    ref_coords: np.ndarray,
    ref_regions: pd.Series,
    k: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    ref_regions = ref_regions.astype(str).reset_index(drop=True)
    categories = pd.Index(sorted(ref_regions.unique()))
    codes = categories.get_indexer(ref_regions)
    tree = cKDTree(ref_coords)
    _, idx = tree.query(query_coords, k=min(k, len(ref_regions)))
    if idx.ndim == 1:
        idx = idx[:, None]
    neighbor_codes = codes[idx]
    votes = np.zeros((query_coords.shape[0], len(categories)), dtype=np.int16)
    row_idx = np.arange(query_coords.shape[0])
    for col in range(neighbor_codes.shape[1]):
        np.add.at(votes, (row_idx, neighbor_codes[:, col]), 1)
    best = votes.argmax(axis=1)
    confidence = votes.max(axis=1) / float(neighbor_codes.shape[1])
    return categories.take(best).astype(str).to_numpy(), confidence.astype(np.float32)


def plot_labelled_spatial(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: pd.Series | np.ndarray,
    colors: dict[str, str],
    point_size: float,
    alpha: float = 0.72,
) -> None:
    labels = pd.Series(labels, dtype="object").astype(str).reset_index(drop=True)
    for label, idx in labels.groupby(labels).groups.items():
        sub = coords[np.asarray(list(idx), dtype=int)]
        ax.scatter(sub[:, 0], sub[:, 1], s=point_size, c=[colors[label]], alpha=alpha, linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_axis_off()


def plot_continuous_spatial(
    ax: plt.Axes,
    background_coords: np.ndarray,
    coords: np.ndarray,
    values: np.ndarray,
    cmap: LinearSegmentedColormap,
) -> plt.Collection:
    ax.scatter(
        background_coords[:, 0],
        background_coords[:, 1],
        s=0.35,
        c="#D9D9D9",
        alpha=0.32,
        linewidths=0,
        rasterized=True,
    )
    lo, hi = np.nanpercentile(values, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values) + 1e-6)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=5.4,
        c=np.clip(values, lo, hi),
        cmap=cmap,
        norm=Normalize(vmin=lo, vmax=hi),
        alpha=0.9,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_axis_off()
    return scatter


def ribomap_only_marker_score(aligned: ad.AnnData) -> tuple[np.ndarray, list[str]]:
    marker_path = MODALITY_DIR / "rna_ribo_modality_bias_by_group.csv"
    marker_df = pd.read_csv(marker_path)
    genes = (
        marker_df[(marker_df["group"] == "RIBOMap-only") & (marker_df["ribo_bias"] > 0)]
        .sort_values("ribo_bias", ascending=False)["gene"]
        .astype(str)
        .tolist()
    )
    gene_to_idx = {gene.lower(): i for i, gene in enumerate(aligned.var_names.astype(str))}
    keep = [gene for gene in genes if gene.lower() in gene_to_idx][:32]
    if not keep:
        return np.zeros(aligned.n_obs, dtype=float), []
    idx = [gene_to_idx[gene.lower()] for gene in keep]
    x = matrix_to_dense(aligned.X[:, idx]).astype(float)
    x = np.log1p(np.maximum(x, 0))
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    score = ((x - mean) / std).mean(axis=1)
    return np.asarray(score, dtype=float), keep


def ensure_umap(adata: ad.AnnData, random_state: int = 0) -> ad.AnnData:
    if "X_umap" in adata.obsm:
        return adata
    work = adata.copy()
    n_comps = min(30, work.n_obs - 2, work.n_vars - 1)
    if n_comps < 2:
        raise ValueError("Need at least two observations/features to compute UMAP.")
    sc.pp.pca(work, n_comps=n_comps, random_state=random_state)
    sc.pp.neighbors(work, n_neighbors=min(15, work.n_obs - 1), n_pcs=n_comps)
    sc.tl.umap(work, random_state=random_state)
    return work


def plot_umap(
    ax: plt.Axes,
    adata: ad.AnnData,
    label_col: str,
    colors: dict[str, str],
    point_size: float,
    alpha: float,
) -> None:
    coords = np.asarray(adata.obsm["X_umap"])[:, :2].astype(float)
    labels = adata.obs[label_col].astype(str).reset_index(drop=True)
    for label, idx in labels.groupby(labels).groups.items():
        sub = coords[np.asarray(list(idx), dtype=int)]
        ax.scatter(
            sub[:, 0],
            sub[:, 1],
            s=point_size,
            c=[colors.get(label, "#BDBDBD")],
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")


def plot_umap_overlay(
    ax: plt.Axes,
    background_coords: np.ndarray,
    overlay_coords: np.ndarray,
    labels: pd.Series | np.ndarray,
    colors: dict[str, str],
    point_size: float = 5.0,
) -> None:
    ax.scatter(
        background_coords[:, 0],
        background_coords[:, 1],
        s=0.35,
        c="#D9D9D9",
        alpha=0.26,
        linewidths=0,
        rasterized=True,
    )
    labels = pd.Series(labels, dtype="object").astype(str).reset_index(drop=True)
    for label, idx in labels.groupby(labels).groups.items():
        sub = overlay_coords[np.asarray(list(idx), dtype=int)]
        ax.scatter(
            sub[:, 0],
            sub[:, 1],
            s=point_size,
            c=[colors.get(label, "#BDBDBD")],
            alpha=0.88,
            linewidths=0,
            rasterized=True,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")


def add_celltype_prefix(adata: ad.AnnData) -> ad.AnnData:
    if "celltype" not in adata.obs and "cell_type" in adata.obs:
        adata.obs["celltype"] = adata.obs["cell_type"].astype(str)
    adata.obs["celltype_prefix"] = adata.obs["celltype"].astype(str).map(merge_celltype_prefix)
    return adata


def save_umap_pair(
    left: ad.AnnData,
    right: ad.AnnData,
    colors: dict[str, str],
    stem: str,
    left_point_size: float,
    right_point_size: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.55), constrained_layout=True)
    plot_umap(axes[0], left, "celltype_prefix", colors, left_point_size, alpha=0.68)
    plot_umap(axes[1], right, "celltype_prefix", colors, right_point_size, alpha=0.78)
    save_figure(fig, stem)


def make_panel_j_umap_alignment_quality() -> None:
    deep = add_celltype_prefix(ad.read_h5ad(DEEP_RIBOMAP_H5AD))
    star = add_celltype_prefix(ad.read_h5ad(STARMAP_H5AD))
    ribo = add_celltype_prefix(ad.read_h5ad(RIBOMAP_H5AD))
    aligned = add_celltype_prefix(ensure_umap(ad.read_h5ad(ALIGNED_H5AD), random_state=7))
    deep_label_col = "aligned_deep_ribomap_celltype"
    aligned.obs["aligned_deep_ribomap_celltype_prefix"] = aligned.obs[deep_label_col].astype(str).map(merge_celltype_prefix)

    all_prefixes = sorted(
        set(deep.obs["celltype_prefix"].astype(str))
        | set(star.obs["celltype_prefix"].astype(str))
        | set(ribo.obs["celltype_prefix"].astype(str))
        | set(aligned.obs["celltype_prefix"].astype(str))
        | set(aligned.obs["aligned_deep_ribomap_celltype_prefix"].astype(str))
    )
    all_prefixes = [x for x in all_prefixes if x.strip().lower() not in {"na", "nan", "none", "unknown", ""}]
    colors = categorical_colors(all_prefixes)
    write_palette_csv(colors, "panel_j_umap_celltype_palette", "celltype_prefix")
    standalone_legend_from_categories(all_prefixes, colors, "panel_j_umap_celltype_legend", nrows=3)

    save_umap_pair(
        deep,
        star,
        colors,
        "panel_j_deep_starmap_celltype_umap",
        left_point_size=0.20,
        right_point_size=0.36,
    )
    save_umap_pair(
        ribo,
        aligned,
        colors,
        "panel_j_current_alignment_quality_umap",
        left_point_size=0.36,
        right_point_size=3.2,
    )
    save_umap_pair(
        ribo,
        aligned,
        colors,
        "panel_j_aligned_transfer_visualization",
        left_point_size=0.36,
        right_point_size=3.2,
    )

    starmap_umap = np.asarray(star.obsm["X_umap"])[:, :2].astype(float)
    star_pos = star.obs_names.get_indexer(aligned.obs["original_starmap_obs_name"].astype(str))
    valid = star_pos >= 0
    overlay_coords = starmap_umap[star_pos[valid]]
    overlay = aligned.obs.iloc[np.flatnonzero(valid)].copy()
    exact_agreement = (
        overlay["celltype"].astype(str).to_numpy() == overlay[deep_label_col].astype(str).to_numpy()
    ).mean()
    prefix_agreement = (
        overlay["celltype_prefix"].astype(str).to_numpy()
        == overlay["aligned_deep_ribomap_celltype_prefix"].astype(str).to_numpy()
    ).mean()

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.55), constrained_layout=True)
    plot_umap(axes[0], deep, "celltype_prefix", colors, point_size=0.20, alpha=0.68)
    plot_umap_overlay(
        axes[1],
        starmap_umap,
        overlay_coords,
        overlay["aligned_deep_ribomap_celltype_prefix"],
        colors,
        point_size=4.6,
    )
    plot_umap_overlay(
        axes[2],
        starmap_umap,
        overlay_coords,
        overlay["celltype_prefix"],
        colors,
        point_size=4.6,
    )
    save_figure(fig, "panel_j_alignment_celltype_quality_umap")

    fig, ax = plt.subplots(figsize=(2.55, 2.55), constrained_layout=True)
    plot_umap(ax, deep, "celltype_prefix", colors, point_size=0.20, alpha=0.68)
    save_figure(fig, "panel_j_alignment_quality_1_deep_ribomap_native_umap")

    fig, ax = plt.subplots(figsize=(2.55, 2.55), constrained_layout=True)
    plot_umap_overlay(
        ax,
        starmap_umap,
        overlay_coords,
        overlay["aligned_deep_ribomap_celltype_prefix"],
        colors,
        point_size=4.6,
    )
    save_figure(fig, "panel_j_alignment_quality_2_deep_labels_on_starmap_umap")

    fig, ax = plt.subplots(figsize=(2.55, 2.55), constrained_layout=True)
    plot_umap_overlay(
        ax,
        starmap_umap,
        overlay_coords,
        overlay["celltype_prefix"],
        colors,
        point_size=4.6,
    )
    save_figure(fig, "panel_j_alignment_quality_3_starmap_labels_on_starmap_umap")

    pd.DataFrame(
        [
            {
                "n_aligned_cells_on_starmap_umap": int(valid.sum()),
                "exact_celltype_agreement": float(exact_agreement),
                "prefix_celltype_agreement": float(prefix_agreement),
                "aligned_h5ad": str(ALIGNED_H5AD),
                "expression_source": "deep_brain_ribomap",
                "label_spatial_source": "mouse_brain_starmap_rep2",
            }
        ]
    ).to_csv(OUT_DIR / "panel_j_alignment_celltype_quality_metrics.csv", index=False)

    pd.DataFrame(
        {
            "current_alignment": ["deep_brain_ribomap -> mouse_brain_starmap_rep2"],
            "deep_starmap_panel_note": [
                "Latest leakage-free FGW alignment uses deep RIBOMap expression with STARmap observations, "
                "cell type labels, and spatial coordinates; mouse_brain_ribomap_rep2 is used only as evaluation target."
            ],
        }
    ).to_csv(OUT_DIR / "panel_j_umap_alignment_notes.csv", index=False)


def dataset_spatial_scatter(ax: plt.Axes, adata: ad.AnnData, color: str, title: str, subtitle: str) -> None:
    coords = np.asarray(adata.obsm["spatial"])
    coords = coords[:, :2].astype(float)
    coords = coords - np.nanmin(coords, axis=0)
    denom = np.nanmax(coords, axis=0)
    denom[denom == 0] = 1
    coords = coords / denom
    ax.scatter(coords[:, 0], coords[:, 1], s=1.8, c=color, alpha=0.58, linewidths=0)
    ax.set_title(title, fontsize=11, pad=10)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color="#555555")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)


def color_values(labels: pd.Series, color_col: str, colors: dict[str, str] | None = None) -> list[str]:
    labels = labels.astype(str).reset_index(drop=True)
    invalid = labels.str.strip().str.lower().isin({"na", "nan", "none", "unknown", ""})
    if invalid.all() or labels.nunique(dropna=False) <= 1:
        return ["#BDBDBD"] * len(labels)
    colors = colors or categorical_colors(labels)
    return ["#BDBDBD" if invalid.iloc[i] else colors[value] for i, value in enumerate(labels)]


def merge_celltype_prefix(value: str) -> str:
    label = str(value).strip()
    if label.lower() in {"na", "nan", "none", "unknown", ""}:
        return "NA"
    if " " in label:
        return label.split()[0]
    label = re.sub(r"(_)?\d+$", "", label)
    return re.sub(r"_+$", "", label)


def standalone_spatial_plot(
    adata: ad.AnnData,
    color_col: str,
    stem: str,
    colors: dict[str, str] | None = None,
) -> None:
    coords = np.asarray(adata.obsm["spatial"])[:, :2].astype(float)
    coords = coords - np.nanmin(coords, axis=0)
    denom = np.nanmax(coords, axis=0)
    denom[denom == 0] = 1
    coords = coords / denom
    labels = adata.obs[color_col].astype(str)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=1.15,
        c=color_values(labels, color_col, colors=colors),
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", pad_inches=0, dpi=350)
    plt.close(fig)


def standalone_legend_from_categories(
    categories: list[str],
    colors: dict[str, str],
    stem: str,
    nrows: int | None = None,
) -> None:
    if len(categories) == 1 and categories[0] == "NA":
        categories = ["NA"]
        colors = {"NA": "#BDBDBD"}

    if nrows is not None:
        ncol = math.ceil(len(categories) / nrows)
    elif len(categories) <= 12:
        ncol = len(categories)
    elif len(categories) <= 48:
        ncol = math.ceil(len(categories) / 2)
    else:
        ncol = min(12, len(categories))
    nrow = math.ceil(len(categories) / ncol)
    fig_width = max(3.2, 0.62 * ncol)
    fig_height = max(0.45, 0.36 * nrow)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5.5,
            markerfacecolor=colors[cat],
            markeredgecolor="none",
            label=cat,
        )
        for cat in categories
    ]
    ax.legend(
        handles=handles,
        labels=categories,
        loc="center",
        ncol=ncol,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.75,
        borderaxespad=0,
        fontsize=7.5,
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", pad_inches=0.01, dpi=350)
    plt.close(fig)


def standalone_legend_plot(
    adata: ad.AnnData,
    color_col: str,
    stem: str,
    colors: dict[str, str] | None = None,
) -> None:
    labels = adata.obs[color_col].astype(str)
    invalid = labels.str.strip().str.lower().isin({"na", "nan", "none", "unknown", ""})
    if invalid.all() or labels.nunique(dropna=False) <= 1:
        categories = ["NA"]
        colors = {"NA": "#BDBDBD"}
    else:
        categories = sorted(labels[~invalid].unique())
        colors = colors or categorical_colors(categories)
    standalone_legend_from_categories(categories, colors, stem)


def clear_panel_b_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.glob("panel_b_*"):
        if path.is_file():
            path.unlink()


def write_palette_csv(colors: dict[str, str], stem: str, label_col: str) -> None:
    pd.DataFrame(
        [{label_col: label, "color": color} for label, color in sorted(colors.items())]
    ).to_csv(OUT_DIR / f"{stem}.csv", index=False)


def make_panel_b_separate_spatial() -> None:
    clear_panel_b_outputs()
    datasets = [
        ("deep_ribomap_source", DEEP_RIBOMAP_H5AD),
        ("starmap_source", STARMAP_H5AD),
        ("ribomap_target", RIBOMAP_H5AD),
    ]
    loaded: dict[str, ad.AnnData] = {}
    prefix_categories: set[str] = set()
    for dataset_name, path in datasets:
        adata = ad.read_h5ad(path)
        adata.obs["celltype_prefix"] = adata.obs["celltype"].astype(str).map(merge_celltype_prefix)
        loaded[dataset_name] = adata
        prefix_categories.update(adata.obs["celltype_prefix"].astype(str).unique())

    prefix_categories = {x for x in prefix_categories if x.strip().lower() not in {"na", "nan", "none", "unknown", ""}}
    prefix_colors = categorical_colors(sorted(prefix_categories))
    write_palette_csv(prefix_colors, "panel_b_unified_celltype_palette", "celltype")

    for dataset_name, adata in loaded.items():
        fine_colors = categorical_colors(adata.obs["celltype"].astype(str))
        write_palette_csv(
            fine_colors,
            f"panel_b_fine_grained_celltype_{dataset_name}_palette",
            "celltype",
        )
        standalone_spatial_plot(
            adata,
            "celltype",
            f"panel_b_fine_grained_celltype_{dataset_name}",
            colors=fine_colors,
        )

        stem = f"panel_b_unified_celltype_{dataset_name}"
        standalone_spatial_plot(adata, "celltype_prefix", stem, colors=prefix_colors)

    standalone_legend_from_categories(
        sorted(prefix_colors),
        prefix_colors,
        "panel_b_unified_celltype_legend",
    )
    standalone_legend_from_categories(
        sorted(prefix_colors),
        prefix_colors,
        "panel_b_unified_celltype_legend_3rows",
        nrows=3,
    )
    standalone_legend_from_categories(
        sorted(prefix_colors),
        prefix_colors,
        "panel_b_unified_celltype_legend_4rows",
        nrows=4,
    )


def make_panel_b_data_overview() -> None:
    deep = sample_adata(ad.read_h5ad(DEEP_RIBOMAP_H5AD), 14000, 5)
    star = sample_adata(ad.read_h5ad(STARMAP_H5AD), 12000, 7)
    ribo = sample_adata(ad.read_h5ad(RIBOMAP_H5AD), 12000, 11)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.78, bottom=0.24, wspace=0.22)
    dataset_spatial_scatter(
        axes[0],
        deep,
        to_hex(TAB20[1]),
        "RIBOMap source",
        "deep-brain RIBOMap",
    )
    dataset_spatial_scatter(
        axes[1],
        star,
        to_hex(TAB20[0]),
        "STARmap source",
        "mouse-brain STARmap rep2",
    )
    dataset_spatial_scatter(
        axes[2],
        ribo,
        to_hex(TAB20[2]),
        "RIBOMap target",
        "mouse-brain RIBOMap rep2",
    )
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=10, lw=1.1, color="#333333")
    for start, end, y, label in [
        (0.18, 0.83, 0.15, "RIBOMap-to-RIBOMap transfer"),
        (0.50, 0.83, 0.09, "STARmap-to-RIBOMap transfer"),
    ]:
        arrow = FancyArrowPatch((start, y), (end, y), transform=fig.transFigure, **arrow_kw)
        fig.add_artist(arrow)
        fig.text((start + end) / 2, y + 0.022, label, ha="center", va="bottom", fontsize=8.8, color="#333333")
    save_figure(fig, "panel_b_data_visualization")


def make_panel_g_jaccard() -> None:
    pairs = pd.read_csv(MODALITY_DIR / "panel_gene_jaccard_by_panel_pairwise_overlap.csv")
    pairs["modality_group"] = pairs["pair_type"].apply(
        lambda value: "Same modality" if value in {"Within RIBOMap", "Within STARmap"} else "Cross modality"
    )
    pairs["analysis_set"] = "All methods"
    no_spapros = pairs[(pairs["algorithm_a"] != "spapros") & (pairs["algorithm_b"] != "spapros")].copy()
    no_spapros["analysis_set"] = "Without spapros"
    combined = pd.concat([pairs, no_spapros], ignore_index=True)
    combined.to_csv(OUT_DIR / "panel_g_same_vs_cross_modality_jaccard_values.csv", index=False)

    summary = (
        combined.groupby(["analysis_set", "panel_size", "modality_group"], observed=False)["jaccard"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "panel_g_same_vs_cross_modality_jaccard_summary.csv", index=False)

    plot_df = no_spapros.copy()
    order = ["Same modality", "Cross modality"]
    palette = {"Same modality": to_hex(TAB20[0]), "Cross modality": to_hex(TAB20[1])}
    fig, ax = plt.subplots(figsize=(3.45, 4.2))
    sns.barplot(
        data=plot_df,
        x="panel_size",
        y="jaccard",
        hue="modality_group",
        hue_order=order,
        palette=palette,
        errorbar="se",
        capsize=0.08,
        err_kws={"linewidth": 0.8},
        ax=ax,
    )
    rng = np.random.default_rng(2)
    offsets = {"Same modality": -0.2, "Cross modality": 0.2}
    x_positions = {32: 0, 64: 1, 128: 2}
    for group in order:
        sub = plot_df[plot_df["modality_group"] == group]
        xs = sub["panel_size"].map(x_positions).astype(float).to_numpy()
        xs = xs + offsets[group] + rng.uniform(-0.045, 0.045, size=len(sub))
        ax.scatter(xs, sub["jaccard"], s=17, c="black", edgecolors="white", linewidths=0.35, alpha=0.82, zorder=4)
    ax.set_xlabel("Panel size", fontsize=18, labelpad=8)
    ax.set_ylabel("Jaccard similarity", fontsize=18, labelpad=8)
    ax.set_xticks([0, 1, 2], ["32", "64", "128"])
    ax.tick_params(axis="both", labelsize=16, colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.legend(
        frameon=False,
        title="",
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        fontsize=14,
        handlelength=1.25,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.26, right=0.98, bottom=0.18, top=0.82)
    save_figure(fig, "panel_g_gene_panel_jaccard")


def add_sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, text: str) -> None:
    h = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=0.7, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.15, text, ha="center", va="bottom", fontsize=11)


def q_label(q: float, prefix: str = "q") -> str:
    if not np.isfinite(q):
        return "n.s."
    if q < 0.001:
        return f"{prefix}<0.001"
    return f"{prefix}={q:.3f}"


def make_panel_h_ribo_bias_for_size(panel_size: int, stems: list[str]) -> None:
    values = pd.read_csv(MODALITY_DIR / "rna_ribo_modality_bias_by_panel_values.csv")
    tests = pd.read_csv(MODALITY_DIR / "rna_ribo_modality_bias_by_panel_pairwise_tests.csv")
    df = values[(values["scenario"] == "actual panel runs") & (values["panel_size"] == panel_size)].copy()
    if df.empty:
        raise ValueError(f"No panel H ribo-bias values found for panel size {panel_size}")
    order = ["RIBOMap-only", "Shared", "STARmap-only", "Background"]
    palette = {
        "RIBOMap-only": to_hex(TAB20[0]),
        "Shared": to_hex(TAB20[9]),
        "STARmap-only": to_hex(TAB20[3]),
        "Background": "#BDBDBD",
    }
    rng = np.random.default_rng(8)
    plot_df = []
    for group in order:
        sub = df[df["group"] == group]
        if group == "Background" and len(sub) > 600:
            sub = sub.sample(600, random_state=8)
        plot_df.append(sub)
    plot_df = pd.concat(plot_df, ignore_index=True)
    panel_tests = tests[(tests["scenario"] == "actual panel runs") & (tests["panel_size"] == panel_size)].copy()
    for stem in stems:
        df.to_csv(figure_output_path(stem, "_values.csv"), index=False)
        panel_tests.to_csv(figure_output_path(stem, "_pairwise_tests.csv"), index=False)

    fig, ax = plt.subplots(figsize=(3.55, 4.0))
    sns.violinplot(
        data=df,
        x="group",
        y="ribo_bias",
        order=order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=0.9,
        ax=ax,
    )
    for i, group in enumerate(order):
        sub = plot_df[plot_df["group"] == group]
        jitter = rng.uniform(-0.17, 0.17, len(sub))
        ax.scatter(
            np.full(len(sub), i) + jitter,
            sub["ribo_bias"],
            s=8 if group != "Background" else 5,
            c="black",
            alpha=0.55 if group != "Background" else 0.12,
            linewidths=0,
    )
    test = tests[
        (tests["scenario"] == "actual panel runs")
        & (tests["panel_size"] == panel_size)
        & (tests["group_a"] == "RIBOMap-only")
        & (tests["group_b"] == "STARmap-only")
    ]
    if not test.empty:
        q = float(test.iloc[0]["qvalue_bh"])
        label = q_label(q)
        add_sig_bracket(ax, 0, 2, df["ribo_bias"].quantile(0.985), label)
    test_bg = tests[
        (tests["scenario"] == "actual panel runs")
        & (tests["panel_size"] == panel_size)
        & (tests["group_a"] == "RIBOMap-only")
        & (tests["group_b"] == "Background")
    ]
    if not test_bg.empty:
        q = float(test_bg.iloc[0]["qvalue_bh"])
        label = q_label(q)
        add_sig_bracket(ax, 0, 3, df["ribo_bias"].quantile(0.985) + 1.45, label)
    ax.set_xlabel("")
    ax.set_ylabel("RIBOMap bias score", fontsize=17)
    ax.set_xticks(range(len(order)), ["Deep-RIBOmap", "Shared", "STARmap", "Background"], rotation=30, ha="right")
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis="x", colors="black", bottom=True, top=False, direction="out", length=5, width=0.9, labelsize=14)
    ax.tick_params(axis="y", colors="black", left=True, right=False, direction="out", length=5, width=0.9, labelsize=14)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)
    save_figure_multi(fig, stems)


def make_panel_h_ribo_bias() -> None:
    make_panel_h_ribo_bias_for_size(128, ["panel_h_source_modality_ribo_bias"])


def make_panel_h_ribo_bias_by_panel() -> None:
    for panel_size in PANEL_SIZES:
        stems = [f"{BY_PANEL_DIRNAME}/panel_h_source_modality_ribo_bias_p{panel_size}"]
        if panel_size == 128:
            stems.append("panel_h_source_modality_ribo_bias")
        make_panel_h_ribo_bias_for_size(panel_size, stems)


def wrap_term(label: str) -> str:
    return "\n".join(textwrap.wrap(label, width=38, break_long_words=False))


def wrap_term_short(label: str) -> str:
    return "\n".join(textwrap.wrap(label, width=30, break_long_words=False))


def make_enrichment_dotplot(
    df: pd.DataFrame,
    set_order: list[str],
    stem: str,
    top_terms_per_set: int = 4,
    figsize: tuple[float, float] | None = None,
) -> None:
    rows = []
    for set_name in set_order:
        sub = df[df["set_name"].eq(set_name)].sort_values(["qvalue_by_set", "pvalue"]).head(top_terms_per_set)
        rows.append(sub)
    plot_df = pd.concat(rows, ignore_index=True)
    terms = plot_df.sort_values(["qvalue_by_set", "pvalue", "term_label"])["term_label"].drop_duplicates().tolist()
    plot_df = df[df["set_name"].isin(set_order) & df["term_label"].isin(terms)].copy()
    plot_df["set_name"] = pd.Categorical(plot_df["set_name"], categories=set_order, ordered=True)
    plot_df["term_label"] = pd.Categorical(plot_df["term_label"], categories=terms, ordered=True)
    plot_df["dot_size"] = np.sqrt(plot_df["overlap"].clip(lower=0)) * 32
    plot_df["log2_or"] = np.log2(plot_df["odds_ratio"].replace(0, np.nan)).clip(-1, 5)
    if figsize is None:
        figsize = (5.6, max(4.4, 0.36 * len(terms) + 1.1))
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        plot_df["set_name"].cat.codes,
        plot_df["term_label"].cat.codes,
        s=plot_df["dot_size"],
        c=plot_df["log2_or"],
        cmap="viridis",
        vmin=-1,
        vmax=5,
        edgecolors="black",
        linewidths=0.25,
        alpha=0.9,
    )
    ax.set_xticks(range(len(set_order)), set_order, rotation=25, ha="right")
    ax.set_yticks(range(len(terms)), [wrap_term(x) for x in terms])
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_ylabel("")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.025)
    cbar.set_label("log2 odds ratio")
    for overlap in (1, 4, 9):
        ax.scatter([], [], s=math.sqrt(overlap) * 32, c="white", edgecolors="black", linewidths=0.25, label=str(overlap))
    ax.legend(title="Overlap", frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    save_figure(fig, stem)


def source_pathway_terms() -> tuple[list[str], list[str], dict[str, str]]:
    translatome_terms = [
        "Gpcr Ligand Binding",
        "Synaptic Signaling",
        "Signaling By Gpcr",
        "Class C 3 Metabotropic Glutamate Pheromone Receptors",
        "G Alpha I Signalling Events",
        "Regulation Of Trans Synaptic Signaling",
    ]
    starmap_terms = [
        "Gliogenesis",
        "Oligodendrocyte Differentiation",
        "Glial Cell Differentiation",
        "Cell Morphogenesis Involved In Neuron Differentiation",
        "Cell Projection Morphogenesis",
        "Ensheathment Of Neurons",
    ]
    term_order = translatome_terms + starmap_terms
    display_labels = {
        "Gpcr Ligand Binding": "GPCR ligand binding",
        "Synaptic Signaling": "Synaptic signaling",
        "Signaling By Gpcr": "GPCR signaling",
        "Class C 3 Metabotropic Glutamate Pheromone Receptors": "Metabotropic glutamate receptors",
        "G Alpha I Signalling Events": "G alpha i signaling",
        "Regulation Of Trans Synaptic Signaling": "Trans-synaptic signaling",
        "Gliogenesis": "Gliogenesis",
        "Oligodendrocyte Differentiation": "Oligodendrocyte differentiation",
        "Glial Cell Differentiation": "Glial cell differentiation",
        "Cell Morphogenesis Involved In Neuron Differentiation": "Neuron differentiation morphogenesis",
        "Cell Projection Morphogenesis": "Cell projection morphogenesis",
        "Ensheathment Of Neurons": "Ensheathment of neurons",
    }
    return translatome_terms, starmap_terms, display_labels


def load_panel_i_source_enrichment(panel_size: int) -> pd.DataFrame:
    by_panel_candidates = [
        OUT_DIR / BY_PANEL_DIRNAME / f"panel_i_source_modality_pathway_p{panel_size}_all_enrichment.csv",
        OUT_DIR / f"panel_i_source_modality_pathway_p{panel_size}_all_enrichment.csv",
    ]
    by_panel_path = next((path for path in by_panel_candidates if path.exists()), by_panel_candidates[0])
    if by_panel_path.exists():
        df = pd.read_csv(by_panel_path)
    elif panel_size == 128:
        df = pd.read_csv(MODALITY_DIR / "functional_program_enrichment.csv")
    else:
        raise FileNotFoundError(by_panel_path)
    if "set_name" not in df.columns and "group" in df.columns:
        df = df.rename(columns={"group": "set_name"})
    if "qvalue" not in df.columns:
        q_col = "qvalue_by_set" if "qvalue_by_set" in df.columns else "qvalue_group"
        df["qvalue"] = df[q_col]
    return df


def make_panel_i_source_pathway_for_size(panel_size: int, stems: list[str]) -> None:
    df = load_panel_i_source_enrichment(panel_size)
    set_order = ["RIBOMap-only", "STARmap-only"]
    translatome_terms, starmap_terms, display_labels = source_pathway_terms()
    term_order = translatome_terms + starmap_terms

    base = df[df["set_name"].isin(set_order) & df["term_label"].isin(term_order)].copy()
    full_index = pd.MultiIndex.from_product([set_order, term_order], names=["set_name", "term_label"])
    selected_sizes = base.groupby("set_name")["selected_size"].first().to_dict()
    plot_df = base.set_index(["set_name", "term_label"]).reindex(full_index).reset_index()
    plot_df["display_label"] = plot_df["term_label"].map(display_labels)
    plot_df["pathway_group"] = np.where(plot_df["term_label"].isin(translatome_terms), "Translatome-related", "STARmap-enriched")
    plot_df["overlap"] = plot_df["overlap"].fillna(0)
    plot_df["selected_size"] = plot_df.apply(
        lambda row: row["selected_size"]
        if pd.notna(row["selected_size"])
        else selected_sizes.get(row["set_name"], panel_size),
        axis=1,
    )
    plot_df["term_size"] = plot_df["term_size"].fillna(0)
    plot_df["odds_ratio"] = plot_df["odds_ratio"].fillna(0)
    plot_df["qvalue"] = plot_df["qvalue"].fillna(1)
    plot_df["term_label"] = pd.Categorical(plot_df["term_label"], categories=term_order, ordered=True)
    plot_df["set_name"] = pd.Categorical(plot_df["set_name"], categories=set_order, ordered=True)
    plot_df["neg_log10_q"] = -np.log10(plot_df["qvalue"].clip(lower=np.nextafter(0, 1)))
    plot_df["log2_or"] = np.log2(plot_df["odds_ratio"].replace(0, np.nan)).fillna(-1).clip(-1, 5)
    plot_df["dot_size"] = 18 + plot_df["neg_log10_q"].clip(0, 8) * 22
    export_cols = [
        "set_name",
        "term_label",
        "display_label",
        "pathway_group",
        "overlap",
        "selected_size",
        "term_size",
        "odds_ratio",
        "qvalue",
        "neg_log10_q",
        "log2_or",
        "dot_size",
    ]
    for stem in stems:
        plot_df.to_csv(figure_output_path(stem, "_values.csv"), index=False)
        plot_df[export_cols].to_csv(figure_output_path(stem, "_export.csv"), index=False)

    blue_cmap = LinearSegmentedColormap.from_list(
        "tab20_blue_single",
        ["#DCEBF7", to_hex(TAB20[1]), to_hex(TAB20[0])],
    )
    x_pos = {"RIBOMap-only": 0.0, "STARmap-only": 0.008}
    fig, ax = plt.subplots(figsize=(3.75, 4.9))
    sc = ax.scatter(
        plot_df["set_name"].map(x_pos),
        plot_df["term_label"].cat.codes,
        s=plot_df["dot_size"],
        c=plot_df["log2_or"],
        cmap=blue_cmap,
        vmin=-1,
        vmax=5,
        edgecolors="black",
        linewidths=0.25,
        alpha=0.9,
    )
    ax.axhline(len(translatome_terms) - 0.5, color="#9E9E9E", lw=0.8, ls="--")
    ax.set_xticks(list(x_pos.values()), set_order, rotation=25, ha="right")
    ax.set_xlim(-0.008, 0.016)
    ax.set_yticks(
        range(len(term_order)),
        [wrap_term(display_labels[x]) for x in term_order],
    )
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(color="#D0D0D0", linewidth=0.65)
    ax.tick_params(axis="both", colors="black", length=4, width=0.8)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.025)
    cbar.set_label("log2 odds ratio", fontsize=15)
    cbar.ax.tick_params(labelsize=14)
    save_figure_multi(fig, stems)


def make_panel_i_source_pathway() -> None:
    make_panel_i_source_pathway_for_size(128, ["panel_i_source_modality_pathway"])


def make_panel_i_source_pathway_by_panel() -> None:
    for panel_size in PANEL_SIZES:
        stems = [f"{BY_PANEL_DIRNAME}/panel_i_source_modality_pathway_p{panel_size}"]
        if panel_size == 128:
            stems.append("panel_i_source_modality_pathway")
        make_panel_i_source_pathway_for_size(panel_size, stems)


def make_panel_i_aligned_pathway() -> None:
    df = pd.read_csv(ALIGNED_DIR / "pathway_enrichment.csv")
    set_order = ["SMITH RIBOMap p128", "Aligned avg p128", "SMITH STARmap p128"]
    display_set = {
        "SMITH RIBOMap p128": "RIBOMap-only",
        "Aligned avg p128": "Aligned",
        "SMITH STARmap p128": "STARmap-only",
    }
    translatome_terms = [
        "Gpcr Ligand Binding",
        "Synaptic Signaling",
        "Signaling By Gpcr",
        "Class C 3 Metabotropic Glutamate Pheromone Receptors",
        "G Alpha I Signalling Events",
        "Regulation Of Trans Synaptic Signaling",
    ]
    starmap_terms = [
        "Gliogenesis",
        "Oligodendrocyte Differentiation",
        "Glial Cell Differentiation",
        "Cell Morphogenesis Involved In Neuron Differentiation",
        "Cell Projection Morphogenesis",
        "Ensheathment Of Neurons",
    ]
    term_order = translatome_terms + starmap_terms
    display_labels = {
        "Gpcr Ligand Binding": "GPCR ligand binding",
        "Synaptic Signaling": "Synaptic signaling",
        "Signaling By Gpcr": "GPCR signaling",
        "Class C 3 Metabotropic Glutamate Pheromone Receptors": "Metabotropic glutamate receptors",
        "G Alpha I Signalling Events": "G alpha i signaling",
        "Regulation Of Trans Synaptic Signaling": "Trans-synaptic signaling",
        "Gliogenesis": "Gliogenesis",
        "Oligodendrocyte Differentiation": "Oligodendrocyte differentiation",
        "Glial Cell Differentiation": "Glial cell differentiation",
        "Cell Morphogenesis Involved In Neuron Differentiation": "Neuron differentiation morphogenesis",
        "Cell Projection Morphogenesis": "Cell projection morphogenesis",
        "Ensheathment Of Neurons": "Ensheathment of neurons",
    }

    base = df[df["set_name"].isin(set_order) & df["term_label"].isin(term_order)].copy()
    full_index = pd.MultiIndex.from_product([set_order, term_order], names=["set_name", "term_label"])
    plot_df = base.set_index(["set_name", "term_label"]).reindex(full_index).reset_index()
    plot_df["display_set"] = plot_df["set_name"].map(display_set)
    plot_df["display_label"] = plot_df["term_label"].map(display_labels)
    plot_df["pathway_group"] = np.where(
        plot_df["term_label"].isin(translatome_terms),
        "Translatome-related",
        "STARmap-enriched",
    )
    plot_df["overlap"] = plot_df["overlap"].fillna(0)
    plot_df["selected_size"] = plot_df["selected_size"].fillna(128)
    plot_df["term_size"] = plot_df["term_size"].fillna(0)
    plot_df["odds_ratio"] = plot_df["odds_ratio"].fillna(0)
    plot_df["qvalue_by_set"] = plot_df["qvalue_by_set"].fillna(1)
    plot_df["neg_log10_q"] = -np.log10(plot_df["qvalue_by_set"].clip(lower=np.nextafter(0, 1)))
    plot_df["log2_or"] = np.log2(plot_df["odds_ratio"].replace(0, np.nan)).fillna(-1).clip(-1, 5)
    plot_df["dot_size"] = 18 + plot_df["neg_log10_q"].clip(0, 8) * 22
    plot_df["term_label"] = pd.Categorical(plot_df["term_label"], categories=term_order, ordered=True)
    plot_df["set_name"] = pd.Categorical(plot_df["set_name"], categories=set_order, ordered=True)
    plot_df["display_set"] = pd.Categorical(
        plot_df["display_set"],
        categories=["RIBOMap-only", "Aligned", "STARmap-only"],
        ordered=True,
    )
    plot_df.to_csv(OUT_DIR / "panel_i_aligned_pathway_values.csv", index=False)
    export_cols = [
        "display_set",
        "term_label",
        "display_label",
        "pathway_group",
        "overlap",
        "selected_size",
        "term_size",
        "odds_ratio",
        "qvalue_by_set",
        "neg_log10_q",
        "log2_or",
        "dot_size",
    ]
    plot_df[export_cols].to_csv(OUT_DIR / "panel_i_aligned_pathway_export.csv", index=False)

    blue_cmap = LinearSegmentedColormap.from_list(
        "tab20_blue_single_i_aligned",
        ["#DCEBF7", to_hex(TAB20[1]), to_hex(TAB20[0])],
    )
    x_pos = {"RIBOMap-only": 0.0, "Aligned": 0.008, "STARmap-only": 0.016}
    fig, ax = plt.subplots(figsize=(4.35, 4.9))
    sc = ax.scatter(
        plot_df["display_set"].astype(str).map(x_pos),
        plot_df["term_label"].cat.codes,
        s=plot_df["dot_size"],
        c=plot_df["log2_or"],
        cmap=blue_cmap,
        vmin=-1,
        vmax=5,
        edgecolors="black",
        linewidths=0.25,
        alpha=0.9,
    )
    ax.axhline(len(translatome_terms) - 0.5, color="#9E9E9E", lw=0.8, ls="--")
    ax.set_xticks(
        [x_pos[x] for x in x_pos],
        list(x_pos.keys()),
        rotation=25,
        ha="right",
    )
    ax.set_xlim(-0.008, 0.025)
    ax.set_yticks(
        range(len(term_order)),
        [wrap_term(display_labels[x]) for x in term_order],
    )
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(color="#D0D0D0", linewidth=0.65)
    ax.tick_params(axis="both", colors="black", length=4, width=0.8)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=15)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.025)
    cbar.set_label("log2 odds ratio", fontsize=15)
    cbar.ax.tick_params(labelsize=14)
    save_figure(fig, "panel_i_aligned_pathway")


def make_panel_k_aligned_performance() -> None:
    # Panel K is intentionally limited to panel size 128: this is the current
    # story figure, while smaller-panel clean-fusion runs remain exploratory.
    benchmark = pd.read_csv(
        ROOT
        / "outputs"
        / "analysis"
        / "benchmark_significance_all_panels_truncated_smith"
        / "benchmark_methods_summary_current_best.csv"
    )
    benchmark = benchmark[
        (benchmark["metric"] == "accuracy")
        & (benchmark["method"] == "SMITH")
        & (benchmark["label"].isin(["celltype", "region"]))
        & (benchmark["panel_size"] == 128)
        & (benchmark["dataset"].isin(["ribo_deep_to_rep2", "starmap_rep2_to_rep2"]))
    ].copy()
    benchmark["source"] = benchmark["dataset"].map(
        {
            "ribo_deep_to_rep2": "Deep RIBOMap-only",
            "starmap_rep2_to_rep2": "STARmap-only",
        }
    )

    marker_sets = pd.read_csv(ALIGNED_DIR / "marker_sets.csv")
    aligned_row = marker_sets[marker_sets["set_name"].eq("Aligned avg p128")].iloc[0]
    aligned_std = {"celltype": np.nan, "region": np.nan}
    for summary_path in CURRENT_ALIGNED_EVAL_SUMMARIES:
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        sub = summary[
            (summary["metric"] == "accuracy")
            & (summary["panel_size"] == 128)
            & (summary["panel_csv"].astype(str) == str(aligned_row["marker_file"]))
        ]
        for label in ("celltype", "region"):
            hit = sub[sub["label"].eq(label)]
            if not hit.empty:
                aligned_std[label] = float(hit.iloc[0]["metric_std"])

    aligned = pd.DataFrame(
        [
            {
                "source": "Clean fusion",
                "label": "celltype",
                "panel_size": 128,
                "value_mean": float(aligned_row["celltype_accuracy"]),
                "value_std": aligned_std["celltype"],
                "n_seeds": 5,
                "metric": "accuracy",
            },
            {
                "source": "Clean fusion",
                "label": "region",
                "panel_size": 128,
                "value_mean": float(aligned_row["region_accuracy"]),
                "value_std": aligned_std["region"],
                "n_seeds": 5,
                "metric": "accuracy",
            },
        ]
    )

    base_cols = ["source", "label", "panel_size", "value_mean", "value_std", "n_seeds", "metric"]
    df = pd.concat([benchmark[base_cols], aligned[base_cols]], ignore_index=True)
    df["label"] = df["label"].map({"celltype": "Cell type", "region": "Spatial region"})
    df["source"] = pd.Categorical(
        df["source"],
        categories=["Deep RIBOMap-only", "Clean fusion", "STARmap-only"],
        ordered=True,
    )
    df.to_csv(OUT_DIR / "panel_k_aligned_performance_values.csv", index=False)

    palette = {
        "Deep RIBOMap-only": to_hex(TAB20[0]),
        "Clean fusion": to_hex(TAB20[4]),
        "STARmap-only": to_hex(TAB20[2]),
    }
    label_order = ["Cell type", "Spatial region"]
    source_order = ["Deep RIBOMap-only", "Clean fusion", "STARmap-only"]

    fig, axes = plt.subplots(1, 2, figsize=(4.65, 2.15), sharey=True)
    y = np.arange(len(source_order), dtype=float)
    y_labels = ["Deep\nRIBOMap-only", "Clean\nfusion", "STARmap-only"]
    x_limits = {
        "Cell type": (0.295, 0.355),
        "Spatial region": (0.455, 0.525),
    }
    x_ticks = {
        "Cell type": [0.30, 0.33, 0.35],
        "Spatial region": [0.46, 0.49, 0.52],
    }
    for ax, label in zip(axes, label_order):
        sub = df[df["label"].eq(label)].set_index("source").reindex(source_order)
        for yi, source in zip(y, source_order):
            row = sub.loc[source]
            ax.errorbar(
                row["value_mean"],
                yi,
                xerr=row["value_std"],
                fmt="o",
                markersize=7.0,
                markerfacecolor=palette[source],
                markeredgecolor="black",
                markeredgewidth=0.45,
                ecolor="black",
                elinewidth=0.85,
                capsize=2.2,
                zorder=4,
            )
        ax.set_xlim(*x_limits[label])
        ax.set_xticks(x_ticks[label])
        ax.set_xlabel(f"{label}\naccuracy", fontsize=12)
        ax.set_ylim(-0.55, len(source_order) - 0.45)
        ax.grid(axis="x", color="#D8D8D8", linewidth=0.65)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=10.5, colors="black", bottom=True, direction="out", length=4, width=0.85)
        ax.tick_params(axis="y", labelsize=10.5, colors="black", left=True, direction="out", length=4, width=0.85)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.85)
    axes[0].set_yticks(y, y_labels)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.subplots_adjust(left=0.29, right=0.99, bottom=0.35, top=0.97, wspace=0.24)
    save_figure(fig, "panel_k_aligned_transfer_performance")


def make_panel_l_aligned_ribo_bias() -> None:
    if not (OUT_DIR / "panel_k_aligned_performance_values.csv").exists():
        make_panel_k_aligned_performance()
    df = pd.read_csv(OUT_DIR / "panel_k_aligned_performance_values.csv")
    source_order = ["Deep RIBOMap-only", "Clean fusion", "STARmap-only"]
    label_order = ["Cell type", "Spatial region"]
    palette = {
        "Deep RIBOMap-only": to_hex(TAB20[0]),
        "Clean fusion": to_hex(TAB20[4]),
        "STARmap-only": to_hex(TAB20[2]),
    }
    legend_labels = {
        "Deep RIBOMap-only": "Deep-RIBOmap",
        "Clean fusion": "Clean fusion",
        "STARmap-only": "STARmap",
    }
    plot_df = (
        df.pivot(index="label", columns="source", values="value_mean")
        .reindex(index=label_order, columns=source_order)
    )
    err_df = (
        df.pivot(index="label", columns="source", values="value_std")
        .reindex(index=label_order, columns=source_order)
    )
    export = df[df["source"].isin(source_order) & df["label"].isin(label_order)].copy()
    export.to_csv(OUT_DIR / "panel_l_clean_fusion_performance_values.csv", index=False)

    fig, ax = plt.subplots(figsize=(2.65, 2.18))
    x = np.arange(len(label_order), dtype=float)
    width = 0.22
    offsets = np.array([-width, 0.0, width])
    handles = []
    for idx, source in enumerate(source_order):
        yerr = err_df[source].to_numpy(dtype=float)
        yerr = np.where(np.isfinite(yerr), yerr, 0.0)
        bars = ax.bar(
            x + offsets[idx],
            plot_df[source].to_numpy(dtype=float),
            yerr=yerr,
            width=width,
            color=palette[source],
            edgecolor="black",
            linewidth=0.45,
            error_kw={"elinewidth": 0.75, "ecolor": "black", "capsize": 1.8, "capthick": 0.75},
            alpha=0.92,
            zorder=3,
        )
        handles.append(bars[0])
    ax.set_ylim(0.0, 0.56)
    ax.set_yticks([0.0, 0.2, 0.4])
    ax.set_ylabel("Accuracy", fontsize=12.5)
    ax.set_xlabel("")
    ax.set_xticks(x, ["Cell type", "Spatial\nregion"])
    ax.tick_params(axis="x", labelsize=10.5, colors="black", bottom=True, direction="out", length=4, width=0.85)
    ax.tick_params(axis="y", labelsize=10.5, colors="black", left=True, direction="out", length=4, width=0.85)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.85)
    ax.legend(
        handles,
        [legend_labels[source] for source in source_order],
        loc="upper center",
        bbox_to_anchor=(0.52, 1.20),
        ncol=3,
        frameon=False,
        fontsize=7.8,
        handlelength=1.0,
        columnspacing=0.85,
    )
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.22, top=0.82)
    save_figure(fig, "panel_l_clean_fusion_performance")


def qvalue_column(df: pd.DataFrame) -> str:
    for col in ("qvalue_by_set", "qvalue", "qvalue_group"):
        if col in df.columns:
            return col
    raise KeyError("No q-value column found in enrichment table.")


def enriched_terms(df: pd.DataFrame, set_name: str | None = None, q_cutoff: float = 0.05) -> set[str]:
    sub = df.copy()
    if set_name is not None:
        sub = sub[sub["set_name"].eq(set_name)].copy()
    if sub.empty:
        return set()
    q_col = qvalue_column(sub)
    return set(
        sub[(sub[q_col].astype(float) < q_cutoff) & (sub["odds_ratio"].astype(float) > 1)]["term"].astype(str)
    )


def source_panel_enrichment_by_size() -> dict[int, dict[str, set[str]]]:
    metrics = pd.read_csv(MODALITY_DIR / "gene_modality_metrics.csv")
    universe = set(metrics["gene"].astype(str).str.upper())
    terms = load_terms(DEFAULT_MSIGDB, universe, min_size=5, max_size=500)
    out: dict[int, dict[str, set[str]]] = {}
    for panel_size in PANEL_SIZES:
        marker_sets = {
            f"Deep RIBOMap-only p{panel_size}": set(read_marker_list(DEEP_SMITH_MARKER_128, panel_size)),
            f"STARmap-only p{panel_size}": set(read_marker_list(STARMAP_SMITH_MARKER_128, panel_size)),
        }
        enrichment = run_enrichment(marker_sets, terms, universe)
        out[panel_size] = {
            "Deep RIBOMap-only": enriched_terms(enrichment, f"Deep RIBOMap-only p{panel_size}"),
            "STARmap-only": enriched_terms(enrichment, f"STARmap-only p{panel_size}"),
        }
    return out


def panel_m_signature_recapture_tables() -> dict[int, pd.DataFrame]:
    source_terms = source_panel_enrichment_by_size()
    aligned_enrichment = pd.read_csv(ALIGNED_DIR / "pathway_enrichment.csv")
    method_order = ["Deep RIBOMap-only", "Aligned", "STARmap-only"]
    reference_order = ["Deep RIBOMap-only enriched pathways", "STARmap-only enriched pathways"]
    out_dir = OUT_DIR / PANEL_M_PROPORTION_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[int, pd.DataFrame] = {}
    for panel_size in PANEL_SIZES:
        method_terms = {
            "Deep RIBOMap-only": source_terms[panel_size]["Deep RIBOMap-only"],
            "Aligned": enriched_terms(aligned_enrichment, f"Aligned avg p{panel_size}"),
            "STARmap-only": source_terms[panel_size]["STARmap-only"],
        }
        reference_terms = {
            reference_order[0]: source_terms[panel_size]["Deep RIBOMap-only"],
            reference_order[1]: source_terms[panel_size]["STARmap-only"],
        }
        rows = []
        for method in method_order:
            row: dict[str, object] = {"method": method}
            for reference_name in reference_order:
                denominator = len(reference_terms[reference_name])
                numerator = len(method_terms[method] & reference_terms[reference_name])
                row[reference_name] = numerator / denominator if denominator else np.nan
                row[f"{reference_name} count"] = f"{numerator} / {denominator}"
            rows.append(row)
        table = pd.DataFrame(rows)
        fraction_cols = ["method", *reference_order]
        tables[panel_size] = table[fraction_cols].copy()
        tables[panel_size].to_csv(out_dir / f"panel_m_enriched_proportion_p{panel_size}.csv", index=False)

        simple_dir = OUT_DIR / "panel_m_data_simple"
        simple_dir.mkdir(parents=True, exist_ok=True)
        tables[panel_size].to_csv(simple_dir / f"panel_m_enriched_fraction_p{panel_size}.csv", index=False)
    return tables


def panel_m_signature_recapture_long(panel_size: int = 128) -> pd.DataFrame:
    tables = panel_m_signature_recapture_tables()
    if panel_size not in tables:
        raise KeyError(f"No panel M recapture table for panel size {panel_size}")
    return tables[panel_size].melt(
        id_vars="method",
        var_name="reference_signature",
        value_name="recaptured_fraction",
    )


def _make_panel_m_aligned_pathway_legacy() -> None:
    summary = pd.read_csv(ALIGNED_DIR / "marker_modality_summary.csv")
    pathway = pd.read_csv(ALIGNED_DIR / "pathway_enrichment.csv")
    set_order = ["SMITH RIBOMap p128", "Aligned avg p128", "SMITH STARmap p128"]
    display_set = {
        "SMITH RIBOMap p128": "Deep RIBOMap-only",
        "Aligned avg p128": "Clean fusion",
        "SMITH STARmap p128": "STARmap-only",
    }
    source_order = ["Deep RIBOMap-only", "Clean fusion", "STARmap-only"]
    palette = {
        "Deep RIBOMap-only": to_hex(TAB20[0]),
        "Clean fusion": to_hex(TAB20[4]),
        "STARmap-only": to_hex(TAB20[2]),
    }
    translatome_terms = [
        "Synaptic Signaling",
        "Regulation Of Trans Synaptic Signaling",
        "Gpcr Ligand Binding",
        "Signaling By Gpcr",
        "G Alpha I Signalling Events",
    ]
    starmap_terms = [
        "Gliogenesis",
        "Oligodendrocyte Differentiation",
        "Glial Cell Differentiation",
        "Cell Projection Morphogenesis",
        "Ensheathment Of Neurons",
    ]
    selected_terms = translatome_terms + starmap_terms

    bias = summary[summary["set_name"].isin(set_order)].copy()
    bias["source"] = bias["set_name"].map(display_set)
    bias = bias.set_index("source").reindex(source_order).reset_index()

    path = pathway[
        pathway["set_name"].isin(set_order) & pathway["term_label"].isin(selected_terms)
    ].copy()
    path["source"] = path["set_name"].map(display_set)
    path["pathway_group"] = np.where(
        path["term_label"].isin(translatome_terms),
        "Translatome-related",
        "STARmap-enriched",
    )
    path["log2_odds_ratio"] = np.log2(path["odds_ratio"].replace(0, np.nan)).fillna(0).clip(0, 3)
    path_summary = (
        path.groupby(["source", "pathway_group"], as_index=False)
        .agg(
            mean_log2_odds_ratio=("log2_odds_ratio", "mean"),
            n_significant=("qvalue_by_set", lambda x: int((x < 0.05).sum())),
            n_pathways=("qvalue_by_set", "size"),
        )
    )
    export = bias[
        [
            "source",
            "median_ribo_bias",
            "mean_ribo_bias",
            "frac_ribo_bias_gt0",
        ]
    ].merge(path_summary, on="source", how="left")
    export.to_csv(OUT_DIR / "panel_m_clean_fusion_translatome_signal_values.csv", index=False)

    heat_order = ["Translatome-related", "STARmap-enriched"]
    path_heights = (
        path_summary.pivot(index="source", columns="pathway_group", values="mean_log2_odds_ratio")
        .reindex(index=source_order, columns=heat_order)
        .fillna(0)
    )
    path_sig = (
        path_summary.pivot(index="source", columns="pathway_group", values="n_significant")
        .reindex(index=source_order, columns=heat_order)
        .fillna(0)
        .astype(int)
    )
    path_n = (
        path_summary.pivot(index="source", columns="pathway_group", values="n_pathways")
        .reindex(index=source_order, columns=heat_order)
        .fillna(0)
        .astype(int)
    )

    fig, axes = plt.subplots(1, 2, figsize=(4.65, 2.18), gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.45})
    x = np.arange(len(source_order), dtype=float)
    x_labels = ["Deep-\nRIBOmap", "Clean\nfusion", "STARmap"]
    colors = [palette[source] for source in source_order]

    axes[0].bar(
        x,
        bias["median_ribo_bias"].to_numpy(dtype=float),
        width=0.62,
        color=colors,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.92,
        zorder=3,
    )
    axes[0].scatter(
        x,
        bias["mean_ribo_bias"].to_numpy(dtype=float),
        s=14,
        color="black",
        linewidths=0,
        zorder=4,
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylim(-0.14, 0.20)
    axes[0].set_yticks([-0.1, 0.0, 0.1])
    axes[0].set_ylabel("RIBOMap bias score", fontsize=11.2)

    group_x = np.arange(len(heat_order), dtype=float)
    width = 0.22
    offsets = np.array([-width, 0.0, width])
    for source_idx, source in enumerate(source_order):
        values = path_heights.loc[source].to_numpy(dtype=float)
        axes[1].bar(
            group_x + offsets[source_idx],
            values,
            width=width,
            color=palette[source],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.92,
            zorder=3,
        )
        for group_idx, group in enumerate(heat_order):
            axes[1].text(
                group_x[group_idx] + offsets[source_idx],
                float(path_heights.loc[source, group]) + 0.07,
                f"{path_sig.loc[source, group]}/{path_n.loc[source, group]}",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color="black",
                rotation=90,
            )
    axes[1].set_ylim(0, 2.75)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_ylabel("Mean log2 odds ratio", fontsize=10.4)
    axes[1].set_xticks(group_x, ["Translatome\nrelated", "STARmap\nenriched"])

    for ax_idx, ax in enumerate(axes):
        ax.set_xlabel("")
        if ax_idx == 0:
            ax.set_xticks(x, x_labels)
        ax.tick_params(axis="x", labelsize=7.4 if ax_idx == 0 else 7.8, colors="black", bottom=True, direction="out", length=4, width=0.85)
        ax.tick_params(axis="y", labelsize=9.6, colors="black", left=True, direction="out", length=4, width=0.85)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, zorder=0)
        ax.grid(axis="x", visible=False)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.85)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.25, top=0.96)
    save_figure(fig, "panel_m_clean_fusion_translatome_signal")


def make_panel_m_aligned_pathway() -> None:
    recapture = panel_m_signature_recapture_long(128)
    source_order = ["Deep RIBOMap-only", "Aligned", "STARmap-only"]
    reference_order = ["Deep RIBOMap-only enriched pathways", "STARmap-only enriched pathways"]
    palette = {
        "Deep RIBOMap-only": to_hex(TAB20[0]),
        "Aligned": to_hex(TAB20[1]),
        "STARmap-only": to_hex(TAB20[2]),
    }
    recapture.to_csv(OUT_DIR / "panel_m_clean_fusion_translatome_signal_values.csv", index=False)

    plot_mat = (
        recapture.pivot(index="method", columns="reference_signature", values="recaptured_fraction")
        .reindex(index=source_order, columns=reference_order)
        .fillna(0)
    )
    x = np.arange(len(reference_order), dtype=float)
    width = 0.22
    offsets = np.array([-width, 0.0, width])
    legend_labels = {
        "Deep RIBOMap-only": "Deep-RIBOmap",
        "Aligned": "Aligned",
        "STARmap-only": "STARmap",
    }

    fig, ax = plt.subplots(figsize=(3.55, 2.25))
    handles = []
    for source_idx, source in enumerate(source_order):
        bars = ax.bar(
            x + offsets[source_idx],
            plot_mat.loc[source].to_numpy(dtype=float),
            width=width,
            color=palette[source],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.94,
            zorder=3,
        )
        handles.append(bars[0])
    ax.set_xticks(x, ["Deep-RIBOmap\nenriched", "STARmap\nenriched"])
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel("Recaptured fraction", fontsize=12)
    ax.tick_params(axis="x", labelsize=9.5, colors="black", bottom=True, direction="out", length=4, width=0.85)
    ax.tick_params(axis="y", labelsize=10, colors="black", left=True, direction="out", length=4, width=0.85)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.85)
    ax.legend(
        handles,
        [legend_labels[source] for source in source_order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.26),
        ncol=3,
        frameon=False,
        fontsize=8.2,
        handlelength=1.0,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.24, top=0.78)
    save_figure_multi(fig, ["panel_m_clean_fusion_translatome_signal", "panel_m_aligned_pathway"])


def _make_panel_lm_aligned_signal_summary_legacy() -> None:
    bias_summary = pd.read_csv(ALIGNED_DIR / "marker_modality_summary.csv")
    pathway = pd.read_csv(ALIGNED_DIR / "pathway_enrichment.csv")
    perf = pd.read_csv(OUT_DIR / "panel_k_aligned_performance_values.csv")
    set_order = ["SMITH RIBOMap p128", "Aligned avg p128", "SMITH STARmap p128"]
    display_set = {
        "SMITH RIBOMap p128": "Deep RIBOMap-only",
        "Aligned avg p128": "Clean fusion",
        "SMITH STARmap p128": "STARmap-only",
    }
    source_order = ["Deep RIBOMap-only", "Clean fusion", "STARmap-only"]
    palette = {
        "Deep RIBOMap-only": to_hex(TAB20[0]),
        "Clean fusion": to_hex(TAB20[4]),
        "STARmap-only": to_hex(TAB20[2]),
    }
    translatome_terms = [
        "Synaptic Signaling",
        "Regulation Of Trans Synaptic Signaling",
        "Gpcr Ligand Binding",
        "Signaling By Gpcr",
        "G Alpha I Signalling Events",
    ]
    starmap_terms = [
        "Gliogenesis",
        "Oligodendrocyte Differentiation",
        "Glial Cell Differentiation",
        "Cell Projection Morphogenesis",
        "Ensheathment Of Neurons",
    ]
    selected_terms = translatome_terms + starmap_terms

    bias = bias_summary[bias_summary["set_name"].isin(set_order)].copy()
    bias["source"] = bias["set_name"].map(display_set)
    bias = bias.set_index("source").reindex(source_order).reset_index()

    path = pathway[pathway["set_name"].isin(set_order) & pathway["term_label"].isin(selected_terms)].copy()
    path["source"] = path["set_name"].map(display_set)
    path["pathway_group"] = np.where(
        path["term_label"].isin(translatome_terms),
        "Translatome-related",
        "STARmap-enriched",
    )
    path["minus_log10_q"] = -np.log10(path["qvalue_by_set"].clip(lower=np.nextafter(0, 1)))
    path["log2_odds_ratio"] = np.log2(path["odds_ratio"].replace(0, np.nan)).fillna(0).clip(0, 3)
    agg = (
        path.groupby(["source", "pathway_group"], as_index=False)
        .agg(
            mean_log2_odds_ratio=("log2_odds_ratio", "mean"),
            mean_minus_log10_q=("minus_log10_q", "mean"),
            n_significant=("qvalue_by_set", lambda x: int((x < 0.05).sum())),
            n_pathways=("qvalue_by_set", "size"),
        )
    )
    combined = bias[
        [
            "source",
            "median_ribo_bias",
            "mean_ribo_bias",
            "frac_ribo_bias_gt0",
        ]
    ].merge(agg, on="source", how="left")
    combined.to_csv(OUT_DIR / "panel_lm_aligned_signal_summary_values.csv", index=False)

    perf_order = ["Cell type", "Spatial region"]
    perf_mat = (
        perf.pivot(index="source", columns="label", values="value_mean")
        .reindex(index=source_order, columns=perf_order)
    )

    heat_order = ["Translatome-related", "STARmap-enriched"]
    heat = (
        agg.pivot(index="source", columns="pathway_group", values="mean_log2_odds_ratio")
        .reindex(index=source_order, columns=heat_order)
        .fillna(0)
    )
    sig = (
        agg.pivot(index="source", columns="pathway_group", values="n_significant")
        .reindex(index=source_order, columns=heat_order)
        .fillna(0)
        .astype(int)
    )
    n_pathways = (
        agg.pivot(index="source", columns="pathway_group", values="n_pathways")
        .reindex(index=source_order, columns=heat_order)
        .fillna(0)
        .astype(int)
    )

    fig = plt.figure(figsize=(5.85, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.33)
    ax_score = fig.add_subplot(gs[0, 0])
    ax_path = fig.add_subplot(gs[0, 1])

    legend_labels = {
        "Deep RIBOMap-only": "Deep-RIBOmap",
        "Clean fusion": "Clean fusion",
        "STARmap-only": "STARmap",
    }
    bar_width = 0.22
    offsets = np.array([-bar_width, 0.0, bar_width])
    source_colors = [palette[source] for source in source_order]

    score_groups = ["Cell type\naccuracy", "Spatial region\naccuracy", "RIBOMap\nbias"]
    score_x = np.arange(len(score_groups), dtype=float)
    score_values = pd.DataFrame(
        {
            "Cell type\naccuracy": perf_mat["Cell type"],
            "Spatial region\naccuracy": perf_mat["Spatial region"],
            "RIBOMap\nbias": bias.set_index("source").reindex(source_order)["median_ribo_bias"],
        }
    ).reindex(source_order)
    handles = []
    for source_idx, source in enumerate(source_order):
        bars = ax_score.bar(
            score_x + offsets[source_idx],
            score_values.loc[source].to_numpy(dtype=float),
            width=bar_width,
            color=source_colors[source_idx],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.92,
            label=legend_labels[source],
            zorder=3,
        )
        handles.append(bars[0])
    ax_score.axhline(0, color="black", lw=0.8)
    ax_score.axvline(1.5, color="#A8A8A8", lw=0.8, ls="--", zorder=1)
    ax_score.set_xticks(score_x, score_groups)
    ax_score.set_ylim(-0.13, 0.56)
    ax_score.set_yticks([-0.1, 0.0, 0.2, 0.4])
    ax_score.set_ylabel("Score", fontsize=12.5)
    ax_score.tick_params(axis="x", labelsize=9.5, colors="black", bottom=True, direction="out", length=4, width=0.85)
    ax_score.tick_params(axis="y", labelsize=10, colors="black", left=True, direction="out", length=4, width=0.85)
    ax_score.grid(axis="y", color="#D8D8D8", linewidth=0.65, zorder=0)
    ax_score.grid(axis="x", visible=False)
    for spine in ax_score.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.85)

    path_groups = ["Translatome\nrelated", "STARmap\nenriched"]
    path_x = np.arange(len(path_groups), dtype=float)
    path_heights = (
        agg.pivot(index="source", columns="pathway_group", values="mean_log2_odds_ratio")
        .reindex(index=source_order, columns=heat_order)
        .fillna(0)
    )
    path_sig = (
        agg.pivot(index="source", columns="pathway_group", values="n_significant")
        .reindex(index=source_order, columns=heat_order)
        .fillna(0)
        .astype(int)
    )
    path_n = (
        agg.pivot(index="source", columns="pathway_group", values="n_pathways")
        .reindex(index=source_order, columns=heat_order)
        .fillna(0)
        .astype(int)
    )
    for source_idx, source in enumerate(source_order):
        values = path_heights.loc[source].to_numpy(dtype=float)
        ax_path.bar(
            path_x + offsets[source_idx],
            values,
            width=bar_width,
            color=source_colors[source_idx],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.92,
            zorder=3,
        )
        for group_idx, group in enumerate(heat_order):
            x = path_x[group_idx] + offsets[source_idx]
            y_value = float(path_heights.loc[source, group])
            ax_path.text(
                x,
                y_value + 0.07,
                f"{path_sig.loc[source, group]}/{path_n.loc[source, group]}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="black",
                rotation=90,
            )
    ax_path.set_xticks(path_x, path_groups)
    ax_path.set_ylim(0, 3.05)
    ax_path.set_ylabel("Mean log2 odds ratio", fontsize=12)
    ax_path.tick_params(axis="x", labelsize=9.5, colors="black", bottom=True, direction="out", length=4, width=0.85)
    ax_path.tick_params(axis="y", labelsize=10, colors="black", left=True, direction="out", length=4, width=0.85)
    ax_path.grid(axis="y", color="#D8D8D8", linewidth=0.65, zorder=0)
    ax_path.grid(axis="x", visible=False)
    for spine in ax_path.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.85)

    fig.legend(
        handles,
        [legend_labels[source] for source in source_order],
        loc="upper center",
        bbox_to_anchor=(0.53, 1.04),
        ncol=3,
        frameon=False,
        fontsize=9.2,
        handlelength=1.0,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.29, top=0.85)
    save_figure(fig, "panel_lm_aligned_signal_summary")


def make_panel_lm_aligned_signal_summary() -> None:
    if not (OUT_DIR / "panel_k_aligned_performance_values.csv").exists():
        make_panel_k_aligned_performance()
    perf = pd.read_csv(OUT_DIR / "panel_k_aligned_performance_values.csv")
    bias_summary = pd.read_csv(ALIGNED_DIR / "marker_modality_summary.csv")
    recapture = panel_m_signature_recapture_long(128)

    source_order = ["Deep RIBOMap-only", "Aligned", "STARmap-only"]
    source_colors = {
        "Deep RIBOMap-only": to_hex(TAB20[0]),
        "Aligned": to_hex(TAB20[1]),
        "STARmap-only": to_hex(TAB20[2]),
    }
    legend_labels = {
        "Deep RIBOMap-only": "Deep-RIBOmap",
        "Aligned": "Aligned",
        "STARmap-only": "STARmap",
    }
    bias_set_names = {
        "SMITH RIBOMap p128": "Deep RIBOMap-only",
        "Aligned avg p128": "Aligned",
        "SMITH STARmap p128": "STARmap-only",
    }
    bias = bias_summary[bias_summary["set_name"].isin(bias_set_names)].copy()
    bias["source"] = bias["set_name"].map(bias_set_names)
    bias = bias.set_index("source").reindex(source_order)

    perf = perf.copy()
    perf["source"] = perf["source"].replace({"Clean fusion": "Aligned"})
    perf_order = ["Cell type", "Spatial region"]
    perf_mat = (
        perf.pivot(index="source", columns="label", values="value_mean")
        .reindex(index=source_order, columns=perf_order)
    )
    score_groups = ["Cell type\naccuracy", "Spatial region\naccuracy", "RIBOMap\nbias"]
    score_values = pd.DataFrame(
        {
            score_groups[0]: perf_mat["Cell type"],
            score_groups[1]: perf_mat["Spatial region"],
            score_groups[2]: bias["median_ribo_bias"],
        }
    ).reindex(source_order)
    reference_order = ["Deep RIBOMap-only enriched pathways", "STARmap-only enriched pathways"]
    recapture_mat = (
        recapture.pivot(index="method", columns="reference_signature", values="recaptured_fraction")
        .reindex(index=source_order, columns=reference_order)
        .fillna(0)
    )

    score_export = score_values.reset_index(names="source").melt(
        id_vars="source", var_name="metric", value_name="value"
    )
    score_export["panel"] = "performance_and_bias"
    recapture_export = recapture.rename(
        columns={"method": "source", "reference_signature": "metric", "recaptured_fraction": "value"}
    )
    recapture_export["panel"] = "pathway_recapture"
    pd.concat([score_export, recapture_export], ignore_index=True)[
        ["panel", "source", "metric", "value"]
    ].to_csv(OUT_DIR / "panel_lm_aligned_signal_summary_values.csv", index=False)

    fig = plt.figure(figsize=(5.85, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.33)
    ax_score = fig.add_subplot(gs[0, 0])
    ax_path = fig.add_subplot(gs[0, 1])
    bar_width = 0.22
    offsets = np.array([-bar_width, 0.0, bar_width])

    score_x = np.arange(len(score_groups), dtype=float)
    handles = []
    for source_idx, source in enumerate(source_order):
        bars = ax_score.bar(
            score_x + offsets[source_idx],
            score_values.loc[source].to_numpy(dtype=float),
            width=bar_width,
            color=source_colors[source],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.94,
            label=legend_labels[source],
            zorder=3,
        )
        handles.append(bars[0])
    ax_score.axhline(0, color="black", lw=0.8)
    ax_score.axvline(1.5, color="#A8A8A8", lw=0.8, ls="--", zorder=1)
    ax_score.set_xticks(score_x, score_groups)
    ax_score.set_ylim(-0.13, 0.56)
    ax_score.set_yticks([-0.1, 0.0, 0.2, 0.4])
    ax_score.set_ylabel("Score", fontsize=12.5)

    path_x = np.arange(len(reference_order), dtype=float)
    for source_idx, source in enumerate(source_order):
        ax_path.bar(
            path_x + offsets[source_idx],
            recapture_mat.loc[source].to_numpy(dtype=float),
            width=bar_width,
            color=source_colors[source],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.94,
            zorder=3,
        )
    ax_path.set_xticks(path_x, ["Deep-RIBOmap\nenriched", "STARmap\nenriched"])
    ax_path.set_ylim(0, 1.08)
    ax_path.set_yticks([0.0, 0.5, 1.0])
    ax_path.set_ylabel("Recaptured fraction", fontsize=12)

    for ax in (ax_score, ax_path):
        ax.tick_params(axis="x", labelsize=9.5, colors="black", bottom=True, direction="out", length=4, width=0.85)
        ax.tick_params(axis="y", labelsize=10, colors="black", left=True, direction="out", length=4, width=0.85)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, zorder=0)
        ax.grid(axis="x", visible=False)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.85)

    fig.legend(
        handles,
        [legend_labels[source] for source in source_order],
        loc="upper center",
        bbox_to_anchor=(0.53, 1.04),
        ncol=3,
        frameon=False,
        fontsize=9.2,
        handlelength=1.0,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.29, top=0.85)
    save_figure(fig, "panel_lm_aligned_signal_summary")


def add_horizontal_sig_bracket(ax: plt.Axes, y1: float, y2: float, x: float, text: str) -> None:
    xlim = ax.get_xlim()
    width = 0.08 * (xlim[1] - xlim[0])
    ax.plot([x, x + width, x + width, x], [y1, y1, y2, y2], color="black", lw=0.9, clip_on=False)
    ax.text(x + width * 1.2, (y1 + y2) / 2, text, ha="left", va="center", fontsize=12.5)


def make_panel_n_aligned_ribomap_bias_for_size(panel_size: int, stems: list[str]) -> None:
    values_path = OUT_DIR / "panel_l_aligned_ribo_bias_by_panel_values.csv"
    directional_path = OUT_DIR / "panel_l_aligned_ribo_bias_directional_tests.csv"
    values = pd.read_csv(values_path)
    tests = pd.read_csv(directional_path)
    order = ["Deep RIBOMap-only", "Aligned", "STARmap-only"]
    display_labels = ["Deep-RIBOmap", "Aligned", "STARmap"]
    palette = {
        "Deep RIBOMap-only": to_hex(TAB20[0]),
        "Aligned": to_hex(TAB20[1]),
        "STARmap-only": to_hex(TAB20[3]),
    }
    df = values[(values["panel_size"] == panel_size) & (values["group"].isin(order))].copy()
    if df.empty:
        raise ValueError(f"No panel N ribo-bias values found for panel size {panel_size}")
    df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)
    for stem in stems:
        df.to_csv(figure_output_path(stem, "_values.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.15, 2.45))
    sns.violinplot(
        data=df,
        y="group",
        x="ribo_bias",
        order=order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1.15,
        orient="h",
        ax=ax,
    )
    rng = np.random.default_rng(18 + panel_size)
    for y_idx, group in enumerate(order):
        sub = df[df["group"].astype(str).eq(group)]
        jitter = rng.uniform(-0.12, 0.12, len(sub))
        ax.scatter(
            sub["ribo_bias"],
            np.full(len(sub), y_idx) + jitter,
            s=13,
            c="black",
            alpha=0.43,
            linewidths=0,
            zorder=4,
        )
        median = float(sub["ribo_bias"].median())
        ax.scatter(
            [median],
            [y_idx],
            s=44,
            c="white",
            edgecolors="black",
            linewidths=0.75,
            zorder=5,
        )

    ax.axvline(0, color="black", lw=0.85)
    ax.set_xlim(-8.2, 8.2)
    ax.set_xticks([-5, 0, 5])
    ax.set_xlabel("RIBOmap bias score", fontsize=21)
    ax.set_ylabel("")
    ax.set_yticks(range(len(order)), display_labels)
    ax.tick_params(axis="x", labelsize=17, colors="black", bottom=True, direction="out", length=5, width=0.9)
    ax.tick_params(axis="y", labelsize=19, colors="black", left=True, direction="out", length=5, width=0.9)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)

    deep_vs_aligned = tests[
        (tests["panel_size"] == panel_size)
        & (tests["comparison"] == "Aligned vs Deep RIBOMap-only")
    ]
    aligned_vs_star = tests[
        (tests["panel_size"] == panel_size)
        & (tests["comparison"] == "Aligned > STARmap-only")
    ]
    if not deep_vs_aligned.empty:
        q = float(deep_vs_aligned.iloc[0]["mannwhitney_one_sided_p"])
        label = "n.s.\n" + q_label(q)
        add_horizontal_sig_bracket(ax, 0, 1, 1.95, label)
    if not aligned_vs_star.empty:
        q_col = "mannwhitney_one_sided_q_across_panel_sizes"
        q = float(aligned_vs_star.iloc[0][q_col])
        add_horizontal_sig_bracket(ax, 1, 2, 3.25, q_label(q))
    fig.subplots_adjust(left=0.26, right=0.90, bottom=0.30, top=0.98)
    save_figure_multi(fig, stems)


def make_panel_n_aligned_ribomap_bias_by_panel() -> None:
    for panel_size in PANEL_SIZES:
        stems = [f"{BY_PANEL_DIRNAME}/panel_n_aligned_ribomap_bias_violin_horizontal_p{panel_size}"]
        if panel_size == 128:
            stems.append("panel_n_aligned_ribomap_bias_violin_horizontal")
        make_panel_n_aligned_ribomap_bias_for_size(panel_size, stems)


def make_panel_j_data_visualization() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    star = ad.read_h5ad(STARMAP_H5AD)
    ribo = ad.read_h5ad(RIBOMAP_H5AD)
    aligned = ad.read_h5ad(ALIGNED_H5AD)

    star_coords = np.asarray(star.obsm["spatial"])[:, :2].astype(float)
    ribo_coords = np.asarray(ribo.obsm["spatial"])[:, :2].astype(float)
    star_norm = normalize_coords(star_coords)
    ribo_norm = normalize_coords(ribo_coords)
    aligned_norm = normalize_coords(np.asarray(aligned.obsm["spatial"])[:, :2].astype(float), ref_coords=star_coords)

    star_space_regions, star_space_conf = spatial_knn_transfer_regions(
        star_norm,
        ribo_norm,
        ribo.obs["region"],
        k=31,
    )
    aligned_space_regions, aligned_space_conf = spatial_knn_transfer_regions(
        aligned_norm,
        ribo_norm,
        ribo.obs["region"],
        k=31,
    )
    aligned.obs["starmap_space_region"] = aligned_space_regions
    aligned.obs["starmap_space_region_confidence"] = aligned_space_conf

    region_order = [
        "Isocortex",
        "Hippocampal region",
        "Cortical subplate",
        "Olfactory areas",
        "Cerebal nuclei",
        "Thalamus",
        "Hypothalamus",
        "Fiber tracts",
        "other",
    ]
    present = [region for region in region_order if region in set(ribo.obs["region"].astype(str))]
    present += sorted(set(ribo.obs["region"].astype(str)) - set(present))
    region_colors = {region: to_hex(TAB20[i % len(TAB20)]) for i, region in enumerate(present)}
    standalone_legend_from_categories(present, region_colors, "panel_j_region_legend", nrows=2)

    score, score_genes = ribomap_only_marker_score(aligned)
    pd.DataFrame(
        {
            "starmap_obs_name": star.obs_names.astype(str),
            "starmap_space_region": star_space_regions,
            "starmap_space_region_confidence": star_space_conf,
        }
    ).to_csv(OUT_DIR / "panel_j_starmap_space_region_transfer.csv", index=False)
    pd.DataFrame(
        {
            "aligned_obs_name": aligned.obs_names.astype(str),
            "starmap_space_region": aligned_space_regions,
            "starmap_space_region_confidence": aligned_space_conf,
            "ribomap_only_marker_score": score,
        }
    ).to_csv(OUT_DIR / "panel_j_aligned_ribomap_expression_score.csv", index=False)
    (OUT_DIR / "panel_j_ribomap_only_marker_score_genes.txt").write_text("\n".join(score_genes) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.55), constrained_layout=True)
    plot_labelled_spatial(axes[0], ribo_norm, ribo.obs["region"], region_colors, point_size=0.75)
    plot_labelled_spatial(axes[1], star_norm, star_space_regions, region_colors, point_size=0.75)
    blue_cmap = LinearSegmentedColormap.from_list(
        "tab20_blue_single_j",
        ["#EAF2FB", to_hex(TAB20[1]), to_hex(TAB20[0])],
    )
    mappable = plot_continuous_spatial(axes[2], star_norm, aligned_norm, score, blue_cmap)
    cbar = fig.colorbar(mappable, ax=axes[2], fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.6)
    save_figure(fig, "panel_j_aligned_transfer_visualization")


def selected_panels(text: str) -> set[str]:
    wanted = {x.strip().lower() for x in text.split(",") if x.strip()}
    return wanted or {"all"}


def wants_panel(wanted: set[str], panel: str) -> bool:
    return "all" in wanted or panel in wanted


def main() -> None:
    global ALIGNED_H5AD, ALIGNED_EVAL_SUMMARY, ALIGNED_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panels",
        default="all",
        help="Comma-separated panels to regenerate, e.g. j,k,l,m. Default: all.",
    )
    parser.add_argument(
        "--aligned-h5ad",
        type=Path,
        default=ALIGNED_H5AD,
        help="Aligned h5ad file used for panel J.",
    )
    parser.add_argument(
        "--aligned-eval-summary",
        type=Path,
        default=ALIGNED_EVAL_SUMMARY,
        help="Aligned eval_summary.csv used for panel K.",
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=ALIGNED_DIR,
        help="Aligned pathway analysis directory used for panels L/M.",
    )
    args = parser.parse_args()
    wanted = selected_panels(args.panels)

    configure_plotting()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ALIGNED_H5AD = args.aligned_h5ad
    ALIGNED_EVAL_SUMMARY = args.aligned_eval_summary
    ALIGNED_DIR = args.aligned_dir
    write_prompts()
    if wants_panel(wanted, "b"):
        make_panel_b_separate_spatial()
    if wants_panel(wanted, "g"):
        make_panel_g_jaccard()
    if wants_panel(wanted, "h"):
        make_panel_h_ribo_bias()
    if wants_panel(wanted, "h_by_panel"):
        make_panel_h_ribo_bias_by_panel()
    if wants_panel(wanted, "i"):
        make_panel_i_source_pathway()
    if wants_panel(wanted, "i_by_panel"):
        make_panel_i_source_pathway_by_panel()
    if wants_panel(wanted, "i_aligned"):
        make_panel_i_aligned_pathway()
    if wants_panel(wanted, "j"):
        make_panel_j_umap_alignment_quality()
        make_panel_j_data_visualization()
    if wants_panel(wanted, "k"):
        make_panel_k_aligned_performance()
    if wants_panel(wanted, "l"):
        make_panel_l_aligned_ribo_bias()
    if wants_panel(wanted, "m"):
        make_panel_m_aligned_pathway()
    if wants_panel(wanted, "l") or wants_panel(wanted, "m") or wants_panel(wanted, "lm"):
        make_panel_lm_aligned_signal_summary()
    if wants_panel(wanted, "n") or wants_panel(wanted, "n_by_panel"):
        make_panel_n_aligned_ribomap_bias_by_panel()

    manifest = {
        "panel_a": "panel_a_schematic_prompt.txt",
        "panel_b": {
            "unified_celltype": [
                "panel_b_unified_celltype_deep_ribomap_source.pdf/png",
                "panel_b_unified_celltype_starmap_source.pdf/png",
                "panel_b_unified_celltype_ribomap_target.pdf/png",
                "panel_b_unified_celltype_legend.pdf/png",
                "panel_b_unified_celltype_legend_3rows.pdf/png",
                "panel_b_unified_celltype_legend_4rows.pdf/png",
                "panel_b_unified_celltype_palette.csv",
            ],
            "fine_grained_celltype": [
                "panel_b_fine_grained_celltype_deep_ribomap_source.pdf/png",
                "panel_b_fine_grained_celltype_starmap_source.pdf/png",
                "panel_b_fine_grained_celltype_ribomap_target.pdf/png",
                "panel_b_fine_grained_celltype_*_palette.csv",
            ],
        },
        "panel_c_to_f": "reserved for benchmark panels generated separately",
        "panel_g": "panel_g_gene_panel_jaccard.pdf/png",
        "panel_h": {
            "main": "panel_h_source_modality_ribo_bias.pdf/png",
            "by_panel": [
                f"{BY_PANEL_DIRNAME}/panel_h_source_modality_ribo_bias_p{panel_size}.pdf/png"
                for panel_size in PANEL_SIZES
            ],
        },
        "panel_i": {
            "main": "panel_i_source_modality_pathway.pdf/png",
            "by_panel": [
                f"{BY_PANEL_DIRNAME}/panel_i_source_modality_pathway_p{panel_size}.pdf/png"
                for panel_size in PANEL_SIZES
            ],
            "note": "Source-modality unique gene sets; shared genes are removed, so selected sizes can be smaller than nominal panel sizes.",
        },
        "panel_j": [
            "panel_j_alignment_schematic_prompt.txt",
            "panel_j_aligned_transfer_visualization.pdf/png",
            "panel_j_alignment_quality_1_deep_ribomap_native_umap.pdf/png",
            "panel_j_alignment_quality_2_deep_labels_on_starmap_umap.pdf/png",
            "panel_j_alignment_quality_3_starmap_labels_on_starmap_umap.pdf/png",
        ],
        "panel_k": "panel_k_aligned_transfer_performance.pdf/png",
        "panel_l": {
            "recommended": "panel_l_aligned_performance_compact.pdf/png",
            "bar_version": "panel_l_clean_fusion_performance.pdf/png",
            "legacy_bias": "panel_l_aligned_ribo_bias.pdf/png",
        },
        "panel_m": "panel_m_aligned_pathway.pdf/png",
        "panel_n": {
            "main": "panel_n_aligned_ribomap_bias_violin_horizontal.pdf/png",
            "by_panel": [
                f"{BY_PANEL_DIRNAME}/panel_n_aligned_ribomap_bias_violin_horizontal_p{panel_size}.pdf/png"
                for panel_size in PANEL_SIZES
            ],
        },
        "panel_lm": "panel_lm_aligned_signal_summary.pdf/png",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[info] wrote ribo figure panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
