from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = REPO_ROOT / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_paired_source"
DEFAULT_MULTI_SEED_RESULT_DIR = REPO_ROOT / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_5seed_panels"
DEFAULT_MERFISH = REPO_ROOT / "data/liver_merfish/adata_healthy_merfish.h5ad"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures/agent_visium_retrieval_refined"
DEFAULT_OUTPUT_PREFIX = DEFAULT_FIGURE_DIR / "liver_merfish_visium_agent_block"
DEFAULT_PANEL_DIR = DEFAULT_FIGURE_DIR
ARIAL_FONT_FILES = [
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
]

PALETTE = {
    "source": "#BDBDBD",
    "source_dark": "#7A7A7A",
    "integrated": "#2F7FB9",
    "integrated_light": "#9DBCDC",
    "visium": "#6EA45E",
    "shared": "#9AA6B2",
    "gain": "#6E4AA5",
    "loss": "#B85C5C",
    "hepatocyte": "#D9B36C",
    "stromal": "#5E8C8A",
    "immune": "#8A6AA8",
    "endothelial": "#6E8FB9",
    "background": "#FFFFFF",
    "ink": "#222222",
}


def _configure_matplotlib() -> None:
    for font_file in ARIAL_FONT_FILES:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    if "Arial" not in available_fonts:
        print("Warning: Arial font is not registered in matplotlib; falling back to DejaVu Sans.")
    mpl.rcParams.update(
        {
            "font.family": "Arial" if "Arial" in available_fonts else "DejaVu Sans",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.55,
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "legend.frameon": False,
        }
    )


def _read_result(result_dir: Path) -> dict[str, Any]:
    with (result_dir / "formal_benchmark_result.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric(summary: pd.DataFrame, panel: str, metric: str) -> float:
    row = summary[(summary["panel"] == panel) & (summary["metric"] == metric)]
    if row.empty:
        raise KeyError(f"Missing metric {metric} for panel {panel}")
    return float(row["value"].iloc[0])


def _clean_gene_set(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "gene" in out.columns and "gene_symbol" not in out.columns:
        out = out.rename(columns={"gene": "gene_symbol"})
    return out


def _matrix_for_genes(adata: ad.AnnData, genes: list[str]) -> np.ndarray:
    x = adata[:, genes].X
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _load_panel_genes(panel_path: str | Path, panel_size: int = 64) -> list[str]:
    panel_path = Path(panel_path)
    sep = "\t" if panel_path.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(panel_path, sep=sep)
    gene_col = next(
        (col for col in df.columns if str(col).lower() in {"gene", "genes", "gene_symbol", "target"}),
        df.columns[0],
    )
    genes = [str(gene).strip().upper() for gene in df[gene_col].tolist() if str(gene).strip()]
    return genes[:panel_size]


def _compute_cell_type_repeats(
    result: dict[str, Any],
    merfish_file: Path,
    output_tsv: Path,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    panel_size: int = 64,
    label_column: str = "Cell_Type",
) -> pd.DataFrame:
    if output_tsv.exists():
        cached = pd.read_csv(output_tsv, sep="\t")
        if set(cached["seed"].astype(int)) >= set(seeds):
            return cached

    panels = {
        "paired snRNA only": result["source_panel_tsv"],
        "snRNA + multi-Visium": result["integrated_panel_tsv"],
    }
    adata = ad.read_h5ad(merfish_file)
    try:
        labels = adata.obs[label_column].astype(str).to_numpy()
        valid = pd.notna(labels) & (labels != "") & (labels != "nan")
        labels = labels[valid]
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
        rows = []
        for panel_label, panel_path in panels.items():
            genes = [gene for gene in _load_panel_genes(panel_path, panel_size=panel_size) if gene in adata.var_names]
            x = _matrix_for_genes(adata, genes)[valid]
            indices = np.arange(x.shape[0])
            for seed in seeds:
                train_idx, test_idx = train_test_split(
                    indices,
                    test_size=0.2,
                    random_state=seed,
                    stratify=y,
                )
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        n_jobs=1,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                )
                model.fit(x[train_idx], y[train_idx])
                pred = model.predict(x[test_idx])
                rows.extend(
                    [
                        {
                            "panel": panel_label,
                            "seed": seed,
                            "metric": "Accuracy",
                            "value": float(accuracy_score(y[test_idx], pred)),
                        },
                        {
                            "panel": panel_label,
                            "seed": seed,
                            "metric": "Balanced accuracy",
                            "value": float(balanced_accuracy_score(y[test_idx], pred)),
                        },
                        {
                            "panel": panel_label,
                            "seed": seed,
                            "metric": "Macro F1",
                            "value": float(f1_score(y[test_idx], pred, average="macro")),
                        },
                    ]
                )
    finally:
        del adata

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(output_tsv, sep="\t", index=False)
    return out


def _load_training_seed_panel_size_repeats(result_dir: Path, metric: str = "cell_type_macro_f1") -> pd.DataFrame | None:
    metrics_tsv = result_dir / "multi_seed_panel_size_metrics.tsv"
    if not metrics_tsv.exists():
        return None
    df = pd.read_csv(metrics_tsv, sep="\t")
    required = {"training_seed", "panel_size", "panel", "metric", "value"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"{metrics_tsv} is missing required columns: {missing}")
    df = df[df["metric"] == metric].copy()
    if df.empty:
        available = sorted(pd.read_csv(metrics_tsv, sep="\t")["metric"].astype(str).unique())
        raise ValueError(f"Metric `{metric}` was not found in {metrics_tsv}. Available metrics: {available}")
    panel_labels = {
        "source_smith": "snRNA only",
        "multi_visium_smith": "snRNA + multi-Visium",
    }
    df["panel"] = df["panel"].map(panel_labels).fillna(df["panel"].astype(str))
    df["seed"] = df["training_seed"].astype(int)
    metric_labels = {
        "cell_type_accuracy": "Cell type classification accuracy",
        "cell_type_balanced_accuracy": "Held-out MERFISH cell balanced accuracy",
        "cell_type_macro_f1": "Held-out MERFISH cell macro F1",
        "cell_type_weighted_f1": "Held-out MERFISH cell weighted F1",
    }
    df["metric_label"] = metric_labels.get(metric, metric)
    return df


def _load_or_compute_merfish_gene_expression_support(
    merfish_file: Path,
    output_tsv: Path,
) -> pd.DataFrame:
    if output_tsv.exists():
        return pd.read_csv(output_tsv, sep="\t")

    adata = ad.read_h5ad(merfish_file)
    try:
        x = adata.X
        if sparse.issparse(x):
            mean_expression = np.asarray(x.mean(axis=0)).ravel()
            detection_rate = np.asarray((x > 0).mean(axis=0)).ravel()
        else:
            x = np.asarray(x, dtype=np.float32)
            mean_expression = np.asarray(x.mean(axis=0)).ravel()
            detection_rate = np.asarray((x > 0).mean(axis=0)).ravel()
        genes = [str(gene).strip().upper() for gene in adata.var_names.astype(str).tolist()]
        out = pd.DataFrame(
            {
                "gene_symbol": genes,
                "mean_expression": mean_expression.astype(float),
                "detection_rate": detection_rate.astype(float),
            }
        )
    finally:
        del adata

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_tsv, sep="\t", index=False)
    return out


def _load_panel_support_repeats(multi_seed_result_dir: Path, merfish_file: Path) -> pd.DataFrame:
    support = _load_or_compute_merfish_gene_expression_support(
        merfish_file,
        multi_seed_result_dir / "diagnostics/merfish_gene_expression_support.tsv",
    )
    expression_map = dict(zip(support["gene_symbol"].astype(str).str.upper(), support["mean_expression"].astype(float)))
    rows = []
    for seed_dir in sorted(multi_seed_result_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        for panel_size in (32, 64, 128):
            panel_files = {
                "snRNA only": seed_dir / "panels" / f"source_top_{panel_size}_panel.tsv",
                "snRNA + multi-Visium": seed_dir / "panels" / f"multi_visium_top_{panel_size}_panel.tsv",
            }
            for panel_label, panel_file in panel_files.items():
                if not panel_file.exists():
                    continue
                genes = _load_panel_genes(panel_file, panel_size=panel_size)
                values = [expression_map[gene] for gene in genes if gene in expression_map]
                if not values:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "panel_size": panel_size,
                        "panel": panel_label,
                        "value": float(np.mean(values)),
                        "n_genes": len(values),
                        "metric_label": "Mean MERFISH expression",
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No panel support rows could be built from {multi_seed_result_dir}.")
    support_tsv = multi_seed_result_dir / "diagnostics/panel_size_merfish_expression_support.tsv"
    support_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(support_tsv, sep="\t", index=False)
    return out


def _load_marker_expression(merfish_file: Path, genes: list[str], label_column: str = "Cell_Type") -> pd.DataFrame:
    adata = ad.read_h5ad(merfish_file)
    try:
        genes_present = [gene for gene in genes if gene in adata.var_names]
        x = _matrix_for_genes(adata, genes_present)
        labels = adata.obs[label_column].astype(str).to_numpy()
        rows = []
        for gene_idx, gene in enumerate(genes_present):
            values = x[:, gene_idx]
            for label in sorted(set(labels)):
                mask = labels == label
                rows.append({"gene_symbol": gene, "cell_type": label, "mean_expression": float(values[mask].mean())})
        expr = pd.DataFrame(rows)
    finally:
        del adata
    if expr.empty:
        return expr
    expr["z_expression"] = expr.groupby("gene_symbol")["mean_expression"].transform(
        lambda value: (value - value.mean()) / (value.std(ddof=0) + 1e-8)
    )
    return expr


def _plot_reference_schematic(ax: plt.Axes, result: dict[str, Any]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def rounded_box(x: float, y: float, w: float, h: float, text: str, color: str, text_color: str = "white") -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.035",
            facecolor=color,
            edgecolor="black",
            linewidth=0.35,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.2, color=text_color)

    def matrix_icon(x: float, y: float, color: str, label: str) -> None:
        for i in range(4):
            for j in range(3):
                ax.add_patch(
                    Rectangle(
                        (x + i * 0.026, y + j * 0.026),
                        0.022,
                        0.022,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=0.2,
                        alpha=0.88 - 0.07 * ((i + j) % 3),
                    )
                )
        ax.text(x + 0.052, y - 0.023, label, ha="center", va="top", fontsize=5.7)

    matrix_icon(0.06, 0.57, PALETTE["source_dark"], "paired snRNA")
    for i in range(5):
        matrix_icon(0.05 + i * 0.055, 0.22, PALETTE["visium"], f"V{i + 1}")

    rounded_box(0.42, 0.43, 0.18, 0.18, "SMITH\nrank fusion", "#E8EEF4", PALETTE["ink"])
    rounded_box(0.76, 0.43, 0.15, 0.18, "64-gene\npanel", PALETTE["integrated"], "white")

    ax.annotate("", xy=(0.41, 0.52), xytext=(0.20, 0.62), arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#555555"))
    for i in range(5):
        ax.annotate(
            "",
            xy=(0.43, 0.45),
            xytext=(0.08 + i * 0.055, 0.31),
            arrowprops=dict(arrowstyle="-", lw=0.55, color="#8AA37E"),
        )
    ax.annotate("", xy=(0.75, 0.52), xytext=(0.61, 0.52), arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#555555"))

    ax.text(0.02, 0.94, "Reference-aware panel selection", ha="left", va="top", fontsize=6.8)
    ax.text(0.42, 0.27, "paired transcriptome\n+ in situ spatial context", ha="left", va="top", fontsize=5.8, color="#444444")
    n_refs = len(result["reference_runs"])
    ax.text(0.08, 0.08, f"{n_refs} liver Visium references", ha="left", va="center", fontsize=5.8, color="#444444")


def _plot_performance(ax: plt.Axes, repeats: pd.DataFrame) -> None:
    if "panel_size" in repeats.columns:
        panel_sizes = sorted(repeats["panel_size"].astype(int).unique())
        panels = ["snRNA only", "snRNA + multi-Visium"]
        colors = [PALETTE["source"], PALETTE["integrated_light"]]
        edge_colors = [PALETTE["source_dark"], PALETTE["integrated"]]
        offsets = [-0.16, 0.16]
        rng = np.random.default_rng(17)
        for size_idx, panel_size in enumerate(panel_sizes):
            for panel_idx, panel in enumerate(panels):
                values = repeats.loc[
                    (repeats["panel_size"].astype(int) == panel_size) & (repeats["panel"] == panel),
                    "value",
                ].to_numpy(dtype=float)
                if values.size == 0:
                    continue
                pos = size_idx + offsets[panel_idx]
                violin = ax.violinplot(
                    [values],
                    positions=[pos],
                    widths=0.26,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                body = violin["bodies"][0]
                body.set_facecolor(colors[panel_idx])
                body.set_edgecolor("none")
                body.set_alpha(0.40)
                jitter = rng.normal(0, 0.018, size=len(values))
                ax.scatter(
                    np.full(len(values), pos) + jitter,
                    values,
                    s=10,
                    color=colors[panel_idx],
                    edgecolor="black",
                    linewidth=0.25,
                    zorder=3,
                )
                median = float(np.median(values))
                ax.plot([pos - 0.10, pos + 0.10], [median, median], color=edge_colors[panel_idx], lw=0.75, zorder=4)
        ax.set_xticks(np.arange(len(panel_sizes)), [str(size) for size in panel_sizes])
        ax.set_xlabel("Panel size")
        metric_label = str(repeats["metric_label"].iloc[0]) if "metric_label" in repeats.columns else "MERFISH test performance"
        ax.set_ylabel(metric_label)
        ax.xaxis.label.set_size(9.0)
        ax.yaxis.label.set_size(9.0)
        ax.tick_params(axis="both", labelsize=8.2)
        finite_values = repeats["value"].to_numpy(dtype=float)
        ymin = max(0.0, float(np.nanmin(finite_values)) - 0.035)
        ymax = min(1.0, float(np.nanmax(finite_values)) + 0.055)
        if ymax - ymin < 0.12:
            center = (ymin + ymax) / 2
            ymin = max(0.0, center - 0.06)
            ymax = min(1.0, center + 0.06)
        ax.set_ylim(ymin, ymax)
        y_span = ymax - ymin
        stat_base = ymax - 0.11 * y_span
        stat_step = 0.045 * y_span
        for size_idx, panel_size in enumerate(panel_sizes):
            subset = repeats[repeats["panel_size"].astype(int) == panel_size]
            paired = subset.pivot_table(index="seed", columns="panel", values="value", aggfunc="mean")
            if not set(panels).issubset(paired.columns) or paired.dropna().shape[0] < 2:
                continue
            paired = paired.dropna(subset=panels)
            try:
                p_value = wilcoxon(
                    paired["snRNA + multi-Visium"],
                    paired["snRNA only"],
                    alternative="greater",
                    zero_method="wilcox",
                ).pvalue
            except ValueError:
                continue
            x0 = size_idx + offsets[0]
            x1 = size_idx + offsets[1]
            y = stat_base + (size_idx % 2) * stat_step
            ax.plot([x0, x0, x1, x1], [y, y + stat_step * 0.25, y + stat_step * 0.25, y], color="#333333", lw=0.45)
            ax.text((x0 + x1) / 2, y + stat_step * 0.36, f"p={p_value:.3f}", ha="center", va="bottom", fontsize=8.0)
        handles = [
            Rectangle((0, 0), 1, 1, facecolor=colors[0], edgecolor="black", linewidth=0.35, alpha=0.65),
            Rectangle((0, 0), 1, 1, facecolor=colors[1], edgecolor="black", linewidth=0.35, alpha=0.65),
        ]
        ax.legend(handles, panels, loc="lower right", fontsize=6.3, handlelength=1.1, borderaxespad=0.3)
        return

    metrics = ["Accuracy", "Balanced accuracy", "Macro F1"]
    panels = ["paired snRNA only", "snRNA + multi-Visium"]
    colors = [PALETTE["source"], PALETTE["integrated_light"]]
    offsets = [-0.17, 0.17]
    rng = np.random.default_rng(11)
    for metric_idx, metric in enumerate(metrics):
        for panel_idx, panel in enumerate(panels):
            values = repeats.loc[(repeats["metric"] == metric) & (repeats["panel"] == panel), "value"].to_numpy()
            pos = metric_idx + offsets[panel_idx]
            violin = ax.violinplot(
                [values],
                positions=[pos],
                widths=0.28,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            body = violin["bodies"][0]
            body.set_facecolor(colors[panel_idx])
            body.set_edgecolor("none")
            body.set_alpha(0.36)
            jitter = rng.normal(0, 0.018, size=len(values))
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=9,
                color=colors[panel_idx],
                edgecolor="black",
                linewidth=0.25,
                zorder=3,
            )
            median = float(np.median(values))
            ax.plot([pos - 0.11, pos + 0.11], [median, median], color="black", lw=0.55, zorder=4)
    ax.set_xticks(np.arange(len(metrics)), ["Accuracy", "Balanced\naccuracy", "Macro F1"])
    ax.set_ylim(0.68, 0.91)
    ax.set_ylabel("MERFISH test performance")
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=colors[0], edgecolor="black", linewidth=0.35, alpha=0.65),
        Rectangle((0, 0), 1, 1, facecolor=colors[1], edgecolor="black", linewidth=0.35, alpha=0.65),
    ]
    ax.legend(handles, panels, loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=5.4, handlelength=1.0, borderaxespad=0.0)


def _plot_panel_shift(ax: plt.Axes, result: dict[str, Any]) -> None:
    source_panel = set(result["aggregation"]["source_top_panel"])
    integrated_panel = set(result["aggregation"]["integrated_top_panel"])
    shared = len(source_panel & integrated_panel)
    source_only = len(source_panel - integrated_panel)
    integrated_only = len(integrated_panel - source_panel)
    jaccard = result["aggregation"]["jaccard"]

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    left = Circle((0.40, 0.54), radius=0.25, facecolor=PALETTE["source"], alpha=0.45, edgecolor="black", lw=0.45)
    right = Circle((0.60, 0.54), radius=0.25, facecolor=PALETTE["integrated_light"], alpha=0.55, edgecolor="black", lw=0.45)
    ax.add_patch(left)
    ax.add_patch(right)
    ax.text(0.27, 0.54, str(source_only), ha="center", va="center", fontsize=12, color=PALETTE["source_dark"])
    ax.text(0.50, 0.54, str(shared), ha="center", va="center", fontsize=12, color="#333333")
    ax.text(0.73, 0.54, str(integrated_only), ha="center", va="center", fontsize=12, color=PALETTE["integrated"])
    ax.text(0.27, 0.24, "source\nonly", ha="center", va="center", fontsize=5.8, color=PALETTE["source_dark"])
    ax.text(0.50, 0.24, "shared", ha="center", va="center", fontsize=6.2, color="#53606A")
    ax.text(0.73, 0.24, "Visium-\npromoted", ha="center", va="center", fontsize=6.2, color=PALETTE["integrated"])
    ax.text(0.50, 0.94, "Panel shift after spatial-reference fusion", ha="center", va="top", fontsize=6.8)
    ax.text(0.50, 0.08, f"64 genes; Jaccard = {jaccard:.3f}", ha="center", va="center", fontsize=6.3, color="#46545F")


def _plot_merfish_support(ax: plt.Axes, changed: pd.DataFrame) -> None:
    data = [
        changed.loc[changed["set"] == "source_only", "mutual_info"].dropna().to_numpy(),
        changed.loc[changed["set"] == "integrated_only", "mutual_info"].dropna().to_numpy(),
    ]
    positions = [0, 1]
    parts = ax.violinplot(data, positions=positions, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], [PALETTE["source"], PALETTE["integrated"]]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.22)
    rng = np.random.default_rng(7)
    for pos, values, color in zip(positions, data, [PALETTE["source"], PALETTE["integrated"]]):
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(np.full(len(values), pos) + jitter, values, s=8, color=color, alpha=0.90, linewidths=0)
        median = float(np.median(values))
        ax.plot([pos - 0.20, pos + 0.20], [median, median], color=color, lw=1.2)
        ax.text(pos + 0.02, median + 0.026, f"{median:.3f}", ha="center", va="bottom", fontsize=5.4, color=color)
    ax.set_xticks(positions, ["source-only\nreplaced", "Visium-promoted\nadded"])
    ax.set_ylabel("MERFISH cell-type\nmutual information")
    ax.set_title("MERFISH support for Visium-promoted genes", loc="left", fontsize=6.8)
    ax.grid(axis="y", color="#E7E7E7", lw=0.45)


def _plot_panel_support(ax: plt.Axes, repeats: pd.DataFrame) -> None:
    panel_sizes = sorted(repeats["panel_size"].astype(int).unique())
    panels = ["snRNA only", "snRNA + multi-Visium"]
    colors = [PALETTE["source"], PALETTE["integrated_light"]]
    edge_colors = [PALETTE["source_dark"], PALETTE["integrated"]]
    offsets = [-0.16, 0.16]
    rng = np.random.default_rng(23)
    for size_idx, panel_size in enumerate(panel_sizes):
        for panel_idx, panel in enumerate(panels):
            values = repeats.loc[
                (repeats["panel_size"].astype(int) == panel_size) & (repeats["panel"] == panel),
                "value",
            ].to_numpy(dtype=float)
            if values.size == 0:
                continue
            pos = size_idx + offsets[panel_idx]
            violin = ax.violinplot(
                [values],
                positions=[pos],
                widths=0.26,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            body = violin["bodies"][0]
            body.set_facecolor(colors[panel_idx])
            body.set_edgecolor("none")
            body.set_alpha(0.40)
            jitter = rng.normal(0, 0.018, size=len(values))
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=10,
                color=colors[panel_idx],
                edgecolor="black",
                linewidth=0.25,
                zorder=3,
            )
            median = float(np.median(values))
            ax.plot([pos - 0.10, pos + 0.10], [median, median], color=edge_colors[panel_idx], lw=0.75, zorder=4)

    ax.set_xticks(np.arange(len(panel_sizes)), [str(size) for size in panel_sizes])
    ax.set_xlabel("Panel size")
    ax.set_ylabel("Mean MERFISH\nexpression", labelpad=4)
    ax.xaxis.label.set_size(9.0)
    ax.yaxis.label.set_size(9.0)
    ax.tick_params(axis="both", labelsize=8.2)
    finite_values = repeats["value"].to_numpy(dtype=float)
    ymin = max(0.0, float(np.nanmin(finite_values)) - 0.01)
    ymax = float(np.nanmax(finite_values)) + 0.025
    ax.set_ylim(ymin, ymax)
    y_span = ymax - ymin
    stat_base = ymax - 0.11 * y_span
    stat_step = 0.045 * y_span
    for size_idx, panel_size in enumerate(panel_sizes):
        subset = repeats[repeats["panel_size"].astype(int) == panel_size]
        paired = subset.pivot_table(index="seed", columns="panel", values="value", aggfunc="mean")
        if not set(panels).issubset(paired.columns) or paired.dropna().shape[0] < 2:
            continue
        paired = paired.dropna(subset=panels)
        try:
            p_value = wilcoxon(
                paired["snRNA + multi-Visium"],
                paired["snRNA only"],
                alternative="greater",
                zero_method="wilcox",
            ).pvalue
        except ValueError:
            continue
        x0 = size_idx + offsets[0]
        x1 = size_idx + offsets[1]
        y = stat_base + (size_idx % 2) * stat_step
        ax.plot([x0, x0, x1, x1], [y, y + stat_step * 0.25, y + stat_step * 0.25, y], color="#333333", lw=0.45)
        ax.text((x0 + x1) / 2, y + stat_step * 0.36, f"p={p_value:.3f}", ha="center", va="bottom", fontsize=8.0)



def _plot_delta_f1(ax: plt.Axes, delta: pd.DataFrame) -> None:
    order = delta.sort_values("delta_f1", ascending=True)
    y = np.arange(order.shape[0])
    colors = [PALETTE["gain"] if value >= 0 else PALETTE["loss"] for value in order["delta_f1"]]
    ax.axvline(0, color="#777777", lw=0.7)
    ax.hlines(y, 0, order["delta_f1"], color=colors, lw=1.6, alpha=0.85)
    ax.scatter(order["delta_f1"], y, s=24, color=colors, zorder=3)
    ax.set_yticks(y, order["class"])
    ax.set_xlabel("Change in F1\n(snRNA + Visium minus snRNA)")
    ax.set_title("Gains concentrate in stromal and immune cells", loc="left", fontsize=8, weight="bold")
    ax.set_xlim(-0.06, 0.30)
    ax.grid(axis="x", color="#E6E2D8", lw=0.5)


def _plot_marker_heatmap(ax: plt.Axes, expr: pd.DataFrame, marker_genes: list[str]) -> None:
    if expr.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No marker expression available", ha="center", va="center")
        return
    cell_order = ["Hep_1", "Hep_2", "Hep_3", "HSC_1", "HSC_2", "LSEC", "Macrophage_1", "Macrophage_2", "Cholangiocyte"]
    genes = [gene for gene in marker_genes if gene in set(expr["gene_symbol"])]
    mat = (
        expr[expr["gene_symbol"].isin(genes) & expr["cell_type"].isin(cell_order)]
        .pivot(index="gene_symbol", columns="cell_type", values="z_expression")
        .reindex(index=genes, columns=cell_order)
        .fillna(0.0)
    )
    image = ax.imshow(mat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1.8, vmax=1.8)
    ax.set_xticks(np.arange(len(cell_order)), cell_order, rotation=45, ha="right", fontsize=5.4)
    ax.set_yticks(np.arange(len(genes)), genes, fontsize=6.1)
    ax.set_title("Representative Visium-promoted genes in MERFISH", loc="left", fontsize=8, weight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(image, ax=ax, fraction=0.040, pad=0.018)
    cbar.set_label("z-scored mean expression", fontsize=5.8)
    cbar.ax.tick_params(labelsize=5.2, length=2)


def _save_single_panel(
    output_dir: Path,
    stem: str,
    plotter: Any,
    *,
    figsize: tuple[float, float],
    title: str | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    plotter(ax)
    if title:
        fig.suptitle(title, x=0.02, y=0.99, ha="left", va="top", fontsize=9, weight="bold")
        fig.subplots_adjust(top=0.88)
    paths = {
        "pdf": str((output_dir / stem).with_suffix(".pdf")),
        "svg": str((output_dir / stem).with_suffix(".svg")),
        "png": str((output_dir / stem).with_suffix(".png")),
    }
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def build_separate_panels(
    result_dir: Path,
    merfish_file: Path,
    output_dir: Path,
    multi_seed_result_dir: Path,
    performance_metric: str,
) -> dict[str, dict[str, str]]:
    _configure_matplotlib()
    result = _read_result(result_dir)
    repeats = _load_training_seed_panel_size_repeats(multi_seed_result_dir, metric=performance_metric)
    if repeats is None:
        repeats = _compute_cell_type_repeats(
            result,
            merfish_file,
            result_dir / "diagnostics/cell_type_classification_5seed_repeats.tsv",
        )
    support_repeats = _load_panel_support_repeats(multi_seed_result_dir, merfish_file)

    panels = {
        "reference_schematic": _save_single_panel(
            output_dir,
            "01_reference_schematic",
            lambda ax: _plot_reference_schematic(ax, result),
            figsize=(2.75, 1.55),
        ),
        "merfish_performance": _save_single_panel(
            output_dir,
            "02_merfish_performance",
            lambda ax: _plot_performance(ax, repeats),
            figsize=(2.25, 2.25),
        ),
        "merfish_gene_support": _save_single_panel(
            output_dir,
            "04_merfish_gene_support",
            lambda ax: _plot_panel_support(ax, support_repeats),
            figsize=(2.25, 2.25),
        ),
    }
    return panels


def build_figure(result_dir: Path, merfish_file: Path, output_prefix: Path) -> dict[str, str]:
    _configure_matplotlib()
    result = _read_result(result_dir)
    summary = pd.read_csv(result_dir / "formal_benchmark_summary.tsv", sep="\t")
    changed = _clean_gene_set(pd.read_csv(result_dir / "diagnostics/changed_gene_merfish_discriminativeness.tsv", sep="\t"))
    delta = pd.read_csv(result_dir / "diagnostics/cell_type_delta_report.tsv", sep="\t")
    marker_genes = ["COL1A1", "ACTA2", "CD5L", "IL4R", "AHNAK", "TFPI", "SDS", "ASGR2"]
    expr = _load_marker_expression(merfish_file, marker_genes)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 6.8), facecolor="white")
    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=[0.95, 1.25, 1.55],
        width_ratios=[1, 1, 1, 1, 1, 1],
        hspace=0.72,
        wspace=0.72,
    )
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])
    ax_f = fig.add_subplot(gs[2, 0:6])

    _plot_reference_schematic(ax_a, result)
    _plot_performance(ax_b, summary)
    _plot_panel_shift(ax_c, result)
    _plot_merfish_support(ax_d, changed)
    _plot_delta_f1(ax_e, delta)
    _plot_marker_heatmap(ax_f, expr, marker_genes)

    panel_labels = [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"), (ax_e, "e"), (ax_f, "f")]
    for ax, label in panel_labels:
        ax.text(-0.11, 1.08, label, transform=ax.transAxes, fontsize=10, weight="bold", va="top", ha="left")

    fig.suptitle(
        "Multi-Visium retrieval improves liver MERFISH panel design through spatial-context genes",
        x=0.02,
        y=0.995,
        ha="left",
        va="top",
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.02,
        0.962,
        "Paired snRNA source and five in situ Visium references were fused to select a 64-gene panel, then evaluated on held-out MERFISH cells.",
        ha="left",
        va="top",
        fontsize=6.5,
        color="#46545F",
    )
    fig.subplots_adjust(top=0.91, bottom=0.08, left=0.08, right=0.985)

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
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--merfish-file", default=str(DEFAULT_MERFISH))
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--panel-dir", default=str(DEFAULT_PANEL_DIR))
    parser.add_argument("--multi-seed-result-dir", default=str(DEFAULT_MULTI_SEED_RESULT_DIR))
    parser.add_argument("--performance-metric", default="cell_type_macro_f1")
    parser.add_argument("--separate-panels", action="store_true")
    parser.add_argument("--composite", action="store_true")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    merfish_file = Path(args.merfish_file)
    payload: dict[str, Any] = {}
    if args.separate_panels or not args.composite:
        payload["separate_panels"] = build_separate_panels(
            result_dir,
            merfish_file,
            Path(args.panel_dir),
            Path(args.multi_seed_result_dir),
            str(args.performance_metric),
        )
    if args.composite:
        payload["composite"] = build_figure(result_dir, merfish_file, Path(args.output_prefix))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
