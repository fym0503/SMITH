"""Manuscript-aligned analysis helpers for RIBOMap Figure 4c-h."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else float("nan")


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    standard_deviation = np.nanstd(values)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.full(values.shape, np.nan)
    return (values - np.nanmean(values)) / standard_deviation


def ribomap_bias(ribomap_mean: np.ndarray, starmap_mean: np.ndarray) -> np.ndarray:
    """Compute z(log1p RIBOMap mean) - z(log1p STARmap mean)."""
    return zscore(np.log1p(ribomap_mean)) - zscore(np.log1p(starmap_mean))


def bias_table_from_objects(ribomap, starmap) -> pd.DataFrame:
    """Compute RIBOMap bias from currently loaded AnnData objects."""
    from reproducibility.workflows.common import gene_symbols
    from scipy import sparse

    def means(adata):
        matrix = adata.X
        values = np.asarray(matrix.mean(axis=0)).ravel() if sparse.issparse(matrix) else np.asarray(matrix).mean(axis=0)
        return {gene: float(value) for gene, value in zip(gene_symbols(adata), values) if gene}

    ribo, star = means(ribomap), means(starmap)
    shared = sorted(set(ribo) & set(star))
    return pd.DataFrame({
        "gene_symbol": shared,
        "ribomap_bias": ribomap_bias(
            np.asarray([ribo[gene] for gene in shared], dtype=float),
            np.asarray([star[gene] for gene in shared], dtype=float),
        ),
    })


def jaccard_from_panel_records(panel_records: list[dict]) -> pd.DataFrame:
    """Calculate same/cross-modality overlap from in-memory panel gene lists."""
    import itertools

    rows = []
    for size in sorted({int(row["panel_size"]) for row in panel_records}):
        current = [row for row in panel_records if int(row["panel_size"]) == size]
        for left, right in itertools.combinations(current, 2):
            left_genes, right_genes = set(left["panel_genes"]), set(right["panel_genes"])
            rows.append({
                "panel_size": size,
                "modality_group": "Same modality" if left["source"] == right["source"] else "Cross modality",
                "source_a": left["source"], "source_b": right["source"],
                "method_a": left.get("method", "SMITH"), "method_b": right.get("method", "SMITH"),
                "jaccard": jaccard_similarity(left_genes, right_genes),
                "overlap": len(left_genes & right_genes), "union": len(left_genes | right_genes),
            })
    return pd.DataFrame(rows)


def bh_adjust(pvalues: list[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.clip(ranked, 0, 1)
    return result


def bias_group(gene: str, ribomap_panel: set[str], starmap_panel: set[str]) -> str:
    if gene in ribomap_panel - starmap_panel:
        return "RIBOMap-only"
    if gene in ribomap_panel & starmap_panel:
        return "Shared"
    if gene in starmap_panel - ribomap_panel:
        return "STARmap-only"
    return "Background"


def performance_paired_tests(values: pd.DataFrame, comparator: str = "PERSIST-class") -> pd.DataFrame:
    rows = []
    groups = ["source", "label", "panel_size"]
    for keys, frame in values.groupby(groups, observed=True):
        pivot = frame.pivot_table(index="evaluation_seed", columns="method", values="accuracy", aggfunc="mean")
        if "SMITH" not in pivot or comparator not in pivot:
            paired = pd.DataFrame()
            pvalue = np.nan
        else:
            paired = pivot[["SMITH", comparator]].dropna()
            if len(paired) < 2:
                pvalue = np.nan
            else:
                try:
                    pvalue = float(wilcoxon(paired["SMITH"], paired[comparator], alternative="greater").pvalue)
                except ValueError:
                    pvalue = 1.0
        rows.append(dict(zip(groups, keys, strict=True), method_a="SMITH", method_b=comparator,
                         n_pairs=len(paired), pvalue=pvalue, alternative="greater",
                         pairing="evaluation_seed"))
    return pd.DataFrame(rows)


def bias_pairwise_tests(values: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("RIBOMap-only", "STARmap-only"),
        ("RIBOMap-only", "Background"),
        ("STARmap-only", "Background"),
        ("Shared", "Background"),
    ]
    rows = []
    for panel_size, frame in values.groupby("panel_size", observed=True):
        for group_a, group_b in comparisons:
            left = frame.loc[frame["group"] == group_a, "ribomap_bias"].dropna().to_numpy(float)
            right = frame.loc[frame["group"] == group_b, "ribomap_bias"].dropna().to_numpy(float)
            if len(left) < 2 or len(right) < 2:
                statistic = pvalue = cliff = np.nan
            else:
                statistic, pvalue = mannwhitneyu(left, right, alternative="two-sided")
                right_sorted = np.sort(right)
                greater = np.searchsorted(right_sorted, left, side="left").sum()
                less = (len(right_sorted) - np.searchsorted(right_sorted, left, side="right")).sum()
                cliff = (greater - less) / (len(left) * len(right))
            rows.append({"panel_size": panel_size, "group_a": group_a, "group_b": group_b,
                         "n_a": len(left), "n_b": len(right), "mannwhitney_u": statistic,
                         "pvalue": pvalue, "cliffs_delta": cliff, "alternative": "two-sided"})
    result = pd.DataFrame(rows)
    result["qvalue_bh"] = np.nan
    valid = result["pvalue"].notna()
    if valid.any():
        result.loc[valid, "qvalue_bh"] = bh_adjust(result.loc[valid, "pvalue"].tolist())
    return result


def write_statistical_analysis(metrics_path: str | Path, bias_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    performance = performance_paired_tests(pd.read_csv(metrics_path, sep="\t"))
    performance_path = output_dir / "figure4_c_f_paired_tests.tsv"
    performance.to_csv(performance_path, sep="\t", index=False)
    bias = pd.read_csv(bias_path, sep="\t")
    if "method" in bias:
        bias = bias[bias["method"] == "SMITH"]
    bias_tests = bias_pairwise_tests(bias)
    bias_path_out = output_dir / "figure4_h_pairwise_tests.tsv"
    bias_tests.to_csv(bias_path_out, sep="\t", index=False)
    return {"performance_tests": str(performance_path), "bias_tests": str(bias_path_out)}
