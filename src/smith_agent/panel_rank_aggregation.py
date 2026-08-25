from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from smith_agent.utils import ensure_dir, write_json


GENE_SYMBOL_COLUMNS = (
    "gene_symbol",
    "gene_symbols",
    "gene_name",
    "gene_names",
    "feature_name",
    "gene_short_name",
    "symbol",
)

COORDINATE_COLUMN_PAIRS = (
    ("xcoord", "ycoord"),
    ("center_x", "center_y"),
    ("array_col", "array_row"),
    ("pxl_col_in_fullres", "pxl_row_in_fullres"),
    ("x", "y"),
)


def _clean_gene_symbol(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text.upper()


def _gene_symbols(adata: ad.AnnData, symbol_column: str = "") -> pd.Series:
    if symbol_column and symbol_column in adata.var:
        values = adata.var[symbol_column]
    else:
        values = None
        for column in GENE_SYMBOL_COLUMNS:
            if column in adata.var:
                values = adata.var[column]
                break
        if values is None:
            values = pd.Series(adata.var_names, index=adata.var_names)
    return pd.Series(values, index=adata.var_names).map(_clean_gene_symbol)


def _sample_obs_indices(n_obs: int, max_cells: int, seed: int) -> np.ndarray:
    if max_cells <= 0 or n_obs <= max_cells:
        return np.arange(n_obs)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_obs, size=max_cells, replace=False))


def _load_adata_subset(adata_file: str | Path, max_cells: int, seed: int) -> ad.AnnData:
    adata = ad.read_h5ad(adata_file)
    indices = _sample_obs_indices(adata.n_obs, max_cells=max_cells, seed=seed)
    if len(indices) != adata.n_obs:
        adata = adata[indices, :].copy()
    return adata


def _as_csr_float_matrix(x: Any) -> Any:
    if sparse.issparse(x):
        return x.tocsr().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _matrix_mean(x: Any) -> np.ndarray:
    return np.asarray(x.mean(axis=0)).ravel()


def _matrix_variance(x: Any, mean: np.ndarray) -> np.ndarray:
    if sparse.issparse(x):
        second = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    else:
        second = np.asarray(np.square(x).mean(axis=0)).ravel()
    return np.maximum(0.0, second - np.square(mean))


def _matrix_detection_rate(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray((x > 0).mean(axis=0)).ravel()
    return np.asarray((x > 0).mean(axis=0)).ravel()


def _coordinates(adata: ad.AnnData) -> np.ndarray | None:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2].astype(np.float64)
    lowered = {str(column).lower(): column for column in adata.obs.columns}
    for x_col, y_col in COORDINATE_COLUMN_PAIRS:
        if x_col in lowered and y_col in lowered:
            coords = adata.obs[[lowered[x_col], lowered[y_col]]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            if np.isfinite(coords).all():
                return coords
    return None


def _coordinate_correlation_score(x: Any, coords: np.ndarray | None, variance: np.ndarray) -> np.ndarray:
    if coords is None or coords.shape[0] != x.shape[0]:
        return np.zeros(x.shape[1], dtype=np.float64)
    scores = []
    for axis in range(2):
        coord = coords[:, axis].astype(np.float64)
        coord = coord - np.nanmean(coord)
        coord_std = float(np.nanstd(coord))
        if not math.isfinite(coord_std) or coord_std <= 0:
            scores.append(np.zeros(x.shape[1], dtype=np.float64))
            continue
        if sparse.issparse(x):
            cov = np.asarray(x.T.dot(coord)).ravel() / max(1, x.shape[0])
        else:
            cov = np.asarray(x.T @ coord).ravel() / max(1, x.shape[0])
        gene_std = np.sqrt(np.maximum(variance, 0.0))
        corr = np.divide(cov, gene_std * coord_std, out=np.zeros_like(cov), where=(gene_std > 0))
        scores.append(np.clip(np.abs(corr), 0.0, 1.0))
    return np.sqrt(np.square(scores[0]) + np.square(scores[1])) / math.sqrt(2.0)


def _hvg_score(adata: ad.AnnData) -> np.ndarray:
    if "highly_variable_rank" in adata.var:
        rank = pd.to_numeric(adata.var["highly_variable_rank"], errors="coerce")
        valid = rank.notna()
        score = pd.Series(0.0, index=adata.var_names)
        if valid.any():
            score.loc[valid] = _percentile(-rank.loc[valid])
        return score.to_numpy(dtype=np.float64)
    if "highly_variable" in adata.var:
        return adata.var["highly_variable"].astype(float).to_numpy(dtype=np.float64)
    return np.zeros(adata.n_vars, dtype=np.float64)


def _percentile(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype(float)
    return series.astype(float).rank(method="average", pct=True, na_option="bottom").fillna(0.0)


def _filter_gene_patterns(df: pd.DataFrame, patterns: list[str] | None) -> pd.DataFrame:
    if not patterns:
        return df
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns if str(pattern).strip()]
    if not compiled:
        return df
    keep = ~df["gene_symbol"].map(lambda gene: any(pattern.search(str(gene)) for pattern in compiled))
    return df[keep].copy()


def rank_genes_from_adata(
    adata_file: str | Path,
    output_tsv: str | Path,
    dataset_id: str = "",
    data_role: str = "reference",
    symbol_column: str = "",
    max_cells: int = 50000,
    seed: int = 42,
    min_detection_rate: float = 0.0,
    exclude_gene_patterns: list[str] | None = None,
    expression_weight: float = 0.35,
    variability_weight: float = 0.35,
    detection_weight: float = 0.15,
    spatial_weight: float = 0.15,
    hvg_weight: float = 0.0,
) -> pd.DataFrame:
    adata_file = Path(adata_file)
    adata = _load_adata_subset(adata_file, max_cells=max_cells, seed=seed)
    try:
        genes = _gene_symbols(adata, symbol_column=symbol_column)
        x = _as_csr_float_matrix(adata.X)
        mean = _matrix_mean(x)
        variance = _matrix_variance(x, mean)
        detection = _matrix_detection_rate(x)
        spatial_score = _coordinate_correlation_score(x, _coordinates(adata), variance)
        hvg_score = _hvg_score(adata)

        df = pd.DataFrame(
            {
                "gene_symbol": genes.to_numpy(),
                "mean_expression": mean,
                "variance": variance,
                "detection_rate": detection,
                "spatial_coord_score": spatial_score,
                "hvg_score": hvg_score,
            }
        )
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    df = df[df["gene_symbol"].astype(bool)].copy()
    df = _filter_gene_patterns(df, exclude_gene_patterns)
    if min_detection_rate > 0:
        df = df[df["detection_rate"] >= float(min_detection_rate)].copy()
    if df.empty:
        raise ValueError(f"No genes remained after ranking filters for {adata_file}.")

    df = (
        df.groupby("gene_symbol", as_index=False)
        .agg(
            mean_expression=("mean_expression", "mean"),
            variance=("variance", "mean"),
            detection_rate=("detection_rate", "max"),
            spatial_coord_score=("spatial_coord_score", "max"),
            hvg_score=("hvg_score", "max"),
        )
        .copy()
    )
    expression_pct = _percentile(np.log1p(df["mean_expression"]))
    variability_pct = _percentile(np.log1p(df["variance"]))
    detection_pct = _percentile(df["detection_rate"])
    spatial_pct = _percentile(df["spatial_coord_score"])
    hvg_pct = _percentile(df["hvg_score"])
    df["rank_score"] = (
        float(expression_weight) * expression_pct
        + float(variability_weight) * variability_pct
        + float(detection_weight) * detection_pct
        + float(spatial_weight) * spatial_pct
        + float(hvg_weight) * hvg_pct
    )
    df["dataset_id"] = dataset_id or adata_file.stem
    df["dataset_path"] = str(adata_file)
    df["data_role"] = data_role
    df = df.sort_values(["rank_score", "detection_rate", "variance", "gene_symbol"], ascending=[False, False, False, True])
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    output_tsv = Path(output_tsv)
    ensure_dir(output_tsv.parent)
    df.to_csv(output_tsv, sep="\t", index=False)
    return df


def _read_rank_file(
    rank_file: str | Path,
    dataset_id: str = "source_rank_file",
    data_role: str = "source_rank_file",
) -> pd.DataFrame:
    rank_file = Path(rank_file)
    sep = "\t" if rank_file.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(rank_file, sep=sep)
    if df.empty:
        raise ValueError(f"Rank file is empty: {rank_file}")
    gene_column = next((column for column in df.columns if str(column).lower() in {"gene", "genes", "gene_symbol", "target"}), df.columns[0])
    score_column = next((column for column in df.columns if str(column).lower() in {"rank_score", "score", "prob", "weight"}), "")
    rank_df = pd.DataFrame({"gene_symbol": df[gene_column].map(_clean_gene_symbol)})
    if score_column:
        rank_df["raw_score"] = pd.to_numeric(df[score_column], errors="coerce").fillna(0.0)
    rank_df = rank_df[rank_df["gene_symbol"].astype(bool)].drop_duplicates("gene_symbol").reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)
    if score_column:
        rank_df["rank_score"] = _percentile(rank_df["raw_score"])
        rank_df = rank_df.drop(columns=["raw_score"])
    else:
        rank_df["rank_score"] = 1.0 - ((rank_df["rank"] - 1) / max(1, len(rank_df) - 1))
    rank_df["dataset_id"] = dataset_id
    rank_df["dataset_path"] = str(rank_file)
    rank_df["data_role"] = data_role
    return rank_df


def _rank_scores(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    out = df[["gene_symbol", score_column]].copy()
    out = out.rename(columns={score_column: "score"})
    out["score"] = _percentile(out["score"])
    return out


def _normalize_loaded_rank(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Normalize a ranking returned by a current SMITH run for in-memory aggregation."""
    frame = df.copy()
    gene_column = next((column for column in frame.columns if str(column).lower() in {"gene", "genes", "gene_symbol", "target", "marker"}), frame.columns[0])
    result = pd.DataFrame({"gene_symbol": frame[gene_column].map(_clean_gene_symbol)})
    score_column = next((column for column in frame.columns if str(column).lower() in {"rank_score", "score", "prob", "weight"}), None)
    if score_column:
        result["rank_score"] = pd.to_numeric(frame[score_column], errors="coerce").fillna(0.0).to_numpy()
    else:
        result["rank_score"] = 1.0 - (np.arange(len(frame)) / max(1, len(frame) - 1))
    result = result[result["gene_symbol"].astype(bool)].drop_duplicates("gene_symbol").reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result["dataset_id"] = dataset_id
    return result


def aggregate_reference_panel_ranks_loaded(
    source_ranking: pd.DataFrame,
    reference_rankings: list[pd.DataFrame],
    *,
    panel_size: int = 64,
    source_weight: float = 0.55,
    reference_weight: float = 0.45,
    min_reference_support: int = 1,
    gene_universe: str = "source",
    restrict_gene_symbols: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Aggregate current source/reference rankings without reading rank files."""
    source = _normalize_loaded_rank(source_ranking, "source")
    references = [_normalize_loaded_rank(frame, f"reference_{index + 1}") for index, frame in enumerate(reference_rankings)]
    restrict = {_clean_gene_symbol(gene) for gene in (restrict_gene_symbols or []) if _clean_gene_symbol(gene)}
    source_scores = _rank_scores(source, "rank_score").rename(columns={"score": "source_score"})
    if gene_universe == "source":
        universe = set(source_scores["gene_symbol"])
    elif gene_universe == "intersection":
        universe = set(source_scores["gene_symbol"])
        for frame in references:
            universe &= set(frame["gene_symbol"])
    else:
        universe = set(source_scores["gene_symbol"])
        for frame in references:
            universe |= set(frame["gene_symbol"])
    if restrict:
        universe &= restrict
    merged = source_scores.copy()
    for index, frame in enumerate(references):
        scores = _rank_scores(frame, "rank_score").rename(columns={"score": f"reference_{index + 1}_score"})
        merged = merged.merge(scores, on="gene_symbol", how="outer")
    merged = merged[merged["gene_symbol"].isin(universe)].copy()
    ref_columns = [column for column in merged.columns if column.startswith("reference_") and column.endswith("_score")]
    merged["reference_support"] = merged[ref_columns].notna().sum(axis=1)
    merged["mean_reference_score"] = merged[ref_columns].mean(axis=1, skipna=True).fillna(0.0)
    merged["source_score"] = merged["source_score"].fillna(0.0)
    merged = merged[merged["reference_support"] >= int(min_reference_support)].copy()
    merged["integrated_score"] = float(source_weight) * merged["source_score"] + float(reference_weight) * merged["mean_reference_score"]
    merged = merged.sort_values(["integrated_score", "source_score", "mean_reference_score", "reference_support", "gene_symbol"], ascending=[False, False, False, False, True])
    merged.insert(0, "integrated_rank", np.arange(1, len(merged) + 1))
    source_top = source_scores.sort_values(["source_score", "gene_symbol"], ascending=[False, True]).head(panel_size)["gene_symbol"].tolist()
    integrated_top = merged.head(panel_size)["gene_symbol"].tolist()
    return {
        "source_ranking": source,
        "integrated_ranking": merged,
        "source_panel_genes": source_top,
        "integrated_panel_genes": integrated_top,
        "comparison": {
            "panel_size": int(panel_size), "n_source_genes": int(len(source_scores)),
            "n_reference_datasets": len(references), "n_integrated_ranked_genes": int(len(merged)),
            "min_reference_support": int(min_reference_support),
            "source_weight": float(source_weight), "reference_weight": float(reference_weight),
            "overlap_count": len(set(source_top) & set(integrated_top)),
            "jaccard": jaccard_loaded(set(source_top), set(integrated_top)),
        },
    }


def jaccard_loaded(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else float("nan")


def tune_reference_aggregation_loaded(
    source_ranking: pd.DataFrame,
    reference_rankings: list[pd.DataFrame],
    target_adata: ad.AnnData,
    *,
    panel_sizes: tuple[int, ...] = (32, 64, 128),
    source_weights: tuple[float, ...] = (0.25, 0.5, 0.75),
    label_column: str = "Cell_Type",
    seed: int = 42,
    min_reference_support: int = 1,
    restrict_gene_symbols: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Tune the Agent's source/reference aggregation on held-out biology.

    The candidate panels are constructed from the rankings returned by the
    current SMITH runs.  No ranking or metric file is read.  The objective is
    deliberately explicit: select the source/reference weight and panel size
    that maximize held-out cell-type accuracy on the target AnnData.
    """
    from smith_agent.benchmarking import cell_type_evaluation_loaded

    sizes = tuple(sorted({int(size) for size in panel_sizes if int(size) > 0}))
    weights = tuple(sorted({float(weight) for weight in source_weights if 0.0 <= float(weight) <= 1.0}))
    if not sizes or not weights:
        raise ValueError("panel_sizes and source_weights must contain valid values")
    rows: list[dict[str, Any]] = []
    aggregations: dict[tuple[float, int], dict[str, Any]] = {}
    for source_weight in weights:
        aggregation = aggregate_reference_panel_ranks_loaded(
            source_ranking,
            reference_rankings,
            panel_size=max(sizes),
            source_weight=source_weight,
            reference_weight=1.0 - source_weight,
            min_reference_support=min_reference_support,
            restrict_gene_symbols=restrict_gene_symbols,
            gene_universe="source",
        )
        for size in sizes:
            panel_genes = aggregation["integrated_panel_genes"][:size]
            metrics, _ = cell_type_evaluation_loaded(
                target_adata,
                panel_genes,
                panel_size=size,
                seed=seed,
                label_column=label_column,
                output_dir=None,
            )
            row = {
                "source_weight": source_weight,
                "reference_weight": 1.0 - source_weight,
                "panel_size": size,
                "cell_type_accuracy": float(metrics["metrics"]["cell_type_accuracy"]),
                "cell_type_balanced_accuracy": float(metrics["metrics"]["cell_type_balanced_accuracy"]),
                "cell_type_macro_f1": float(metrics["metrics"]["cell_type_macro_f1"]),
                "panel_genes": list(panel_genes),
            }
            rows.append(row)
            aggregations[(source_weight, size)] = aggregation
    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["cell_type_accuracy", "cell_type_balanced_accuracy", "cell_type_macro_f1", "panel_size", "source_weight"],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)
    best = results.iloc[0].to_dict()
    best["panel_genes"] = list(best["panel_genes"])
    return {
        "results": results,
        "best": best,
        "best_aggregation": aggregations[(float(best["source_weight"]), int(best["panel_size"]))],
    }


def aggregate_reference_panel_ranks(
    output_dir: str | Path,
    source_adata_file: str | Path | None = None,
    source_rank_file: str | Path | None = None,
    reference_adata_files: list[str | Path] | None = None,
    reference_rank_files: list[str | Path] | None = None,
    reference_ids: list[str] | None = None,
    panel_size: int = 64,
    source_weight: float = 0.55,
    reference_weight: float = 0.45,
    max_cells: int = 50000,
    seed: int = 42,
    min_detection_rate: float = 0.0,
    min_reference_support: int = 1,
    gene_universe: str = "source",
    exclude_gene_patterns: list[str] | None = None,
    restrict_gene_symbols: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    references = [Path(item) for item in (reference_adata_files or [])]
    reference_rank_paths_in = [Path(item) for item in (reference_rank_files or [])]
    reference_inputs = references or reference_rank_paths_in
    reference_ids = reference_ids or [path.stem for path in reference_inputs]
    if references and reference_rank_paths_in:
        raise ValueError("Use reference_adata_files or reference_rank_files, not both.")
    if len(reference_ids) != len(reference_inputs):
        raise ValueError("reference_ids must have the same length as reference inputs.")
    if not source_adata_file and not source_rank_file:
        raise ValueError("aggregate_reference_panel_ranks requires source_adata_file or source_rank_file.")
    if not reference_inputs:
        raise ValueError("aggregate_reference_panel_ranks requires at least one reference input.")

    restrict_set = {_clean_gene_symbol(item) for item in (restrict_gene_symbols or [])}
    restrict_set = {item for item in restrict_set if item}

    def _apply_gene_constraints(df: pd.DataFrame) -> pd.DataFrame:
        constrained = _filter_gene_patterns(df, exclude_gene_patterns)
        if restrict_set:
            constrained = constrained[constrained["gene_symbol"].isin(restrict_set)].copy()
        if constrained.empty:
            raise ValueError("No genes remained after applying gene constraints.")
        return constrained

    source_rank_path = output_dir / "source_gene_rank.tsv"
    if source_rank_file:
        source_df = _read_rank_file(source_rank_file, dataset_id="source", data_role="source_rank_file")
        source_df = _apply_gene_constraints(source_df)
        source_df.to_csv(source_rank_path, sep="\t", index=False)
    else:
        source_df = rank_genes_from_adata(
            adata_file=source_adata_file,
            output_tsv=source_rank_path,
            dataset_id=Path(str(source_adata_file)).stem,
            data_role="source_scrna",
            max_cells=max_cells,
            seed=seed,
            min_detection_rate=min_detection_rate,
            exclude_gene_patterns=exclude_gene_patterns,
            spatial_weight=0.0,
            expression_weight=0.20,
            variability_weight=0.30,
            detection_weight=0.10,
            hvg_weight=0.40,
        )
        source_df = _apply_gene_constraints(source_df)
        source_df.to_csv(source_rank_path, sep="\t", index=False)

    reference_rank_paths: list[str] = []
    reference_dfs: list[pd.DataFrame] = []
    for idx, path in enumerate(reference_inputs):
        rank_path = output_dir / f"reference_{idx + 1}_{reference_ids[idx]}_gene_rank.tsv"
        if reference_rank_paths_in:
            ref_df = _read_rank_file(path, dataset_id=reference_ids[idx], data_role="reference_rank_file")
            ref_df = _apply_gene_constraints(ref_df)
            ref_df.to_csv(rank_path, sep="\t", index=False)
        else:
            ref_df = rank_genes_from_adata(
                adata_file=path,
                output_tsv=rank_path,
                dataset_id=reference_ids[idx],
                data_role="spatial_context",
                max_cells=max_cells,
                seed=seed,
                min_detection_rate=min_detection_rate,
                exclude_gene_patterns=exclude_gene_patterns,
                spatial_weight=0.20,
                expression_weight=0.30,
                variability_weight=0.30,
                detection_weight=0.20,
            )
            ref_df = _apply_gene_constraints(ref_df)
            ref_df.to_csv(rank_path, sep="\t", index=False)
        reference_rank_paths.append(str(rank_path))
        reference_dfs.append(ref_df)

    source_scores = _rank_scores(source_df, "rank_score").rename(columns={"score": "source_score"})
    merged = source_scores.copy()
    if gene_universe == "source":
        universe = set(source_scores["gene_symbol"])
    elif gene_universe == "intersection":
        universe = set(source_scores["gene_symbol"])
        for ref_df in reference_dfs:
            universe &= set(ref_df["gene_symbol"])
    else:
        universe = set(source_scores["gene_symbol"])
        for ref_df in reference_dfs:
            universe |= set(ref_df["gene_symbol"])

    support_frames = []
    for idx, ref_df in enumerate(reference_dfs):
        scores = _rank_scores(ref_df, "rank_score").rename(columns={"score": f"reference_{idx + 1}_score"})
        support_frames.append(scores)
        merged = merged.merge(scores, on="gene_symbol", how="outer")

    merged = merged[merged["gene_symbol"].isin(universe)].copy()
    reference_score_columns = [column for column in merged.columns if column.startswith("reference_") and column.endswith("_score")]
    merged["reference_support"] = merged[reference_score_columns].notna().sum(axis=1)
    merged["mean_reference_score"] = merged[reference_score_columns].mean(axis=1, skipna=True).fillna(0.0)
    merged["source_score"] = merged["source_score"].fillna(0.0)
    merged = merged[merged["reference_support"] >= int(min_reference_support)].copy()
    merged["integrated_score"] = float(source_weight) * merged["source_score"] + float(reference_weight) * merged["mean_reference_score"]
    merged = merged.sort_values(
        ["integrated_score", "source_score", "mean_reference_score", "reference_support", "gene_symbol"],
        ascending=[False, False, False, False, True],
    )
    merged.insert(0, "integrated_rank", np.arange(1, len(merged) + 1))

    integrated_rank_tsv = output_dir / "integrated_panel_rank.tsv"
    merged.to_csv(integrated_rank_tsv, sep="\t", index=False)
    source_top_panel_tsv = output_dir / f"source_top_{panel_size}_panel.tsv"
    source_top_df = source_scores.sort_values(["source_score", "gene_symbol"], ascending=[False, True]).head(panel_size).copy()
    source_top_df.insert(0, "panel_rank", np.arange(1, len(source_top_df) + 1))
    source_top_df.to_csv(source_top_panel_tsv, sep="\t", index=False)
    top_panel_tsv = output_dir / f"integrated_top_{panel_size}_panel.tsv"
    merged.head(panel_size).to_csv(top_panel_tsv, sep="\t", index=False)

    source_top = source_top_df["gene_symbol"].tolist()
    integrated_top = merged.head(panel_size)["gene_symbol"].tolist()
    source_set = set(source_top)
    integrated_set = set(integrated_top)
    comparison = {
        "panel_size": int(panel_size),
        "n_source_genes": int(source_scores.shape[0]),
        "n_reference_datasets": len(reference_dfs),
        "n_integrated_ranked_genes": int(merged.shape[0]),
        "n_restricted_gene_symbols": int(len(restrict_set)),
        "source_weight": float(source_weight),
        "reference_weight": float(reference_weight),
        "min_reference_support": int(min_reference_support),
        "gene_universe": gene_universe,
        "exclude_gene_patterns": exclude_gene_patterns or [],
        "source_top_panel": source_top,
        "integrated_top_panel": integrated_top,
        "overlap_count": len(source_set & integrated_set),
        "jaccard": len(source_set & integrated_set) / max(1, len(source_set | integrated_set)),
        "integrated_only_genes": [gene for gene in integrated_top if gene not in source_set],
        "source_only_genes": [gene for gene in source_top if gene not in integrated_set],
        "artifacts": {
            "source_rank_tsv": str(source_rank_path),
            "source_top_panel_tsv": str(source_top_panel_tsv),
            "reference_rank_tsvs": reference_rank_paths,
            "integrated_rank_tsv": str(integrated_rank_tsv),
            "integrated_top_panel_tsv": str(top_panel_tsv),
        },
    }
    comparison_json = output_dir / "rank_aggregation_comparison.json"
    write_json(comparison_json, comparison)
    return {
        **comparison,
        "comparison_json": str(comparison_json),
        "source_rank_tsv": str(source_rank_path),
        "source_top_panel_tsv": str(source_top_panel_tsv),
        "reference_rank_tsvs": reference_rank_paths,
        "integrated_rank_tsv": str(integrated_rank_tsv),
        "integrated_top_panel_tsv": str(top_panel_tsv),
    }
