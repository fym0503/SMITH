"""Manuscript-aligned analysis helpers for C. elegans Figure 3c-f."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def lineage_overlap(train_file: str | Path, test_file: str | Path, column: str = "cell_name") -> int:
    """Return the number of shared lineage/cell identifiers across a split."""
    import anndata as ad

    train = ad.read_h5ad(train_file, backed="r")
    test = ad.read_h5ad(test_file, backed="r")
    try:
        if column not in train.obs or column not in test.obs:
            raise KeyError(f"Split identifier {column!r} is absent from one input")
        return len(set(train.obs[column].astype(str)) & set(test.obs[column].astype(str)))
    finally:
        train.file.close()
        test.file.close()


def paired_wilcoxon(
    values: pd.DataFrame,
    metric: str,
    comparator: str = "PERSIST-class",
    pair_columns: tuple[str, ...] = ("dataset", "panel_size"),
) -> pd.DataFrame:
    """Run the paper's one-sided paired split test, preserving the pairing keys."""
    rows: list[dict[str, object]] = []
    for keys, frame in values.groupby(list(pair_columns), dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        pivot = frame.pivot_table(index="split", columns="method", values=metric, aggfunc="mean")
        if "SMITH" not in pivot or comparator not in pivot:
            pvalue = np.nan
            n_pairs = 0
        else:
            paired = pivot[["SMITH", comparator]].dropna()
            n_pairs = len(paired)
            if n_pairs < 2:
                pvalue = np.nan
            else:
                try:
                    pvalue = float(wilcoxon(paired["SMITH"], paired[comparator], alternative="greater").pvalue)
                except ValueError:
                    pvalue = 1.0
        row = dict(zip(pair_columns, keys, strict=True))
        row.update({"metric": metric, "method_a": "SMITH", "method_b": comparator,
                    "n_pairs": n_pairs, "pvalue": pvalue,
                    "alternative": "greater", "pairing": "split"})
        rows.append(row)
    return pd.DataFrame(rows)


def statistical_analysis(values: pd.DataFrame, comparator: str = "PERSIST-class") -> pd.DataFrame:
    """Compute manuscript paired tests directly from current split-level results."""
    rows = [
        paired_wilcoxon(values, metric, comparator=comparator)
        for metric in ("cell_type_accuracy", "developmental_time_pearson")
    ]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_statistical_analysis(values_path: str | Path, output_dir: str | Path) -> Path:
    """Write split-level paired-test metadata; missing baselines are explicit."""
    values = pd.read_csv(values_path, sep="\t")
    result = statistical_analysis(values)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "figure3_c_f_paired_tests.tsv"
    result.to_csv(path, sep="\t", index=False)
    (output_dir / "figure3_c_f_statistics.json").write_text(
        json.dumps({"tests": str(path), "pairing": "split", "alternative": "greater",
                    "status": "computed" if not result.empty else "insufficient_methods"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
