from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from smith_agent.adapters.external import add_sys_paths


def evaluate_cross_dataset_panel(
    smith_root: str | Path,
    panel_path: str | Path,
    train_adata_file: str | Path,
    test_adata_file: str | Path,
    panel_size: int = 32,
    label: str | None = None,
    obsm_key: str | None = "X_pca",
    time_label: str | None = None,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    add_sys_paths([smith_root])
    from smith.eval import (
        evaluate_split_knn_classification,
        evaluate_split_obsm_regression,
        evaluate_split_time_knn_regression,
        load_panel_genes,
    )

    panel_genes = load_panel_genes(panel_path, panel_size=panel_size)
    train_adata = ad.read_h5ad(train_adata_file)
    test_adata = ad.read_h5ad(test_adata_file)
    train_adata.var_names = train_adata.var_names.str.upper()
    test_adata.var_names = test_adata.var_names.str.upper()

    results: list[dict[str, Any]] = []

    if label and label in train_adata.obs.columns and label in test_adata.obs.columns:
        accuracy, report = evaluate_split_knn_classification(
            train_adata,
            test_adata,
            panel_genes,
            label_name=label,
            n_neighbors=n_neighbors,
        )
        results.append(
            {
                "evaluation": "split_knn_classification",
                "metric": "accuracy",
                "label": label,
                "value": float(accuracy),
                "details": str(sorted(report.keys())[:5]),
            }
        )

    if obsm_key and obsm_key in train_adata.obsm and obsm_key in test_adata.obsm:
        ev_reg, corr_reg = evaluate_split_obsm_regression(train_adata, test_adata, panel_genes, obsm_key)
        results.append(
            {
                "evaluation": "split_obsm_regression",
                "metric": "pearson_correlation",
                "label": obsm_key,
                "value": float(corr_reg),
                "details": "",
            }
        )

    if time_label and time_label in train_adata.obs.columns and time_label in test_adata.obs.columns:
        time_corr = evaluate_split_time_knn_regression(
            train_adata,
            test_adata,
            panel_genes,
            time_label=time_label,
            n_neighbors=n_neighbors,
        )
        results.append(
            {
                "evaluation": "split_time_knn_regression",
                "metric": "pearson_correlation",
                "label": time_label,
                "value": float(time_corr),
                "details": "",
            }
        )

    return pd.DataFrame(results)
