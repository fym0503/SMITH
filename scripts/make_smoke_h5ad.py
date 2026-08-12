#!/usr/bin/env python3
"""Create a tiny synthetic AnnData file for SMITH smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def build_smoke_adata(n_cells: int, n_genes: int, seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n_celltypes = 3
    n_regions = 2
    n_pathology = 2

    celltype_ids = np.arange(n_cells) % n_celltypes
    region_ids = (np.arange(n_cells) // n_celltypes) % n_regions
    pathology_ids = (np.arange(n_cells) // (n_celltypes * n_regions)) % n_pathology

    base = rng.gamma(shape=1.5, scale=1.0, size=(n_cells, n_genes)).astype(np.float32)
    signal = np.zeros_like(base)
    for ct in range(n_celltypes):
        signal[celltype_ids == ct, ct::n_celltypes] += 1.5
    for region in range(n_regions):
        signal[region_ids == region, (region + 3)::4] += 0.8
    for pathology in range(n_pathology):
        signal[pathology_ids == pathology, (pathology + 5)::5] += 0.7

    x = np.log1p(base + signal).astype(np.float32)
    var_names = [f"GENE{i:03d}" for i in range(n_genes)]
    obs_names = [f"cell{i:03d}" for i in range(n_cells)]
    celltypes = [f"type_{idx}" for idx in celltype_ids]

    obs = pd.DataFrame(
        {
            "celltype": celltypes,
            "cell_type": celltypes,
            "region": [f"region_{idx}" for idx in region_ids],
            "pathology": [f"state_{idx}" for idx in pathology_ids],
        },
        index=obs_names,
    )
    var = pd.DataFrame(index=var_names)
    coords = np.column_stack(
        [
            celltype_ids + rng.normal(0, 0.05, size=n_cells),
            region_ids + rng.normal(0, 0.05, size=n_cells),
        ]
    ).astype(np.float32)

    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers["raw"] = sparse.csr_matrix(np.expm1(x).astype(np.float32))
    adata.obsm["spatial"] = coords
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic SMITH smoke-test H5AD file.")
    parser.add_argument("--output", default="data/smoke/smoke_panel.h5ad")
    parser.add_argument("--n-cells", type=int, default=72)
    parser.add_argument("--n-genes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    adata = build_smoke_adata(args.n_cells, args.n_genes, args.seed)
    adata.write_h5ad(output)
    print(f"Wrote {output} with shape {adata.n_obs} x {adata.n_vars}")


if __name__ == "__main__":
    main()
