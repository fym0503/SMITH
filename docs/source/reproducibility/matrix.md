# Reproducibility matrix

| Manuscript panel | Public tutorial | Real starting input | Freshly generated analysis | Boundary |
|---|---|---|---|---|
| Figure 2 | Whole mouse brain | Unavailable | None | Source and data unavailable; no mock result |
| Figure 3c-f | Regulatory activity | Five TF and five miRNA lineage-aware train/test H5AD splits | Seven-method panels, held-out identity/time metrics, grouped bars and split-level tests | Hosted notebook shows a one-split SMITH run; h-k need auxiliary annotations |
| Figure 4c-h | RIBOMap transfer | Deep-RIBOmap, STARmap and target RIBOMap H5AD | Shared-gene inputs, seven-method panels, target metrics, Jaccard and RIBOMap bias | Hosted notebook runs SMITH; i and j-n require versioned pathway/alignment inputs |
| Figure 6c-d | SMITH-Agent | Liver snRNA, MERFISH and five spatial references | Per-seed rankings, integrated panels, MERFISH metrics and paired tests | Hosted notebook uses two references/two seeds; e-j require external backends/HPO |

Each figure table consumed by a notebook is written under that run's
`figure_data/` directory. Files under `reproducibility/reference_outputs/` are
optional post-run comparisons only and are never tutorial inputs.
