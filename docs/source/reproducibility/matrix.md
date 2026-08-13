# Reproducibility matrix

| Figure | Public tutorial | Starting input | Newly generated outputs | Boundary |
|---|---|---|---|---|
| Figure 2 | Whole mouse brain | Unavailable | None | Source and data unavailable |
| Figure 3 | Regulatory activity | C. elegans train/test H5AD | SMITH ranking, panel, identity/time metrics | Tutorial runs one real split |
| Figure 4 | RIBOMap transfer | RIBOMap source/target H5AD | Shared-gene H5AD, SMITH/baseline panels, metrics | Manuscript adds directions/baselines/seeds |
| Figure 6 | SMITH-Agent | liver snRNA, MERFISH, spatial references | source/reference rankings, integrated panel, MERFISH metrics | Probe backends are separate |

The public package intentionally omits the controlled in-house disease chapter. Reference output tables under `reproducibility/reference_outputs/` are optional comparisons and are never workflow inputs.
