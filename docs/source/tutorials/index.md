# Biological panel-design tutorials

These executed notebooks demonstrate three workflows using TF activity, brain RIBOMap and liver MERFISH data. The matching `_source.ipynb` files in the repository can be opened and run in Jupyter.

```{toctree}
:maxdepth: 1

notebooks/regulatory_section/02_SMITH_Regulatory_Activity_executed
notebooks/ribomap_section/03_SMITH_RIBOMap_Transfer_executed
notebooks/agent_section/05_SMITH_Agent_Evaluation_executed
```

| Tutorial profile | Analysis | Reference comparison |
|---|---|---|
| `TF_profile` | Five TF lineage splits; 15 cell-type and developmental-time evaluations | 15/15 SMITH results match the recorded reference values |
| `ribomap_profile` | Four configurations; 12 accuracy endpoints, panel overlap and expression bias | Accuracy means differ by at most 0.000154; overlap uses the stored multi-method result table |
| `Agent_profile` | Five spatial references and five seeds; MERFISH panel evaluation | 30/30 accuracy and expression points match the recorded reference values |

The probe-screen panels use the recorded 12,160-gene screen summaries. H5AD inputs are listed with expected paths and SHA-256 checksums in `reproducibility/data_manifest.yaml`. Data archives will be linked there when their public release is ready; use `SMITH_TUTORIAL_DATA` to point notebooks at an existing data directory.
