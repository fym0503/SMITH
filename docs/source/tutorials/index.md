# Biological Panel-Design Tutorials

Each chapter below is a runnable notebook. It starts from real H5AD inputs,
trains SMITH, creates a new gene ranking and panel, evaluates the biological
endpoint, and renders the manuscript-matched figures on the same page.

Download a case from its versioned Zenodo archive:

```bash
python scripts/download_tutorial_data.py --case 02_regulatory_activity --data-root data/tutorials
```

Read the Docs serves pre-executed notebooks with their figure outputs. It does
not download large data or train SMITH during documentation builds. The source
notebook is linked from each page for a fresh local run.

Old aggregate manuscript tables are retained under `reproducibility/reference_outputs/` only for optional comparison after a run. They are not tutorial inputs or part of the end-to-end workflow.

```{toctree}
:maxdepth: 1

01_wmb
notebooks/regulatory_section/02_SMITH_Regulatory_Activity_executed
notebooks/ribomap_section/03_SMITH_RIBOMap_Transfer_executed
notebooks/agent_section/05_SMITH_Agent_Evaluation_executed
```
