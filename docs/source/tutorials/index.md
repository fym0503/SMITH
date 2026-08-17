# Biological Panel-Design Tutorials

These tutorials start from real H5AD inputs. Each workflow creates a new gene ranking, panel, evaluation table and `run_manifest.json`; the notebooks analyze only those new files.

Download a case from its versioned Zenodo archive:

```bash
python scripts/download_tutorial_data.py --case 02_regulatory_activity --data-root data/tutorials
```

Read the Docs serves pre-executed notebooks. It does not download large data or train SMITH during documentation builds. Editable source notebooks are stored beside the executed copies.

Old aggregate manuscript tables are retained under `reproducibility/reference_outputs/` only for optional comparison after a run. They are not tutorial inputs or part of the end-to-end workflow.

```{toctree}
:maxdepth: 1

01_wmb
02_regulatory_activity
03_ribomap_transfer
05_agent
```
