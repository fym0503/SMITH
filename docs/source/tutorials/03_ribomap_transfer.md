# RIBOMap transfer

This workflow loads real deep-brain and mouse-brain RIBOMap objects, recomputes their shared-gene inputs, trains SMITH, selects a variance baseline, and evaluates both newly generated panels on target data.

```bash
python scripts/download_tutorial_data.py --case 03_ribomap_transfer --data-root data/tutorials
python reproducibility/workflows/ribomap_transfer/run_tutorial.py \
  --data-root data/tutorials \
  --output-dir outputs/tutorials/ribomap \
  --panel-size 64 --epochs 30 --device cpu
```

Prepared H5AD files are written under `prepared_data/`; ranking, both panels, evaluation tables and `run_manifest.json` remain in the selected output directory.

```{toctree}
:maxdepth: 1

notebooks/ribomap_section/03_SMITH_RIBOMap_Transfer_executed
```
