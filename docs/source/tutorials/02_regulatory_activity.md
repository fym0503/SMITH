# Regulatory activity

The workflow begins with a real C. elegans `train.h5ad` and `test.h5ad`, trains SMITH with reconstruction, cell-type and developmental-time objectives, writes a new ranking/panel, and evaluates it on the held-out split.

```bash
python scripts/download_tutorial_data.py --case 02_regulatory_activity --data-root data/tutorials
python reproducibility/workflows/regulatory_activity/run_tutorial.py \
  --data-root data/tutorials \
  --output-dir outputs/tutorials/regulatory \
  --dataset elegans_tf --split split_1 \
  --panel-size 32 --epochs 30 --device cpu
```

Outputs are written under `outputs/tutorials/regulatory/`: `smith/ranking/`, `smith/panel_top32.csv`, `evaluation/metrics.tsv`, developmental-time predictions and `run_manifest.json`.

```{toctree}
:maxdepth: 1

notebooks/regulatory_section/02_SMITH_Regulatory_Activity_executed
```
