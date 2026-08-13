# SMITH-Agent panel evaluation

The Agent tutorial starts from real healthy liver snRNA-seq, MERFISH and two spatial references. It trains a source panel and reference panels, aggregates their new rankings, then evaluates source-only and multi-reference panels on MERFISH cell type and spatial coordinates.

```bash
python scripts/download_tutorial_data.py --case 05_agent --data-root data/tutorials
python reproducibility/workflows/agent/run_tutorial.py \
  --data-root data/tutorials \
  --output-dir outputs/tutorials/agent \
  --panel-size 64 --epochs 30 --device cpu
```

Repeat `--reference agent/references/<file>.h5ad` to run the full paper reference set. ODT, OligoMiner and ProbeDealer are separate optional backends; the tutorial reports an explicit `not_run` state when they are not configured and never fabricates a pass rate.

```{toctree}
:maxdepth: 1

notebooks/agent_section/05_SMITH_Agent_Evaluation_executed
```
