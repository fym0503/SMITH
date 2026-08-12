# Scripts

Main entry points copied into this clean package:

- `main.py`: original SMITH training and target-panel selection runner.
- `eval.py`: original downstream panel evaluation helper.
- `submit_eval.py`: original batch/evaluation helper.
- `make_smoke_h5ad.py`: creates a tiny synthetic AnnData file for tests and smoke runs.
- `odt_property_only_runner.py`, `run_*`, `prepare_*`, `build_*`: probe-feasibility demonstration helpers used by `smith_agent.feasibility`.
- `agent_examples/`: exploratory and figure-generation scripts copied from the SMITH-Agent workspace. Some of these scripts still expect manuscript-scale data or external tools and are kept as examples rather than core package smoke tests.

Run the optimizer from the package root:

```bash
python scripts/make_smoke_h5ad.py --output data/smoke/smoke_panel.h5ad
python scripts/main.py \
  --adata_file data/smoke/smoke_panel.h5ad \
  --saving_dir outputs/smoke/saving \
  --log_dir outputs/smoke/logs \
  --tasks recon,cls,region,pathology,coordination \
  --task_name celltype \
  --panel_size 8 \
  --epoch 2 \
  --record 1 \
  --batch_size 16 \
  --rep_dim 8 \
  --rep_hidden_dims 8 \
  --head_hidden_dims 8 \
  --dim 8 \
  --device cpu \
  --balance_mode off
```
