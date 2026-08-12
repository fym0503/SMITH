# Quickstart

Install the package in editable mode:

```bash
python -m pip install -e .
```

Build a distributable source archive and wheel:

```bash
python -m build --sdist --wheel
```

## Run Without External Data

Generate a synthetic AnnData file:

```bash
python scripts/make_smoke_h5ad.py --output data/smoke/smoke_panel.h5ad
```

Run a short SMITH example on CPU:

```bash
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

Inspect the SMITH-Agent registry from the source tree:

```bash
smith-cli --config configs/agent/agent.yaml tools
```

After installing the wheel, the same default registry is embedded in the package:

```bash
smith-cli tools
```

Run tests:

```bash
python -m pytest tests
```

## Expected H5AD Fields

The core optimizer reads `adata.X` by default, or `adata.layers[<name>]` when `--layer` is provided. Common annotations are `obs['celltype']` or `obs['cell_type']`, `obs['region']`, `obs['pathology']`, and `obsm['spatial']`.
