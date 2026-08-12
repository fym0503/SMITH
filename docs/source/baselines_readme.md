# SMITH Baselines: Setup and Run Guide

This workspace contains the baseline methods from `GPS_tools-main` and a validated local setup for running them on:

- SC data: `/workspace/fanyimin/SMITH_tool-main/SC_dataset_processed.h5ad`
- ST data: `/workspace/fanyimin/SMITH_tool-main/ST_dataset_processed.h5ad`

The code was patched to:

- remove hard-coded absolute paths from another machine
- support both sparse and dense `.h5ad` matrices
- use local package imports for bundled baseline code
- work with current `spapros` and `numpy` versions

## Repository Layout

- Baseline code: `/workspace/fanyimin/SMITH_baselines/GPS_tools-main`
- Sampled smoke-test inputs: `/workspace/fanyimin/SMITH_baselines/smoke_inputs`
- Smoke-test outputs: `/workspace/fanyimin/SMITH_baselines/smoke_outputs`
- Subset sampler: `/workspace/fanyimin/SMITH_baselines/scripts/sample_balanced_subset.py`
- End-to-end smoke runner: `/workspace/fanyimin/SMITH_baselines/scripts/run_smoke_baselines.sh`
- End-to-end full-data runner: `/workspace/fanyimin/SMITH_baselines/scripts/run_full_baselines.sh`

## Validated Environments

Two conda envs were created because `scGIST` needs a TensorFlow-compatible stack separate from the other baselines.

- Common env: `/workspace/fanyimin/.conda/envs/smith-common`
- scGIST env: `/workspace/fanyimin/.conda/envs/smith-scgist`

### Recreate `smith-common`

```bash
conda create -y -p /workspace/fanyimin/.conda/envs/smith-common python=3.11 pip
/workspace/fanyimin/.conda/envs/smith-common/bin/python -m pip install --no-user \
  scanpy==1.10.3 anndata==0.10.9 spapros==0.1.6
/workspace/fanyimin/.conda/envs/smith-common/bin/python -m pip install --no-user \
  torch --index-url https://download.pytorch.org/whl/cpu
```

### Recreate `smith-scgist`

```bash
conda create -y -p /workspace/fanyimin/.conda/envs/smith-scgist python=3.11 pip
/workspace/fanyimin/.conda/envs/smith-scgist/bin/python -m pip install --no-user \
  tensorflow==2.13.0 scanpy==1.9.6 anndata==0.9.2 numpy==1.24.3 pandas==1.5.3 \
  scipy==1.11.4 matplotlib==3.7.5 seaborn==0.11.2 scikit-learn==1.3.2
```

## Dataset Notes

The provided datasets both contain a usable supervised label column:

- `celltype`

Observed shapes:

- SC: `269758 x 293`
- ST: `219953 x 293`

## Quick Smoke Test

Run the full smoke suite on balanced subsets sampled from the SC and ST inputs:

```bash
bash /workspace/fanyimin/SMITH_baselines/scripts/run_smoke_baselines.sh
```

This script will:

1. sample up to `1000` cells/spots per `celltype` into 8k-cell subset files
2. run `scGeneFit`, `activeSVM`, `persist_sup`, `persist_unsup`, `spapros`, and `scGIST`
3. write outputs under `smoke_outputs/end_to_end`

You can override defaults:

```bash
SC_DATA=/workspace/fanyimin/SMITH_tool-main/SC_dataset_processed.h5ad \
ST_DATA=/workspace/fanyimin/SMITH_tool-main/ST_dataset_processed.h5ad \
NUM_MARKERS=8 \
SAMPLE_PER_CLASS=1000 \
OUT_DIR=/workspace/fanyimin/SMITH_baselines/smoke_outputs/end_to_end \
bash /workspace/fanyimin/SMITH_baselines/scripts/run_smoke_baselines.sh
```

## Sample a Subset Only

```bash
/workspace/fanyimin/.conda/envs/smith-common/bin/python \
  /workspace/fanyimin/SMITH_baselines/scripts/sample_balanced_subset.py \
  --input /workspace/fanyimin/SMITH_tool-main/SC_dataset_processed.h5ad \
  --output /workspace/fanyimin/SMITH_baselines/smoke_inputs/SC_smoke_8000.h5ad \
  --label celltype \
  --max-per-class 1000
```

Repeat the same command for the ST dataset.

## Run Individual Baselines

### scGeneFit

```bash
PYTHON_BIN=/workspace/fanyimin/.conda/envs/smith-common/bin/python \
bash /workspace/fanyimin/SMITH_baselines/GPS_tools-main/baselines/scGeneFit/run.sh \
  /workspace/fanyimin/SMITH_tool-main/SC_dataset_processed.h5ad \
  celltype \
  8 \
  /workspace/fanyimin/SMITH_baselines/out/scGeneFit_sc
```

### activeSVM

Use the Python entrypoint directly if you want to control smoke-test arguments such as `--num_samples` or `--max_iter`.

```bash
/workspace/fanyimin/.conda/envs/smith-common/bin/python \
  /workspace/fanyimin/SMITH_baselines/GPS_tools-main/baselines/activeSVM/run.py \
  --adata /workspace/fanyimin/SMITH_tool-main/SC_dataset_processed.h5ad \
  --label celltype \
  --num_markers 8 \
  --num_samples 2000 \
  --max_iter 20 \
  --output /workspace/fanyimin/SMITH_baselines/out/activeSVM_sc
```

### persist supervised

```bash
/workspace/fanyimin/.conda/envs/smith-common/bin/python \
  /workspace/fanyimin/SMITH_baselines/GPS_tools-main/baselines/persist/run_sup.py \
  --adata /workspace/fanyimin/SMITH_baselines/smoke_inputs/SC_smoke_8000.h5ad \
  --label celltype \
  --num_markers 8 \
  --max_epochs 1 \
  --output /workspace/fanyimin/SMITH_baselines/out/persist_sup_sc
```

### persist unsupervised

```bash
/workspace/fanyimin/.conda/envs/smith-common/bin/python \
  /workspace/fanyimin/SMITH_baselines/GPS_tools-main/baselines/persist/run_unsup.py \
  --adata /workspace/fanyimin/SMITH_baselines/smoke_inputs/ST_smoke_8000.h5ad \
  --num_markers 8 \
  --max_epochs 1 \
  --output /workspace/fanyimin/SMITH_baselines/out/persist_unsup_st
```

### spapros

`spapros` works after the local compatibility shim, but it is slower. For first validation, use the subset inputs.

```bash
PYTHON_BIN=/workspace/fanyimin/.conda/envs/smith-common/bin/python \
bash /workspace/fanyimin/SMITH_baselines/GPS_tools-main/baselines/spapros/run.sh \
  /workspace/fanyimin/SMITH_baselines/smoke_inputs/ST_smoke_8000.h5ad \
  celltype \
  8 \
  /workspace/fanyimin/SMITH_baselines/out/spapros_st
```

### scGIST

Use the TensorFlow env.

```bash
/workspace/fanyimin/.conda/envs/smith-scgist/bin/python \
  /workspace/fanyimin/SMITH_baselines/GPS_tools-main/baselines/scGIST/run.py \
  --adata /workspace/fanyimin/SMITH_baselines/smoke_inputs/ST_smoke_8000.h5ad \
  --label celltype \
  --num_markers 8 \
  --epochs 1 \
  --output /workspace/fanyimin/SMITH_baselines/out/scGIST_st
```

## What Was Verified Here

Completed successfully:

- `scGeneFit` on full SC and full ST
- `activeSVM` on full SC and full ST
- `persist_sup` on sampled SC and sampled ST
- `persist_unsup` on sampled ST
- `spapros` on sampled SC and sampled ST
- `scGIST` on sampled SC and sampled ST

Generated smoke outputs:

- `/workspace/fanyimin/SMITH_baselines/smoke_outputs/scGeneFit/sc/marker_8.csv`
- `/workspace/fanyimin/SMITH_baselines/smoke_outputs/scGeneFit/st/marker_8.csv`
- `/workspace/fanyimin/SMITH_baselines/smoke_outputs/activeSVM/sc/marker_8.csv`
- `/workspace/fanyimin/SMITH_baselines/smoke_outputs/activeSVM/st/marker_8.csv`
- `/workspace/fanyimin/SMITH_baselines/smoke_outputs/persist_sup_subset/sc/marker_8.csv`
- `/workspace/fanyimin/SMITH_baselines/smoke_outputs/persist_sup_subset/st/marker_8.csv`
- `/workspace/fanyimin/SMITH_baselines/smoke_outputs/persist_unsup_subset/st/marker_8.csv`
- `/workspace/fanyimin/SMITH_baselines/smoke_outputs/spapros_subset/sc/marker_8.csv`
- `/workspace/fanyimin/SMITH_baselines/smoke_outputs/spapros_subset/st/marker_8.csv`
- `/workspace/fanyimin/SMITH_baselines/smoke_outputs/scGIST_subset/sc/marker_8.csv`
- `/workspace/fanyimin/SMITH_baselines/smoke_outputs/scGIST_subset/st/marker_8.csv`

## Practical Recommendation

For debugging:

- start with the subset workflow in `scripts/run_smoke_baselines.sh`

For production runs:

- `scGeneFit` and `activeSVM` can be tried directly on the full inputs
- `persist`, `spapros`, and `scGIST` should be started on subsets first and then scaled carefully

## Full-Data or Other-Dataset Runs

For later non-smoke execution on your own datasets, use:

```bash
bash /workspace/fanyimin/SMITH_baselines/scripts/run_full_baselines.sh
```

This runner does not sample subsets. It runs directly on the dataset paths you provide.

Default inputs are your current datasets:

- `SC_DATA=/workspace/fanyimin/SMITH_tool-main/SC_dataset_processed.h5ad`
- `ST_DATA=/workspace/fanyimin/SMITH_tool-main/ST_dataset_processed.h5ad`
- `UNSUP_DATA=$ST_DATA`

Important defaults:

- `NUM_MARKERS=32`
- `ACTIVE_SVM_NUM_SAMPLES=3600`
- `ACTIVE_SVM_MAX_ITER=200`
- `PERSIST_MAX_EPOCHS=200`
- `SCGIST_EPOCHS=200`

Example on other datasets:

```bash
SC_DATA=/path/to/other_sc.h5ad \
ST_DATA=/path/to/other_st.h5ad \
UNSUP_DATA=/path/to/other_st.h5ad \
LABEL=celltype \
NUM_MARKERS=32 \
OUT_DIR=/path/to/output_dir \
bash /workspace/fanyimin/SMITH_baselines/scripts/run_full_baselines.sh
```

You can disable individual baselines by setting the corresponding flag to `0`:

```bash
RUN_SPAPROS=0 RUN_SCGIST=0 \
bash /workspace/fanyimin/SMITH_baselines/scripts/run_full_baselines.sh
```

Available flags:

- `RUN_SCGENEFIT`
- `RUN_ACTIVESVM`
- `RUN_PERSIST_SUP`
- `RUN_PERSIST_UNSUP`
- `RUN_SPAPROS`
- `RUN_SCGIST`

Notes:

- supervised baselines require the `LABEL` column to exist in `.obs`
- `persist_unsup` does not need a label and runs on `UNSUP_DATA`
- full runs for `persist`, `spapros`, and `scGIST` can be much slower than the smoke workflow
