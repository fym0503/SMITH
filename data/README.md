# Data Directory

Store input H5AD files here. Tutorial paths and checksums are listed in `reproducibility/data_manifest.yaml`.

Place external H5AD files under this directory when you want to run the packaged examples. The agent dataset registry uses these relative paths by default:

- `data/smith/SC_dataset_processed.h5ad`
- `data/smith/ST_dataset_processed.h5ad`
- `data/htapp/HTAPP-313-SMP-932/scrna.h5ad`
- `data/htapp/HTAPP-313-SMP-932/merfish_1.h5ad`
- `data/elegans/elegans_scrna.h5ad`
- `data/elegans/elegans_tf.h5ad`
- `data/elegans/elegans_mirna.h5ad`
- `data/ribomap/*.h5ad`

For a self-contained functional test, generate a synthetic dataset:

```bash
python scripts/make_smoke_h5ad.py --output data/smoke/smoke_panel.h5ad
```

Then run SMITH on the generated file:

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
  --device cpu \
  --balance_mode off
```

## Tutorial data

Download the complete data bundle from [Zenodo](https://doi.org/10.5281/zenodo.24137308) (7.10 GB download; about 11.2 GB of input data). One download supplies all three tutorials, including the five TF lineage splits and five liver spatial references.

From the SMITH repository root, run:

```bash
python scripts/download_tutorial_data.py --data-root data/tutorials
export SMITH_TUTORIAL_DATA="$PWD/data/tutorials"
jupyter lab docs/source/tutorials/notebooks
```

The script verifies the archive SHA-256, extracts the inputs into `data/tutorials/`, and checks each input file. Allow about 20 GB of free disk space for the archive and extracted data, plus space for training outputs.

If you already downloaded the archive, add `--archive /path/to/SMITH_tutorial_complete_20261110.tar.gz` to the download command. If you extracted it manually, set `SMITH_TUTORIAL_DATA` to the extracted `SMITH_tutorial_complete/` directory. Keep its `regulatory_activity/`, `ribomap_transfer/` and `agent/` subdirectories together.
