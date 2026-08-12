# Data Directory

Large SMITH benchmark datasets are not included in this clean package.

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
