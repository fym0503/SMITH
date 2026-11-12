# Reproducibility workflows

This directory contains the input manifest, task-profile settings, reference metrics and runnable workflows for the TF activity, RIBOMap and liver MERFISH examples. The notebooks start from H5AD inputs and write panels, predictions, figures and run manifests.

## Check and run

```bash
smith-repro list
smith-repro check --data-root data/tutorials
smith-repro run 03_ribomap_transfer --data-root data/tutorials
```

Expected input paths, file sizes and checksums are listed in `data_manifest.yaml`. H5AD data bundles are managed separately from the Git repository.

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
