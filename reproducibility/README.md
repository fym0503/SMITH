# Reproducibility workflows

This directory contains the input manifest, task-profile settings, reference metrics and runnable workflows for the TF activity, RIBOMap and liver MERFISH examples. The notebooks start from H5AD inputs and write panels, predictions, figures and run manifests.

## Check and run

```bash
smith-repro list
smith-repro check --data-root data/tutorials
smith-repro run 03_ribomap_transfer --data-root data/tutorials
```

Expected input paths, file sizes and checksums are listed in `data_manifest.yaml`. H5AD data bundles are managed separately from the Git repository.
