# Biological panel-design tutorials

These executed notebooks demonstrate three workflows using TF activity, brain RIBOMap and liver MERFISH data. The matching `_source.ipynb` files in the repository can be opened and run in Jupyter.

```{toctree}
:maxdepth: 1

notebooks/regulatory_section/02_SMITH_Regulatory_Activity_executed
notebooks/ribomap_section/03_SMITH_RIBOMap_Transfer_executed
notebooks/agent_section/05_SMITH_Agent_Evaluation_executed
```

| Tutorial profile | Analysis | Reference comparison |
|---|---|---|
| `TF_profile` | Five TF lineage splits; 15 cell-type and developmental-time evaluations | 15/15 SMITH results match the recorded reference values |
| `ribomap_profile` | Four configurations; 12 accuracy endpoints, panel overlap and expression bias | Accuracy means differ by at most 0.000154; overlap uses the stored multi-method result table |
| `Agent_profile` | Five spatial references and five seeds; MERFISH panel evaluation | 30/30 panels and expression points match; accuracy differs by at most 0.000332 in the fresh-environment run |

The probe-screen panels use the recorded 12,160-gene screen summaries. H5AD inputs are listed with expected paths and SHA-256 checksums in `reproducibility/data_manifest.yaml`.

## Download and run

Download the complete data bundle from [Zenodo](https://doi.org/10.5281/zenodo.24137308) (7.10 GB download; about 11.2 GB of input data). One download supplies all three tutorials, including the five TF lineage splits and five liver spatial references.

From the SMITH repository root, run:

```bash
python scripts/download_tutorial_data.py --data-root data/tutorials
export SMITH_TUTORIAL_DATA="$PWD/data/tutorials"
jupyter lab docs/source/tutorials/notebooks
```

The script verifies the archive SHA-256, extracts the inputs into `data/tutorials/`, and checks each input file. Allow about 20 GB of free disk space for the archive and extracted data, plus space for training outputs.

If you already downloaded the archive, add `--archive /path/to/SMITH_tutorial_complete_20261110.tar.gz` to the download command. If you extracted it manually, set `SMITH_TUTORIAL_DATA` to the extracted `SMITH_tutorial_complete/` directory. Keep its `regulatory_activity/`, `ribomap_transfer/` and `agent/` subdirectories together.

### Tutorial execution environment

The reference results use CUDA training. For the same execution mode, use Python 3.12, install `requirements-reproducibility.txt` alongside `.[notebooks]`, and set `SMITH_TUTORIAL_DEVICE=cuda:0`. Notebooks select CUDA when available and otherwise use CPU. CPU and CUDA training can produce different rankings even with the same seed.
