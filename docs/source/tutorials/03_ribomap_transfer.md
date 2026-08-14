# Reproduce Figure 4c-h

## Biological question

Can a compact gene panel transfer cell-type and brain-region biology from
reference modalities into RIBOMap? Deep-RIBOmap and STARmap measure related but
non-identical molecular views. The goal is therefore to preserve stable tissue
structure without erasing ribosome-associated expression differences.

## Data, model, and analysis

The workflow starts from the real Deep-RIBOmap, STARmap, and target RIBOMap H5AD
files and constructs shared-gene training objects. SMITH is trained on each
source with reconstruction and cell-type objectives, plus spatial coordination
when coordinates are available. New source panels are evaluated on target
RIBOMap cell types and regions. Cell-type and region accuracy test transfer of
identity and anatomy; same- versus cross-modality Jaccard asks which biological
features are stable across references; RIBOMap bias asks whether panel genes
retain translatome-specific abundance rather than only generic expression.

```bash
python scripts/download_tutorial_data.py \
  --case 03_ribomap_transfer --data-root data/tutorials

python reproducibility/workflows/ribomap_transfer/run_tutorial.py \
  --data-root data/tutorials \
  --output-dir outputs/tutorials/ribomap \
  --sources Deep-RIBOmap,STARmap \
  --methods SMITH \
  --panel-sizes 32,64,128 \
  --training-seeds 1,2 \
  --evaluation-seeds 1,2,3 \
  --epochs 30 --device cpu --force

python reproducibility/workflows/ribomap_transfer/plot_figure4.py \
  --metrics outputs/tutorials/ribomap/figure_data/figure4_c_f_values.tsv \
  --overlap outputs/tutorials/ribomap/figure_data/figure4_g_jaccard.tsv \
  --bias outputs/tutorials/ribomap/figure_data/figure4_h_ribomap_bias.tsv \
  --output-dir outputs/tutorials/ribomap/figures
```

The run first writes shared-gene H5AD files, then trains SMITH and writes new
rankings, panels, evaluations, metrics, logs, and `run_manifest.json`. The
notebook reads those outputs rather than reusing aggregate result tables.
Figure 4c-h are exported one panel per file. Panels c-f retain the near-square
benchmark geometry from the manuscript, while g and h keep their original
portrait proportions. The shared method legend is a separate export.

The executed page uses both real source modalities but only SMITH. The complete
paper command adds all six manuscript baselines and five seeds; the grouped bars
and their repeated-run points are generated from those fresh outputs.

Figure 4i is omitted until the exact versioned Reactome/GO gene-set snapshot is
distributed. Figure 4j-n require the manuscript clean-fusion aligned H5AD. The
tutorial does not insert placeholder panels or unrelated summaries for either
missing input.

```{toctree}
:maxdepth: 1

notebooks/ribomap_section/03_SMITH_RIBOMap_Transfer_executed
```
