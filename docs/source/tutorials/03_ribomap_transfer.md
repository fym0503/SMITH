# Reproduce Figure 4c-h

This tutorial begins with the real Deep-RIBOmap, STARmap, and target RIBOMap
H5AD files. It recomputes the shared-gene training objects, selects panels, and
evaluates the new panels on target cell type and brain region. The plotting code
then rebuilds the manuscript analyses in Figure 4c-h: four transfer benchmarks,
same- versus cross-modality panel Jaccard similarity, and gene-level RIBOMap bias.

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
  --epochs 30 --device cpu

python reproducibility/workflows/ribomap_transfer/plot_figure4.py \
  --metrics outputs/tutorials/ribomap/figure_data/figure4_c_f_values.tsv \
  --overlap outputs/tutorials/ribomap/figure_data/figure4_g_jaccard.tsv \
  --bias outputs/tutorials/ribomap/figure_data/figure4_h_ribomap_bias.tsv \
  --output-dir outputs/tutorials/ribomap/figures
```

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
