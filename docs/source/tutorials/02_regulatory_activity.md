# Which regulatory features preserve *C. elegans* identity and development?

## Biological question

Which compact set of regulatory features is sufficient to preserve *C. elegans*
cell identity and developmental progression? TF and miRNA activity are biological
regulatory readouts, not a generic feature-selection table. A useful panel should
recover discrete lineage labels and the continuous developmental-time signal in
cells that were not used for training.

## Data, model, and analysis

The `train.h5ad`/`test.h5ad` files contain lineage-aware TF or miRNA activity,
cell-type labels, and absolute developmental time. The workflow trains SMITH on
the training split with reconstruction, cell-type classification, and
developmental-time objectives. Its learned ranking is converted into new panels,
then evaluated on held-out cells. Cell-type accuracy asks whether lineage
information survives compression; developmental-time Pearson correlation asks
whether the ordered trajectory survives as well. These are biological tests of
identity and progression, not technical return-on-investment scores.

The short executed notebook below uses one real split and SMITH only so that the
page remains runnable on a CPU. It is a real end-to-end run, but it is not
presented as the complete seven-method paper benchmark.

```bash
python scripts/download_tutorial_data.py \
  --case 02_regulatory_activity --data-root data/tutorials

python reproducibility/workflows/regulatory_activity/run_tutorial.py \
  --data-root data/tutorials \
  --output-dir outputs/tutorials/regulatory \
  --datasets elegans_tf,elegans_mirna \
  --splits split_1 \
  --methods SMITH \
  --epochs 30 --device cpu --force

python reproducibility/workflows/regulatory_activity/plot_figure3.py \
  --values outputs/tutorials/regulatory/figure_data/figure3_c_f_values.tsv \
  --output-dir outputs/tutorials/regulatory/figures
```

The run starts from the H5AD files, invokes `scripts/main.py` to train SMITH, and
writes ranking, panel, evaluation, metrics, logs, and `run_manifest.json` under
`outputs/tutorials/regulatory`. The notebook then reads only those newly created
files. The plotter writes Figure 3c, d, e and f as separate PNG/PDF/SVG/TIFF files.
Each panel uses the manuscript panel proportions; the shared seven-method legend
is exported as its own asset instead of changing the chart canvases.

For the manuscript-scale c-f panels, use all five splits and the six external
baseline backends. `--baseline-python METHOD=PATH` can be repeated when PERSIST
or scGIST is installed in a separate environment. The workflow never replaces a
missing paper baseline with a variance ranking.

Figure 3h-k are not silently approximated: reproducing them additionally requires
the versioned developmental-module annotations, TF-pair definitions, and
scRNA-to-TF transfer inputs used in those manuscript panels.

```{toctree}
:maxdepth: 1

notebooks/regulatory_section/02_SMITH_Regulatory_Activity_executed
```
