# Reproduce Figure 3c-f

This tutorial starts from the lineage-aware C. elegans TF and miRNA activity
`train.h5ad`/`test.h5ad` splits used in the manuscript. It trains each requested
panel-selection method, evaluates every newly selected panel on held-out cells,
and draws the four quantitative panels from Figure 3: cell-type accuracy and
developmental-time correlation for TF and miRNA activity.

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
  --epochs 30 --device cpu

python reproducibility/workflows/regulatory_activity/plot_figure3.py \
  --values outputs/tutorials/regulatory/figure_data/figure3_c_f_values.tsv \
  --output-prefix outputs/tutorials/regulatory/figures/figure3_c_f_reproduced
```

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
