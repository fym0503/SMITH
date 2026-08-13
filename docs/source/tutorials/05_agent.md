# Reproduce Figure 6c-d

This tutorial starts from real healthy-liver snRNA-seq, MERFISH, and spatial
reference H5AD files. For each training seed it runs SMITH on the source and every
reference, aggregates the new rankings, selects source-only and multi-reference
panels at 32/64/128 genes, and evaluates both panels on MERFISH. The final code
rebuilds Figure 6c (cell-type accuracy) and Figure 6d (mean MERFISH expression)
with paired seed-level points and one-sided Wilcoxon tests.

```bash
python scripts/download_tutorial_data.py \
  --case 05_agent --data-root data/tutorials

python reproducibility/workflows/agent/run_tutorial.py \
  --data-root data/tutorials \
  --output-dir outputs/tutorials/agent \
  --panel-sizes 32,64,128 \
  --training-seeds 1,2,3,4,5 \
  --epochs 200 --device cuda:0

python reproducibility/workflows/agent/plot_figure6.py \
  --accuracy outputs/tutorials/agent/figure_data/figure6_c_cell_type_accuracy.tsv \
  --expression outputs/tutorials/agent/figure_data/figure6_d_merfish_expression.tsv \
  --output-prefix outputs/tutorials/agent/figures/figure6_c_d_reproduced
```

The hosted notebook uses two real references and two seeds; the manuscript run
uses all five references listed in the data manifest and five seeds. Figure 6e-g
requires separately installed ODT, OligoMiner, and ProbeDealer backends, while
Figure 6h-j requires the validation-guided hyperparameter search. Those stages
remain explicit `not_run` boundaries and are never filled with fabricated rates.

```{toctree}
:maxdepth: 1

notebooks/agent_section/05_SMITH_Agent_Evaluation_executed
```
