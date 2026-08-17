# Which genes preserve liver cell identity in MERFISH?

## Biological question

Which genes should be measured in a spatial liver assay so that cell identities
and their expression programs remain interpretable? The source snRNA-seq gives
broad cell-state coverage, while spatial references add tissue context. We
therefore compare a source-only panel with a panel informed by both kinds of
biological evidence before reading out MERFISH.

## Data, model, and analysis

The inputs are healthy-liver snRNA-seq, MERFISH, and spatial-reference H5AD
files. MERFISH is held out for evaluation; the source and spatial references are
training views. SMITH is trained separately on the source and each reference
after restricting them to the MERFISH gene universe. Their rankings are
aggregated into source-only and multi-reference panels. MERFISH cell-type
accuracy tests whether selected genes retain cellular identity in the assay that
will be measured; mean MERFISH expression tests whether the panel is supported
by detectable biology. The paired comparison asks whether spatial references add
information beyond the source transcriptome.

```bash
python scripts/download_tutorial_data.py \
  --case 05_agent --data-root data/tutorials

python reproducibility/workflows/agent/run_tutorial.py \
  --data-root data/tutorials \
  --output-dir outputs/tutorials/agent \
  --panel-sizes 32,64,128 \
  --training-seeds 1,2,3,4,5 \
  --epochs 200 --device cpu --force

python reproducibility/workflows/agent/plot_figure6.py \
  --accuracy outputs/tutorials/agent/figure_data/figure6_c_cell_type_accuracy.tsv \
  --expression outputs/tutorials/agent/figure_data/figure6_d_merfish_expression.tsv \
  --output-dir outputs/tutorials/agent/figures
```

The run trains fresh source/reference models, aggregates their rankings, and
writes new panels, MERFISH evaluation tables, metrics, logs, and
`run_manifest.json` under the output directory. Figure 6c and d are separate square panels, matching the original `2.25 x 2.25`
inch plotting canvases. Each is exported independently as PNG/PDF/SVG/TIFF.

The hosted notebook uses two real references and two seeds; the manuscript run
uses all five references listed in the data manifest and five seeds. Figure 6e-g
requires separately installed ODT, OligoMiner, and ProbeDealer backends, while
Figure 6h-j requires the validation-guided hyperparameter search. Those stages
remain explicit `not_run` boundaries and are never filled with fabricated rates.

```{toctree}
:maxdepth: 1

notebooks/agent_section/05_SMITH_Agent_Evaluation_executed
```
