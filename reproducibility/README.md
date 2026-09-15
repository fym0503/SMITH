# SMITH reproducibility workflows

The runnable cases start from real H5AD inputs and create fresh rankings, panels, metrics and run manifests. See `data_manifest.yaml` for external data and `workflows/` for portable entry points.

Historical aggregate tables are stored in `reference_outputs/` only for optional post-run comparison. They are not tutorial inputs. Controlled in-house disease data/workflows are excluded from the public package.

For a completed multi-split C. elegans run, Figure 3h can be exported from the
generated panel manifest with `scripts/compute_elegans_module_coverage.py`:

```bash
python scripts/compute_elegans_module_coverage.py \
  --panels outputs/paper/regulatory/figure_data/generated_panels.tsv \
  --module-file data/tutorials/regulatory_activity/elegans/annotations/tf_spatiotemporal_modules.tsv \
  --output outputs/paper/regulatory/figure_data/figure3_h_module_miss_rate.tsv
```

The command requires the full method/split/seed/panel-size grid by default and
fails explicitly when a panel is missing. Grid completeness alone does not
establish manuscript equivalence; compare the resulting values and rendered
structure against `validation/figure3_validation.yaml` before calling a panel
reproduced.
