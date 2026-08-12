# In-House Disease Transfer

Raw participant-level human brain data are controlled access. This example therefore recomputes results from a de-identified aggregate transfer-robustness table.

```bash
smith-repro run 04_inhouse_disease
```

The output reports multi-seed changes in rank correlation and top-64 overlap for each cohort comparison. It cannot regenerate participant-level panels, UMAPs, imputation results or spatial gene examples without authorized input data. This limitation is part of the manifest, not an implicit tutorial failure.
