# Whole Mouse Brain Selection

This example exercises the algorithmic path underlying the WMB section: AnnData loading, five task tensors, stochastic-gate training and target export. It uses deterministic synthetic observations so it can run in CI and does not reproduce donor-level benchmark values.

```bash
smith-repro run 01_wmb
```

Expected artifacts:

- `smoke_panel.h5ad`: deterministic annotated input.
- `selection/epoch_0.csv`: ranked eight-target panel.
- `summary.json`: selected targets and manuscript mapping.

The full Figure 2 workflow additionally requires WMB references, donor-aware splits, baseline panels, five seeds, transfer evaluation, ablations and runtime benchmarking. Those requirements are recorded in `reproducibility/manifests/01_wmb.yaml`.
