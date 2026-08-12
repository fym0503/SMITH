# Regulatory Activity

This example recomputes representative TF and miRNA summaries from the five-run evaluation table. It focuses on the joint cell-identity and developmental-time claim.

```bash
smith-repro run 02_regulatory_activity
```

Inspect `summary.json` for the best validated SMITH result at each dataset and panel size. Full Figure 3 regeneration also needs lineage-aware splits, panel training, baseline runs, module coverage, co-activity reconstruction and scRNA-to-TF transfer.
