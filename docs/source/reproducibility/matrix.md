# Reproducibility Matrix

The matrix prevents three different claims from being conflated: code-path verification, representative analysis reproduction and full-paper regeneration.

| Results section | Example | Bundled input | Representative output | Full-paper status |
|---|---|---|---|---|
| WMB benchmark | Source-availability record | None; original WMB code/data are unavailable | No fabricated Figure 2 output | Original WMB workflow and data must be supplied |
| Regulatory activity | Real-output notebook | Completed TF/miRNA five-run metrics | SMITH vs baseline identity/time comparison | Public atlas download and training required |
| RIBOMap transfer | Real-output notebook | Completed RIBOMap/STARmap benchmark metrics | Transfer accuracy comparison | Public spatial objects and pathway workflow required |
| Human disease atlas | Real-output notebook | Real de-identified per-seed robustness metrics | Recomputed aggregate transfer audit | Controlled-access raw data required |
| SMITH-Agent | Real-output notebook | Real five-seed MERFISH metrics and 12,160-gene gate summary | Reference/feasibility comparison | External references and probe backends required |

Every input fixture is SHA-256 pinned. Run `smith-repro check` before analysis. Every output includes the case id, manuscript section, figure and claim so that artifacts remain interpretable after they are moved.

## What the examples do not claim

- They do not reproduce every panel of every main and supplementary figure.
- Aggregate examples do not rerun target selection from raw H5AD files.
- The human-disease example does not make restricted human data public.
- The WMB page is not evidence that manuscript-scale Figure 2 has been repeated; its source boundary is explicit because the original code/data are unavailable.
