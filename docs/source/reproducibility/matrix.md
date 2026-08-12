# Reproducibility Matrix

The matrix prevents three different claims from being conflated: code-path verification, representative analysis reproduction and full-paper regeneration.

| Results section | Example | Bundled input | Representative output | Full-paper status |
|---|---|---|---|---|
| WMB benchmark | Complete small-data training | Synthetic H5AD generated at runtime | Ranked eight-target panel | External WMB data and multi-seed benchmarks required |
| Regulatory activity | Aggregate recomputation | TF/miRNA five-run metrics | Best SMITH identity/time summary | Public atlas download and training required |
| RIBOMap transfer | Aggregate recomputation | RIBOMap/STARmap benchmark metrics | Transfer accuracy summary | Public spatial objects and pathway workflow required |
| Human disease atlas | Aggregate recomputation | De-identified robustness table | Multi-seed transfer audit | Controlled-access raw data required |
| SMITH-Agent | Aggregate recomputation | MERFISH metrics and feasibility example | Reference/feasibility summary | External references and probe backends required |

Every input fixture is SHA-256 pinned. Run `smith-repro check` before analysis. Every output includes the case id, manuscript section, figure and claim so that artifacts remain interpretable after they are moved.

## What the examples do not claim

- They do not reproduce every panel of every main and supplementary figure.
- Aggregate examples do not rerun target selection from raw H5AD files.
- The human-disease example does not make restricted human data public.
- Running the small WMB example is not evidence that manuscript-scale HPO has been repeated.
