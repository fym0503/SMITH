# Core Concepts

SMITH separates panel selection from workflow orchestration.

- `smith` contains the stochastic-gate optimizer, task heads, losses, dataset loading and evaluation helpers.
- `smith_agent` handles intent parsing, reference retrieval, registered tools, feasibility backends, reporting and session state.
- `smith.reproducibility` maps manuscript sections to versioned inputs and representative analysis artifacts.

## Selection modes

The manuscript uses three design settings:

- **scRNA reference**: select a panel from transcriptome-wide single-cell data.
- **spatial reference transfer**: reuse an existing spatial assay as a design prior for a new assay.
- **aligned reference design**: combine single-cell coverage with spatial or cross-modality context before ranking targets.

## Objectives

The core runner supports reconstruction, cell type classification, region classification, pathology classification and spatial-coordinate regression. Multi-task gradients are balanced through the minimum-norm solver.

## Paper examples

Each main Results section has one executable example. Every example states what it computes, which compact inputs it uses, and what additional data or tools would be required for the full manuscript analysis. These requirements can include large public downloads, restricted data, external tools, GPU time or hyperparameter searches.
