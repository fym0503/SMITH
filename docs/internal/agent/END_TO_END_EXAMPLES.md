# Smith-Agent End-to-End Examples

This document defines practical end-to-end example prompts for `smith-cli`.

The goal is not only to show that SMITH can run, but to exercise the broader `smith-agent` workflow described in the SMITH-Agent workflow design:

- reference dataset retrieval
- feasibility-aware filtering
- panel optimization
- cross-dataset evaluation
- final decision summary

## Example 1. Formal SC-to-ST Transfer Workflow

This is the current best fully grounded example in the repo.

It exercises:

- user-provided dataset mounting via `@/absolute/path`
- formal SMITH panel selection
- cross-dataset evaluation
- session artifact persistence

### Prompt

```text
Use @/workspace/fanyimin/SMITH_unified/code/SMITH_tool-main/SC_dataset_processed.h5ad to run a formal SMITH panel selection with tasks recon, cls, region, and pathology. Generate a 64-gene panel, then evaluate the resulting panel on @/workspace/fanyimin/SMITH_unified/code/SMITH_tool-main/ST_dataset_processed.h5ad, and summarize the transfer performance.
```

### What It Should Trigger

Expected high-level tool chain:

1. `run_smith_selection`
2. `evaluate_cross_dataset_panel`
3. final session summary

### Main Outputs

Expected artifact types:

- `smith_selection/saving/epoch_*.csv`
- `smith_selection/saving/smith_run_manifest.json`
- `cross_dataset_evaluation.csv`

### Why This Matters

This example corresponds most directly to:

- panel `a` overall workflow
- panel `d` task-aware optimization
- panel `e` final report / output package

It is the best current example for validating the orchestration layer before adding richer feasibility or probe export execution.

## Example 2. Reference-Retrieval-Oriented Workflow

This example is designed to exercise the retrieval side of the agent figure.

### Prompt

```text
Inspect @/workspace/fanyimin/SMITH_unified/code/SMITH_tool-main/SC_dataset_processed.h5ad, retrieve the most relevant reference datasets for transfer-aware panel design, and explain why they are compatible with this request.
```

### What It Should Trigger

Expected high-level tool chain:

1. `resolve_dataset_context`
2. `score_reference_transferability`
3. final session summary

### Main Outputs

Expected artifact types:

- `context_summary.json`
- `reference_candidates.tsv`
- `reference_score_matrix.tsv`
- `reference_score_matrix.png`
- `reference_selection.json`

### Why This Matters

This example corresponds most directly to:

- panel `b` reference dataset retrieval
- the figure-level claim that the agent does not optimize on a fixed dataset blindly

The implemented score decomposition includes species, tissue, modality compatibility, query gene coverage, label compatibility, disease-context support, and spatial support. This makes the output directly usable for the proposed Panel B ranked-table or heatmap visualization.

## Example 2b. CELLxGENE Public Source Retrieval

This example is the public-atlas version of Panel B. The experimental input is paired or tissue-matched scRNA-seq; the prospective MERFISH/imaging-ST target is held out and is not retrieved or used during panel selection. The query terms are parameters, not hard-coded; `liver` can be replaced by another tissue.

### Prompt

```text
For a prospective liver MERFISH panel-design experiment, use paired liver scRNA-seq as the source input. Query CELLxGENE for Homo sapiens liver single-cell RNA references with at least 10000 cells, rank the top 10 candidates, keep only datasets that can be materialized as H5AD or Census subsets, and do not retrieve or use liver MERFISH target data.
```

### Equivalent Tool Call

```bash
smith-cli run-tool query_cellxgene_metadata --arguments-json '{"query":{"query_role":"panel_design_source","organism":"Homo sapiens","tissue":"liver","assay_family":"single_cell_rna","exclude_assay_families":["imaging_spatial","sequencing_spatial"],"target_assay_family":"imaging_spatial","tissue_mode":"containing","min_cells":10000,"materializable_only":true},"max_results":10}'
```

### Main Outputs

- `cellxgene_reference_candidates.tsv`
- `cellxgene_reference_candidates.json`

The candidate table includes `materializable_in_census` and `downloadable_h5ad` so the agent can decide whether to slice through CELLxGENE Census or download a Discover H5AD asset before local subsetting.

The `query_role=panel_design_source` guardrail makes the retrieval source-only: if the request mentions MERFISH/Xenium/CosMx/MERSCOPE/seqFISH/osmFISH, the agent still retrieves scRNA-seq source references and excludes imaging/spatial target assay families. If a paired study is known, add `collection_id` or `collection_ids` to restrict retrieval to that CELLxGENE collection and `exclude_dataset_ids` to hide the held-out target dataset.

## Example 3. Search-Guided Discovery Workflow

This example uses the new search layer to let a user discover available resources before asking for an analysis run.

### Prompt

```text
Search for feasibility filtering support, ProbeDealer integration, and example end-to-end workflows in smith_agent.
```

### What It Should Trigger

Expected high-level tool chain:

1. `search_smith_agent`
2. optional follow-up inspection or execution requests

### Why This Matters

This example is useful when the user wants to understand:

- which skills and tools already exist
- which design documents mention feasibility filtering
- which current implementation pieces are grounded versus still planned

## Example 4. Figure-Plan Target Workflow

This is the most important conceptual example, even though not all modules are fully executable yet.

It is the best prompt to align development with the manuscript figure.

### Prompt

```text
Use @/workspace/fanyimin/SMITH_unified/code/SMITH_tool-main/SC_dataset_processed.h5ad as the source dataset and @/workspace/fanyimin/SMITH_unified/code/SMITH_tool-main/ST_dataset_processed.h5ad as the transfer test dataset. First inspect the source data and retrieve relevant references. Then apply feasibility-aware filtering to a candidate gene pool, run formal SMITH panel selection under a 64-gene budget, evaluate transfer performance on the test dataset, and generate a decision-oriented summary of the final panel.
```

### Figure-Plan Coverage

This prompt is designed to cover:

- `a` overall workflow
- `b` reference retrieval
- `c` feasibility-aware filtering
- `c2` budget-aware compression
- `d` task-aware optimization
- `e` final report / package
- `e2` decision trace

### Current State

Currently grounded and runnable:

- dataset inspection
- reference ranking
- formal SMITH selection
- cross-dataset evaluation
- session-level result summary
- local search across registries, docs, configs, and outputs

Currently scaffolded but not yet fully operational end to end:

- multi-backend feasibility execution using ODT / OligoMiner / PaintSHOP / ProbeDealer in one CLI request
- constraint-aware replacement mode
- budget-aware compression as a first-class interactive tool
- final probe export packaging
- richer decision-trace reporting from completed runs

This prompt should therefore be treated as the north-star end-to-end CLI example for ongoing implementation.

## Suggested CLI Session

Recommended manual session order:

```text
/search feasibility filtering
/search ProbeDealer
Inspect @/workspace/fanyimin/SMITH_unified/code/SMITH_tool-main/SC_dataset_processed.h5ad
Use @/workspace/fanyimin/SMITH_unified/code/SMITH_tool-main/SC_dataset_processed.h5ad to run a formal SMITH panel selection with tasks recon, cls, region, and pathology. Generate a 64-gene panel, then evaluate the resulting panel on @/workspace/fanyimin/SMITH_unified/code/SMITH_tool-main/ST_dataset_processed.h5ad, and summarize the transfer performance.
/summary
```

## Development Note

The most important future improvement is not just adding more tools.

It is making the final agent response summarize:

- which references were selected
- which genes were filtered or retained
- which panel was chosen
- how transfer performance changed
- which artifacts the user should inspect next

That final response is what makes the CLI example look like a real `smith-agent` workflow rather than a sequence of disconnected tool calls.
