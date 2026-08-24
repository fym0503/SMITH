# SMITH-Agent Feasibility Layer

Integration notes for adding external probe-design and feasibility backends to the SMITH agent workflow.

## Goal

This functionality now lives in `smith_agent.feasibility`, which lets SMITH-Agent call multiple external tools during a `feasibility-aware filtering` stage before final panel optimization.

Recommended scope for the interface layer:

- normalize tool inputs from a candidate gene list and assay context
- run one or more backends
- collect tool outputs into a common schema
- expose a unified feasibility summary back to the agent

This layer should not reimplement probe design algorithms. It should orchestrate them.

## Recommended Backends

We currently recommend integrating these four backends:

1. `ODT / SCRINSHOT`
2. `PaintSHOP`
3. `OligoMiner`
4. `ProbeDealer`

We explicitly do **not** recommend integrating `SPAPROS` into this repo at this stage because it overlaps with panel-selection logic and is less suitable as a modular feasibility backend.

## Why These Four

These backends are complementary rather than redundant:

- `ODT / SCRINSHOT`: sequence-property feasibility and probe-set construction
- `PaintSHOP`: transcript-aware annotation and isoform-flattened probe sufficiency
- `OligoMiner`: off-target specificity and genome/transcriptome-aware filtering
- `ProbeDealer`: multiplex-ready probe construction, including readout/primer-aware design

Together they cover the most useful practical filtering axes:

- transcript validity
- sequence chemistry
- off-target specificity
- probe-set sufficiency
- multiplex deployment compatibility

## Backend Summaries

### 1. ODT / SCRINSHOT

Role in SMITH-agent:

- main `property-based feasibility` backend
- evaluate whether a candidate gene can support a valid probe set after sequence-level filtering

What it is good for:

- GC content filtering
- melting temperature filtering
- homopolymer filtering
- arm-length constraints
- non-overlapping probe-set construction
- reporting `best_set_size` and `n_oligo_sets`

Why it is useful:

- it directly connects candidate genes to deployable probe design
- it is the most natural backend for a `feasibility-aware pre-filter`

Local code already present:

- `/workspace/fanyimin/spapros_info/odt_h5ad_feasibility_profile.py`
- `/workspace/fanyimin/spapros_info/odt_scrinshot_feasibility_demo.py`
- `/workspace/fanyimin/spapros_info/h5ad_profile_merged/feasibility_profile_summary.json`
- `/workspace/fanyimin/spapros_info/h5ad_profile_merged/feasibility_profile_merged.tsv`

Primary source:

- Schiller et al., Oligo Designer Toolsuite / SCRINSHOT-oriented probe design framework
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11621025/

Suggested interface output:

- `status`
- `transcript_id`
- `candidate_oligos_initial`
- `candidate_oligos_after_property_filters`
- `n_oligo_sets`
- `best_set_size`
- `feasible_property_only`

### 2. PaintSHOP

Role in SMITH-agent:

- `transcript-aware annotation` backend
- supports isoform-aware or isoform-flattened design decisions

What it is good for:

- transcript annotation harmonization
- isoform-flattened region handling
- probe count sufficiency reasoning
- practical design for large RNA FISH panels

Why it is useful:

- many candidate genes are gene-level objects in scRNA-seq, while probe design happens on transcript sequence
- this backend helps explain whether a gene is robustly targetable across isoforms

Primary source:

- PaintSHOP: a platform for the interactive design of transcriptome- and genome-scale oligonucleotide FISH experiments
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8349872/

Suggested interface output:

- `canonical_or_flattened_transcript_model`
- `effective_target_span_nt`
- `estimated_probe_capacity`
- `isoform_robustness_flag`
- `annotation_notes`

### 3. OligoMiner

Role in SMITH-agent:

- `specificity` backend
- used to penalize or remove genes whose candidate probes are prone to off-target binding

What it is good for:

- transcriptome/genome specificity checks
- k-mer or alignment-based off-target evaluation
- repeat-aware filtering
- large-scale oligo screening

Why it is useful:

- a gene may be sequence-feasible but still be risky due to off-target hybridization
- this adds a second, orthogonal feasibility axis beyond GC/Tm-style property checks

Primary source:

- OligoMiner: a rapid, flexible environment for the design of genome-scale oligonucleotide in situ hybridization probes
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5877937/

Suggested interface output:

- `specificity_pass`
- `offtarget_score`
- `repeat_mask_flag`
- `alignment_summary`
- `specificity_notes`

### 4. ProbeDealer

Role in SMITH-agent:

- `deployment-oriented multiplex design` backend
- used when the user wants a panel that is closer to assay-ready output

What it is good for:

- multiplexed FISH probe construction
- readout/primer/barcode-aware design
- end-to-end probe design for imaging assays

Why it is useful:

- this is the most practical backend for the final stage of assay deployment
- it helps bridge from a selected gene panel to a more complete probe-design package

Primary source:

- ProbeDealer: automated design of probes for highly multiplexed FISH and sequential imaging
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7745008/

Suggested interface output:

- `deployment_ready_flag`
- `n_constructible_probes`
- `barcode_or_readout_compatibility`
- `primer_compatibility`
- `deployment_notes`

## Recommended Common Schema

All backend adapters should map their results into a shared per-gene schema.

Suggested minimum fields:

```text
gene
backend
status
feasible
score
reason
transcript_id
effective_target_span_nt
candidate_oligos_initial
candidate_oligos_after_filters
n_oligo_sets
best_set_size
specificity_pass
offtarget_score
deployment_ready_flag
notes
```

Then aggregate across backends into a final `agent-facing` summary:

```text
gene
overall_feasible
overall_score
property_feasible
specificity_feasible
isoform_robust
deployment_ready
recommended_action
```

## Proposed Agent Workflow

Recommended order inside `SMITH-agent`:

1. collect candidate genes from user constraints and SMITH search space
2. harmonize identifiers and transcript annotations
3. run `ODT / SCRINSHOT` for property feasibility
4. run `OligoMiner` for specificity filtering
5. run `PaintSHOP` for transcript / isoform / sufficiency annotation
6. optionally run `ProbeDealer` for deployment-oriented validation
7. return an `optimization-ready pool` to SMITH
8. run panel optimization or HPO on the filtered pool
9. emit a final report with both biological and assay-design rationale

## Recommended Interface Modules

Suggested repo structure:

```text
smith_agent/feasibility/
  README.md
  smith_agent/feasibility/
    __init__.py
    schemas.py
    orchestrator.py
    backends/
      odt_scrinshot.py
      paintshop.py
      oligominer.py
      probedealer.py
    adapters/
      gene_ids.py
      transcript_annotation.py
    reports/
      summarize.py
```

## Suggested Priority

Implementation priority:

1. `ODT / SCRINSHOT`
2. `OligoMiner`
3. `PaintSHOP`
4. `ProbeDealer`

Reason:

- `ODT / SCRINSHOT` gives the clearest immediate practical-filtering value
- `OligoMiner` adds the most important orthogonal signal: specificity
- `PaintSHOP` improves transcript-level robustness and explainability
- `ProbeDealer` is most useful after the first three are working

## Suggested Paper Language

Use language like:

> We integrated four complementary probe-design backends into the agentic workflow to bridge algorithmic panel selection and deployable assay design, covering sequence-property feasibility, transcript-aware probe sufficiency, off-target specificity and multiplex-ready probe construction.

Avoid language like:

- "we wrapped several tools"
- "we used Fabian's tool"
- "the agent just calls external packages"

The intended framing is:

- `feasibility-aware candidate pruning`
- `assay-aware panel design`
- `bridging algorithmic selection and deployable probe construction`

## Current Local Assets

Useful local starting points:

- `/workspace/fanyimin/spapros_info`
- `/workspace/fanyimin/.codex/skills/smith-hpo/SKILL.md`
- `/workspace/fanyimin/SMITH_unified/code/workflows/htapp313_smith_multimodal/search_smith_htapp313_multimodal.py`

## Hand-off Notes

If another agent takes over implementation, ask it to:

1. define a stable per-gene result schema first
2. implement `ODT / SCRINSHOT` adapter first
3. keep each backend isolated behind a common interface
4. avoid coupling feasibility logic to SMITH training internals
5. produce machine-readable summaries that the main SMITH agent can consume directly
