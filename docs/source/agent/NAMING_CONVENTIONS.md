# Smith-Agent Naming Conventions

This document defines the recommended naming rules for `smith-agent`.

The goal is to keep the system easy to extend, test, and reason about as new skills, tools, backends, registries, and artifacts are added.

## Core Principle

Use naming to preserve architectural boundaries:

- `skill` names describe workflow intent
- `tool` names describe deterministic actions
- `adapter` names describe backend-specific execution wrappers
- `backend` names describe external systems or method families
- `artifact` names describe saved outputs
- `session state` keys describe canonical workflow objects

Do not mix these layers in one name.

## Global Style Rules

Use `snake_case` for:

- skill ids
- tool ids
- registry ids
- Python module names
- YAML config ids
- session state keys
- artifact stems where possible

Use lowercase only.

Do not use:

- spaces
- camelCase
- PascalCase
- vague abbreviations unless already standard in the field

Prefer short explicit names over compressed clever names.

Good:

- `reference_retrieval`
- `run_spapros`
- `feasibility_table.tsv`

Bad:

- `ReferenceRetrieval`
- `spRun`
- `do_filtering_and_optimization`

## Layer-Specific Rules

## 1. Skills

Skills represent workflow phases or reusable reasoning patterns.

Skill names should:

- be noun-like or stage-like
- describe a domain capability
- avoid implementation details
- avoid backend names unless the skill is truly backend-specific

Recommended pattern:

```text
<domain_capability>
```

Examples:

- `task_intake`
- `reference_retrieval`
- `candidate_gene_assembly`
- `feasibility_filtering`
- `panel_selection`
- `baseline_comparison`
- `probe_generation`
- `reporting`

Avoid:

- `run_smith_and_compare`
- `use_odt_then_paintshop`
- `do_panel_thing`

Reason:

Skills decide what kind of work should happen, not which exact executable should be invoked.

## 2. Tools

Tools represent deterministic executable actions.

Tool names should:

- start with a verb
- describe one concrete action
- avoid policy language
- avoid planner logic
- avoid multi-step orchestration in one name

Recommended patterns:

```text
run_<method_or_backend>
build_<artifact>
load_<object>
list_<registry_object>
select_<object>
score_<object>
evaluate_<metric_family>
compare_<result_family>
export_<artifact>
package_<artifact>
normalize_<entity>
validate_<entity>
merge_<entity_set>
aggregate_<result_family>
apply_<policy>
```

Examples:

- `run_smith_selection`
- `run_spapros`
- `run_odt_screen`
- `run_probedealer_export`
- `score_reference_transferability`
- `aggregate_feasibility_results`
- `apply_feasibility_policy`
- `build_decision_trace`

Avoid:

- `run_spapros_if_needed`
- `try_run_odt`
- `auto_select_reference`
- `do_feasibility_and_repair`
- `maybe_export_probes`

Reason:

Whether a tool is needed is decided by the planner or skill layer, not encoded into the tool name.

## 3. Adapters

Adapters are code wrappers around a specific backend or package.

Adapter names should:

- reflect the backend
- reflect the execution role
- remain internal and implementation-oriented

Recommended patterns:

```text
<backend>_runner
<backend>_adapter
<backend>_client
<backend>_exporter
<backend>_parser
```

Examples:

- `odt_runner`
- `paintshop_adapter`
- `probedealer_exporter`
- `spapros_runner`

Use adapter names in implementation code, not as user-facing workflow names.

## 4. Backends

Backends are the actual external tools, algorithms, or method families.

Keep backend ids stable and close to community usage.

Examples:

- `smith`
- `scgist`
- `scgenefit`
- `activesvm`
- `spapros`
- `odt`
- `oligominer`
- `paintshop`
- `probedealer`

Do not invent alternate aliases unless required for compatibility.

## 5. Registries

Registries should use plural nouns for directories and singular ids for entries.

Directory examples:

- `configs/skills/`
- `configs/datasets/`
- `configs/references/`
- `configs/models/`
- `configs/baselines/`
- `configs/feasibility_backends/`
- `configs/probe_backends/`
- `configs/policies/`

Entry id examples:

- `htapp213`
- `ribomap_mouse_l3`
- `smith_default`
- `spapros_default`
- `odt_default`

## 6. Artifacts

Artifacts are saved outputs from tools or skills.

Artifact names should:

- describe the object, not the command that produced it
- remain stable across runs
- use suffixes like `.json`, `.tsv`, `.csv`, `.md`, `.fasta`

Recommended examples:

- `request.json`
- `context_summary.json`
- `reference_candidates.tsv`
- `reference_selection.json`
- `candidate_gene_pool.tsv`
- `feasibility_table.tsv`
- `filter_decisions.tsv`
- `panel_candidate_rank.tsv`
- `selected_panel.tsv`
- `selection_metadata.json`
- `baseline_summary.tsv`
- `panel_comparison.tsv`
- `decision_trace.tsv`
- `final_probe_table.tsv`
- `final_probe_sequences.fasta`
- `deliverable_manifest.json`
- `final_report.md`

Avoid:

- `smith_output_final_v2_reallyfinal.tsv`
- `run1_results.tsv`
- `tmp_table.tsv`
- `panel_after_some_filtering.tsv`

If run-specific separation is needed, keep stable artifact names inside a run directory rather than encoding run metadata into every file name.

Good:

```text
sessions/<session_id>/runs/<run_id>/selected_panel.tsv
```

Bad:

```text
selected_panel_session12_run4_20260410.tsv
```

## 7. Session State Keys

Session keys should match canonical workflow objects.

Recommended examples:

- `active_request`
- `active_dataset`
- `active_reference_selection`
- `candidate_gene_pool`
- `feasibility_table`
- `selected_panel`
- `baseline_summary`
- `decision_trace`
- `final_probe_bundle`

Avoid using tool names as state keys unless the state stores raw tool outputs intentionally.

Bad:

- `ran_odt`
- `did_spapros`
- `smith_done`

Good:

- `feasibility_table`
- `panel_comparison`

## 8. Policies and Modes

Policies and modes should use noun phrases, not executable verbs.

Examples:

- `default_policy`
- `strict_feasibility`
- `balanced_transfer`
- `budget_compression`
- `constraint_replacement`

Avoid:

- `compress_if_too_large`
- `replace_bad_genes`

These are decision policies, not tool actions.

## Decision Words: Where They Belong

These words usually belong in `skills`, `policies`, or planner logic, not tool names:

- `if_needed`
- `auto`
- `smart`
- `adaptive`
- `dynamic`
- `try`
- `maybe`
- `best`
- `optimal`

Examples:

- `baseline_comparison` may decide whether to call `run_spapros`
- `strict_feasibility` may decide whether to drop borderline genes
- `panel_selection` may choose `budget_compression` versus `constraint_replacement`

But the tool names remain:

- `run_spapros`
- `apply_feasibility_policy`
- `run_budget_compression`
- `run_constraint_replacement`

## Recommended Verb Vocabulary For Tools

Use a small stable verb set.

- `list_`: enumerate registry entries
- `describe_`: summarize one registered object
- `load_`: read a stored object into runtime
- `resolve_`: infer or map identifiers/context
- `normalize_`: standardize names or schema
- `validate_`: enforce expected structure
- `extract_`: pull structured content from data
- `merge_`: combine multiple sources
- `score_`: compute ranking signals
- `select_`: choose from scored candidates
- `run_`: execute a model or backend
- `aggregate_`: combine backend outputs
- `apply_`: enforce a rule or policy
- `evaluate_`: compute metrics on outputs
- `compare_`: contrast methods or runs
- `build_`: generate structured summaries
- `export_`: write final outward-facing outputs
- `package_`: bundle multiple outputs together

Avoid adding many near-synonyms unless necessary.

Prefer `build_report` over both `generate_report` and `create_report` if one convention is already chosen.

## Naming By Responsibility

Use this test before naming anything:

1. Is this deciding what should happen?
   Use a `skill` or `policy` name.

2. Is this performing one concrete execution step?
   Use a `tool` name.

3. Is this wrapping a specific external package?
   Use an `adapter` name.

4. Is this a saved object?
   Use an `artifact` name.

If one name seems to answer more than one of these questions, the boundary is probably wrong.

## Recommended Canonical Vocabulary For Smith-Agent

Use these terms consistently across configs, code, and documents:

- `reference` instead of mixing `ref`, `retrieval target`, `source dataset`
- `candidate_gene_pool` instead of alternating with `candidate list`, `gene universe`, `search pool`
- `feasibility_filtering` for early screening
- `panel_selection` for final optimization
- `budget_compression` for shrinking oversized panels
- `constraint_replacement` for repairing infeasible genes
- `probe_generation` for final assay export
- `decision_trace` for explainability output

This vocabulary should stay stable across:

- skill ids
- tool names
- artifact names
- figure labels
- report sections

## Good End-to-End Example

```text
Skill:
  feasibility_filtering

Tools:
  run_odt_screen
  run_oligominer_screen
  run_paintshop_screen
  aggregate_feasibility_results
  apply_feasibility_policy

Artifacts:
  feasibility_table.tsv
  filter_decisions.tsv

Policy:
  strict_feasibility
```

This is clear because each layer has one role.

## Bad End-to-End Example

```text
Skill:
  do_probe_feasibility_and_maybe_filter

Tools:
  smart_odt_if_needed
  maybe_run_paintshop
  auto_pick_best_filter

Artifacts:
  finalish_filter_results_v3.tsv
```

This is unclear because planning, execution, and output semantics are mixed together.

## Final Rule

When in doubt:

- make `skills` broader
- make `tools` narrower
- make `artifacts` more noun-like
- keep backend names literal

The naming system should make the architecture readable without reading the implementation.
