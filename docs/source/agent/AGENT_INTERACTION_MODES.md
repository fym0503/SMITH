# Smith-Agent Interaction Modes

`smith-agent` uses two top-level controllers.

## Pipeline Mode

Pipeline mode is a deterministic controller for standard panel-design deliverables. The workflow graph is deterministic, but its parameters are extracted by the LLM into a typed JSON request before any tool runs.

Use it when the user asks to produce or update one of the core outputs:

- selected gene panel
- transcript-level probe candidate inputs
- feasibility-filtered panel
- cross-dataset evaluation
- plots and final report

Canonical flow:

```text
source scRNA-seq dataset mount or CELLxGENE source retrieval
-> SMITH panel selection
-> transcript/probe candidate bridge
-> feasibility backends
-> optional held-out target evaluation
-> plots
-> report
```

The JSON request includes fields such as:

- `train_adata_file`
- `test_adata_file`
- `panel_size`
- `objective`
- `tasks`
- `label`
- `species`
- `run_feasibility`
- `run_evaluation`
- `build_report`
- `skip_odt`

For prospective imaging-ST assays such as MERFISH, Xenium, CosMx, MERSCOPE, seqFISH, and osmFISH, the source input is paired or tissue-matched scRNA-seq. Target imaging-ST data are held out and are not used for retrieval or panel selection unless the user explicitly provides a test dataset path for post-selection evaluation.

Pipeline mode is rule-routed before ReAct mode because formal panel requests often contain words like "summarize" or "report". Those should not make the agent stop and analyze before the deliverable exists.

## ReAct Mode

ReAct mode is the general agentic controller for everything else.

Use it for:

- search and retrieval
- dataset inspection
- explicit tool execution
- result interpretation
- report or table analysis
- backend comparison
- error diagnosis
- follow-up plotting
- replacement suggestions

ReAct mode follows this loop:

```text
reason over session state
-> choose registered tool
-> observe tool result
-> re-plan
-> answer when enough evidence exists
```

The LLM chooses actions, but registered tools perform all state-changing work.

## Routing Rule

```text
if user asks for a standard panel-design deliverable:
    Pipeline Mode
else:
    ReAct Mode
```

Slash commands remain shell shortcuts. They are not a separate mode; they bypass the router only for interaction speed.

In short:

```text
Pipeline builds standardized panel-design outputs.
ReAct handles analysis, search, inspection, explicit tools, and follow-up reasoning.
```
