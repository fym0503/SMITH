# Reference outputs

These files are **precomputed manuscript or backend outputs**, not inputs to the
core SMITH training tutorials. They are used only where rerunning an external
probe-design backend is impractically slow for a hosted notebook.

- `agent_full_tool_pass_summary.tsv` contains the completed 12,160-gene
  Figure 6f backend summary used in the manuscript.
- `agent_probedealer_offtarget_examples.tsv` contains the ten Figure 6g genes
  extracted from the completed 12,160-gene ProbeDealer scan on pc157 on
  2026-08-26. The source table was
  `/workspace/fanyimin/tutorial_outputs/agent_figure6fg_20260826/probedealer_full_scan/probe_risk_summary.tsv`.

The Agent notebook labels these plots as reference-output visualizations. Its
SMITH training, panel aggregation, hyperparameter example and MERFISH evaluation
continue to run from H5AD inputs.
