# Tutorial reference tables

These compact, version-controlled tables provide the metrics and panel rankings used to compare tutorial runs with the reported analyses. They contain summary values and target lists; source H5AD files are listed separately in `reproducibility/data_manifest.yaml`.

- `tf_paper_metrics.tsv` and `tf_paper_panels.tsv`: five-split TF cell-type and developmental-time values, with ordered 128-target rankings.
- `tf_transfer_paper_metrics.tsv` and `tf_transfer_paper_panel.tsv`: scRNA-seq-to-TF evaluation values and ordered target ranking.
- `ribomap_paper_accuracy.tsv` and `ribomap_paper_panels/`: endpoint accuracy means and associated ordered panels/configurations.
- `ribomap_panel_overlap.tsv`: same- and cross-modality comparisons for SMITH, PERSIST-class and scGIST.
- `agent_multi_reference_metrics.tsv`: five-seed liver MERFISH panel metrics.
- `agent_full_tool_pass_summary.tsv` and `agent_probedealer_offtarget_examples.tsv`: probe-screen summary values and selected examples used by the Agent notebook.
