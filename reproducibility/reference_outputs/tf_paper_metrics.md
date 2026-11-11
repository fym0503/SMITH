# TF tutorial reference metrics

`tf_paper_metrics.tsv` contains the five-split reference values for TF cell-type accuracy and developmental-time correlation. `tf_paper_panels.tsv` stores the ordered 128-gene ranking for each split; smaller panels use the leading 32 or 64 genes.

`tf_transfer_paper_metrics.tsv` contains the scRNA-seq-to-TF evaluation values. `tf_transfer_paper_panel.tsv` stores the ordered transfer ranking. The notebook computes these metrics from the selected panels and held-out data.
