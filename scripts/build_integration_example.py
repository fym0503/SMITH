#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smith_agent.feasibility.integration import (
    IntegrationThresholds,
    build_integration_summary,
    write_passing_probe_fasta,
    write_passing_targets_json,
)


def main() -> None:
    run_dir = ROOT_DIR / "demo_outputs" / "mouse_10_gene_all_tools"
    summary_df = build_integration_summary(
        manifest_tsv=run_dir / "mouse_10_genes_manifest.tsv",
        odt_summary_tsv=run_dir / "odt_scrinshot_property_only" / "property_only_summary.tsv",
        oligominer_specificity_tsv=run_dir / "oligominer_specificity" / "oligominer_specificity_summary.tsv",
        probedealer_summary_tsv=run_dir / "probedealer_py" / "probedealer_summary.tsv",
        paintshop_rna_probe_tsv=run_dir
        / "paintshop"
        / "pipeline_output"
        / "03_output_files"
        / "02_rna_probes_all"
        / "mouse_10_gene_demo_refseq_newBalance.tsv",
        thresholds=IntegrationThresholds(),
    )

    integration_summary_tsv = run_dir / "integration_summary.tsv"
    summary_df.to_csv(integration_summary_tsv, sep="\t", index=False)

    passing_json = run_dir / "passing_targets.json"
    write_passing_targets_json(summary_df, passing_json)

    final_probe_fasta = run_dir / "final_probe_sequences.fasta"
    write_passing_probe_fasta(
        final_probe_fasta=run_dir / "probedealer_py" / "final_probes.fa",
        passing_transcript_ids=summary_df.loc[summary_df["overall_pass"], "transcript_id"].tolist(),
        output_fasta=final_probe_fasta,
    )

    summary_payload = {
        "integration_summary_tsv": str(integration_summary_tsv),
        "passing_targets_json": str(passing_json),
        "final_probe_sequences_fasta": str(final_probe_fasta),
        "n_total_targets": int(len(summary_df)),
        "n_passing_targets": int(summary_df["overall_pass"].sum()),
    }
    out_json = run_dir / "integration_outputs.json"
    out_json.write_text(json.dumps(summary_payload, indent=2) + "\n")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
