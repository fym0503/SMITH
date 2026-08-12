from __future__ import annotations

from pathlib import Path

import pandas as pd

from smith_agent.feasibility.integration import IntegrationThresholds, apply_hard_constraints, write_passing_targets_json
from smith_agent.utils import ensure_dir


def build_three_backend_feasibility_summary(
    manifest_tsv: str | Path,
    odt_summary_tsv: str | Path | None,
    oligominer_summary_tsv: str | Path,
    probedealer_summary_tsv: str | Path,
    thresholds: IntegrationThresholds,
    skip_property_gate: bool = False,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_tsv, sep="\t")
    if odt_summary_tsv:
        odt = pd.read_csv(odt_summary_tsv, sep="\t").rename(
            columns={
                "gene": "gene_symbol",
                "candidate_oligos_after_property_filters": "property_probe_count",
                "candidate_oligos_initial": "odt_initial_probe_count",
                "feasible_property_only": "odt_feasible_property_only",
            }
        )
    else:
        odt = manifest[["gene_symbol"]].copy()
        odt["property_probe_count"] = 999999 if skip_property_gate else pd.NA
        odt["odt_initial_probe_count"] = pd.NA
        odt["odt_feasible_property_only"] = True if skip_property_gate else pd.NA
    oligominer = pd.read_csv(oligominer_summary_tsv, sep="\t").rename(
        columns={
            "candidate_probe_count": "oligominer_candidate_probe_count",
            "specific_probe_count": "oligominer_specific_probe_count",
        }
    )
    probedealer = pd.read_csv(probedealer_summary_tsv, sep="\t").rename(
        columns={
            "initial_probe_count": "probedealer_initial_probe_count",
            "final_probe_count": "probedealer_final_probe_count",
        }
    )

    df = manifest.merge(
        odt[["gene_symbol", "property_probe_count", "odt_initial_probe_count", "odt_feasible_property_only"]],
        on="gene_symbol",
        how="left",
    )
    df = df.merge(oligominer, on="transcript_id", how="left")
    df = df.merge(probedealer, on="transcript_id", how="left")
    df["paintshop_on_target_mean"] = pd.NA
    df["paintshop_off_target_mean"] = pd.NA
    df["paintshop_off_target_max"] = pd.NA
    return apply_hard_constraints(df, thresholds)


def write_three_backend_outputs(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, str]:
    out_dir = ensure_dir(output_dir)
    summary_tsv = out_dir / "integration_summary.tsv"
    passing_json = out_dir / "passing_targets.json"
    df.to_csv(summary_tsv, sep="\t", index=False)
    write_passing_targets_json(df, passing_json)
    return {
        "integration_summary_tsv": str(summary_tsv),
        "passing_targets_json": str(passing_json),
    }
