from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from Bio import SeqIO


@dataclass(frozen=True)
class IntegrationThresholds:
    min_property_probes: int = 20
    min_specific_probes: int = 10
    min_deployment_probes: int = 20
    min_paintshop_on_target_mean: float = 95.0
    max_paintshop_off_target_mean: float = 10.0
    max_paintshop_off_target_max: float = 100.0
    require_transcript_gate: bool = False


def load_gene_manifest(manifest_tsv: str | Path) -> pd.DataFrame:
    return pd.read_csv(manifest_tsv, sep="\t")


def load_odt_property_summary(summary_tsv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(summary_tsv, sep="\t")
    return df.rename(
        columns={
            "gene": "gene_symbol",
            "candidate_oligos_after_property_filters": "property_probe_count",
            "candidate_oligos_initial": "odt_initial_probe_count",
            "feasible_property_only": "odt_feasible_property_only",
        }
    )


def load_oligominer_specificity_summary(summary_tsv: str | Path) -> pd.DataFrame:
    return pd.read_csv(summary_tsv, sep="\t").rename(
        columns={
            "candidate_probe_count": "oligominer_candidate_probe_count",
            "specific_probe_count": "oligominer_specific_probe_count",
        }
    )


def load_probedealer_summary(summary_tsv: str | Path) -> pd.DataFrame:
    return pd.read_csv(summary_tsv, sep="\t").rename(
        columns={
            "initial_probe_count": "probedealer_initial_probe_count",
            "final_probe_count": "probedealer_final_probe_count",
        }
    )


def load_paintshop_summary(rna_probe_tsv: str | Path) -> pd.DataFrame:
    cols = [
        "chromosome",
        "start",
        "stop",
        "sequence",
        "Tm",
        "on_target",
        "off_target",
        "repeat",
        "max_kmer",
        "strand",
        "refseq",
        "transcript_id",
        "gene_id",
    ]
    df = pd.read_csv(rna_probe_tsv, sep="\t", header=None, names=cols)
    return (
        df.groupby("transcript_id", as_index=False)
        .agg(
            paintshop_probe_count=("sequence", "size"),
            paintshop_on_target_mean=("on_target", "mean"),
            paintshop_off_target_mean=("off_target", "mean"),
            paintshop_off_target_max=("off_target", "max"),
        )
        .reset_index(drop=True)
    )


def apply_hard_constraints(
    merged_df: pd.DataFrame,
    thresholds: IntegrationThresholds,
) -> pd.DataFrame:
    df = merged_df.copy()
    df["pass_property"] = df["property_probe_count"].fillna(0) >= thresholds.min_property_probes
    df["pass_specificity"] = df["oligominer_specific_probe_count"].fillna(0) >= thresholds.min_specific_probes
    df["pass_deployment"] = df["probedealer_final_probe_count"].fillna(0) >= thresholds.min_deployment_probes
    df["pass_transcript"] = (
        (df["paintshop_on_target_mean"].fillna(0) >= thresholds.min_paintshop_on_target_mean)
        & (df["paintshop_off_target_mean"].fillna(float("inf")) <= thresholds.max_paintshop_off_target_mean)
        & (df["paintshop_off_target_max"].fillna(float("inf")) <= thresholds.max_paintshop_off_target_max)
    )

    if thresholds.require_transcript_gate:
        df["overall_pass"] = (
            df["pass_property"] & df["pass_specificity"] & df["pass_deployment"] & df["pass_transcript"]
        )
    else:
        df["overall_pass"] = df["pass_property"] & df["pass_specificity"] & df["pass_deployment"]

    actions: list[str] = []
    reasons: list[str] = []
    for row in df.itertuples(index=False):
        if not row.pass_property:
            actions.append("drop_property")
            reasons.append("property_probe_count below threshold")
        elif not row.pass_specificity:
            actions.append("drop_specificity")
            reasons.append("oligominer_specific_probe_count below threshold")
        elif thresholds.require_transcript_gate and not row.pass_transcript:
            actions.append("deprioritize")
            reasons.append("paintshop transcript-quality gate failed")
        elif not row.pass_deployment:
            actions.append("drop_deployment")
            reasons.append("probedealer_final_probe_count below threshold")
        elif not row.pass_transcript:
            actions.append("keep_with_caution")
            reasons.append("passed hard gates, but PaintSHOP transcript quality is weaker")
        else:
            actions.append("keep")
            reasons.append("passed all hard constraints")
    df["recommended_action"] = actions
    df["reason"] = reasons
    return df


def build_integration_summary(
    manifest_tsv: str | Path,
    odt_summary_tsv: str | Path,
    oligominer_specificity_tsv: str | Path,
    probedealer_summary_tsv: str | Path,
    paintshop_rna_probe_tsv: str | Path,
    thresholds: IntegrationThresholds,
) -> pd.DataFrame:
    manifest = load_gene_manifest(manifest_tsv)
    odt = load_odt_property_summary(odt_summary_tsv)
    oligominer = load_oligominer_specificity_summary(oligominer_specificity_tsv)
    probedealer = load_probedealer_summary(probedealer_summary_tsv)
    paintshop = load_paintshop_summary(paintshop_rna_probe_tsv)

    df = manifest.merge(
        odt[["gene_symbol", "property_probe_count", "odt_initial_probe_count", "odt_feasible_property_only"]],
        on="gene_symbol",
        how="left",
    )
    df = df.merge(oligominer, on="transcript_id", how="left")
    df = df.merge(probedealer, on="transcript_id", how="left")
    df = df.merge(paintshop, on="transcript_id", how="left")
    return apply_hard_constraints(df, thresholds)


def write_passing_targets_json(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    records = df[df["overall_pass"]].to_dict(orient="records")
    output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def write_passing_probe_fasta(
    final_probe_fasta: str | Path,
    passing_transcript_ids: Iterable[str],
    output_fasta: str | Path,
) -> None:
    output_fasta = Path(output_fasta)
    passing = set(passing_transcript_ids)
    pattern = re.compile(r"_(ENSMUST[0-9]+)_Seq_")
    selected_records = []
    for record in SeqIO.parse(str(final_probe_fasta), "fasta"):
        match = pattern.search(record.id)
        if not match:
            continue
        transcript_id = match.group(1)
        if transcript_id in passing:
            selected_records.append(record)
    with output_fasta.open("w", encoding="utf-8") as handle:
        SeqIO.write(selected_records, handle, "fasta")

