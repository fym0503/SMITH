#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from Bio import SeqIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from smith_agent.adapters.feasibility_backends import run_probedealer_backend_screen
from scripts.run_scrna_probedealer_risk_scan import (
    SOURCE_GENE_METADATA_H5AD,
    load_gene_symbols_from_h5ad,
    summarize_probe_hits,
)
from scripts.run_top128_feasibility_filtering import (
    HUMAN_REFERENCE_DIR,
    SMITH_PACKAGE_ROOT,
    build_local_probe_candidate_manifest,
    load_transcript_to_gene,
)


DEFAULT_RANK_TSV = PROJECT_ROOT / "outputs/liver_merfish_benchmark/liver_source_gene_rank.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/scrna_probedealer_full_gene_scan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable ProbeDealer scan over the full ranked scRNA gene universe "
            "and classify target feasibility / known off-target risks."
        )
    )
    parser.add_argument("--rank-tsv", default=str(DEFAULT_RANK_TSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=0, help="0 means all ranked genes.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means run all batches.")
    parser.add_argument("--species", default="homo_sapiens")
    parser.add_argument("--min-target-probes", type=int, default=20)
    parser.add_argument("--min-known-offtarget-probes", type=int, default=20)
    parser.add_argument("--min-offtarget-fraction-known", type=float, default=0.5)
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--force-batches", action="store_true")
    return parser.parse_args()


def read_ranked_panel(rank_tsv: str | Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(rank_tsv, sep="\t")
    if "gene_symbol" not in df.columns:
        raise ValueError(f"Rank file must contain `gene_symbol`: {rank_tsv}")
    if top_n and top_n > 0:
        df = df.head(top_n).copy()
    else:
        df = df.copy()
    if "rank" not in df.columns:
        df.insert(0, "rank", range(1, len(df) + 1))
    return df[["rank", "gene_symbol"]].copy()


def write_batch_fasta(
    manifest_batch: pd.DataFrame,
    records_by_id: dict[str, Any],
    output_fasta: Path,
) -> None:
    wanted = manifest_batch["transcript_id"].astype(str).tolist()
    with output_fasta.open("w", encoding="utf-8") as handle:
        SeqIO.write([records_by_id[transcript_id] for transcript_id in wanted], handle, "fasta")


def valid_batch_summary(summary_path: Path, expected_rows: int) -> bool:
    if not summary_path.exists():
        return False
    try:
        df = pd.read_csv(summary_path, sep="\t")
    except Exception:
        return False
    return len(df) == expected_rows and {"transcript_id", "initial_probe_count", "final_probe_count"}.issubset(df.columns)


def run_probedealer_batches(
    manifest: pd.DataFrame,
    transcript_fasta: Path,
    output_dir: Path,
    species: str,
    batch_size: int,
    max_batches: int,
    force: bool,
) -> list[dict[str, Any]]:
    batch_root = output_dir / "probedealer_batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    records_by_id = SeqIO.to_dict(SeqIO.parse(str(transcript_fasta), "fasta"))
    batch_logs: list[dict[str, Any]] = []
    n_batches_total = (len(manifest) + batch_size - 1) // batch_size
    n_batches_to_run = n_batches_total if not max_batches or max_batches <= 0 else min(max_batches, n_batches_total)

    for batch_idx in range(n_batches_to_run):
        start = batch_idx * batch_size
        stop = min(start + batch_size, len(manifest))
        batch = manifest.iloc[start:stop].copy()
        batch_dir = batch_root / f"batch_{batch_idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_manifest = batch_dir / "batch_manifest.tsv"
        batch_fasta = batch_dir / "batch_transcripts.fa"
        summary_path = batch_dir / "probedealer_summary.tsv"
        batch.to_csv(batch_manifest, sep="\t", index=False)
        if force or not batch_fasta.exists():
            write_batch_fasta(batch, records_by_id, batch_fasta)

        started = time.time()
        if not force and valid_batch_summary(summary_path, len(batch)):
            status = "skipped_existing"
            result: dict[str, Any] = {"status": status, "output_files": {"summary_tsv": str(summary_path)}}
        else:
            result = run_probedealer_backend_screen(
                package_root=SMITH_PACKAGE_ROOT,
                transcript_fasta=batch_fasta,
                output_dir=batch_dir,
                use_transcriptome_reference=True,
                species=species,
            )
            status = str(result.get("status", "unknown"))

        elapsed = time.time() - started
        batch_log = {
            "batch_index": batch_idx,
            "start_row": start,
            "stop_row": stop,
            "n_genes": int(len(batch)),
            "status": status,
            "elapsed_seconds": elapsed,
            "batch_dir": str(batch_dir),
            "summary_tsv": str(summary_path),
        }
        batch_logs.append(batch_log)
        print(
            f"batch {batch_idx + 1}/{n_batches_total}: {status}, "
            f"genes={len(batch)}, elapsed={elapsed:.1f}s",
            flush=True,
        )
    return batch_logs


def summarize_batches(
    manifest: pd.DataFrame,
    output_dir: Path,
    transcript_to_gene: dict[str, str],
    gene_id_to_symbol: dict[str, str],
    batch_size: int,
) -> pd.DataFrame:
    batch_root = output_dir / "probedealer_batches"
    rows: list[pd.DataFrame] = []
    n_batches_total = (len(manifest) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches_total):
        start = batch_idx * batch_size
        stop = min(start + batch_size, len(manifest))
        batch = manifest.iloc[start:stop].copy()
        batch_dir = batch_root / f"batch_{batch_idx:04d}"
        summary_path = batch_dir / "probedealer_summary.tsv"
        if not valid_batch_summary(summary_path, len(batch)):
            continue
        rows.append(
            summarize_probe_hits(
                manifest=batch,
                probedealer_dir=batch_dir,
                transcript_to_gene=transcript_to_gene,
                gene_id_to_symbol=gene_id_to_symbol,
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def classify_risk(risk: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    risk = risk.copy()
    risk["resolved"] = risk["transcript_id"].notna()
    risk["geneid_probedealer_fail"] = risk["geneid_target_only_probe_count"].fillna(0) < args.min_target_probes
    risk["symbol_known_probedealer_fail"] = (
        risk["symbol_target_only_probe_count_known"].fillna(0) < args.min_target_probes
    )
    known_classified = (
        risk["symbol_target_only_probe_count_known"].fillna(0)
        + risk["probes_with_known_different_symbol"].fillna(0)
    )
    risk["known_classified_probe_count"] = known_classified
    risk["target_probe_fraction_known"] = (
        risk["symbol_target_only_probe_count_known"] / known_classified
    ).where(known_classified > 0)
    risk["known_offtarget_fraction_initial"] = (
        risk["probes_with_known_different_symbol"] / risk["initial_probe_count"]
    ).where(risk["initial_probe_count"] > 0)
    risk["unknown_fraction_initial"] = (
        risk["unknown_symbol_probe_count"] / risk["initial_probe_count"]
    ).where(risk["initial_probe_count"] > 0)
    risk["probe_feasible_symbol_level"] = (
        risk["resolved"] & (risk["symbol_target_only_probe_count_known"].fillna(0) >= args.min_target_probes)
    )
    risk["same_symbol_rescue_candidate"] = (
        risk["resolved"]
        & risk["geneid_probedealer_fail"]
        & (risk["symbol_target_only_probe_count_known"].fillna(0) >= args.min_target_probes)
    )
    risk["known_offtarget_risk"] = (
        risk["resolved"]
        & risk["symbol_known_probedealer_fail"]
        & (risk["probes_with_known_different_symbol"].fillna(0) >= args.min_known_offtarget_probes)
        & (risk["different_symbol_fraction_known"].fillna(0) >= args.min_offtarget_fraction_known)
    )
    risk["clean_known_offtarget_risk"] = (
        risk["known_offtarget_risk"]
        & (risk["clean_different_symbol_probe_count"].fillna(0) >= args.min_known_offtarget_probes)
    )
    risk["annotation_limited_failure"] = (
        risk["resolved"]
        & risk["symbol_known_probedealer_fail"]
        & ~risk["known_offtarget_risk"]
        & (risk["unknown_symbol_probe_count"].fillna(0) >= args.min_target_probes)
    )
    risk["low_probe_count_failure"] = (
        risk["resolved"]
        & risk["symbol_known_probedealer_fail"]
        & ~risk["known_offtarget_risk"]
        & ~risk["annotation_limited_failure"]
    )

    risk["probe_feasibility_class"] = "unresolved"
    risk.loc[risk["probe_feasible_symbol_level"], "probe_feasibility_class"] = "feasible_symbol_specific"
    risk.loc[risk["same_symbol_rescue_candidate"], "probe_feasibility_class"] = "annotation_rescue"
    risk.loc[risk["known_offtarget_risk"], "probe_feasibility_class"] = "known_offtarget_risk"
    risk.loc[risk["annotation_limited_failure"], "probe_feasibility_class"] = "annotation_limited_failure"
    risk.loc[risk["low_probe_count_failure"], "probe_feasibility_class"] = "low_probe_count_failure"
    return risk


def summarize_rank_bins(risk: pd.DataFrame) -> pd.DataFrame:
    bins = [64, 128, 256, 512, 1024, 2048, 4096, len(risk)]
    rows: list[dict[str, Any]] = []
    for top_n in dict.fromkeys(min(item, len(risk)) for item in bins):
        subset = risk.head(top_n)
        rows.append(
            {
                "top_n": int(top_n),
                "n_genes": int(len(subset)),
                "resolved": int(subset["resolved"].sum()),
                "probe_feasible_symbol_level": int(subset["probe_feasible_symbol_level"].sum()),
                "annotation_rescue": int(subset["same_symbol_rescue_candidate"].sum()),
                "known_offtarget_risk": int(subset["known_offtarget_risk"].sum()),
                "clean_known_offtarget_risk": int(subset["clean_known_offtarget_risk"].sum()),
                "annotation_limited_failure": int(subset["annotation_limited_failure"].sum()),
                "feasible_fraction_total": float(subset["probe_feasible_symbol_level"].mean()),
                "known_offtarget_risk_fraction_total": float(subset["known_offtarget_risk"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(risk: pd.DataFrame, output_dir: Path, args: argparse.Namespace, batch_logs: list[dict[str, Any]]) -> None:
    risk_path = output_dir / "probe_risk_summary.tsv"
    risk.to_csv(risk_path, sep="\t", index=False)

    risk[risk["known_offtarget_risk"]].sort_values(
        ["probes_with_known_different_symbol", "different_symbol_fraction_known", "rank"],
        ascending=[False, False, True],
    ).to_csv(output_dir / "known_offtarget_risk_genes.tsv", sep="\t", index=False)

    risk[risk["clean_known_offtarget_risk"]].sort_values(
        ["clean_different_symbol_probe_count", "probes_with_known_different_symbol", "rank"],
        ascending=[False, False, True],
    ).to_csv(output_dir / "clean_known_offtarget_risk_genes.tsv", sep="\t", index=False)

    risk[risk["same_symbol_rescue_candidate"]].sort_values(
        ["symbol_target_only_probe_count_known", "rank"],
        ascending=[False, True],
    ).to_csv(output_dir / "annotation_rescue_genes.tsv", sep="\t", index=False)

    class_summary = (
        risk["probe_feasibility_class"]
        .value_counts(dropna=False)
        .rename_axis("probe_feasibility_class")
        .reset_index(name="gene_count")
    )
    class_summary.to_csv(output_dir / "probe_feasibility_class_summary.tsv", sep="\t", index=False)

    rank_bin_summary = summarize_rank_bins(risk)
    rank_bin_summary.to_csv(output_dir / "rank_bin_feasibility_summary.tsv", sep="\t", index=False)

    summary = {
        "rank_tsv": str(Path(args.rank_tsv).resolve()),
        "top_n_requested": int(args.top_n),
        "batch_size": int(args.batch_size),
        "thresholds": {
            "min_target_probes": int(args.min_target_probes),
            "min_known_offtarget_probes": int(args.min_known_offtarget_probes),
            "min_offtarget_fraction_known": float(args.min_offtarget_fraction_known),
        },
        "n_input_genes": int(len(risk)),
        "n_resolved_genes": int(risk["resolved"].sum()),
        "n_unresolved_genes": int((~risk["resolved"]).sum()),
        "n_probe_feasible_symbol_level": int(risk["probe_feasible_symbol_level"].sum()),
        "n_annotation_rescue": int(risk["same_symbol_rescue_candidate"].sum()),
        "n_known_offtarget_risk": int(risk["known_offtarget_risk"].sum()),
        "n_clean_known_offtarget_risk": int(risk["clean_known_offtarget_risk"].sum()),
        "n_annotation_limited_failure": int(risk["annotation_limited_failure"].sum()),
        "n_low_probe_count_failure": int(risk["low_probe_count_failure"].sum()),
        "outputs": {
            "probe_risk_summary_tsv": str(risk_path),
            "known_offtarget_risk_genes_tsv": str(output_dir / "known_offtarget_risk_genes.tsv"),
            "clean_known_offtarget_risk_genes_tsv": str(output_dir / "clean_known_offtarget_risk_genes.tsv"),
            "annotation_rescue_genes_tsv": str(output_dir / "annotation_rescue_genes.tsv"),
            "class_summary_tsv": str(output_dir / "probe_feasibility_class_summary.tsv"),
            "rank_bin_summary_tsv": str(output_dir / "rank_bin_feasibility_summary.tsv"),
        },
        "batch_logs": batch_logs,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = read_ranked_panel(args.rank_tsv, args.top_n)
    panel.to_csv(output_dir / "selected_scrna_ranked_genes.tsv", sep="\t", index=False)
    (output_dir / "selected_scrna_ranked_genes.txt").write_text(
        "\n".join(panel["gene_symbol"].astype(str)) + "\n",
        encoding="utf-8",
    )

    probe_dir = output_dir / "probe_candidates"
    manifest_tsv = probe_dir / "probe_candidate_manifest.tsv"
    transcript_fasta = probe_dir / "probe_candidate_transcripts.fa"
    if args.force_manifest or not (manifest_tsv.exists() and transcript_fasta.exists()):
        manifest_result = build_local_probe_candidate_manifest(
            output_dir=probe_dir,
            species=args.species,
            panel=panel,
        )
    else:
        manifest_result = {
            "manifest_tsv": str(manifest_tsv),
            "transcript_fasta": str(transcript_fasta),
            "failures_json": str(probe_dir / "probe_candidate_failures.json"),
        }

    manifest = pd.read_csv(manifest_result["manifest_tsv"], sep="\t")
    batch_logs = run_probedealer_batches(
        manifest=manifest,
        transcript_fasta=Path(manifest_result["transcript_fasta"]),
        output_dir=output_dir,
        species=args.species,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        force=args.force_batches,
    )

    transcript_to_gene = load_transcript_to_gene(HUMAN_REFERENCE_DIR / "transcript_to_gene.tsv")
    gene_id_to_symbol = load_gene_symbols_from_h5ad(SOURCE_GENE_METADATA_H5AD)
    gene_id_to_symbol.update(
        dict(zip(manifest["gene_id"].astype(str), manifest["gene_symbol"].astype(str), strict=False))
    )
    gene_id_to_symbol.update(
        {
            "ENSG00000277101": "ARHGEF26",
            "ENSG00000285132": "CTSB",
            "ENSG00000292149": "TCF7L1",
            "ENSG00000281344": "HELLPAR",
        }
    )

    summarized = summarize_batches(
        manifest=manifest,
        output_dir=output_dir,
        transcript_to_gene=transcript_to_gene,
        gene_id_to_symbol=gene_id_to_symbol,
        batch_size=args.batch_size,
    )
    risk = panel.merge(summarized, on="gene_symbol", how="left")
    risk = classify_risk(risk, args)
    write_outputs(risk, output_dir, args, batch_logs)


if __name__ == "__main__":
    main()
