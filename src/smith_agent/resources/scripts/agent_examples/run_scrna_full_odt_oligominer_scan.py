#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from Bio import SeqIO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from smith_agent.adapters.feasibility_backends import run_oligominer_specificity_screen
from scripts.run_top128_feasibility_filtering import (
    HUMAN_REFERENCE_DIR,
    ODT_PYTHON,
    SMITH_PACKAGE_ROOT,
    fastq_ids,
    load_transcript_to_gene,
    parse_two_column_hits,
    summarize_gene_aware_hits,
    write_local_odt_runner,
)


DEFAULT_RANK_TSV = PROJECT_ROOT / "outputs/liver_merfish_benchmark/liver_source_gene_rank.tsv"
DEFAULT_PROBEDEALER_SCAN_DIR = PROJECT_ROOT / "outputs/scrna_probedealer_full_gene_scan"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/scrna_full_three_tool_feasibility"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full scRNA ODT and OligoMiner scans, then merge them with the "
            "existing full scRNA ProbeDealer scan into a three-tool feasibility table."
        )
    )
    parser.add_argument("--rank-tsv", default=str(DEFAULT_RANK_TSV))
    parser.add_argument("--probedealer-scan-dir", default=str(DEFAULT_PROBEDEALER_SCAN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--species", default="homo_sapiens")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["odt", "oligominer", "merge"],
        choices=["odt", "oligominer", "merge"],
    )
    parser.add_argument("--odt-batch-size", type=int, default=32)
    parser.add_argument("--odt-inner-batch-size", type=int, default=8)
    parser.add_argument("--odt-max-workers", type=int, default=4)
    parser.add_argument("--oligominer-batch-size", type=int, default=256)
    parser.add_argument("--oligominer-max-workers", type=int, default=2)
    parser.add_argument("--oligominer-temperature-c", type=int, default=42)
    parser.add_argument("--min-property-probes", type=int, default=20)
    parser.add_argument("--min-specific-probes", type=int, default=10)
    parser.add_argument("--min-deployment-probes", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all batches; useful for debugging.")
    parser.add_argument("--force-odt", action="store_true")
    parser.add_argument("--force-oligominer", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def read_ranked_panel(rank_tsv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(rank_tsv, sep="\t")
    if "gene_symbol" not in df.columns:
        raise ValueError(f"Rank file must contain `gene_symbol`: {rank_tsv}")
    df = df.copy()
    if "rank" not in df.columns:
        df.insert(0, "rank", range(1, len(df) + 1))
    return df[["rank", "gene_symbol"]].copy()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def summary_path_for_steps(output_dir: Path, steps: list[str]) -> Path:
    if "merge" in steps:
        return output_dir / "run_summary.json"
    return output_dir / f"run_summary_{'_'.join(steps)}.json"


def valid_tsv(path: Path, expected_rows: int, required_cols: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return False
    return len(df) == expected_rows and required_cols.issubset(df.columns)


def manifest_batches(manifest: pd.DataFrame, batch_size: int, max_batches: int = 0) -> list[tuple[int, pd.DataFrame]]:
    batches = [
        (idx, manifest.iloc[start : start + batch_size].copy())
        for idx, start in enumerate(range(0, len(manifest), batch_size))
    ]
    if max_batches and max_batches > 0:
        return batches[:max_batches]
    return batches


def write_batch_fasta(batch: pd.DataFrame, transcript_fasta: Path, output_fasta: Path) -> None:
    wanted = set(batch["transcript_id"].astype(str))
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with output_fasta.open("w", encoding="utf-8") as handle:
        SeqIO.write(
            [record for record in SeqIO.parse(str(transcript_fasta), "fasta") if record.id in wanted],
            handle,
            "fasta",
        )


def run_odt_batch(
    runner_path: Path,
    manifest_tsv: Path,
    transcript_fasta: Path,
    output_dir: Path,
    inner_batch_size: int,
    max_workers: int,
    set_size_min: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "odt_batch.log"
    cmd = [
        str(ODT_PYTHON),
        str(runner_path),
        "--manifest",
        str(manifest_tsv.resolve()),
        "--transcript-fasta",
        str(transcript_fasta.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--batch-size",
        str(inner_batch_size),
        "--max-workers",
        str(max_workers),
        "--set-size-min",
        str(set_size_min),
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, cwd=PROJECT_ROOT)
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": int(proc.returncode),
        "seconds": round(time.time() - started, 3),
        "log": str(log_path),
        "summary_tsv": str(output_dir / "property_only_summary.tsv"),
    }


def run_odt_batches(
    manifest: pd.DataFrame,
    transcript_fasta: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    odt_root = output_dir / "odt_property_batches"
    odt_root.mkdir(parents=True, exist_ok=True)
    runner_path = odt_root / "run_local_odt_property.py"
    write_local_odt_runner(runner_path)

    logs: list[dict[str, Any]] = []
    required_cols = {
        "gene",
        "status",
        "transcript_id",
        "candidate_oligos_initial",
        "candidate_oligos_after_property_filters",
        "feasible_property_only",
    }
    batches = manifest_batches(manifest, args.odt_batch_size, args.max_batches)
    for idx, batch in batches:
        batch_dir = odt_root / f"batch_{idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_manifest = batch_dir / "batch_manifest.tsv"
        summary_path = batch_dir / "property_only_summary.tsv"
        batch.to_csv(batch_manifest, sep="\t", index=False)

        if not args.force_odt and valid_tsv(summary_path, len(batch), required_cols):
            result = {
                "batch_index": idx,
                "n_genes": int(len(batch)),
                "status": "skipped_existing",
                "seconds": 0.0,
                "summary_tsv": str(summary_path),
            }
        else:
            result = run_odt_batch(
                runner_path=runner_path,
                manifest_tsv=batch_manifest,
                transcript_fasta=transcript_fasta,
                output_dir=batch_dir,
                inner_batch_size=args.odt_inner_batch_size,
                max_workers=args.odt_max_workers,
                set_size_min=2,
            )
            result.update({"batch_index": idx, "n_genes": int(len(batch))})

        logs.append(result)
        print(
            f"ODT batch {idx + 1}/{(len(manifest) + args.odt_batch_size - 1) // args.odt_batch_size}: "
            f"{result['status']}, genes={len(batch)}, seconds={result['seconds']}",
            flush=True,
        )
        if result["status"] == "error" and not args.continue_on_error:
            raise RuntimeError(f"ODT batch {idx} failed; see {result.get('log')}")

    merge_odt_summaries(manifest, odt_root, args.odt_batch_size, output_dir / "odt_property_summary.tsv")
    return logs


def _run_oligominer_worker(
    batch_idx: int,
    batch_fasta: str,
    batch_dir: str,
    species: str,
    temperature_c: int,
) -> dict[str, Any]:
    started = time.time()
    result = run_oligominer_specificity_screen(
        package_root=SMITH_PACKAGE_ROOT,
        transcript_fasta=batch_fasta,
        output_dir=batch_dir,
        temperature_c=temperature_c,
        species=species,
    )
    return {
        "batch_index": int(batch_idx),
        "status": str(result.get("status", "unknown")),
        "seconds": round(time.time() - started, 3),
        "result": result,
        "summary_tsv": str(Path(batch_dir) / "oligominer_specificity_summary.tsv"),
    }


def run_oligominer_batches(
    manifest: pd.DataFrame,
    transcript_fasta: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    oligominer_root = output_dir / "oligominer_batches"
    oligominer_root.mkdir(parents=True, exist_ok=True)
    required_cols = {"transcript_id", "candidate_probe_count", "specific_probe_count"}
    batches = manifest_batches(manifest, args.oligominer_batch_size, args.max_batches)

    pending: list[tuple[int, pd.DataFrame, Path, Path]] = []
    logs: list[dict[str, Any]] = []
    for idx, batch in batches:
        batch_dir = oligominer_root / f"batch_{idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_manifest = batch_dir / "batch_manifest.tsv"
        batch_fasta = batch_dir / "batch_transcripts.fa"
        summary_path = batch_dir / "oligominer_specificity_summary.tsv"
        batch.to_csv(batch_manifest, sep="\t", index=False)
        if not batch_fasta.exists() or args.force_oligominer:
            write_batch_fasta(batch, transcript_fasta, batch_fasta)
        if not args.force_oligominer and valid_tsv(summary_path, len(batch), required_cols):
            logs.append(
                {
                    "batch_index": idx,
                    "n_genes": int(len(batch)),
                    "status": "skipped_existing",
                    "seconds": 0.0,
                    "summary_tsv": str(summary_path),
                }
            )
        else:
            pending.append((idx, batch, batch_fasta, batch_dir))

    total_batches = (len(manifest) + args.oligominer_batch_size - 1) // args.oligominer_batch_size
    with ProcessPoolExecutor(max_workers=args.oligominer_max_workers) as executor:
        futures = {
            executor.submit(
                _run_oligominer_worker,
                idx,
                str(batch_fasta.resolve()),
                str(batch_dir.resolve()),
                args.species,
                args.oligominer_temperature_c,
            ): (idx, len(batch))
            for idx, batch, batch_fasta, batch_dir in pending
        }
        for future in as_completed(futures):
            idx, n_genes = futures[future]
            try:
                result = future.result()
                result["n_genes"] = int(n_genes)
            except Exception as exc:
                result = {
                    "batch_index": idx,
                    "n_genes": int(n_genes),
                    "status": "error",
                    "seconds": None,
                    "error": str(exc),
                }
            logs.append(result)
            print(
                f"OligoMiner batch {idx + 1}/{total_batches}: "
                f"{result['status']}, genes={n_genes}, seconds={result.get('seconds')}",
                flush=True,
            )
            if result["status"] == "error" and not args.continue_on_error:
                raise RuntimeError(f"OligoMiner batch {idx} failed: {result.get('error') or result.get('result')}")

    merge_oligominer_summaries(
        manifest,
        oligominer_root,
        args.oligominer_batch_size,
        output_dir / "oligominer_specificity_summary.tsv",
    )
    build_full_oligominer_gene_aware_summary(
        manifest,
        oligominer_root,
        args.oligominer_batch_size,
        output_dir / "oligominer_geneaware_summary.tsv",
    )
    return sorted(logs, key=lambda item: item["batch_index"])


def merge_odt_summaries(manifest: pd.DataFrame, odt_root: Path, batch_size: int, output_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for idx, batch in manifest_batches(manifest, batch_size):
        path = odt_root / f"batch_{idx:04d}" / "property_only_summary.tsv"
        if path.exists():
            frames.append(pd.read_csv(path, sep="\t"))
    if frames:
        merged = pd.concat(frames, ignore_index=True)
    else:
        merged = pd.DataFrame(
            columns=[
                "gene",
                "status",
                "reason",
                "transcript_id",
                "candidate_oligos_initial",
                "candidate_oligos_after_property_filters",
                "feasible_property_only",
            ]
        )
    order = manifest[["gene_symbol"]].assign(_order=range(len(manifest)))
    merged = (
        merged.merge(order, left_on="gene", right_on="gene_symbol", how="left")
        .sort_values("_order")
        .drop(columns=["gene_symbol", "_order"])
    )
    merged.to_csv(output_path, sep="\t", index=False)
    return merged


def merge_oligominer_summaries(
    manifest: pd.DataFrame,
    oligominer_root: Path,
    batch_size: int,
    output_path: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for idx, _batch in manifest_batches(manifest, batch_size):
        path = oligominer_root / f"batch_{idx:04d}" / "oligominer_specificity_summary.tsv"
        if path.exists():
            frames.append(pd.read_csv(path, sep="\t"))
    if frames:
        merged = pd.concat(frames, ignore_index=True)
    else:
        merged = pd.DataFrame(columns=["transcript_id", "candidate_probe_count", "specific_probe_count"])
    order = manifest[["transcript_id"]].assign(_order=range(len(manifest)))
    merged = merged.merge(order, on="transcript_id", how="left").sort_values("_order").drop(columns=["_order"])
    merged.to_csv(output_path, sep="\t", index=False)
    return merged


def build_full_oligominer_gene_aware_summary(
    manifest: pd.DataFrame,
    oligominer_root: Path,
    batch_size: int,
    output_path: Path,
) -> pd.DataFrame:
    transcript_to_gene = load_transcript_to_gene(HUMAN_REFERENCE_DIR / "transcript_to_gene.tsv")
    rows: list[dict[str, Any]] = []
    for idx, batch in manifest_batches(manifest, batch_size):
        batch_dir = oligominer_root / f"batch_{idx:04d}"
        for row in batch.itertuples(index=False):
            transcript_id = str(row.transcript_id)
            transcript_dir = batch_dir / transcript_id
            query_ids = fastq_ids(transcript_dir / f"{transcript_id}.fastq")
            hits = parse_two_column_hits(transcript_dir / f"{transcript_id}.sam", transcript_to_gene)
            summary = summarize_gene_aware_hits(query_ids, hits, str(row.gene_id))
            rows.append({"transcript_id": transcript_id, **summary})
    result = pd.DataFrame(rows)
    result.to_csv(output_path, sep="\t", index=False)
    return result


def failure_reason(row: Any) -> str:
    reasons: list[str] = []
    if not bool(row.transcript_resolved):
        reasons.append("transcript_unresolved")
    if not bool(row.pass_odt_property_20):
        reasons.append("low_odt_property_probe_count")
    if not bool(row.pass_oligominer_geneaware_10):
        reasons.append("low_oligominer_geneaware_specificity")
    if not bool(row.pass_probedealer_target_20):
        reasons.append("low_probedealer_target_probe_count")
    return ";".join(reasons) if reasons else "pass"


def merge_three_tool_table(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    rank = read_ranked_panel(args.rank_tsv)
    probedealer_scan_dir = Path(args.probedealer_scan_dir).resolve()
    manifest = pd.read_csv(
        probedealer_scan_dir / "probe_candidates" / "probe_candidate_manifest.tsv",
        sep="\t",
    )
    odt = pd.read_csv(output_dir / "odt_property_summary.tsv", sep="\t").rename(
        columns={
            "gene": "gene_symbol",
            "candidate_oligos_initial": "odt_initial_probe_count",
            "candidate_oligos_after_property_filters": "odt_property_probe_count",
            "feasible_property_only": "odt_feasible_property_only",
        }
    )
    oligominer = pd.read_csv(output_dir / "oligominer_specificity_summary.tsv", sep="\t").rename(
        columns={
            "candidate_probe_count": "oligominer_candidate_probe_count",
            "specific_probe_count": "oligominer_strict_specific_probe_count",
        }
    )
    oligominer_geneaware = pd.read_csv(output_dir / "oligominer_geneaware_summary.tsv", sep="\t").rename(
        columns={
            "target_gene_hit_probe_count": "oligominer_target_gene_hit_probe_count",
            "cross_gene_probe_count": "oligominer_cross_gene_probe_count",
            "off_target_only_probe_count": "oligominer_off_target_only_probe_count",
            "no_hit_probe_count": "oligominer_no_hit_probe_count",
            "gene_aware_specific_probe_count": "oligominer_geneaware_specific_probe_count",
        }
    )
    probedealer = pd.read_csv(probedealer_scan_dir / "probe_risk_summary.tsv", sep="\t").rename(
        columns={
            "initial_probe_count": "probedealer_initial_probe_count",
            "geneid_target_only_probe_count": "probedealer_geneid_target_probe_count",
            "symbol_target_only_probe_count_known": "probedealer_target_final_probe_count",
            "probes_with_known_different_symbol": "probedealer_known_offtarget_probe_count",
            "unknown_symbol_probe_count": "probedealer_unknown_symbol_probe_count",
            "no_hit_probe_count": "probedealer_no_hit_probe_count",
        }
    )

    table = rank.merge(manifest, on="gene_symbol", how="left")
    table["transcript_resolved"] = table["transcript_id"].notna()
    table = table.merge(
        odt[
            [
                "gene_symbol",
                "status",
                "reason",
                "transcript_id",
                "odt_initial_probe_count",
                "odt_property_probe_count",
                "odt_feasible_property_only",
            ]
        ].rename(
            columns={
                "status": "odt_status",
                "reason": "odt_reason",
                "transcript_id": "odt_transcript_id",
            }
        ),
        on="gene_symbol",
        how="left",
    )
    table = table.merge(oligominer, on="transcript_id", how="left")
    table = table.merge(oligominer_geneaware, on="transcript_id", how="left")
    table = table.merge(
        probedealer[
            [
                "gene_symbol",
                "probedealer_initial_probe_count",
                "probedealer_geneid_target_probe_count",
                "probedealer_target_final_probe_count",
                "probedealer_known_offtarget_probe_count",
                "probedealer_unknown_symbol_probe_count",
                "probedealer_no_hit_probe_count",
                "probe_feasibility_class",
            ]
        ],
        on="gene_symbol",
        how="left",
    )

    numeric_cols = [
        "odt_initial_probe_count",
        "odt_property_probe_count",
        "oligominer_candidate_probe_count",
        "oligominer_strict_specific_probe_count",
        "oligominer_target_gene_hit_probe_count",
        "oligominer_cross_gene_probe_count",
        "oligominer_off_target_only_probe_count",
        "oligominer_no_hit_probe_count",
        "oligominer_geneaware_specific_probe_count",
        "probedealer_initial_probe_count",
        "probedealer_geneid_target_probe_count",
        "probedealer_target_final_probe_count",
        "probedealer_known_offtarget_probe_count",
        "probedealer_unknown_symbol_probe_count",
        "probedealer_no_hit_probe_count",
    ]
    for col in numeric_cols:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0).astype(int)

    table["pass_odt_property_20"] = table["odt_property_probe_count"] >= args.min_property_probes
    table["pass_oligominer_strict_10"] = table["oligominer_strict_specific_probe_count"] >= args.min_specific_probes
    table["pass_oligominer_geneaware_10"] = (
        table["oligominer_geneaware_specific_probe_count"] >= args.min_specific_probes
    )
    table["pass_probedealer_target_20"] = table["probedealer_target_final_probe_count"] >= args.min_deployment_probes
    table["pass_three_tool_feasibility"] = (
        table["transcript_resolved"]
        & table["pass_odt_property_20"]
        & table["pass_oligominer_geneaware_10"]
        & table["pass_probedealer_target_20"]
    )
    table["primary_failure_reason"] = [failure_reason(row) for row in table.itertuples(index=False)]

    table_path = output_dir / "three_tool_feasibility_table.tsv"
    table.to_csv(table_path, sep="\t", index=False)

    overlap_cols = ["pass_odt_property_20", "pass_oligominer_geneaware_10", "pass_probedealer_target_20"]
    overlap = (
        table.groupby(overlap_cols, dropna=False)
        .size()
        .reset_index(name="gene_count")
        .sort_values("gene_count", ascending=False)
    )
    overlap["tool_pattern"] = overlap.apply(
        lambda row: "+".join(
            name
            for name, passed in [
                ("ODT", row["pass_odt_property_20"]),
                ("OligoMiner", row["pass_oligominer_geneaware_10"]),
                ("ProbeDealer", row["pass_probedealer_target_20"]),
            ]
            if bool(passed)
        )
        or "none",
        axis=1,
    )
    overlap.to_csv(output_dir / "three_tool_overlap_counts.tsv", sep="\t", index=False)

    pass_summary = pd.DataFrame(
        [
            {"gate": "transcript_resolved", "pass_count": int(table["transcript_resolved"].sum()), "total_count": int(len(table))},
            {"gate": "ODT_property_ge20", "pass_count": int(table["pass_odt_property_20"].sum()), "total_count": int(len(table))},
            {
                "gate": "OligoMiner_strict_specific_ge10",
                "pass_count": int(table["pass_oligominer_strict_10"].sum()),
                "total_count": int(len(table)),
            },
            {
                "gate": "OligoMiner_geneaware_specific_ge10",
                "pass_count": int(table["pass_oligominer_geneaware_10"].sum()),
                "total_count": int(len(table)),
            },
            {
                "gate": "ProbeDealer_target_final_ge20",
                "pass_count": int(table["pass_probedealer_target_20"].sum()),
                "total_count": int(len(table)),
            },
            {
                "gate": "three_tool_feasibility",
                "pass_count": int(table["pass_three_tool_feasibility"].sum()),
                "total_count": int(len(table)),
            },
        ]
    )
    pass_summary.to_csv(output_dir / "tool_pass_summary.tsv", sep="\t", index=False)
    return table


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    probedealer_scan_dir = Path(args.probedealer_scan_dir).resolve()
    manifest_tsv = probedealer_scan_dir / "probe_candidates" / "probe_candidate_manifest.tsv"
    transcript_fasta = probedealer_scan_dir / "probe_candidates" / "probe_candidate_transcripts.fa"
    if not manifest_tsv.exists() or not transcript_fasta.exists():
        raise FileNotFoundError(f"Missing full scRNA ProbeDealer manifest/FASTA under {probedealer_scan_dir}")
    if not (probedealer_scan_dir / "probe_risk_summary.tsv").exists():
        raise FileNotFoundError(f"Missing full scRNA ProbeDealer risk summary under {probedealer_scan_dir}")

    manifest = pd.read_csv(manifest_tsv, sep="\t")
    run_summary: dict[str, Any] = {
        "rank_tsv": str(Path(args.rank_tsv).resolve()),
        "probedealer_scan_dir": str(probedealer_scan_dir),
        "n_resolved_transcripts": int(len(manifest)),
        "steps_requested": args.steps,
        "thresholds": {
            "min_property_probes": int(args.min_property_probes),
            "min_specific_probes": int(args.min_specific_probes),
            "min_deployment_probes": int(args.min_deployment_probes),
        },
        "outputs": {},
    }

    if "odt" in args.steps:
        start = time.time()
        odt_logs = run_odt_batches(manifest, transcript_fasta, output_dir, args)
        run_summary["odt"] = {"seconds": round(time.time() - start, 3), "batch_logs": odt_logs}
        run_summary["outputs"]["odt_property_summary_tsv"] = str(output_dir / "odt_property_summary.tsv")
        write_json(output_dir / "run_summary_odt.partial.json", run_summary)

    if "oligominer" in args.steps:
        start = time.time()
        oligominer_logs = run_oligominer_batches(manifest, transcript_fasta, output_dir, args)
        run_summary["oligominer"] = {"seconds": round(time.time() - start, 3), "batch_logs": oligominer_logs}
        run_summary["outputs"]["oligominer_specificity_summary_tsv"] = str(output_dir / "oligominer_specificity_summary.tsv")
        run_summary["outputs"]["oligominer_geneaware_summary_tsv"] = str(output_dir / "oligominer_geneaware_summary.tsv")
        write_json(output_dir / "run_summary_oligominer.partial.json", run_summary)

    if "merge" in args.steps:
        table = merge_three_tool_table(args, output_dir)
        run_summary["merged"] = {
            "n_input_genes": int(len(table)),
            "n_transcript_resolved": int(table["transcript_resolved"].sum()),
            "n_odt_pass": int(table["pass_odt_property_20"].sum()),
            "n_oligominer_geneaware_pass": int(table["pass_oligominer_geneaware_10"].sum()),
            "n_probedealer_pass": int(table["pass_probedealer_target_20"].sum()),
            "n_three_tool_pass": int(table["pass_three_tool_feasibility"].sum()),
        }
        run_summary["outputs"]["three_tool_feasibility_table_tsv"] = str(output_dir / "three_tool_feasibility_table.tsv")
        run_summary["outputs"]["tool_pass_summary_tsv"] = str(output_dir / "tool_pass_summary.tsv")
        run_summary["outputs"]["three_tool_overlap_counts_tsv"] = str(output_dir / "three_tool_overlap_counts.tsv")

    write_json(summary_path_for_steps(output_dir, args.steps), run_summary)
    print(json.dumps(run_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
