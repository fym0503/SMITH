#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smith_agent.probedealer import OligoDesignConfig, build_oligo_array, load_fasta_records
from smith_agent.probedealer.transcriptome import (
    fetch_biomart_transcriptome,
    filter_probes_by_transcriptome,
    load_transcript_to_gene,
    make_blast_db,
)


DEFAULT_FASTA = Path(
    "third_party/probedealer_examples/ExampleTargetFiles/sequential RNA FISH/"
    "sequential RNA FISH example input_mouse_gencode vM25.fasta"
)
DEFAULT_SUPPLEMENT = Path("41598_2020_76439_MOESM2_ESM123.xlsx")
BLASTN_PATH = ROOT_DIR / "third_party/envs/probedealer_blast/bin/blastn"
MAKEBLASTDB_PATH = ROOT_DIR / "third_party/envs/probedealer_blast/bin/makeblastdb"
DEFAULT_REF_DIR = ROOT_DIR / "third_party/reference_data/probedealer_supplement_tx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Python ProbeDealer-like initial/final counts against supplementary counts."
    )
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--output", type=Path, default=Path("probedealer_final_compare.tsv"))
    parser.add_argument(
        "--full-transcriptome",
        action="store_true",
        help="Fetch all mouse transcripts from Ensembl Biomart instead of only supplement transcript IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supplement = pd.read_excel(args.supplement).rename(
        columns={
            "Transcript ID": "transcript_id",
            "ProbeDealer_initial probe count": "supp_probe_count",
            "ProbeDealer_final probe count after transcriptome BLAST": "supp_final_probe_count",
            "FPKM": "supp_fpkm",
        }
    )
    transcript_ids = None
    if not args.full_transcriptome:
        transcript_ids = supplement["transcript_id"].dropna().astype(str).tolist()

    short_fasta = args.ref_dir / "TxShortHeader.fa"
    mapping_tsv = args.ref_dir / "transcript_to_gene.tsv"
    if not short_fasta.exists() or not mapping_tsv.exists():
        print(f"Fetching transcriptome into {args.ref_dir} ...", flush=True)
        short_fasta, mapping_tsv = fetch_biomart_transcriptome(args.ref_dir, transcript_ids=transcript_ids)

    db_prefix = args.ref_dir / "TxShortHeader"
    if not (args.ref_dir / "TxShortHeader.nsq").exists():
        print(f"Building BLAST db at {db_prefix} ...", flush=True)
        make_blast_db(short_fasta, db_prefix, MAKEBLASTDB_PATH)
    transcript_to_gene = load_transcript_to_gene(mapping_tsv)

    records = load_fasta_records(args.fasta)
    designed = build_oligo_array(records, OligoDesignConfig())

    py_initial_counts: list[int] = []
    py_final_counts: list[int] = []
    for transcript_id, _seq in records:
        print(f"Filtering transcript {transcript_id} ...", flush=True)
        probes = designed[transcript_id]
        py_initial_counts.append(len(probes))
        filtered = filter_probes_by_transcriptome(
            probes,
            transcript_to_gene=transcript_to_gene,
            blast_db_prefix=db_prefix,
            blastn_path=BLASTN_PATH,
            work_dir=args.ref_dir / f"blast_{transcript_id}",
        )
        py_final_counts.append(len(filtered))

    observed = pd.DataFrame(
        {
            "transcript_id": [sequence_id for sequence_id, _ in records],
            "py_probe_count": py_initial_counts,
            "py_final_probe_count": py_final_counts,
        }
    )

    merged = observed.merge(supplement, how="left", on="transcript_id")
    merged["count_delta"] = merged["py_probe_count"] - merged["supp_probe_count"]
    merged["final_count_delta"] = merged["py_final_probe_count"] - merged["supp_final_probe_count"]
    merged["abs_count_delta"] = merged["count_delta"].abs()
    merged["abs_final_count_delta"] = merged["final_count_delta"].abs()
    merged = merged[
        [
            "transcript_id",
            "supp_fpkm",
            "py_probe_count",
            "supp_probe_count",
            "count_delta",
            "py_final_probe_count",
            "supp_final_probe_count",
            "final_count_delta",
            "abs_count_delta",
            "abs_final_count_delta",
        ]
    ].sort_values("transcript_id")

    args.output.write_text(merged.to_csv(sep="\t", index=False))

    comparable = merged.dropna(subset=["supp_probe_count", "supp_final_probe_count"])
    print(f"Initial MAE: {comparable['abs_count_delta'].mean():.2f}")
    print(f"Final MAE: {comparable['abs_final_count_delta'].mean():.2f}")
    print(f"Wrote: {args.output}")
    print()
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
