#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT_DIR / "demo_outputs" / "mouse_100_gene_run"
MANIFEST = RUN_DIR / "mouse_100_genes_manifest.tsv"
TRANSCRIPT_FASTA = RUN_DIR / "mouse_100_genes_transcripts.fa"
PAINTSHOP_FASTA = RUN_DIR / "paintshop_pseudogenome.fa"
PAINTSHOP_GTF = RUN_DIR / "paintshop_pseudogenome.gtf"


def fetch_sequence(transcript_id: str) -> str:
    headers = {"Content-Type": "application/json"}
    r = requests.get(
        f"https://rest.ensembl.org/sequence/id/{transcript_id}?type=cdna",
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["seq"].upper()


def main() -> None:
    manifest = pd.read_csv(MANIFEST, sep="\t")
    seq_records: list[SeqRecord] = []
    with PAINTSHOP_GTF.open("w") as gtf_handle:
        for row in manifest.itertuples(index=False):
            seq = fetch_sequence(row.transcript_id)
            seq_records.append(
                SeqRecord(Seq(seq), id=row.transcript_id, description=row.transcript_id)
            )
            attrs = (
                f'gene_id "{row.gene_id}"; transcript_id "{row.transcript_id}"; '
                f'exon_number "1"; exon_id "{row.transcript_id}.1"; gene_name "{row.gene_symbol}";'
            )
            gtf_handle.write(
                f"{row.transcript_id}\tcustom\texon\t1\t{len(seq)}\t.\t+\t.\t{attrs}\n"
            )

    with TRANSCRIPT_FASTA.open("w") as handle:
        SeqIO.write(seq_records, handle, "fasta")
    with PAINTSHOP_FASTA.open("w") as handle:
        SeqIO.write(seq_records, handle, "fasta")

    print(TRANSCRIPT_FASTA)
    print(PAINTSHOP_FASTA)
    print(PAINTSHOP_GTF)


if __name__ == "__main__":
    main()
