#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smith_agent.feasibility import (
    ODTScrinshotBackend,
    OligoMinerBackend,
    PaintSHOPBackend,
    ProbeDealerBackend,
)


GENES = [
    "Actb",
    "Gad1",
    "Aqp4",
    "Aldoc",
    "Mbp",
    "Mobp",
    "Slc17a7",
    "Pcp4",
    "Snap25",
    "Gfap",
]


def fetch_mouse_canonical_transcript(gene_symbol: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    lookup = requests.get(
        f"https://rest.ensembl.org/lookup/symbol/mus_musculus/{gene_symbol}?expand=1",
        headers=headers,
        timeout=60,
    )
    lookup.raise_for_status()
    data = lookup.json()
    transcripts = data.get("Transcript") or []
    canonical = [tx for tx in transcripts if tx.get("is_canonical") == 1]
    transcript = canonical[0] if canonical else transcripts[0]
    transcript_id = transcript["id"]

    seq_resp = requests.get(
        f"https://rest.ensembl.org/sequence/id/{transcript_id}?type=cdna",
        headers=headers,
        timeout=60,
    )
    seq_resp.raise_for_status()
    seq = seq_resp.json()["seq"].upper()

    return {
        "gene_symbol": gene_symbol,
        "gene_id": data["id"],
        "transcript_id": transcript_id,
        "sequence": seq,
    }


def write_transcript_fasta(records: list[dict[str, str]], fasta_path: Path) -> None:
    seq_records = []
    for record in records:
        seq_records.append(
            SeqRecord(
                seq=Seq(record["sequence"]),
                id=record["transcript_id"],
                description=f"gene_symbol={record['gene_symbol']} gene_id={record['gene_id']}",
            )
        )
    with fasta_path.open("w") as handle:
        SeqIO.write(seq_records, handle, "fasta")


def write_paintshop_pseudogenome(records: list[dict[str, str]], genome_fasta: Path, gtf_path: Path) -> None:
    write_transcript_fasta(records, genome_fasta)
    with gtf_path.open("w") as handle:
        for record in records:
            seqid = record["transcript_id"]
            seq_len = len(record["sequence"])
            gene_id = record["gene_id"]
            transcript_id = record["transcript_id"]
            gene_name = record["gene_symbol"]
            attrs = (
                f'gene_id "{gene_id}"; transcript_id "{transcript_id}"; '
                f'exon_number "1"; exon_id "{transcript_id}.1"; gene_name "{gene_name}";'
            )
            handle.write(
                f"{seqid}\tcustom\texon\t1\t{seq_len}\t.\t+\t.\t{attrs}\n"
            )


def main() -> None:
    output_root = ROOT_DIR / "demo_outputs" / "mouse_10_gene_all_tools"
    output_root.mkdir(parents=True, exist_ok=True)

    records = [fetch_mouse_canonical_transcript(gene) for gene in GENES]

    transcript_fasta = output_root / "mouse_10_genes_transcripts.fa"
    write_transcript_fasta(records, transcript_fasta)

    gene_manifest = output_root / "mouse_10_genes_manifest.tsv"
    with gene_manifest.open("w") as handle:
        handle.write("gene_symbol\tgene_id\ttranscript_id\tsequence_length\n")
        for record in records:
            handle.write(
                f"{record['gene_symbol']}\t{record['gene_id']}\t{record['transcript_id']}\t{len(record['sequence'])}\n"
            )

    pseudo_genome_fa = output_root / "paintshop_pseudogenome.fa"
    pseudo_gtf = output_root / "paintshop_pseudogenome.gtf"
    write_paintshop_pseudogenome(records, pseudo_genome_fa, pseudo_gtf)

    results: dict[str, dict] = {
        "example_genes": {
            "genes": GENES,
            "manifest_tsv": str(gene_manifest),
            "transcript_fasta": str(transcript_fasta),
            "paintshop_genome_fasta": str(pseudo_genome_fa),
            "paintshop_gtf": str(pseudo_gtf),
        }
    }

    results["odt_scrinshot"] = ODTScrinshotBackend().run_gene_symbols_property_only(
        genes=GENES,
        species="mus_musculus",
        output_dir=output_root / "odt_scrinshot_property_only",
        set_size_min=2,
    ).to_dict()

    results["oligominer"] = OligoMinerBackend().run_multi_transcript_fasta(
        fasta_path=transcript_fasta,
        output_dir=output_root / "oligominer",
    ).to_dict()

    results["probedealer_py"] = ProbeDealerBackend().run_transcript_fasta(
        fasta_path=transcript_fasta,
        output_dir=output_root / "probedealer_py",
    ).to_dict()

    results["paintshop"] = PaintSHOPBackend().run_custom(
        assembly="mouse_10_gene_demo",
        genome_fasta=pseudo_genome_fa,
        annotation_file=pseudo_gtf,
        work_dir=output_root / "paintshop",
        cores=1,
    ).to_dict()

    output_json = output_root / "all_tools_results.json"
    output_json.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"\nWrote: {output_json}")


if __name__ == "__main__":
    main()
