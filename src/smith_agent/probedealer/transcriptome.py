from __future__ import annotations

import subprocess
import urllib.parse
from pathlib import Path
from typing import Iterable

import requests

from .core import ProbeCandidate


BIOMART_URL = "https://www.ensembl.org/biomart/martservice"


def biomart_dataset_for_species(species: str) -> str:
    normalized = species.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "human": "hsapiens_gene_ensembl",
        "homo_sapiens": "hsapiens_gene_ensembl",
        "hsapiens": "hsapiens_gene_ensembl",
        "mouse": "mmusculus_gene_ensembl",
        "mus_musculus": "mmusculus_gene_ensembl",
        "mmusculus": "mmusculus_gene_ensembl",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported species `{species}` for ProbeDealer/OligoMiner transcriptome references. "
            "Supported species: homo_sapiens, mus_musculus."
        ) from exc


def reference_slug_for_species(species: str) -> str:
    dataset = biomart_dataset_for_species(species)
    if dataset == "hsapiens_gene_ensembl":
        return "human"
    if dataset == "mmusculus_gene_ensembl":
        return "mouse"
    raise AssertionError(f"Unhandled BioMart dataset: {dataset}")


def _build_biomart_query(
    transcript_ids: Iterable[str] | None = None,
    species: str = "mus_musculus",
) -> str:
    filter_xml = ""
    if transcript_ids is not None:
        joined = ",".join(transcript_ids)
        filter_xml = f'<Filter name = "ensembl_transcript_id" value = "{joined}"/>'
    dataset = biomart_dataset_for_species(species)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "1" count = "" datasetConfigVersion = "0.6" >
  <Dataset name = "{dataset}" interface = "default" >
    {filter_xml}
    <Attribute name = "cdna" />
    <Attribute name = "ensembl_transcript_id" />
    <Attribute name = "ensembl_gene_id" />
  </Dataset>
</Query>"""


def fetch_biomart_transcriptome(
    output_dir: str | Path,
    transcript_ids: Iterable[str] | None = None,
    species: str = "mus_musculus",
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    short_fasta = output_dir / "TxShortHeader.fa"
    mapping_tsv = output_dir / "transcript_to_gene.tsv"

    xml = _build_biomart_query(transcript_ids, species=species)
    url = BIOMART_URL + "?query=" + urllib.parse.quote(xml)
    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()

    with short_fasta.open("w") as fasta_handle, mapping_tsv.open("w") as mapping_handle:
        mapping_handle.write("transcript_id\tgene_id\n")
        line_iter = response.iter_lines(decode_unicode=True)
        header_line = next(line_iter)
        while header_line is not None and not header_line.strip():
            header_line = next(line_iter)

        if header_line is None:
            raise RuntimeError("Biomart returned no header line")

        header = header_line.split("\t")
        seq_idx = header.index("cDNA sequences")
        tx_idx = header.index("Transcript stable ID")
        gene_idx = header.index("Gene stable ID")

        for line in line_iter:
            if not line or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) <= max(seq_idx, tx_idx, gene_idx):
                continue
            seq = parts[seq_idx].strip().upper()
            transcript_id = parts[tx_idx].strip()
            gene_id = parts[gene_idx].strip()
            if not seq or not transcript_id or not gene_id:
                continue
            fasta_handle.write(f">{transcript_id}\n")
            for i in range(0, len(seq), 80):
                fasta_handle.write(seq[i : i + 80] + "\n")
            mapping_handle.write(f"{transcript_id}\t{gene_id}\n")

    return short_fasta, mapping_tsv


def load_transcript_to_gene(mapping_tsv: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with Path(mapping_tsv).open() as handle:
        next(handle)
        for line in handle:
            transcript_id, gene_id = line.rstrip("\n").split("\t")
            mapping[transcript_id] = gene_id
    return mapping


def make_blast_db(
    fasta_path: str | Path,
    db_prefix: str | Path,
    makeblastdb_path: str | Path,
) -> None:
    subprocess.run(
        [
            str(makeblastdb_path),
            "-in",
            str(fasta_path),
            "-dbtype",
            "nucl",
            "-parse_seqids",
            "-out",
            str(db_prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def filter_probes_by_transcriptome(
    probes: list[ProbeCandidate],
    transcript_to_gene: dict[str, str],
    blast_db_prefix: str | Path,
    blastn_path: str | Path,
    work_dir: str | Path,
) -> list[ProbeCandidate]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    query_fasta = work_dir / "probe_queries.fa"
    output_tsv = work_dir / "blast_hits.tsv"

    with query_fasta.open("w") as handle:
        for probe in probes:
            handle.write(f">{probe.header}\n{probe.sequence}\n")

    subprocess.run(
        [
            str(blastn_path),
            "-query",
            str(query_fasta),
            "-db",
            str(blast_db_prefix),
            "-outfmt",
            "6 qseqid sseqid",
            "-out",
            str(output_tsv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    hits_by_query: dict[str, set[str]] = {}
    if output_tsv.exists():
        with output_tsv.open() as handle:
            for line in handle:
                qseqid, sseqid = line.rstrip("\n").split("\t")
                gene_id = transcript_to_gene.get(sseqid)
                if gene_id is None:
                    continue
                hits_by_query.setdefault(qseqid, set()).add(gene_id)

    passed: list[ProbeCandidate] = []
    for probe in probes:
        gene_hits = hits_by_query.get(probe.header, set())
        if len(gene_hits) == 1:
            passed.append(probe)

    return passed
