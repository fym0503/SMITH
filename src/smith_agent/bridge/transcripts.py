from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


REST_ROOT = "https://rest.ensembl.org"
CACHE_PATH = Path(__file__).resolve().parents[3] / ".cache" / "transcript_resolution.json"


@dataclass(frozen=True)
class TranscriptRecord:
    gene_symbol: str
    gene_id: str
    transcript_id: str
    sequence: str

    @property
    def sequence_length(self) -> int:
        return len(self.sequence)


def _get_json(url: str) -> dict:
    request = Request(url, headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urlopen(request, timeout=60) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _write_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")


def _lookup_symbol(species: str, symbol: str) -> dict:
    url = f"{REST_ROOT}/lookup/symbol/{quote(species)}/{quote(symbol)}?expand=1"
    return _get_json(url)


def _sequence_for_transcript(transcript_id: str) -> str:
    url = f"{REST_ROOT}/sequence/id/{quote(transcript_id)}?type=cdna"
    payload = _get_json(url)
    sequence = str(payload.get("seq", "")).upper()
    if not sequence:
        raise ValueError(f"No cDNA sequence returned for transcript `{transcript_id}`.")
    return sequence


def _symbol_variants(symbol: str, species: str) -> list[str]:
    variants = [symbol]
    compact = symbol.strip()
    if compact not in variants:
        variants.append(compact)
    upper = compact.upper()
    if upper not in variants:
        variants.append(upper)
    title = compact.title()
    if title not in variants:
        variants.append(title)
    if species.lower() in {"mus_musculus", "mouse"}:
        mouse_style = compact[:1].upper() + compact[1:].lower() if compact else compact
        if mouse_style not in variants:
            variants.append(mouse_style)
    return [item for item in variants if item]


def fetch_canonical_transcript(symbol: str, species: str) -> TranscriptRecord:
    cache = _load_cache()
    cache_key = f"{species}::{symbol.strip()}"
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and cached.get("transcript_id") and cached.get("sequence"):
        return TranscriptRecord(
            gene_symbol=str(cached["gene_symbol"]),
            gene_id=str(cached["gene_id"]),
            transcript_id=str(cached["transcript_id"]),
            sequence=str(cached["sequence"]),
        )

    errors: list[str] = []
    for candidate_symbol in _symbol_variants(symbol, species):
        try:
            payload = _lookup_symbol(species, candidate_symbol)
        except (HTTPError, URLError, ValueError) as exc:  # noqa: PERF203
            errors.append(f"{candidate_symbol}: {exc}")
            continue
        transcripts = payload.get("Transcript") or []
        if not transcripts:
            errors.append(f"{candidate_symbol}: no transcripts")
            continue
        canonical = [tx for tx in transcripts if tx.get("is_canonical") == 1]
        transcript = canonical[0] if canonical else max(
            transcripts,
            key=lambda tx: abs(int(tx.get("end", 0)) - int(tx.get("start", 0))),
        )
        transcript_id = str(transcript["id"])
        sequence = _sequence_for_transcript(transcript_id)
        record = TranscriptRecord(
            gene_symbol=candidate_symbol,
            gene_id=str(payload.get("id", "")),
            transcript_id=transcript_id,
            sequence=sequence,
        )
        cache[cache_key] = {
            "gene_symbol": record.gene_symbol,
            "gene_id": record.gene_id,
            "transcript_id": record.transcript_id,
            "sequence": record.sequence,
        }
        _write_cache(cache)
        return record
    raise ValueError(f"Failed to resolve transcript for `{symbol}` in `{species}`. Tried: {errors}")


def panel_genes_from_file(panel_path: str | Path, panel_size: int = 64) -> list[str]:
    path = Path(panel_path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.iloc[:panel_size, 0].astype(str).tolist()
    genes = [line.strip().split(",")[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return genes[:panel_size]


def build_probe_candidate_manifest(
    output_dir: str | Path,
    species: str,
    genes: Iterable[str] | None = None,
    panel_path: str | Path | None = None,
    panel_size: int = 64,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if genes is None:
        if not panel_path:
            raise ValueError("Either genes or panel_path must be provided.")
        gene_list = panel_genes_from_file(panel_path, panel_size=panel_size)
    else:
        gene_list = [str(gene).strip() for gene in genes if str(gene).strip()]

    transcript_records: list[TranscriptRecord] = []
    failures: list[dict[str, str]] = []
    gene_order = {gene.lower(): idx for idx, gene in enumerate(gene_list)}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(gene_list)))) as executor:
        future_to_gene = {executor.submit(fetch_canonical_transcript, gene, species): gene for gene in gene_list}
        for future in as_completed(future_to_gene):
            gene = future_to_gene[future]
            try:
                transcript_records.append(future.result())
            except Exception as exc:  # noqa: BLE001
                failures.append({"gene_symbol": gene, "reason": str(exc)})
    transcript_records.sort(key=lambda item: gene_order.get(item.gene_symbol.lower(), 10**9))

    if not transcript_records:
        raise ValueError("No transcripts could be resolved for the requested genes.")

    manifest_path = out_dir / "probe_candidate_manifest.tsv"
    manifest_df = pd.DataFrame(
        [
            {
                "gene_symbol": record.gene_symbol,
                "gene_id": record.gene_id,
                "transcript_id": record.transcript_id,
                "sequence_length": record.sequence_length,
            }
            for record in transcript_records
        ]
    )
    manifest_df.to_csv(manifest_path, sep="\t", index=False)

    fasta_records = [
        SeqRecord(Seq(record.sequence), id=record.transcript_id, description=record.transcript_id)
        for record in transcript_records
    ]
    transcript_fasta = out_dir / "probe_candidate_transcripts.fa"
    with transcript_fasta.open("w", encoding="utf-8") as handle:
        SeqIO.write(fasta_records, handle, "fasta")

    paintshop_fasta = out_dir / "paintshop_pseudogenome.fa"
    with paintshop_fasta.open("w", encoding="utf-8") as handle:
        SeqIO.write(fasta_records, handle, "fasta")

    paintshop_gtf = out_dir / "paintshop_pseudogenome.gtf"
    with paintshop_gtf.open("w", encoding="utf-8") as gtf_handle:
        for record in transcript_records:
            attrs = (
                f'gene_id "{record.gene_id}"; transcript_id "{record.transcript_id}"; '
                f'exon_number "1"; exon_id "{record.transcript_id}.1"; gene_name "{record.gene_symbol}";'
            )
            gtf_handle.write(
                f"{record.transcript_id}\tcustom\texon\t1\t{record.sequence_length}\t.\t+\t.\t{attrs}\n"
            )

    failures_json = out_dir / "probe_candidate_failures.json"
    failures_json.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")

    return {
        "manifest_tsv": str(manifest_path),
        "transcript_fasta": str(transcript_fasta),
        "paintshop_fasta": str(paintshop_fasta),
        "paintshop_gtf": str(paintshop_gtf),
        "failures_json": str(failures_json),
    }
