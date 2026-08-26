#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from Bio import SeqIO

from smith_agent.adapters.feasibility_backends import run_probedealer_backend_screen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_top128_feasibility_filtering import (
    HUMAN_REFERENCE_DIR,
    SMITH_PACKAGE_ROOT,
    build_local_probe_candidate_manifest,
    load_transcript_to_gene,
)


DEFAULT_RANK_TSV = PROJECT_ROOT / "outputs/liver_merfish_benchmark/liver_source_gene_rank.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/scrna_probedealer_risk_scan_top1024"
SOURCE_GENE_METADATA_H5AD = PROJECT_ROOT / "data/cellxgene_liver_scrna/fe4bc7fc-0035-4ebb-919b-2d9097ec5dd4.h5ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan ranked scRNA genes for ProbeDealer cross-gene specificity risks."
    )
    parser.add_argument("--rank-tsv", default=str(DEFAULT_RANK_TSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=1024)
    parser.add_argument("--species", default="homo_sapiens")
    parser.add_argument("--min-target-probes", type=int, default=20)
    parser.add_argument("--min-different-symbol-probes", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gene-metadata-h5ad", default=str(SOURCE_GENE_METADATA_H5AD))
    return parser.parse_args()


def read_ranked_panel(rank_tsv: str | Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(rank_tsv, sep="\t")
    if "gene_symbol" not in df.columns:
        raise ValueError(f"Rank file must contain `gene_symbol`: {rank_tsv}")
    df = df.head(top_n).copy()
    if "rank" not in df.columns:
        df.insert(0, "rank", range(1, len(df) + 1))
    return df[["rank", "gene_symbol"]].copy()


def load_gene_symbols_from_h5ad(h5ad_path: str | Path) -> dict[str, str]:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    var = adata.var.copy()
    adata.file.close()
    mapping: dict[str, str] = {}
    if "feature_name" in var.columns:
        for gene_id, symbol in zip(var.index.astype(str), var["feature_name"].astype(str), strict=False):
            if gene_id.startswith("ENSG") and symbol:
                mapping.setdefault(gene_id, symbol)
    return mapping


def fasta_ids(fasta_path: str | Path) -> list[str]:
    path = Path(fasta_path)
    if not path.exists():
        return []
    return [record.id for record in SeqIO.parse(str(path), "fasta")]


def parse_blast_hits(path: str | Path, transcript_to_gene: dict[str, str]) -> dict[str, set[str]]:
    hits: dict[str, set[str]] = defaultdict(set)
    path = Path(path)
    if not path.exists():
        return hits
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            query_id, subject_id = parts[:2]
            gene_id = transcript_to_gene.get(subject_id)
            if gene_id:
                hits[query_id].add(gene_id)
    return hits


def summarize_probe_hits(
    manifest: pd.DataFrame,
    probedealer_dir: str | Path,
    transcript_to_gene: dict[str, str],
    gene_id_to_symbol: dict[str, str],
) -> pd.DataFrame:
    probedealer_dir = Path(probedealer_dir)
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        gene_symbol = str(row.gene_symbol)
        gene_id = str(row.gene_id)
        transcript_id = str(row.transcript_id)
        blast_dir = probedealer_dir / f"blast_{transcript_id}"
        query_ids = fasta_ids(blast_dir / "probe_queries.fa")
        hits = parse_blast_hits(blast_dir / "blast_hits.tsv", transcript_to_gene)

        geneid_target_only = 0
        symbol_target_only_known = 0
        probes_with_known_different_symbol = 0
        clean_different_symbol = 0
        unknown_symbol_probe_count = 0
        no_hit_probe_count = 0
        off_gene_counter: Counter[str] = Counter()
        off_symbol_counter: Counter[str] = Counter()

        for query_id in query_ids:
            hit_gene_ids = hits.get(query_id, set())
            if not hit_gene_ids:
                no_hit_probe_count += 1
                continue
            if hit_gene_ids == {gene_id}:
                geneid_target_only += 1

            known_symbols: list[str] = []
            has_unknown = False
            has_known_different = False
            for hit_gene_id in hit_gene_ids:
                hit_symbol = gene_symbol if hit_gene_id == gene_id else gene_id_to_symbol.get(hit_gene_id)
                if hit_gene_id != gene_id:
                    off_gene_counter[hit_gene_id] += 1
                    off_symbol_counter[hit_symbol or "?"] += 1
                if hit_symbol is None:
                    has_unknown = True
                    continue
                known_symbols.append(hit_symbol)
                if hit_symbol != gene_symbol:
                    has_known_different = True

            if has_unknown:
                unknown_symbol_probe_count += 1
            if has_known_different:
                probes_with_known_different_symbol += 1
            if not has_unknown and has_known_different:
                clean_different_symbol += 1
            if not has_unknown and known_symbols and all(symbol == gene_symbol for symbol in known_symbols):
                symbol_target_only_known += 1

        known_classified = symbol_target_only_known + probes_with_known_different_symbol
        different_symbol_fraction = (
            probes_with_known_different_symbol / known_classified if known_classified else float("nan")
        )
        rows.append(
            {
                "gene_symbol": gene_symbol,
                "gene_id": gene_id,
                "transcript_id": transcript_id,
                "sequence_length": int(row.sequence_length),
                "initial_probe_count": len(query_ids),
                "geneid_target_only_probe_count": geneid_target_only,
                "symbol_target_only_probe_count_known": symbol_target_only_known,
                "probes_with_known_different_symbol": probes_with_known_different_symbol,
                "clean_different_symbol_probe_count": clean_different_symbol,
                "unknown_symbol_probe_count": unknown_symbol_probe_count,
                "no_hit_probe_count": no_hit_probe_count,
                "different_symbol_fraction_known": different_symbol_fraction,
                "top_cross_gene_ids": ";".join(
                    f"{gene}:{gene_id_to_symbol.get(gene, '?')}:{count}"
                    for gene, count in off_gene_counter.most_common(8)
                ),
                "top_cross_symbols": ";".join(
                    f"{symbol}:{count}" for symbol, count in off_symbol_counter.most_common(8)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = read_ranked_panel(args.rank_tsv, args.top_n)
    panel.to_csv(out_dir / "selected_scrna_ranked_genes.tsv", sep="\t", index=False)
    (out_dir / "selected_scrna_ranked_genes.txt").write_text(
        "\n".join(panel["gene_symbol"].astype(str)) + "\n",
        encoding="utf-8",
    )

    probe_dir = out_dir / "probe_candidates"
    manifest_tsv = probe_dir / "probe_candidate_manifest.tsv"
    transcript_fasta = probe_dir / "probe_candidate_transcripts.fa"
    if args.force or not (manifest_tsv.exists() and transcript_fasta.exists()):
        manifest_result = build_local_probe_candidate_manifest(
            output_dir=probe_dir,
            species=args.species,
            panel=panel,
            gene_metadata_h5ad=args.gene_metadata_h5ad,
        )
    else:
        manifest_result = {
            "manifest_tsv": str(manifest_tsv),
            "transcript_fasta": str(transcript_fasta),
            "failures_json": str(probe_dir / "probe_candidate_failures.json"),
        }

    probedealer_dir = out_dir / "probedealer_backend"
    probedealer_summary = probedealer_dir / "probedealer_summary.tsv"
    if args.force or not probedealer_summary.exists():
        probedealer_result = run_probedealer_backend_screen(
            package_root=SMITH_PACKAGE_ROOT,
            transcript_fasta=transcript_fasta,
            output_dir=probedealer_dir,
            use_transcriptome_reference=True,
            species=args.species,
        )
    else:
        probedealer_result = {
            "status": "skipped_existing",
            "output_files": {"summary_tsv": str(probedealer_summary)},
        }

    manifest = pd.read_csv(manifest_tsv, sep="\t")
    transcript_to_gene = load_transcript_to_gene(HUMAN_REFERENCE_DIR / "transcript_to_gene.tsv")
    gene_id_to_symbol = load_gene_symbols_from_h5ad(SOURCE_GENE_METADATA_H5AD)
    gene_id_to_symbol.update(dict(zip(manifest["gene_id"].astype(str), manifest["gene_symbol"].astype(str), strict=False)))
    # Known alternate-locus annotations observed during the MERFISH scan.
    gene_id_to_symbol.update(
        {
            "ENSG00000277101": "ARHGEF26",
            "ENSG00000285132": "CTSB",
            "ENSG00000292149": "TCF7L1",
            "ENSG00000281344": "HELLPAR",
        }
    )

    risk = summarize_probe_hits(
        manifest=manifest,
        probedealer_dir=probedealer_dir,
        transcript_to_gene=transcript_to_gene,
        gene_id_to_symbol=gene_id_to_symbol,
    )
    risk = panel.merge(risk, on="gene_symbol", how="left")
    risk["resolved"] = risk["transcript_id"].notna()
    risk["geneid_probedealer_fail"] = risk["geneid_target_only_probe_count"].fillna(0) < args.min_target_probes
    risk["symbol_known_probedealer_fail"] = (
        risk["symbol_target_only_probe_count_known"].fillna(0) < args.min_target_probes
    )
    risk["igf1_like"] = (
        risk["resolved"]
        & risk["symbol_known_probedealer_fail"]
        & (risk["probes_with_known_different_symbol"].fillna(0) >= args.min_different_symbol_probes)
        & (risk["different_symbol_fraction_known"].fillna(0) >= 0.5)
    )
    risk["same_symbol_rescue_candidate"] = (
        risk["resolved"]
        & risk["geneid_probedealer_fail"]
        & (risk["symbol_target_only_probe_count_known"].fillna(0) >= args.min_target_probes)
    )

    risk_path = out_dir / "probe_risk_summary.tsv"
    risk.to_csv(risk_path, sep="\t", index=False)
    candidates = risk[risk["igf1_like"]].sort_values(
        ["probes_with_known_different_symbol", "different_symbol_fraction_known"],
        ascending=False,
    )
    candidate_path = out_dir / "igf1_like_candidates.tsv"
    candidates.to_csv(candidate_path, sep="\t", index=False)

    summary = {
        "rank_tsv": str(Path(args.rank_tsv).resolve()),
        "top_n": int(args.top_n),
        "n_input_genes": int(len(panel)),
        "n_resolved_genes": int(risk["resolved"].sum()),
        "n_unresolved_genes": int((~risk["resolved"]).sum()),
        "n_geneid_probedealer_fail": int(risk["geneid_probedealer_fail"].sum()),
        "n_symbol_known_probedealer_fail": int(risk["symbol_known_probedealer_fail"].sum()),
        "n_same_symbol_rescue_candidates": int(risk["same_symbol_rescue_candidate"].sum()),
        "n_igf1_like_candidates": int(risk["igf1_like"].sum()),
        "outputs": {
            "manifest_tsv": str(manifest_tsv),
            "manifest_failures_json": manifest_result.get("failures_json", str(probe_dir / "probe_candidate_failures.json")),
            "probedealer_summary_tsv": str(probedealer_summary),
            "probe_risk_summary_tsv": str(risk_path),
            "igf1_like_candidates_tsv": str(candidate_path),
        },
        "probedealer_result": probedealer_result,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
