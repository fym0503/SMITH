#!/usr/bin/env python3

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import requests
from Bio.SeqUtils import MeltingTemp as mt
from oligo_designer_toolsuite.pipelines._scrinshot_probe_designer import (
    TargetProbeDesigner,
)


DEFAULT_GENES = ["EPCAM", "KRT19", "VIM", "PECAM1", "SCGB1A1"]

SCRINSHOT_DEFAULTS = {
    "max_graph_size": 5000,
    "n_attempts": 100000,
    "heuristic": True,
    "heuristic_n_attempts": 100,
    "target_probe_Tm_parameters": {
        "check": True,
        "strict": True,
        "c_seq": None,
        "shift": 0,
        "nn_table": mt.DNA_NN3,
        "tmm_table": mt.DNA_TMM1,
        "imm_table": mt.DNA_IMM1,
        "de_table": mt.DNA_DE1,
        "dnac1": 50,
        "dnac2": 0,
        "selfcomp": False,
        "saltcorr": 7,
        "Na": 39,
        "K": 75,
        "Tris": 20,
        "Mg": 10,
        "dNTPs": 0,
    },
    "target_probe_Tm_chem_correction_parameters": {
        "DMSO": 0,
        "fmd": 20,
        "DMSOfactor": 0.75,
        "fmdfactor": 0.65,
        "fmdmethod": 1,
        "GC": None,
    },
    "target_probe_Tm_salt_correction_parameters": None,
}


def fetch_canonical_cdna(symbol: str, species: str) -> dict:
    headers = {"Content-Type": "application/json"}
    lookup_url = f"https://rest.ensembl.org/lookup/symbol/{species}/{symbol}?expand=1"
    lookup = requests.get(lookup_url, headers=headers, timeout=60)
    lookup.raise_for_status()
    data = lookup.json()

    transcripts = data.get("Transcript") or []
    if not transcripts:
        raise ValueError(f"No transcripts returned for {symbol}")

    canonical = [tx for tx in transcripts if tx.get("is_canonical") == 1]
    if canonical:
        transcript = canonical[0]
    else:
        transcript = max(
            transcripts,
            key=lambda tx: abs(int(tx.get("end", 0)) - int(tx.get("start", 0))),
        )

    transcript_id = transcript["id"]
    seq_url = f"https://rest.ensembl.org/sequence/id/{transcript_id}?type=cdna"
    seq_resp = requests.get(seq_url, headers=headers, timeout=60)
    seq_resp.raise_for_status()
    seq_data = seq_resp.json()
    seq = (seq_data.get("seq") or "").upper()
    if not seq:
        raise ValueError(f"No cDNA sequence returned for {symbol} ({transcript_id})")

    return {
        "symbol": symbol,
        "transcript_id": transcript_id,
        "gene_id": data.get("id"),
        "sequence": seq,
        "sequence_length": len(seq),
    }


def write_fasta(records: list[dict], fasta_path: Path) -> None:
    with fasta_path.open("w") as handle:
        for record in records:
            header = (
                f"{record['symbol']}::transcript_id={record['transcript_id']}"
                f"::transcript:1-{record['sequence_length']}(+)"
            )
            handle.write(f">{header}\n")
            seq = record["sequence"]
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")


def count_oligos_by_region(oligo_database) -> dict[str, int]:
    return {
        region_id: len(oligo_database.database[region_id])
        for region_id in oligo_database.get_regionid_list()
    }


def summarize_oligosets(oligo_database) -> dict[str, dict]:
    summary = {}
    for region_id in oligo_database.get_regionid_list():
        if region_id not in oligo_database.oligosets:
            continue
        oligosets = oligo_database.oligosets[region_id]
        if oligosets is None or getattr(oligosets, "empty", True):
            continue
        oligo_cols = [col for col in oligosets.columns if col.startswith("oligo_")]
        if not oligo_cols:
            continue
        set_sizes = oligosets[oligo_cols].notna().sum(axis=1)
        summary[region_id] = {
            "n_oligo_sets": int(len(oligosets)),
            "best_set_size": int(set_sizes.max()),
        }
    return summary


def run_property_only_scrinshot(
    fasta_path: Path,
    gene_ids: list[str],
    output_dir: Path,
    set_size_min: int,
    set_size_opt: int,
    n_sets: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, dict]]:
    defaults = copy.deepcopy(SCRINSHOT_DEFAULTS)
    designer = TargetProbeDesigner(str(output_dir), n_jobs=1)

    database = designer.create_oligo_database(
        gene_ids=gene_ids,
        oligo_length_min=40,
        oligo_length_max=45,
        files_fasta_oligo_database=[str(fasta_path)],
        min_oligos_per_gene=set_size_min,
        isoform_consensus=0,
    )
    counts_initial = count_oligos_by_region(database)

    database = designer.filter_by_property(
        oligo_database=database,
        GC_content_min=40,
        GC_content_max=60,
        Tm_min=65,
        Tm_max=75,
        detect_oligo_length_min=15,
        detect_oligo_length_max=40,
        min_thymines=2,
        arm_length_min=10,
        arm_Tm_dif_max=2,
        arm_Tm_min=50,
        arm_Tm_max=60,
        homopolymeric_base_n={"A": 5, "T": 5, "C": 5, "G": 5},
        Tm_parameters=defaults["target_probe_Tm_parameters"],
        Tm_chem_correction_parameters=defaults["target_probe_Tm_chem_correction_parameters"],
        Tm_salt_correction_parameters=defaults["target_probe_Tm_salt_correction_parameters"],
    )
    counts_property = count_oligos_by_region(database)

    database = designer.create_oligo_sets(
        oligo_database=database,
        isoform_weight=2,
        GC_content_min=40,
        GC_content_opt=50,
        GC_content_max=60,
        GC_weight=1,
        Tm_min=65,
        Tm_opt=70,
        Tm_max=75,
        Tm_weight=1,
        Tm_parameters=defaults["target_probe_Tm_parameters"],
        Tm_chem_correction_parameters=defaults["target_probe_Tm_chem_correction_parameters"],
        Tm_salt_correction_parameters=defaults["target_probe_Tm_salt_correction_parameters"],
        set_size_opt=set_size_opt,
        set_size_min=set_size_min,
        distance_between_oligos=0,
        n_sets=n_sets,
        max_graph_size=defaults["max_graph_size"],
        n_attempts=defaults["n_attempts"],
        heuristic=defaults["heuristic"],
        heuristic_n_attempts=defaults["heuristic_n_attempts"],
    )
    oligoset_summary = summarize_oligosets(database)

    database.write_database_to_table(
        attributes=[
            "target",
            "isoform_consensus",
            "GC_content",
            "TmNN",
            "detect_oligo_length",
        ],
        flatten_attribute=False,
        filename="property_filtered_oligos",
    )
    database.write_oligosets_to_table(dir_output="oligosets")

    return counts_initial, counts_property, oligoset_summary


def build_summary(
    fetched_records: list[dict],
    fetch_failures: dict[str, str],
    counts_initial: dict[str, int],
    counts_property: dict[str, int],
    oligoset_summary: dict[str, dict],
    set_size_min: int,
) -> pd.DataFrame:
    rows = []
    fetched_by_symbol = {record["symbol"]: record for record in fetched_records}
    all_symbols = sorted(set(fetch_failures) | set(fetched_by_symbol))

    for symbol in all_symbols:
        if symbol in fetch_failures:
            rows.append(
                {
                    "gene": symbol,
                    "status": "fetch_failed",
                    "reason": fetch_failures[symbol],
                    "transcript_id": None,
                    "sequence_length": None,
                    "candidate_oligos_initial": 0,
                    "candidate_oligos_after_property_filters": 0,
                    "n_oligo_sets": 0,
                    "best_set_size": 0,
                    "feasible_property_only": False,
                }
            )
            continue

        record = fetched_by_symbol[symbol]
        sets = oligoset_summary.get(symbol, {})
        best_set_size = int(sets.get("best_set_size", 0))
        rows.append(
            {
                "gene": symbol,
                "status": "ok",
                "reason": None,
                "transcript_id": record["transcript_id"],
                "sequence_length": record["sequence_length"],
                "candidate_oligos_initial": int(counts_initial.get(symbol, 0)),
                "candidate_oligos_after_property_filters": int(counts_property.get(symbol, 0)),
                "n_oligo_sets": int(sets.get("n_oligo_sets", 0)),
                "best_set_size": best_set_size,
                "feasible_property_only": best_set_size >= set_size_min,
            }
        )

    return pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a few human transcripts from Ensembl and run a reduced "
            "SCRINSHOT probe-feasibility check with ODT."
        )
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        default=DEFAULT_GENES,
        help="Human gene symbols to test.",
    )
    parser.add_argument(
        "--species",
        default="homo_sapiens",
        help="Ensembl species name. Default: homo_sapiens",
    )
    parser.add_argument(
        "--output-dir",
        default="demo_run",
        help="Directory for generated files and summary tables.",
    )
    parser.add_argument(
        "--set-size-min",
        type=int,
        default=3,
        help="Minimum probe-set size to count as feasible.",
    )
    parser.add_argument(
        "--set-size-opt",
        type=int,
        default=5,
        help="Target probe-set size used during set selection.",
    )
    parser.add_argument(
        "--n-sets",
        type=int,
        default=20,
        help="Number of probe sets to generate per gene.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fetched_records = []
    fetch_failures = {}
    for gene in args.genes:
        try:
            fetched_records.append(fetch_canonical_cdna(gene, args.species))
        except Exception as exc:
            fetch_failures[gene] = str(exc)

    fasta_path = output_dir / "input_transcripts.fa"
    if fetched_records:
        write_fasta(fetched_records, fasta_path)
        counts_initial, counts_property, oligoset_summary = run_property_only_scrinshot(
            fasta_path=fasta_path,
            gene_ids=[record["symbol"] for record in fetched_records],
            output_dir=output_dir,
            set_size_min=args.set_size_min,
            set_size_opt=args.set_size_opt,
            n_sets=args.n_sets,
        )
    else:
        counts_initial, counts_property, oligoset_summary = {}, {}, {}

    summary = build_summary(
        fetched_records=fetched_records,
        fetch_failures=fetch_failures,
        counts_initial=counts_initial,
        counts_property=counts_property,
        oligoset_summary=oligoset_summary,
        set_size_min=args.set_size_min,
    )

    summary_path = output_dir / "feasibility_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    metadata = {
        "species": args.species,
        "genes": args.genes,
        "mode": "property_only",
        "notes": [
            "This run used real Ensembl cDNA sequences.",
            "BLAST-based specificity and cross-hybridization filters were skipped because blastn is not installed on this machine.",
            "Feasible_property_only means ODT could form at least one non-overlapping SCRINSHOT probe set of size >= set_size_min after sequence-property filtering.",
        ],
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(summary.to_string(index=False))
    print(f"\nSummary written to: {summary_path}")
    print(f"Metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
