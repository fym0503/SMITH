#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
from Bio.SeqUtils import MeltingTemp as mt
from oligo_designer_toolsuite.pipelines._scrinshot_probe_designer import TargetProbeDesigner

from odt_scrinshot_feasibility_demo import fetch_canonical_cdna, write_fasta  # noqa: E402


SCRINSHOT_DEFAULTS = {
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


def count_oligos_by_region(oligo_database) -> dict[str, int]:
    return {
        region_id: len(oligo_database.database[region_id])
        for region_id in oligo_database.get_regionid_list()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes", nargs="+", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--set-size-min", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fetched = []
    failures = {}
    for gene in args.genes:
        try:
            fetched.append(fetch_canonical_cdna(gene, args.species))
        except Exception as exc:
            failures[gene] = str(exc)

    fasta_path = output_dir / "input_transcripts.fa"
    if fetched:
        write_fasta(fetched, fasta_path)
        defaults = copy.deepcopy(SCRINSHOT_DEFAULTS)
        designer = TargetProbeDesigner(str(output_dir), n_jobs=1)
        database = designer.create_oligo_database(
            gene_ids=[record["symbol"] for record in fetched],
            oligo_length_min=40,
            oligo_length_max=45,
            files_fasta_oligo_database=[str(fasta_path)],
            min_oligos_per_gene=args.set_size_min,
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
    else:
        counts_initial = {}
        counts_property = {}

    rows = []
    for gene in args.genes:
        record = next((r for r in fetched if r["symbol"] == gene), None)
        if record is None:
            rows.append(
                {
                    "gene": gene,
                    "status": "fetch_failed",
                    "reason": failures.get(gene),
                    "transcript_id": None,
                    "candidate_oligos_initial": 0,
                    "candidate_oligos_after_property_filters": 0,
                    "feasible_property_only": False,
                }
            )
            continue
        rows.append(
            {
                "gene": gene,
                "status": "ok",
                "reason": None,
                "transcript_id": record["transcript_id"],
                "candidate_oligos_initial": int(counts_initial.get(gene, 0)),
                "candidate_oligos_after_property_filters": int(counts_property.get(gene, 0)),
                "feasible_property_only": int(counts_property.get(gene, 0)) >= args.set_size_min,
            }
        )

    summary_path = output_dir / "property_only_summary.tsv"
    pd.DataFrame(rows).to_csv(summary_path, sep="\t", index=False)
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "genes": args.genes,
                "species": args.species,
                "mode": "property_filter_only",
                "notes": [
                    "This lightweight demo skips oligoset construction to keep multi-gene runs practical.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(summary_path)


if __name__ == "__main__":
    main()
