#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smith_agent.feasibility import ODTScrinshotBackend, OligoMinerBackend, PaintSHOPBackend, ProbeDealerBackend


RUN_DIR = ROOT_DIR / "demo_outputs" / "mouse_100_gene_run"
MANIFEST = RUN_DIR / "mouse_100_genes_manifest.tsv"
TRANSCRIPT_FASTA = RUN_DIR / "mouse_100_genes_transcripts.fa"
PAINTSHOP_FASTA = RUN_DIR / "paintshop_pseudogenome.fa"
PAINTSHOP_GTF = RUN_DIR / "paintshop_pseudogenome.gtf"


def main() -> None:
    import pandas as pd

    manifest = pd.read_csv(MANIFEST, sep="\t")
    genes = manifest["gene_symbol"].tolist()

    results = {
        "manifest_tsv": str(MANIFEST),
        "transcript_fasta": str(TRANSCRIPT_FASTA),
        "paintshop_fasta": str(PAINTSHOP_FASTA),
        "paintshop_gtf": str(PAINTSHOP_GTF),
    }

    print("Running ODT property-only on 100 genes...", flush=True)
    results["odt_scrinshot"] = ODTScrinshotBackend().run_gene_symbols_property_only(
        genes=genes,
        species="mus_musculus",
        output_dir=RUN_DIR / "odt_scrinshot_property_only",
        set_size_min=2,
    ).to_dict()

    print("Running ProbeDealer on 100 genes...", flush=True)
    results["probedealer_py"] = ProbeDealerBackend().run_transcript_fasta(
        fasta_path=TRANSCRIPT_FASTA,
        output_dir=RUN_DIR / "probedealer_py",
    ).to_dict()

    print("Running OligoMiner official specificity path on 100 genes...", flush=True)
    results["oligominer"] = OligoMinerBackend().run_multi_transcript_specificity(
        fasta_path=TRANSCRIPT_FASTA,
        output_dir=RUN_DIR / "oligominer_specificity",
        temperature_c=42,
    ).to_dict()

    print("Running PaintSHOP custom pipeline on 100 genes...", flush=True)
    results["paintshop"] = PaintSHOPBackend().run_custom(
        assembly="mouse_100_gene_demo",
        genome_fasta=PAINTSHOP_FASTA,
        annotation_file=PAINTSHOP_GTF,
        work_dir=RUN_DIR / "paintshop",
        cores=8,
    ).to_dict()

    out = RUN_DIR / "all_tools_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
