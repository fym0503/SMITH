#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from Bio import SeqIO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smith_agent.feasibility import (
    ODTScrinshotBackend,
    OligoMinerBackend,
    PaintSHOPBackend,
    ProbeDealerBackend,
)


def write_single_transcript_fasta(source_fasta: Path, transcript_id: str, output_fasta: Path) -> None:
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    for record in SeqIO.parse(str(source_fasta), "fasta"):
        if record.id == transcript_id:
            with output_fasta.open("w") as handle:
                SeqIO.write([record], handle, "fasta")
            return
    raise ValueError(f"Transcript {transcript_id} not found in {source_fasta}")


def main() -> None:
    demo_root = ROOT_DIR / "demo_outputs"
    demo_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    odt = ODTScrinshotBackend()
    print("Running ODT/SCRINSHOT demo...", flush=True)
    results["odt_scrinshot"] = odt.run_gene_symbols(
        genes=["Actb"],
        species="mus_musculus",
        output_dir=demo_root / "odt_scrinshot_mouse_actb",
        n_sets=5,
    ).to_dict()

    oligominer_fasta = demo_root / "oligominer_mouse" / "ENSMUST00000112132.fa"
    write_single_transcript_fasta(
        ROOT_DIR
        / "third_party/probedealer_examples/ExampleTargetFiles/sequential RNA FISH/sequential RNA FISH example input_mouse_gencode vM25.fasta",
        "ENSMUST00000112132",
        oligominer_fasta,
    )
    oligominer = OligoMinerBackend()
    print("Running OligoMiner demo...", flush=True)
    results["oligominer"] = oligominer.run_blockparse(
        fasta_path=oligominer_fasta,
        output_dir=demo_root / "oligominer_mouse",
        output_stem="ENSMUST00000112132_blockparse",
    ).to_dict()

    probedealer = ProbeDealerBackend()
    print("Running ProbeDealer demo...", flush=True)
    results["probedealer_py"] = probedealer.run_transcript_fasta(
        fasta_path=ROOT_DIR
        / "third_party/probedealer_examples/ExampleTargetFiles/sequential RNA FISH/sequential RNA FISH example input_mouse_gencode vM25.fasta",
        output_dir=demo_root / "probedealer_mouse",
    ).to_dict()

    paintshop = PaintSHOPBackend()
    print("Running PaintSHOP example demo...", flush=True)
    results["paintshop_example"] = paintshop.run_example(cores=1).to_dict()

    output_path = demo_root / "backend_demo_results.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
