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


DEFAULT_FASTA = Path(
    "third_party/probedealer_examples/ExampleTargetFiles/sequential RNA FISH/"
    "sequential RNA FISH example input_mouse_gencode vM25.fasta"
)
DEFAULT_SUPPLEMENT = Path("41598_2020_76439_MOESM2_ESM123.xlsx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Python ProbeDealer-like initial probe counts against supplementary counts."
    )
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--output", type=Path, default=Path("probedealer_demo_compare.tsv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_fasta_records(args.fasta)
    designed = build_oligo_array(records, OligoDesignConfig())

    supplement = pd.read_excel(args.supplement)
    supplement = supplement.rename(
        columns={
            "Transcript ID": "transcript_id",
            "ProbeDealer_initial probe count": "supp_probe_count",
            "ProbeDealer_final probe count after transcriptome BLAST": "supp_final_probe_count",
            "FPKM": "supp_fpkm",
        }
    )

    observed = pd.DataFrame(
        {
            "transcript_id": [sequence_id for sequence_id, _ in records],
            "py_probe_count": [len(designed[sequence_id]) for sequence_id, _ in records],
        }
    )

    merged = observed.merge(supplement, how="left", on="transcript_id")
    merged["count_delta"] = merged["py_probe_count"] - merged["supp_probe_count"]
    merged["abs_count_delta"] = merged["count_delta"].abs()
    merged = merged[
        [
            "transcript_id",
            "supp_fpkm",
            "py_probe_count",
            "supp_probe_count",
            "supp_final_probe_count",
            "count_delta",
            "abs_count_delta",
        ]
    ].sort_values("transcript_id")

    args.output.write_text(merged.to_csv(sep="\t", index=False))

    comparable = merged.dropna(subset=["supp_probe_count"])
    mae = comparable["abs_count_delta"].mean()
    max_abs = comparable["abs_count_delta"].max()
    exact = int((comparable["count_delta"] == 0).sum())

    print(f"Input fasta: {args.fasta}")
    print(f"Supplement: {args.supplement}")
    print(f"Rows compared: {len(comparable)}")
    print(f"Exact matches: {exact}/{len(comparable)}")
    print(f"Mean absolute error: {mae:.2f}")
    print(f"Max absolute error: {int(max_abs)}")
    print(f"Wrote: {args.output}")
    print()
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
