#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(os.environ.get("SMITH_PACKAGE_ROOT", Path(__file__).resolve().parents[1]))
ODT_PYTHON = Path(os.environ.get("SMITH_ODT_PYTHON", sys.executable))
RUNNER = Path(os.environ.get("SMITH_ODT_LIGHT_SCRIPT", ROOT_DIR / "scripts" / "odt_property_only_runner.py"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--set-size-min", type=int, default=2)
    return parser.parse_args()


def run_batch(batch_genes: list[str], species: str, batch_dir: Path, set_size_min: int) -> Path:
    batch_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ODT_PYTHON),
        str(RUNNER),
        "--species",
        species,
        "--output-dir",
        str(batch_dir),
        "--set-size-min",
        str(set_size_min),
        "--genes",
        *batch_genes,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"Batch failed: {batch_genes}")
    return batch_dir / "property_only_summary.tsv"


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, sep="\t")
    genes = manifest["gene_symbol"].astype(str).tolist()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = [
        genes[i : i + args.batch_size]
        for i in range(0, len(genes), args.batch_size)
    ]

    summary_paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(run_batch, batch, args.species, output_dir / f"batch_{idx:02d}", args.set_size_min): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            summary_paths.append(future.result())

    frames = [pd.read_csv(path, sep="\t") for path in summary_paths]
    merged = pd.concat(frames, ignore_index=True).sort_values("gene").reset_index(drop=True)
    merged.to_csv(output_dir / "property_only_summary.tsv", sep="\t", index=False)
    print(output_dir / "property_only_summary.tsv")


if __name__ == "__main__":
    main()
