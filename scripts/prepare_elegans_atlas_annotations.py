#!/usr/bin/env python3
"""Prepare Figure 3 module and TF-pair annotations from the source atlas."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

SOURCE_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41592-021-01216-1/MediaObjects/"
    "41592_2021_1216_MOESM7_ESM.xlsx"
)
SOURCE_SHA256 = "61c4b9c2075558223b5793ca0d5f0f9e32bddc7469d793e581ceb826d23643db"
ISOFORM_SUFFIX_RE = re.compile(r"\([A-Za-z0-9_.-]+\)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=300) as response, partial.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def canonical_tf(value: object) -> str:
    return ISOFORM_SUFFIX_RE.sub("", str(value).strip().upper())


def prepare(source: Path, data_root: Path) -> dict[str, Path]:
    if sha256(source) != SOURCE_SHA256:
        raise ValueError(f"Source checksum mismatch: {source}")

    annotation_dir = data_root / "regulatory_activity" / "elegans" / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    raw_modules = pd.read_excel(source, sheet_name="a")
    module_columns = {
        "Tissue": "tissue",
        "Progenitor cell lineage (spatial module)": "progenitor_lineage",
        "Temporal module": "temporal_module",
        "TF": "gene_symbol",
    }
    modules = raw_modules[list(module_columns)].rename(columns=module_columns).dropna().copy()
    for column in ("tissue", "progenitor_lineage", "temporal_module"):
        modules[column] = modules[column].astype(str).str.strip()
    modules["gene_symbol"] = modules["gene_symbol"].map(canonical_tf)
    modules.insert(
        0,
        "module_id",
        modules[["tissue", "progenitor_lineage", "temporal_module"]].agg("|".join, axis=1),
    )
    modules = modules.drop_duplicates().sort_values(
        ["tissue", "progenitor_lineage", "temporal_module", "gene_symbol"]
    )

    raw_pairs = pd.read_excel(source, sheet_name="c")
    pair_columns = {
        "Tissue": "tissue",
        "Progenitor cell lineage (spatial module)": "progenitor_lineage",
        "TF1": "gene_a",
        "TF1-specificity": "gene_a_specificity",
        "TF1 temporal module": "gene_a_temporal_module",
        "TF2": "gene_b",
        "TF2-specificity": "gene_b_specificity",
        "TF2 temporal module": "gene_b_temporal_module",
        "Expresssion similarity": "expression_similarity",
        "Regulatory binding (ChIP-seq)": "regulatory_binding",
        "Type": "relationship_type",
    }
    pairs = raw_pairs[list(pair_columns)].rename(columns=pair_columns).copy()
    pairs = pairs.dropna(subset=["tissue", "progenitor_lineage", "gene_a", "gene_b"])
    pairs = pairs[pairs["relationship_type"].astype(str).str.strip() != "."].copy()
    for column in pairs.columns:
        pairs[column] = pairs[column].astype(str).str.strip()
    pairs["gene_a"] = pairs["gene_a"].map(canonical_tf)
    pairs["gene_b"] = pairs["gene_b"].map(canonical_tf)
    pairs = pairs[pairs["gene_a"] != pairs["gene_b"]]
    pairs = pairs.drop_duplicates().sort_values(
        ["tissue", "progenitor_lineage", "gene_a", "gene_b", "relationship_type"]
    )

    module_path = annotation_dir / "tf_spatiotemporal_modules.tsv"
    pair_path = annotation_dir / "tf_regulatory_pairs.tsv"
    modules.to_csv(module_path, sep="\t", index=False, lineterminator="\n")
    pairs.to_csv(pair_path, sep="\t", index=False, lineterminator="\n")
    return {"modules": module_path, "pairs": pair_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and normalize the source-atlas annotations used by SMITH Figure 3h-i."
    )
    parser.add_argument("--data-root", default="data/tutorials")
    parser.add_argument("--source-xlsx", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    source = (
        Path(args.source_xlsx).expanduser().resolve()
        if args.source_xlsx
        else data_root / ".downloads" / "elegans_tf_atlas_supplementary_table_5.xlsx"
    )
    if not source.is_file() or args.force:
        if args.source_xlsx:
            raise FileNotFoundError(source)
        source.parent.mkdir(parents=True, exist_ok=True)
        download(SOURCE_URL, source)
    outputs = prepare(source, data_root)
    for label, path in outputs.items():
        print(f"{label}: {path} ({path.stat().st_size} bytes, sha256={sha256(path)})")


if __name__ == "__main__":
    main()
