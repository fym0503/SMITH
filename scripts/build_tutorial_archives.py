#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_files(case: dict, variant: str | None = None) -> list[dict]:
    variants = case.get("archive_variants", {})
    if variants:
        if variant is None:
            raise ValueError(
                "This case has multiple data archives; choose --variant "
                + ", ".join(sorted(variants))
            )
        if variant not in variants:
            raise KeyError(f"Unknown archive variant: {variant}")
        return list(variants[variant].get("files", []))
    if variant is not None:
        raise ValueError(f"Case has no archive variants; remove --variant {variant!r}")
    specs = list(case.get("files", []))
    specs.extend(case.get("paper_inputs", {}).get("files", []))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local tutorial archive before an authorized Zenodo upload.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--variant", help="Archive variant for cases with full/reproducibility packages.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=str(ROOT / "reproducibility" / "data_manifest.yaml"))
    args = parser.parse_args()
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    case = manifest["cases"][args.case]
    data_root = Path(args.data_root).resolve()
    variants = case.get("archive_variants", {})
    archive_spec = variants.get(args.variant, {}) if variants else case
    if variants and not args.variant:
        raise ValueError("This case requires --variant (reproducibility or full).")
    output = Path(args.output_dir).resolve() / archive_spec["archive_name"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for spec in case_files(case, args.variant):
            source = data_root / spec["path"]
            if not source.is_file():
                raise FileNotFoundError(source)
            if source.stat().st_size != int(spec["bytes"]) or sha256(source) != spec["sha256"]:
                raise ValueError(f"Source file does not match the manifest: {source}")
            archive.add(source.resolve(), arcname=spec["path"], recursive=False)
    print(f"archive: {output}")
    print(f"bytes: {output.stat().st_size}")
    print(f"sha256: {sha256(output)}")


if __name__ == "__main__":
    main()
