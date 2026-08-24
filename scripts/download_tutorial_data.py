#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path

import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reproducibility" / "data_manifest.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Archive member escapes data root: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not allowed: {member.name}")
        handle.extractall(destination, filter="data")


def download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    headers = {"Range": f"bytes={partial.stat().st_size}-"} if partial.exists() else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        if headers and response.status_code == 200:
            partial.unlink()
            mode = "wb"
        else:
            response.raise_for_status()
            mode = "ab" if response.status_code == 206 else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def verify_files(case: dict, data_root: Path) -> list[str]:
    errors = []
    specs = list(case.get("files", []))
    paper_inputs = case.get("paper_inputs", {})
    for spec in paper_inputs.get("files", []):
        # Pending source-workspace inputs are documented but cannot be verified
        # or fetched until their Zenodo checksum is recorded.
        if spec.get("bytes") is not None and spec.get("sha256"):
            specs.append(spec)
    for spec in specs:
        path = data_root / spec["path"]
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        if int(spec["bytes"]) != path.stat().st_size:
            errors.append(f"size mismatch: {path}")
        if str(spec["sha256"]).lower() != sha256(path):
            errors.append(f"checksum mismatch: {path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify a versioned SMITH tutorial data archive.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--data-root", default="data/tutorials")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    if args.case not in manifest["cases"]:
        raise KeyError(f"Unknown tutorial case: {args.case}")
    case = manifest["cases"][args.case]
    data_root = Path(args.data_root).resolve()
    existing_errors = verify_files(case, data_root)
    if not existing_errors:
        print(f"All files for {args.case} already exist and pass checksum validation.")
        return
    url = case.get("archive_url")
    if not url:
        raise RuntimeError(
            f"The {args.case} archive has been prepared but not published. "
            "Set archive_url and archive_sha256 after the Zenodo record is released."
        )
    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / ".downloads" / case["archive_name"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if args.force and archive.exists():
        archive.unlink()
    download(url, archive)
    expected = case.get("archive_sha256")
    if expected and sha256(archive) != expected:
        archive.unlink(missing_ok=True)
        raise ValueError(f"Archive checksum mismatch for {archive}")
    staging = data_root / ".extracting" / args.case
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        safe_extract(archive, staging)
        for child in staging.iterdir():
            target = data_root / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    errors = verify_files(case, data_root)
    if errors:
        raise ValueError("Data verification failed:\n" + "\n".join(errors))
    print(f"Downloaded and verified {args.case} under {data_root}")


if __name__ == "__main__":
    main()
