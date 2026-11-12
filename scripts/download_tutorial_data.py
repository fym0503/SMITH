#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        if headers and response.status_code == 200:
            partial.unlink()
            mode = "wb"
        else:
            response.raise_for_status()
            if response.status_code == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Download server returned an unexpected byte range")
            mode = "ab" if response.status_code == 206 else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def file_specs(case: dict) -> list[dict]:
    return list(case.get("files", [])) + list(case.get("paper_inputs", {}).get("files", []))


def verify_files(case: dict, data_root: Path) -> list[str]:
    errors = []
    for spec in file_specs(case):
        path = data_root / spec["path"]
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        if int(spec["bytes"]) != path.stat().st_size:
            errors.append(f"size mismatch: {path}")
        if str(spec["sha256"]).lower() != sha256(path):
            errors.append(f"checksum mismatch: {path}")
    return errors


def install_archive(archive: Path, data_root: Path, archive_spec: dict, cases: list[dict]) -> None:
    """Verify the complete archive and install only manifest-listed inputs."""
    expected_size = archive_spec.get("archive_bytes")
    if expected_size is not None and archive.stat().st_size != int(expected_size):
        raise ValueError(f"Archive size mismatch: {archive}")
    if sha256(archive) != archive_spec["archive_sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    import tempfile
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extracting-", dir=data_root) as directory:
        staging = Path(directory)
        safe_extract(archive, staging)
        inputs = staging / archive_spec.get("archive_root", "")
        errors = [error for case in cases for error in verify_files(case, inputs)]
        if errors:
            raise ValueError("Data verification failed:\n" + "\n".join(errors))
        paths = {spec["path"] for case in cases for spec in file_specs(case)}
        for relative in sorted(paths):
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Invalid input path: {relative}")
            target = data_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            (inputs / relative_path).replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the complete SMITH tutorial data bundle.")
    parser.add_argument("--case", default="all", help="Case to verify/install, or all (default). The download contains all cases.")
    parser.add_argument("--data-root", default="data/tutorials")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--archive", type=Path, help="Use an already downloaded complete archive.")
    parser.add_argument("--force", action="store_true", help="Reinstall inputs, downloading a fresh archive unless --archive is supplied.")
    args = parser.parse_args()
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    if args.case != "all" and args.case not in manifest["cases"]:
        parser.error(f"Unknown tutorial case: {args.case}")
    cases = list(manifest["cases"].values()) if args.case == "all" else [manifest["cases"][args.case]]
    data_root = Path(args.data_root).expanduser().resolve()
    if not args.force and not any(verify_files(case, data_root) for case in cases):
        print(f"All files for {args.case} already exist and pass checksum validation.")
        return
    archive_spec = manifest["complete_archive"]
    archive = args.archive.expanduser().resolve() if args.archive else data_root / ".downloads" / archive_spec["archive_name"]
    if args.archive:
        if not archive.is_file():
            raise FileNotFoundError(archive)
    elif args.force or not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {archive_spec['archive_bytes']:,} bytes from Zenodo...", flush=True)
        download(archive_spec["archive_url"], archive)
    print("Verifying and extracting tutorial inputs...", flush=True)
    install_archive(archive, data_root, archive_spec, cases)
    print(f"Verified {args.case} under {data_root}")
    print(f"Set SMITH_TUTORIAL_DATA to {data_root}")


if __name__ == "__main__":
    main()
