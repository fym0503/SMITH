#!/usr/bin/env python3
"""Publish validated tutorial archives to Zenodo and update the data manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

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


def request_json(session: requests.Session, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = session.request(method, url, timeout=(30, 300), **kwargs)
    response.raise_for_status()
    return response.json() if response.content else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the versioned SMITH tutorial archives to Zenodo.")
    parser.add_argument("--archive-dir", required=True, help="Directory containing archives from build_tutorial_archives.py.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--token-env", default="ZENODO_TOKEN")
    parser.add_argument("--title", default="SMITH tutorial data release 2026.08")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org instead of zenodo.org.")
    parser.add_argument("--publish", action="store_true", help="Publish the deposition after all uploads succeed.")
    parser.add_argument("--allow-pending-license", action="store_true", help="Allow publication before upstream licenses are verified.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases", {})
    if not cases:
        raise ValueError("The manifest contains no tutorial cases.")
    pending = [case_id for case_id, case in cases.items() if case.get("upstream_license") == "pending_verification"]
    if pending and not args.allow_pending_license:
        raise RuntimeError(
            "Upstream licenses are still pending for: " + ", ".join(pending) + ". "
            "Verify compatibility or pass --allow-pending-license explicitly."
        )

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Set {args.token_env} in the environment; no token was read from disk or committed files.")

    host = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
    api = f"{host}/api"
    archive_dir = Path(args.archive_dir).resolve()
    archives: dict[str, Path] = {}
    for case_id, case in cases.items():
        archive = archive_dir / case["archive_name"]
        if not archive.is_file():
            raise FileNotFoundError(archive)
        expected = case.get("prepared_archive_sha256") or case.get("archive_sha256")
        if expected and sha256(archive) != expected:
            raise ValueError(f"Archive checksum mismatch: {archive}")
        archives[case_id] = archive

    session = requests.Session()
    session.params = {"access_token": token}
    deposition = request_json(
        session,
        "POST",
        f"{api}/deposit/depositions",
        json={
            "metadata": {
                "title": args.title,
                "upload_type": "dataset",
                "description": "Versioned H5AD inputs for the SMITH biological panel-design tutorials.",
                "creators": [{"name": "Fan, Yimin"}],
                "access_right": "open",
                "license": "CC-BY-NC-4.0",
                "keywords": ["SMITH", "spatial transcriptomics", "panel selection", "H5AD"],
            }
        },
    )
    deposition_id = deposition["id"]
    bucket = deposition["links"]["bucket"]
    uploaded: dict[str, dict[str, Any]] = {}
    try:
        for case_id, archive in archives.items():
            with archive.open("rb") as handle:
                file_response = session.put(f"{bucket}/{archive.name}", data=handle, timeout=(30, 3600))
            file_response.raise_for_status()
            uploaded[case_id] = {
                "archive_url": f"{host}/records/{deposition_id}/files/{archive.name}",
                "archive_sha256": sha256(archive),
                "prepared_archive_bytes": archive.stat().st_size,
            }
        if args.publish:
            published = request_json(session, "POST", f"{api}/deposit/depositions/{deposition_id}/actions/publish")
            record_id = published.get("id", deposition_id)
        else:
            record_id = deposition_id
    except Exception:
        session.delete(f"{api}/deposit/depositions/{deposition_id}", timeout=(30, 300))
        raise

    manifest["zenodo_record_url"] = f"{host}/records/{record_id}"
    manifest["zenodo_record_id"] = record_id
    manifest["publication_status"] = "published" if args.publish else "uploaded_unpublished"
    for case_id, values in uploaded.items():
        manifest["cases"][case_id].update(values)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False), encoding="utf-8")
    print(f"Zenodo record: {manifest['zenodo_record_url']}")
    print(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
