from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .registry import ReproducibilityCase, default_reproducibility_root


WORKFLOW_SCRIPTS = {
    "02_regulatory_activity": "regulatory_activity/run_tutorial.py",
    "03_ribomap_transfer": "ribomap_transfer/run_tutorial.py",
    "05_agent": "agent/run_tutorial.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_case(
    case: ReproducibilityCase,
    root: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    repro_root = Path(root).resolve() if root else default_reproducibility_root()
    data_base = Path(data_root).resolve() if data_root else None
    checks = []
    for spec in case.inputs:
        kind = str(spec.get("kind", "reproducibility"))
        base = data_base if kind == "data" and data_base else repro_root
        path = base / str(spec["path"])
        expected = str(spec.get("sha256") or "")
        actual = _sha256(path) if path.is_file() and expected else ""
        expected_bytes = spec.get("bytes")
        checks.append(
            {
                "path": str(path),
                "kind": kind,
                "exists": path.is_file(),
                "size_ok": expected_bytes is None or (path.is_file() and path.stat().st_size == int(expected_bytes)),
                "sha256_ok": not expected or actual == expected,
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
    available = case.full_workflow.get("availability") != "source_unavailable"
    return {
        "case": case.id,
        "ready": available and bool(data_base) and bool(checks) and all(item["exists"] and item["size_ok"] and item["sha256_ok"] for item in checks),
        "availability": case.full_workflow.get("availability"),
        "data_root": str(data_base) if data_base else None,
        "inputs": checks,
    }


def run_case(
    case: ReproducibilityCase,
    output_dir: str | Path,
    root: str | Path | None = None,
    data_root: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    repro_root = Path(root).resolve() if root else default_reproducibility_root()
    status = check_case(case, repro_root, data_root)
    if not status["ready"]:
        raise FileNotFoundError(f"Real H5AD inputs are missing or invalid for `{case.id}`. Pass --data-root after downloading the case archive.")
    relative_script = WORKFLOW_SCRIPTS.get(case.id)
    if relative_script is None:
        raise RuntimeError(f"Case `{case.id}` is not executable from the public package.")
    script = repro_root / "workflows" / relative_script
    if not script.is_file():
        raise RuntimeError(
            "End-to-end tutorial workflows are distributed with the SMITH GitHub checkout. "
            "Clone the repository and run the command shown in docs/source/tutorials/."
        )
    command = [
        sys.executable, str(script), "--data-root", str(Path(data_root).resolve()),
        "--output-dir", str(Path(output_dir).resolve()), *(extra_args or []),
    ]
    subprocess.run(command, cwd=repro_root.parent, check=True)
    manifest = Path(output_dir).resolve() / "run_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Workflow completed without run_manifest.json: {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))
