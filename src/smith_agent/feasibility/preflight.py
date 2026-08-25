"""Non-destructive checks for probe-feasibility backends.

The tutorial uses this module before attempting probe design.  It reports
whether each configured backend has its executable/reference prerequisites;
it never substitutes a reference output for a missing backend.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any

from smith_agent.config import load_agent_config
from smith_agent.registry import load_registries


def _path_status(path: str | Path) -> tuple[bool, str]:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return True, str(candidate)
    resolved = shutil.which(str(path))
    return (resolved is not None, resolved or str(candidate))


def probe_backend_preflight(config_path: str | Path | None = None, *, package_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return live availability checks for configured feasibility/probe tools."""
    config = load_agent_config(config_path)
    registries = load_registries(config)
    root = Path(package_root or config.repo_root).resolve()
    entries = list(registries.feasibility_backends.values()) + list(registries.probe_backends.values())
    rows: list[dict[str, Any]] = []
    for entry in entries:
        backend = entry.backend.lower()
        requirements: list[tuple[str, bool, str]] = []
        if backend in {"odt", "odt_scrinshot"}:
            script = Path(os.environ.get("SMITH_ODT_LIGHT_SCRIPT", root / "scripts" / "odt_property_only_runner.py"))
            ok, detail = _path_status(script)
            requirements.append(("property runner", ok, detail))
        elif backend == "oligominer":
            python = os.environ.get("SMITH_OLIGOMINER_PYTHON", str(root / "third_party/envs/oligominer/bin/python"))
            bowtie = os.environ.get("SMITH_OLIGOMINER_BOWTIE2", str(root / "third_party/envs/oligominer/bin/bowtie2"))
            requirements.extend((name, *_path_status(value)) for name, value in (("OligoMiner runtime", python), ("Bowtie2", bowtie)))
        elif backend == "probedealer":
            light = importlib.util.find_spec("smith_agent.probedealer") is not None
            requirements.append(("lightweight Python backend", light, "smith_agent.probedealer"))
            blast = os.environ.get("SMITH_BLASTN", str(root / "third_party/envs/probedealer_blast/bin/blastn"))
            requirements.append(("BLASTN transcriptome filter", *_path_status(blast)))
        elif backend == "paintshop":
            available = importlib.util.find_spec("paintshop") is not None
            requirements.append(("PaintSHOP Python package", available, "paintshop"))
        else:
            requirements.append(("configured backend", False, f"unknown backend: {entry.backend}"))
        rows.append({
            "id": entry.id,
            "backend": entry.backend,
            "stage": entry.stage,
            "available": bool(requirements) and all(item[1] for item in requirements),
            "requirements": requirements,
            "reason": "ready" if all(item[1] for item in requirements) else "; ".join(f"{name}: unavailable" for name, ok, _ in requirements if not ok),
        })
    return rows
