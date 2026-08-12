from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _normalize_query(value: str) -> str:
    return " ".join(str(value).strip().split())


def search_registry_entries(registries, query: str, limit: int = 20) -> list[dict[str, Any]]:
    normalized = _normalize_query(query).lower()
    if not normalized:
        return []
    groups = [
        ("dataset", registries.datasets),
        ("model", registries.models),
        ("skill", registries.skills),
        ("tool", registries.tools),
        ("baseline", registries.baselines),
        ("feasibility_backend", registries.feasibility_backends),
        ("probe_backend", registries.probe_backends),
        ("policy", registries.policies),
    ]
    results: list[dict[str, Any]] = []
    for kind, mapping in groups:
        for entry_id, entry in mapping.items():
            payload = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
            haystack = json.dumps(payload, ensure_ascii=False).lower()
            if normalized in haystack:
                results.append(
                    {
                        "kind": kind,
                        "id": entry_id,
                        "summary": str(payload.get("description", "") or payload.get("path", "") or payload.get("entrypoint", "")),
                    }
                )
                if len(results) >= limit:
                    return results
    return results


def search_file_tree(
    query: str,
    roots: list[Path],
    limit: int = 20,
    include_content: bool = True,
) -> list[dict[str, Any]]:
    normalized = _normalize_query(query)
    if not normalized:
        return []
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        return []

    results: list[dict[str, Any]] = []
    for root in existing_roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if normalized.lower() in path.name.lower():
                results.append(
                    {
                        "path": str(path),
                        "match_type": "filename",
                        "line": None,
                        "snippet": path.name,
                    }
                )
                if len(results) >= limit:
                    return results

    if include_content and len(results) < limit:
        command = ["rg", "-n", "-i", "-F", "-m", "1", normalized]
        command.extend(str(root) for root in existing_roots)
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode in {0, 1}:
            for raw_line in completed.stdout.splitlines():
                if len(results) >= limit:
                    break
                parts = raw_line.split(":", 2)
                if len(parts) != 3:
                    continue
                path_text, line_no, snippet = parts
                results.append(
                    {
                        "path": path_text,
                        "match_type": "content",
                        "line": int(line_no) if line_no.isdigit() else None,
                        "snippet": snippet.strip(),
                    }
                )
    return results[:limit]


def search_session_artifacts(
    query: str,
    sessions_root: Path,
    outputs_root: Path,
    limit: int = 20,
) -> list[dict[str, Any]]:
    normalized = _normalize_query(query).lower()
    if not normalized:
        return []
    results: list[dict[str, Any]] = []
    for root, kind in [(sessions_root, "session"), (outputs_root, "output")]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if normalized in path.name.lower():
                results.append(
                    {
                        "kind": kind,
                        "path": str(path),
                        "match_type": "filename",
                    }
                )
                if len(results) >= limit:
                    return results
    return results
