from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReproducibilityCase:
    id: str
    order: int
    title: str
    manuscript_section: str
    figure: str
    runtime: str
    data_access: str
    summary: str
    claim: str
    inputs: tuple[dict[str, Any], ...]
    outputs: tuple[str, ...]
    full_workflow: dict[str, Any]
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "order": self.order,
            "title": self.title,
            "manuscript_section": self.manuscript_section,
            "figure": self.figure,
            "runtime": self.runtime,
            "data_access": self.data_access,
            "summary": self.summary,
            "claim": self.claim,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "full_workflow": self.full_workflow,
            "manifest_path": str(self.manifest_path),
        }


def default_reproducibility_root() -> Path:
    source_root = Path(__file__).resolve().parents[3] / "reproducibility"
    if source_root.exists():
        return source_root
    return Path(str(resources.files("smith.reproducibility.resources")))


def load_cases(root: str | Path | None = None) -> dict[str, ReproducibilityCase]:
    repro_root = Path(root).resolve() if root else default_reproducibility_root()
    cases: dict[str, ReproducibilityCase] = {}
    for manifest_path in sorted((repro_root / "manifests").glob("*.yaml")):
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        case = ReproducibilityCase(
            id=str(payload["id"]),
            order=int(payload["order"]),
            title=str(payload["title"]),
            manuscript_section=str(payload["manuscript_section"]),
            figure=str(payload["figure"]),
            runtime=str(payload["runtime"]),
            data_access=str(payload["data_access"]),
            summary=str(payload["summary"]),
            claim=str(payload["claim"]),
            inputs=tuple(payload.get("inputs", [])),
            outputs=tuple(str(item) for item in payload.get("outputs", [])),
            full_workflow=dict(payload.get("full_workflow", {})),
            manifest_path=manifest_path,
        )
        if case.id in cases:
            raise ValueError(f"Duplicate reproducibility case id: {case.id}")
        cases[case.id] = case
    return dict(sorted(cases.items(), key=lambda item: item[1].order))
