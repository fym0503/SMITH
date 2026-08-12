from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BackendResult:
    backend: str
    status: str
    input_summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    output_files: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PanelDesignRequest:
    request_id: str
    dataset_path: str = ""
    species: str = ""
    tissue: str = ""
    modality: str = ""
    panel_budget: int | None = None
    tasks: list[str] = field(default_factory=list)
    objective: str = ""
    must_keep_genes: list[str] = field(default_factory=list)
    forbidden_genes: list[str] = field(default_factory=list)
    candidate_genes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PanelPipelineRequest:
    train_adata_file: str = ""
    test_adata_file: str = ""
    panel_size: int = 64
    objective: str = "balanced"
    tasks: list[str] = field(default_factory=lambda: ["recon", "cls", "region", "pathology"])
    label: str = "pathology"
    obsm_key: str = "X_pca"
    species: str = "homo_sapiens"
    formal: bool = True
    epoch: int = 5
    run_selection: bool = True
    run_feasibility: bool = False
    run_evaluation: bool = False
    build_report: bool = False
    skip_odt: bool = False
    must_keep_genes: list[str] = field(default_factory=list)
    forbidden_genes: list[str] = field(default_factory=list)
    report_format: str = "pdf"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def task_string(self) -> str:
        return ",".join(task for task in self.tasks if task)


@dataclass
class ExternalRoots:
    smith_unified_root: Path
    smith_package_root: Path


@dataclass
class ToolExecutionResult:
    tool: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
