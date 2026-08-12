from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smith_agent.config import AgentConfig
from smith_agent.utils import read_yaml


@dataclass
class SkillRegistryEntry:
    id: str
    description: str
    use_when: list[str] = field(default_factory=list)
    workflow: list[str] = field(default_factory=list)
    recommended_tools: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    next_skills: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolRegistryEntry:
    name: str
    description: str
    domain: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetRegistryEntry:
    id: str
    path: str
    description: str = ""
    species: str = ""
    tissue: str = ""
    modality: str = ""
    tasks: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelRegistryEntry:
    id: str
    backend: str
    entrypoint: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BackendRegistryEntry:
    id: str
    backend: str
    stage: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyRegistryEntry:
    id: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_yaml_dir(path: str | Path) -> list[dict[str, Any]]:
    directory = Path(path)
    if not directory.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for file_path in sorted(directory.glob("*.y*ml")):
        if file_path.name.startswith("."):
            continue
        payloads.append(read_yaml(file_path))
    return payloads


def _build_alias_map(entries: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for entry_id, entry in entries.items():
        aliases[entry_id.lower()] = entry_id
        for alias in getattr(entry, "aliases", []):
            aliases[str(alias).lower()] = entry_id
    return aliases


class RegistryBundle:
    def __init__(
        self,
        skills: dict[str, SkillRegistryEntry],
        tools: dict[str, ToolRegistryEntry],
        datasets: dict[str, DatasetRegistryEntry],
        models: dict[str, ModelRegistryEntry],
        baselines: dict[str, BackendRegistryEntry],
        feasibility_backends: dict[str, BackendRegistryEntry],
        probe_backends: dict[str, BackendRegistryEntry],
        policies: dict[str, PolicyRegistryEntry],
    ):
        self.skills = skills
        self.tools = tools
        self.datasets = datasets
        self.models = models
        self.baselines = baselines
        self.feasibility_backends = feasibility_backends
        self.probe_backends = probe_backends
        self.policies = policies
        self.skill_aliases = _build_alias_map(skills)
        self.tool_aliases = _build_alias_map(tools)
        self.dataset_aliases = _build_alias_map(datasets)
        self.model_aliases = _build_alias_map(models)
        self.policy_aliases = _build_alias_map(policies)

    def resolve_skill_id(self, identifier: str | None) -> str | None:
        if not identifier:
            return None
        return self.skill_aliases.get(str(identifier).strip().lower())

    def resolve_tool_name(self, identifier: str | None) -> str | None:
        if not identifier:
            return None
        return self.tool_aliases.get(str(identifier).strip().lower())

    def resolve_dataset_id(self, identifier: str | None) -> str | None:
        if not identifier:
            return None
        return self.dataset_aliases.get(str(identifier).strip().lower())

    def resolve_model_id(self, identifier: str | None) -> str | None:
        if not identifier:
            return None
        return self.model_aliases.get(str(identifier).strip().lower())

    def resolve_policy_id(self, identifier: str | None) -> str | None:
        if not identifier:
            return None
        return self.policy_aliases.get(str(identifier).strip().lower())


def load_skill_registry(path: str | Path) -> dict[str, SkillRegistryEntry]:
    skills: dict[str, SkillRegistryEntry] = {}
    for payload in _load_yaml_dir(path):
        entry = SkillRegistryEntry(
            id=str(payload["id"]),
            description=str(payload.get("description", "")),
            use_when=[str(item) for item in payload.get("use_when", [])],
            workflow=[str(item) for item in payload.get("workflow", [])],
            recommended_tools=[str(item) for item in payload.get("recommended_tools", [])],
            requires=[str(item) for item in payload.get("requires", [])],
            produces=[str(item) for item in payload.get("produces", [])],
            next_skills=[str(item) for item in payload.get("next_skills", [])],
            aliases=[str(item) for item in payload.get("aliases", [])],
            metadata=dict(payload.get("metadata", {})),
        )
        skills[entry.id] = entry
    return skills


def load_tool_registry(path: str | Path) -> dict[str, ToolRegistryEntry]:
    tools: dict[str, ToolRegistryEntry] = {}
    for payload in _load_yaml_dir(path):
        entry = ToolRegistryEntry(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            domain=str(payload.get("domain", "")),
            input_schema=dict(payload.get("input_schema", {})),
            outputs=[str(item) for item in payload.get("outputs", [])],
            tags=[str(item) for item in payload.get("tags", [])],
            aliases=[str(item) for item in payload.get("aliases", [])],
            metadata=dict(payload.get("metadata", {})),
        )
        tools[entry.name] = entry
    return tools


def load_dataset_registry(path: str | Path) -> dict[str, DatasetRegistryEntry]:
    datasets: dict[str, DatasetRegistryEntry] = {}
    for payload in _load_yaml_dir(path):
        entry = DatasetRegistryEntry(
            id=str(payload["id"]),
            path=str(payload["path"]),
            description=str(payload.get("description", "")),
            species=str(payload.get("species", "")),
            tissue=str(payload.get("tissue", "")),
            modality=str(payload.get("modality", "")),
            tasks=[str(item) for item in payload.get("tasks", [])],
            aliases=[str(item) for item in payload.get("aliases", [])],
            metadata=dict(payload.get("metadata", {})),
        )
        datasets[entry.id] = entry
    return datasets


def load_model_registry(path: str | Path) -> dict[str, ModelRegistryEntry]:
    models: dict[str, ModelRegistryEntry] = {}
    for payload in _load_yaml_dir(path):
        entry = ModelRegistryEntry(
            id=str(payload["id"]),
            backend=str(payload["backend"]),
            entrypoint=str(payload["entrypoint"]),
            description=str(payload.get("description", "")),
            aliases=[str(item) for item in payload.get("aliases", [])],
            capabilities=[str(item) for item in payload.get("capabilities", [])],
            metadata=dict(payload.get("metadata", {})),
        )
        models[entry.id] = entry
    return models


def load_backend_registry(path: str | Path) -> dict[str, BackendRegistryEntry]:
    backends: dict[str, BackendRegistryEntry] = {}
    for payload in _load_yaml_dir(path):
        entry = BackendRegistryEntry(
            id=str(payload["id"]),
            backend=str(payload.get("backend", payload["id"])),
            stage=str(payload.get("stage", "")),
            description=str(payload.get("description", "")),
            aliases=[str(item) for item in payload.get("aliases", [])],
            runtime=dict(payload.get("runtime", {})),
            metadata=dict(payload.get("metadata", {})),
        )
        backends[entry.id] = entry
    return backends


def load_policy_registry(path: str | Path) -> dict[str, PolicyRegistryEntry]:
    policies: dict[str, PolicyRegistryEntry] = {}
    for payload in _load_yaml_dir(path):
        entry = PolicyRegistryEntry(
            id=str(payload["id"]),
            description=str(payload.get("description", "")),
            parameters=dict(payload.get("parameters", {})),
            aliases=[str(item) for item in payload.get("aliases", [])],
            metadata=dict(payload.get("metadata", {})),
        )
        policies[entry.id] = entry
    return policies


def load_registries(config: AgentConfig) -> RegistryBundle:
    return RegistryBundle(
        skills=load_skill_registry(config.skills_dir),
        tools=load_tool_registry(config.tools_dir),
        datasets=load_dataset_registry(config.datasets_dir),
        models=load_model_registry(config.models_dir),
        baselines=load_backend_registry(config.baselines_dir),
        feasibility_backends=load_backend_registry(config.feasibility_backends_dir),
        probe_backends=load_backend_registry(config.probe_backends_dir),
        policies=load_policy_registry(config.policies_dir),
    )
