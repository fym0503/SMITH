from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from smith_agent.utils import read_yaml, repo_root_from_file


def _normalize_base_url(value: str) -> str:
    normalized = str(value).strip().rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized


@dataclass
class LLMSettings:
    provider: str = "openai"
    model: str = ""
    base_url: str = "https://api.babelark.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str = ""
    temperature: float = 0.2
    enabled: bool = True
    timeout_seconds: int = 120
    planner_step_tool_limit: int = 2

    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get(self.api_key_env, "")

    def is_configured(self) -> bool:
        return bool(self.enabled and self.model and self.resolved_api_key())


@dataclass
class AgentConfig:
    repo_root: Path
    reports_root: Path
    sessions_root: Path
    outputs_root: Path
    env_file: Path
    skills_dir: Path
    tools_dir: Path
    datasets_dir: Path
    models_dir: Path
    baselines_dir: Path
    feasibility_backends_dir: Path
    probe_backends_dir: Path
    policies_dir: Path
    max_turn_loops: int = 8
    external_roots: dict[str, Path] = field(default_factory=dict)
    llm: LLMSettings = field(default_factory=LLMSettings)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["repo_root"] = str(self.repo_root)
        payload["reports_root"] = str(self.reports_root)
        payload["sessions_root"] = str(self.sessions_root)
        payload["outputs_root"] = str(self.outputs_root)
        payload["env_file"] = str(self.env_file)
        payload["skills_dir"] = str(self.skills_dir)
        payload["tools_dir"] = str(self.tools_dir)
        payload["datasets_dir"] = str(self.datasets_dir)
        payload["models_dir"] = str(self.models_dir)
        payload["baselines_dir"] = str(self.baselines_dir)
        payload["feasibility_backends_dir"] = str(self.feasibility_backends_dir)
        payload["probe_backends_dir"] = str(self.probe_backends_dir)
        payload["policies_dir"] = str(self.policies_dir)
        payload["external_roots"] = {key: str(value) for key, value in self.external_roots.items()}
        return payload


def _resolve_path(base_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_root / path).resolve()


def _load_dotenv(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _packaged_resources_root() -> Path | None:
    try:
        root = resources.files("smith_agent.resources")
    except (AttributeError, ModuleNotFoundError):
        return None
    path = Path(str(root))
    return path if path.exists() else None


def _infer_config_base(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.name == "agent.yaml" and resolved.parent.name == "agent" and resolved.parent.parent.name == "configs":
        return resolved.parents[2]
    if resolved.name == "agent.yaml" and resolved.parent.name == "configs":
        return resolved.parents[1]
    return resolved.parent


def _select_default_config() -> tuple[Path, Path, Path]:
    source_root = repo_root_from_file(__file__)
    source_agent_config = source_root / "configs" / "agent" / "agent.yaml"
    source_legacy_config = source_root / "configs" / "agent.yaml"
    if source_agent_config.exists():
        return source_agent_config, source_root, source_root
    if source_legacy_config.exists():
        return source_legacy_config, source_root, source_root

    resource_root = _packaged_resources_root()
    if resource_root is not None:
        resource_config = resource_root / "configs" / "agent" / "agent.yaml"
        if resource_config.exists():
            runtime_root = Path(os.environ.get("SMITH_RUNTIME_ROOT", Path.cwd())).resolve()
            return resource_config, runtime_root, resource_root

    return source_agent_config, source_root, source_root


def load_agent_config(config_path: str | Path | None = None) -> AgentConfig:
    if config_path is None:
        selected_config, repo_root, config_base = _select_default_config()
    else:
        selected_config = Path(config_path).resolve()
        config_base = _infer_config_base(selected_config)
        repo_root = Path(os.environ.get("SMITH_RUNTIME_ROOT", config_base)).resolve()
    payload = read_yaml(selected_config)

    env_file = _resolve_path(repo_root, str(payload.get("env_file", ".env")))
    _load_dotenv(env_file)

    external_roots = {
        key: _resolve_path(config_base, str(value))
        for key, value in dict(payload.get("external_roots", {})).items()
    }
    llm_payload = dict(payload.get("llm", {}))
    llm = LLMSettings(
        provider=str(llm_payload.get("provider", "openai")),
        model=str(os.environ.get("OPENAI_MODEL", llm_payload.get("model", ""))),
        base_url=_normalize_base_url(str(os.environ.get("OPENAI_BASE_URL", llm_payload.get("base_url", "https://api.babelark.com/v1")))),
        api_key_env=str(llm_payload.get("api_key_env", "OPENAI_API_KEY")),
        api_key=str(llm_payload.get("api_key", "")),
        temperature=float(llm_payload.get("temperature", 0.2)),
        enabled=bool(llm_payload.get("enabled", True)),
        timeout_seconds=int(llm_payload.get("timeout_seconds", 120)),
        planner_step_tool_limit=int(llm_payload.get("planner_step_tool_limit", 2)),
    )

    return AgentConfig(
        repo_root=repo_root,
        reports_root=_resolve_path(repo_root, str(payload.get("reports_root", "reports"))),
        sessions_root=_resolve_path(repo_root, str(payload.get("sessions_root", "sessions"))),
        outputs_root=_resolve_path(repo_root, str(payload.get("outputs_root", "outputs"))),
        env_file=env_file,
        skills_dir=_resolve_path(config_base, str(payload.get("skills_dir", "configs/skills"))),
        tools_dir=_resolve_path(config_base, str(payload.get("tools_dir", "configs/tools"))),
        datasets_dir=_resolve_path(config_base, str(payload.get("datasets_dir", "configs/datasets"))),
        models_dir=_resolve_path(config_base, str(payload.get("models_dir", "configs/models"))),
        baselines_dir=_resolve_path(config_base, str(payload.get("baselines_dir", "configs/baselines"))),
        feasibility_backends_dir=_resolve_path(
            config_base,
            str(payload.get("feasibility_backends_dir", "configs/feasibility_backends")),
        ),
        probe_backends_dir=_resolve_path(
            config_base,
            str(payload.get("probe_backends_dir", "configs/probe_backends")),
        ),
        policies_dir=_resolve_path(config_base, str(payload.get("policies_dir", "configs/policies"))),
        max_turn_loops=int(payload.get("max_turn_loops", 8)),
        external_roots=external_roots,
        llm=llm,
    )
