from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from smith_agent.config import AgentConfig, load_agent_config
from smith_agent.registry import RegistryBundle, load_registries
from smith_agent.session import AgentSession, SessionStore
from smith_agent.utils import ensure_dir


@dataclass
class ToolRuntime:
    config: AgentConfig
    registries: RegistryBundle
    session_store: SessionStore
    session: AgentSession
    working_dir: Path


def build_runtime(config_path: str | Path | None = None, session_id: str | None = None) -> ToolRuntime:
    config = load_agent_config(config_path)
    registries = load_registries(config)
    session_store = SessionStore(config.sessions_root)
    session = session_store.load(session_id) if session_id else session_store.create()
    working_dir = ensure_dir(config.outputs_root / session.session_id)
    return ToolRuntime(
        config=config,
        registries=registries,
        session_store=session_store,
        session=session,
        working_dir=working_dir,
    )

