from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smith_agent.utils import ensure_dir, iso_timestamp


@dataclass
class SessionMessage:
    role: str
    content: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentSession:
    session_id: str
    created_at: str
    updated_at: str
    active_dataset_id: str = ""
    active_model_id: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    history: list[SessionMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["history"] = [message.to_dict() for message in self.history]
        return payload


class SessionStore:
    def __init__(self, root: str | Path):
        self.root = ensure_dir(root)

    def create(self) -> AgentSession:
        stamp = iso_timestamp()
        session = AgentSession(session_id=uuid.uuid4().hex[:12], created_at=stamp, updated_at=stamp)
        self.save(session)
        return session

    def save(self, session: AgentSession) -> None:
        session.updated_at = iso_timestamp()
        session_dir = ensure_dir(self.root / session.session_id)
        state_path = session_dir / "state.json"
        history_path = session_dir / "history.jsonl"
        payload = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "active_dataset_id": session.active_dataset_id,
            "active_model_id": session.active_model_id,
            "state": session.state,
            "memory": session.memory,
        }
        state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with history_path.open("w", encoding="utf-8") as handle:
            for message in session.history:
                handle.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")

    def load(self, session_id: str) -> AgentSession:
        session_dir = self.root / session_id
        payload = json.loads((session_dir / "state.json").read_text(encoding="utf-8"))
        session = AgentSession(
            session_id=payload["session_id"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            active_dataset_id=str(payload.get("active_dataset_id", "")),
            active_model_id=str(payload.get("active_model_id", "")),
            state=dict(payload.get("state", {})),
            memory=dict(payload.get("memory", {})),
        )
        history_path = session_dir / "history.jsonl"
        if history_path.exists():
            for line in history_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                session.history.append(
                    SessionMessage(
                        role=raw["role"],
                        content=raw["content"],
                        timestamp=raw["timestamp"],
                        metadata=dict(raw.get("metadata", {})),
                    )
                )
        return session

    def append_message(
        self,
        session: AgentSession,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        session.history.append(
            SessionMessage(
                role=role,
                content=content,
                timestamp=iso_timestamp(),
                metadata=metadata or {},
            )
        )
        self.save(session)

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for state_path in sorted(self.root.glob("*/state.json")):
            sessions.append(json.loads(state_path.read_text(encoding="utf-8")))
        return sessions
