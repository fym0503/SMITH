from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from smith_agent.runtime import ToolRuntime

ToolHandler = Callable[[ToolRuntime, dict[str, Any]], dict[str, Any]]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    outputs: list[str]
    handler: ToolHandler

    def to_prompt_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("handler", None)
        return payload


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]):
        self.tools = {tool.name: tool for tool in tools}

    def list_tools(self) -> list[ToolSpec]:
        return list(self.tools.values())

    def execute(self, runtime: ToolRuntime, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        return self.tools[name].handler(runtime, arguments)

