from __future__ import annotations

import argparse
import json
from typing import Any

from smith_agent.config import load_agent_config
from smith_agent.runtime import build_runtime
from smith_agent.interactive import InteractiveSmithAgent
from smith_agent.shell import run_shell
from smith_agent.tools.defaults import build_default_tool_registry


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(prog="smith-cli")
    parser.add_argument("--config", default=None)
    parser.add_argument("--session-id", default=None)
    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("shell")
    resume = subparsers.add_parser("resume")
    resume.add_argument("session_id")
    subparsers.add_parser("skills")
    subparsers.add_parser("tools")
    subparsers.add_parser("datasets")
    subparsers.add_parser("models")
    subparsers.add_parser("sessions")
    describe_skill = subparsers.add_parser("describe-skill")
    describe_skill.add_argument("skill_id")

    run_tool = subparsers.add_parser("run-tool")
    run_tool.add_argument("tool_name")
    run_tool.add_argument("--arguments-json", default="{}")

    new_session = subparsers.add_parser("new-session")
    new_session.add_argument("--print-state", action="store_true")

    args = parser.parse_args()
    interactive_agent = InteractiveSmithAgent(load_agent_config(args.config))

    if args.command in {None, "shell"}:
        raise SystemExit(run_shell(interactive_agent))
    if args.command == "resume":
        raise SystemExit(run_shell(interactive_agent, session_id=args.session_id))

    runtime = build_runtime(config_path=args.config, session_id=args.session_id)
    tool_registry = build_default_tool_registry(runtime)

    if args.command == "skills":
        _print_json([entry.to_dict() for entry in runtime.registries.skills.values()])
        return
    if args.command == "tools":
        _print_json([entry.to_dict() for entry in runtime.registries.tools.values()])
        return
    if args.command == "datasets":
        _print_json([entry.to_dict() for entry in interactive_agent.registries.datasets.values()])
        return
    if args.command == "models":
        _print_json([entry.to_dict() for entry in interactive_agent.registries.models.values()])
        return
    if args.command == "sessions":
        _print_json(interactive_agent.session_store.list_sessions())
        return
    if args.command == "describe-skill":
        skill_id = runtime.registries.resolve_skill_id(args.skill_id)
        if not skill_id:
            raise SystemExit(f"Unknown skill: {args.skill_id}")
        _print_json(runtime.registries.skills[skill_id].to_dict())
        return
    if args.command == "new-session":
        payload = {"session_id": runtime.session.session_id}
        if args.print_state:
            payload["state"] = runtime.session.to_dict()
        _print_json(payload)
        return
    if args.command == "run-tool":
        arguments = json.loads(args.arguments_json)
        result = tool_registry.execute(runtime, args.tool_name, arguments)
        _print_json(result)
        return


if __name__ == "__main__":
    main()
