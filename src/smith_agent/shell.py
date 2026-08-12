from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smith_agent.interactive import InteractiveSmithAgent
from smith_agent.mode_router import route_control_mode

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import FormattedText, HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style as PromptStyle
except Exception:  # noqa: BLE001
    PromptSession = None
    AutoSuggestFromHistory = None
    Completer = object
    Completion = None
    FormattedText = None
    HTML = None
    FileHistory = None
    PromptStyle = None

try:
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
except Exception:  # noqa: BLE001
    box = None
    Columns = None
    Console = None
    Markdown = None
    Panel = None
    Syntax = None
    Table = None
    Text = None


console = Console() if Console is not None else None


@dataclass(frozen=True)
class CommandHint:
    command: str
    usage: str
    description: str

    @property
    def insert_text(self) -> str:
        return f"{self.command} " if self.usage != self.command else self.command


HELP_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Session",
        [
            ("/help", "Show the command palette"),
            ("/status", "Show current session state"),
            ("/summary", "Summarize panel, feasibility, evaluation, and report artifacts"),
            ("/planner", "Show planner model status"),
            ("/mode <message>", "Preview whether a message routes to pipeline or ReAct"),
            ("/history", "Show recent conversation and tool events"),
            ("/reset", "Clear session state and memory"),
            ("/exit", "Exit the shell"),
        ],
    ),
    (
        "Registries",
        [
            ("/datasets", "List registered datasets"),
            ("/models", "List registered SMITH models"),
            ("/skills", "List registered workflow skills"),
            ("/tools", "List registered executable tools"),
            ("/sessions", "List saved sessions"),
        ],
    ),
    (
        "Discovery",
        [
            ("/search <query>", "Search registries, docs, configs, and saved artifacts"),
            ("/use-dataset <id>", "Set active dataset"),
            ("/use-model <id>", "Set active model"),
            ("/run-tool <name> <json>", "Run one registered tool with JSON arguments"),
        ],
    ),
    (
        "Typical Requests",
        [
            ("inspect @/path/data.h5ad", "Inspect an ad hoc dataset"),
            ("run formal panel selection", "Run the deterministic SMITH panel pipeline"),
            ("skip ODT and build report", "Use the faster feasibility path"),
            ("analyze the result", "Use planner-driven analysis tools where possible"),
        ],
    ),
]


def _command_hints() -> list[CommandHint]:
    hints: list[CommandHint] = []
    for _, rows in HELP_SECTIONS:
        for usage, description in rows:
            command = usage.split()[0]
            if command.startswith("/"):
                hints.append(CommandHint(command=command, usage=usage, description=description))
    return hints


class SlashCommandCompleter(Completer):  # type: ignore[misc]
    def __init__(self, hints: list[CommandHint]) -> None:
        self.hints = hints

    def get_completions(self, document, complete_event):  # noqa: D401
        if Completion is None:
            return
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return
        fragment = text.split(None, 1)[0] if " " in text else text
        for hint in self.hints:
            if hint.command.startswith(fragment):
                yield Completion(
                    hint.insert_text,
                    start_position=-len(fragment),
                    display=hint.usage,
                    display_meta=hint.description,
                )


def _plain_print(message: str = "") -> None:
    print(message)


def _print(message: Any = "") -> None:
    if console is None:
        _plain_print(str(message))
    else:
        console.print(message)


def _format_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _render_json(payload: Any) -> None:
    text = _format_payload(payload)
    if console is None or Syntax is None:
        _plain_print(text)
        return
    _print(Syntax(text, "json", theme="ansi_dark", word_wrap=True))


def _print_session_footer(session_id: str) -> None:
    _print(f"[dim]session saved:[/] {session_id}" if console else f"session saved: {session_id}")
    _print(f"[dim]resume with:[/] smith-cli resume {session_id}" if console else f"resume with: smith-cli resume {session_id}")


def _build_prompt_session(agent: InteractiveSmithAgent):
    if PromptSession is None or not sys.stdin.isatty():
        return None
    history_path = Path(agent.config.sessions_root) / ".smith_cli_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    style = PromptStyle.from_dict(
        {
            "prompt.main": "bold #7dd3fc",
            "prompt.sep": "#64748b",
            "bottom-toolbar": "fg:#a8b3c7 bg:#101827",
            "completion-menu.completion": "fg:#dce6f5 bg:#111827",
            "completion-menu.completion.current": "fg:#ffffff bg:#334155",
            "completion-menu.meta.completion": "fg:#9ca3af bg:#111827",
            "completion-menu.meta.completion.current": "fg:#e5e7eb bg:#334155",
        }
    )
    return PromptSession(
        history=FileHistory(str(history_path)) if FileHistory is not None else None,
        auto_suggest=AutoSuggestFromHistory() if AutoSuggestFromHistory is not None else None,
        completer=SlashCommandCompleter(_command_hints()),
        complete_while_typing=True,
        complete_in_thread=True,
        reserve_space_for_menu=8,
        style=style,
        bottom_toolbar=lambda: HTML(
            "<b>Tab</b> commands | <b>@/path.h5ad</b> mount dataset | <b>/summary</b> artifacts"
        )
        if HTML is not None
        else None,
    )


def _prompt_message():
    if FormattedText is None:
        return "smith > "
    return FormattedText(
        [
            ("class:prompt.main", "smith"),
            ("class:prompt.sep", " > "),
        ]
    )


def _read_shell_input(prompt_session) -> str:
    if prompt_session is None:
        return input("smith > ")
    return " ".join(prompt_session.prompt(_prompt_message()).split())


def _logo() -> str:
    return (
        "  ____  __  __ ___ _____ _   _\n"
        " / ___||  \\/  |_ _|_   _| | | |\n"
        " \\___ \\| |\\/| || |  | | | |_| |\n"
        "  ___) | |  | || |  | | |  _  |\n"
        " |____/|_|  |_|___| |_| |_| |_|\n"
        "        A G E N T"
    )


def _render_startup_dashboard(agent: InteractiveSmithAgent, session) -> None:
    status = agent.planner_status()
    if console is None or Panel is None or Table is None or Columns is None:
        _plain_print(f"smith-cli session: {session.session_id}")
        _plain_print(f"active dataset: {session.active_dataset_id or '(none)'}")
        _plain_print(f"active model: {session.active_model_id or '(none)'}")
        _plain_print(f"planner: {status['status']} ({status['detail']})")
        _plain_print("type /help for commands")
        return

    workspace = Table.grid(padding=(0, 1))
    workspace.add_row(f"[bold #7dd3fc]{_logo()}[/]")
    workspace.add_row("[#d1d5db]Panel design agent for SMITH workflows[/]")
    workspace.add_row("")
    workspace.add_row(f"[#93c5fd]session[/]: [bold]{session.session_id}[/]")
    workspace.add_row(f"[#93c5fd]dataset[/]: [bold]{session.active_dataset_id or '(none)'}[/]")
    workspace.add_row(f"[#93c5fd]model[/]: [bold]{session.active_model_id or '(none)'}[/]")
    workspace.add_row(f"[#93c5fd]planner[/]: [bold]{status['status']}[/] [dim]({status['detail']})[/]")

    controls = Table.grid(padding=(0, 1))
    controls.add_row("[bold #f9a8d4]Controls[/]")
    controls.add_row("[#cbd5e1]/help[/] command palette")
    controls.add_row("[#cbd5e1]/summary[/] current artifacts")
    controls.add_row("[#cbd5e1]/mode[/] route preview")
    controls.add_row("[#cbd5e1]/search[/] registry and docs search")
    controls.add_row("[#cbd5e1]/status[/] session state")
    controls.add_row("[#cbd5e1]/tools[/] executable tool list")
    controls.add_row("[#cbd5e1]@/path/file.h5ad[/] mount dataset")

    caps = Table.grid(expand=True)
    caps.add_row(
        f"[bold #22c55e]pipeline[/] formal panel selection   "
        f"[bold #38bdf8]analysis[/] planner-guided tools   "
        f"[bold #f59e0b]reports[/] markdown/html/pdf   "
        f"[bold #a78bfa]registry[/] {len(agent.registries.tools)} tools"
    )

    console.print(
        Columns(
            [
                Panel(workspace, title="[bold #7dd3fc]SMITH-Agent[/]", border_style="#334155", box=box.ROUNDED, padding=(1, 2)),
                Panel(controls, title="[bold #f9a8d4]Command Deck[/]", border_style="#334155", box=box.ROUNDED, padding=(1, 2)),
            ],
            equal=True,
            expand=True,
        )
    )
    console.print(Panel(caps, border_style="#1f2937", box=box.SQUARE, padding=(0, 1)))
    console.print("[dim]Press Ctrl+C or type /exit to leave.[/]\n")


def _render_help() -> None:
    if console is None or Table is None or Panel is None:
        _plain_print(_help_text())
        return
    console.print(
        Panel(
            "[bold #7dd3fc]Command Palette[/]\n[dim]Type / then press Tab for autocomplete.[/]",
            border_style="#334155",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    for section, rows in HELP_SECTIONS:
        table = Table(title=section, box=box.SIMPLE_HEAVY, show_lines=False, padding=(0, 1))
        table.add_column("Command", style="#67e8f9", no_wrap=True)
        table.add_column("Description", style="#e5e7eb")
        for usage, description in rows:
            table.add_row(usage, description)
        console.print(table)


def _help_text() -> str:
    lines = ["Commands:"]
    for _, rows in HELP_SECTIONS:
        for usage, description in rows:
            lines.append(f"{usage:<32} {description}")
    return "\n".join(lines)


def _render_registry_table(title: str, rows: list[dict[str, Any]]) -> None:
    if console is None or Table is None:
        _plain_print(_format_payload(rows))
        return
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False, padding=(0, 1))
    table.add_column("ID", style="#67e8f9", no_wrap=True)
    table.add_column("Kind", style="#a7f3d0", no_wrap=True)
    table.add_column("Description", style="#e5e7eb")
    for row in rows:
        identifier = str(row.get("id") or row.get("name") or row.get("tool_name") or "?")
        kind = str(row.get("domain") or row.get("kind") or row.get("type") or row.get("runtime", "") or "")
        description = str(row.get("description") or row.get("path") or row.get("adapter") or "")
        if len(description) > 110:
            description = description[:107] + "..."
        table.add_row(identifier, kind, description)
    console.print(table)


def _render_sessions(rows: list[dict[str, Any]]) -> None:
    if console is None or Table is None:
        _plain_print(_format_payload(rows))
        return
    table = Table(title="Saved Sessions", box=box.SIMPLE_HEAVY, show_lines=False, padding=(0, 1))
    table.add_column("Session", style="#67e8f9", no_wrap=True)
    table.add_column("Updated", style="#cbd5e1")
    table.add_column("Dataset", style="#e5e7eb")
    table.add_column("Model", style="#e5e7eb")
    for row in rows[-25:]:
        table.add_row(
            str(row.get("session_id", "")),
            str(row.get("updated_at", ""))[:19],
            str(row.get("active_dataset_id", "")),
            str(row.get("active_model_id", "")),
        )
    console.print(table)


def _render_history(session) -> None:
    rows = [message.to_dict() for message in session.history[-20:]]
    if console is None or Table is None:
        _plain_print(_format_payload(rows))
        return
    table = Table(title="Recent History", box=box.SIMPLE_HEAVY, show_lines=False, padding=(0, 1))
    table.add_column("Time", style="#94a3b8", no_wrap=True)
    table.add_column("Role", style="#67e8f9", no_wrap=True)
    table.add_column("Content", style="#e5e7eb")
    for row in rows:
        content = str(row.get("content", "")).replace("\n", " ")
        if len(content) > 120:
            content = content[:117] + "..."
        table.add_row(str(row.get("timestamp", ""))[:19], str(row.get("role", "")), content)
    console.print(table)


def _render_summary(summary: str) -> None:
    if console is None or Markdown is None or Panel is None:
        _plain_print(summary)
        return
    console.print(Panel(Markdown(summary), title="Session Summary", border_style="#334155", box=box.ROUNDED))


def _render_text_response(response: str) -> None:
    if console is None or Markdown is None:
        _plain_print(response)
        return
    console.print()
    console.print(Markdown(response))
    console.print()


def _render_progress(event: dict[str, Any]) -> None:
    message = str(event.get("message", "")).strip()
    if not message:
        return
    event_type = str(event.get("event", ""))
    if console is None:
        _plain_print(message)
        return
    style = {
        "tool_start": "bold #38bdf8",
        "tool_end": "#22c55e",
        "tool_error": "bold red",
        "planner_note": "#cbd5e1",
    }.get(event_type, "#cbd5e1")
    console.print(f"[{style}]{message}[/]")


def run_shell(agent: InteractiveSmithAgent, session_id: str | None = None) -> int:
    session = agent.load_session(session_id) if session_id else agent.create_session()
    agent._runtime(session)
    prompt_session = _build_prompt_session(agent)
    _render_startup_dashboard(agent, session)

    while True:
        try:
            raw = _read_shell_input(prompt_session).strip()
        except (EOFError, KeyboardInterrupt):
            _print()
            _print_session_footer(session.session_id)
            return 0
        if not raw:
            continue
        if raw in {"/exit", "/quit", "quit", "exit", "q"}:
            _print_session_footer(session.session_id)
            return 0
        if raw == "/help":
            _render_help()
            continue
        if raw == "/planner":
            _render_json(agent.planner_status())
            continue
        if raw.startswith("/mode "):
            message = raw.split(" ", 1)[1].strip()
            _render_json({"mode": route_control_mode(message).value, "message": message})
            continue
        if raw == "/summary":
            _render_summary(agent.summarize_session(session, cached=True))
            continue
        if raw == "/status":
            _render_json(agent.execute_tool(session, "inspect_session_state", {}))
            continue
        if raw == "/datasets":
            _render_registry_table("Datasets", [entry.to_dict() for entry in agent.registries.datasets.values()])
            continue
        if raw == "/models":
            _render_registry_table("Models", [entry.to_dict() for entry in agent.registries.models.values()])
            continue
        if raw == "/skills":
            _render_registry_table("Skills", [entry.to_dict() for entry in agent.registries.skills.values()])
            continue
        if raw == "/tools":
            _render_registry_table("Tools", [tool.to_prompt_dict() for tool in agent.tool_registry.list_tools()] if agent.tool_registry else [])
            continue
        if raw == "/sessions":
            _render_sessions(agent.session_store.list_sessions())
            continue
        if raw == "/history":
            _render_history(session)
            continue
        if raw.startswith("/search "):
            query = raw.split(" ", 1)[1].strip()
            _render_json(agent.execute_tool(session, "search_smith_agent", {"query": query}))
            continue
        if raw.startswith("/use-dataset "):
            dataset_id = raw.split(" ", 1)[1].strip()
            _render_json(agent.execute_tool(session, "set_active_dataset", {"dataset_id": dataset_id}))
            continue
        if raw.startswith("/use-model "):
            model_id = raw.split(" ", 1)[1].strip()
            _render_json(agent.execute_tool(session, "set_active_model", {"model_id": model_id}))
            continue
        if raw.startswith("/run-tool "):
            rest = raw.split(" ", 1)[1].strip()
            if " " not in rest:
                _print("[yellow]Usage:[/] /run-tool <name> <json-arguments>" if console else "Usage: /run-tool <name> <json-arguments>")
                continue
            tool_name, arg_text = rest.split(" ", 1)
            arguments = json.loads(arg_text)
            _render_json(agent.execute_tool(session, tool_name, arguments))
            continue
        if raw == "/reset":
            session.state.clear()
            session.memory.clear()
            agent.session_store.save(session)
            _print("[green]session state cleared[/]" if console else "session state cleared")
            continue

        _render_text_response(agent.handle_message(session, raw, progress_callback=_render_progress))
