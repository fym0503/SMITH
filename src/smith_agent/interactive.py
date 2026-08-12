from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from smith_agent.config import AgentConfig, load_agent_config
from smith_agent.llm import LLMClient, fallback_interactive_plan
from smith_agent.mode_router import ControlMode, route_control_mode
from smith_agent.registry import RegistryBundle, load_registries
from smith_agent.result_summary import build_session_summary
from smith_agent.runtime import ToolRuntime
from smith_agent.schemas import PanelPipelineRequest
from smith_agent.session import AgentSession, SessionStore
from smith_agent.tools.defaults import build_default_tool_registry
from smith_agent.tool_registry import ToolRegistry
from smith_agent.utils import iso_timestamp


DATASET_MENTION_RE = re.compile(r"@(/[^ \n\t]+\.h5ad)")


class InteractiveSmithAgent:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or load_agent_config()
        self.registries: RegistryBundle = load_registries(self.config)
        self.session_store = SessionStore(self.config.sessions_root)
        self.llm = LLMClient(self.config.llm)
        self.tool_registry: ToolRegistry | None = None

    def create_session(self) -> AgentSession:
        session = self.session_store.create()
        if self.registries.datasets and not session.active_dataset_id:
            session.active_dataset_id = next(iter(self.registries.datasets.keys()))
        if self.registries.models and not session.active_model_id:
            session.active_model_id = next(iter(self.registries.models.keys()))
        self.session_store.save(session)
        return session

    def load_session(self, session_id: str) -> AgentSession:
        return self.session_store.load(session_id)

    def planner_status(self) -> dict[str, str]:
        if not self.config.llm.enabled:
            return {"status": "disabled", "detail": "planner disabled in config"}
        if self.llm.is_configured():
            return {"status": "enabled", "detail": f"{self.config.llm.model} via {self.config.llm.base_url}"}
        return {"status": "unconfigured", "detail": "missing OPENAI_API_KEY or OPENAI_MODEL"}

    def _runtime(self, session: AgentSession) -> ToolRuntime:
        runtime = ToolRuntime(
            config=self.config,
            registries=self.registries,
            session_store=self.session_store,
            session=session,
            working_dir=(self.config.outputs_root / session.session_id),
        )
        if self.tool_registry is None:
            self.tool_registry = build_default_tool_registry(runtime)
        return runtime

    def execute_tool(self, session: AgentSession, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        runtime = self._runtime(session)
        return self.tool_registry.execute(runtime, tool_name, arguments)

    def summarize_session(self, session: AgentSession, cached: bool = False) -> str:
        del cached
        return build_session_summary(
            {
                "session_id": session.session_id,
                "active_dataset_id": session.active_dataset_id,
                "active_model_id": session.active_model_id,
                "state": session.state,
            }
        )

    def handle_message(
        self,
        session: AgentSession,
        user_message: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        mounted = self._mount_dataset_mentions(session, user_message)
        if mounted:
            self.session_store.append_message(
                session,
                "assistant",
                f"Mounted {len(mounted)} dataset(s) from user-provided @ paths and set the first one active for this session.",
                {"source": "dataset_mount"},
            )
        self.session_store.append_message(session, "user", user_message)
        runtime = self._runtime(session)
        tool_results: list[dict[str, Any]] = []
        mode = route_control_mode(user_message)
        session.memory["last_control_mode"] = mode.value
        self.session_store.save(session)
        allow_llm = self.llm.is_configured()
        planner_failed_this_turn = False

        for _ in range(self.config.max_turn_loops):
            decision, llm_failed = self._plan_turn(
                session,
                user_message,
                tool_results,
                allow_llm=allow_llm,
                mode=mode,
            )
            if llm_failed:
                allow_llm = False
                planner_failed_this_turn = True
            self._apply_state_updates(session, decision.get("state_updates", {}))
            decision = self._repair_tool_dependencies(session, user_message, decision, mode=mode)

            tool_calls = decision.get("tool_calls", [])
            if tool_calls:
                planning_note = str(decision.get("assistant_message", "")).strip()
                if planning_note:
                    self._emit_progress(progress_callback, {"event": "planner_note", "message": planning_note})
                for tool_call in tool_calls[: self.config.llm.planner_step_tool_limit]:
                    self._emit_progress(
                        progress_callback,
                        {"event": "tool_start", "message": self._tool_start_message(tool_call["tool_name"])},
                    )
                    try:
                        result = self.tool_registry.execute(runtime, tool_call["tool_name"], dict(tool_call.get("arguments", {})))
                    except Exception as exc:  # noqa: BLE001
                        message = self._format_tool_error(tool_call["tool_name"], exc)
                        self.session_store.append_message(session, "assistant", message, {"tool_name": tool_call["tool_name"], "status": "error"})
                        self._emit_progress(progress_callback, {"event": "tool_error", "message": message})
                        return message
                    record = {
                        "tool_name": tool_call["tool_name"],
                        "arguments": dict(tool_call.get("arguments", {})),
                        "result": result,
                    }
                    tool_results.append(record)
                    self.session_store.append_message(session, "tool", json.dumps(record, indent=2), {"tool_name": tool_call["tool_name"]})
                    self._emit_progress(
                        progress_callback,
                        {"event": "tool_end", "message": self._tool_end_message(tool_call["tool_name"], result)},
                    )
                if not allow_llm:
                    break
                if decision.get("done"):
                    break
                continue

            assistant_message = str(decision.get("assistant_message", "")).strip()
            if assistant_message:
                self.session_store.append_message(session, "assistant", assistant_message)
                return assistant_message

        if tool_results:
            if self.llm.is_configured() and not planner_failed_this_turn:
                try:
                    final_message = self.llm.summarize_interactive_results(
                        user_message,
                        self._session_snapshot(session),
                        tool_results,
                    )
                except Exception:  # noqa: BLE001
                    final_message = self.summarize_session(session, cached=True)
            else:
                final_message = self.summarize_session(session, cached=True)
            self.session_store.append_message(session, "assistant", final_message)
            return final_message

        fallback = "No action was produced for this turn."
        self.session_store.append_message(session, "assistant", fallback)
        return fallback

    def _mount_dataset_mentions(self, session: AgentSession, user_message: str) -> list[dict[str, Any]]:
        matches = DATASET_MENTION_RE.findall(user_message)
        if not matches:
            return []
        mounted_items: list[dict[str, Any]] = []
        session.state.setdefault("mounted_datasets", {})
        ordered_ids: list[str] = []
        for raw_path in matches:
            mounted_path = Path(raw_path).resolve()
            dataset_id = f"mounted_{mounted_path.stem.lower()}"
            session.state["mounted_datasets"][dataset_id] = {
                "path": str(mounted_path),
                "mounted_at": iso_timestamp(),
            }
            mounted_items.append({"dataset_id": dataset_id, "path": str(mounted_path)})
            ordered_ids.append(dataset_id)
        session.state["last_dataset_mentions"] = ordered_ids
        session.active_dataset_id = ordered_ids[0]
        self.session_store.save(session)
        return mounted_items

    def _plan_turn(
        self,
        session: AgentSession,
        user_message: str,
        tool_results: list[dict[str, Any]],
        allow_llm: bool,
        mode: ControlMode,
    ) -> tuple[dict[str, Any], bool]:
        if mode == ControlMode.PIPELINE:
            specialized = self._pipeline_plan(session, user_message)
            if specialized is not None:
                return specialized, False
            return fallback_interactive_plan(user_message, self._session_snapshot(session)), False

        explicit = self._explicit_tool_plan(user_message)
        if explicit is not None:
            return explicit, False

        snapshot = self._session_snapshot(session)
        recent_history = [
            {"role": message.role, "content": message.content}
            for message in session.history[-6:]
        ]
        if allow_llm and self.llm.is_configured():
            try:
                return self.llm.react_interactive_turn(
                    user_message=user_message,
                    session_snapshot=snapshot,
                    tools=[tool.to_prompt_dict() for tool in self.tool_registry.list_tools()],
                    datasets=[entry.to_dict() for entry in self.registries.datasets.values()],
                    models=[entry.to_dict() for entry in self.registries.models.values()],
                    skills=[entry.to_dict() for entry in self.registries.skills.values()],
                    tool_results=tool_results,
                    recent_history=recent_history,
                ), False
            except Exception as exc:
                session.memory["last_planner_error"] = repr(exc)
                self.session_store.save(session)
                return fallback_interactive_plan(user_message, snapshot), True
        return fallback_interactive_plan(user_message, snapshot), False

    def _pipeline_plan(self, session: AgentSession, user_message: str) -> dict[str, Any] | None:
        request = self._resolve_pipeline_request(session, user_message)
        specialized = self._specialized_plan(session, request)
        if specialized is not None:
            return specialized
        return None

    def _resolve_pipeline_request(self, session: AgentSession, user_message: str) -> PanelPipelineRequest:
        cached = session.memory.get("pipeline_request")
        if isinstance(cached, dict) and cached.get("source_message") == user_message:
            return self._pipeline_request_from_dict(cached)

        if self.llm.is_configured():
            try:
                request = self.llm.extract_panel_pipeline_request(
                    user_message=user_message,
                    session_snapshot=self._session_snapshot(session),
                    mounted_dataset_paths=self._dataset_mention_paths(session, user_message),
                )
            except Exception as exc:  # noqa: BLE001
                session.memory["last_pipeline_request_error"] = repr(exc)
                request = self._fallback_pipeline_request(session, user_message)
        else:
            request = self._fallback_pipeline_request(session, user_message)

        self._complete_pipeline_request(session, user_message, request)
        payload = request.to_dict()
        payload["source_message"] = user_message
        session.memory["pipeline_request"] = payload
        self.session_store.save(session)
        return request

    @staticmethod
    def _pipeline_request_from_dict(payload: dict[str, Any]) -> PanelPipelineRequest:
        return PanelPipelineRequest(**{key: value for key, value in payload.items() if key != "source_message"})

    def _current_pipeline_request(self, session: AgentSession) -> PanelPipelineRequest | None:
        cached = session.memory.get("pipeline_request")
        if not isinstance(cached, dict):
            return None
        try:
            return self._pipeline_request_from_dict(cached)
        except TypeError:
            return None

    def _dataset_mention_paths(self, session: AgentSession, user_message: str) -> list[str]:
        mentions = DATASET_MENTION_RE.findall(user_message)
        if mentions:
            return [str(Path(item).resolve()) for item in mentions]
        ordered_ids = session.state.get("last_dataset_mentions", [])
        mounted = session.state.get("mounted_datasets", {})
        return [str(mounted[item]["path"]) for item in ordered_ids if item in mounted]

    def _complete_pipeline_request(self, session: AgentSession, user_message: str, request: PanelPipelineRequest) -> None:
        train_path, test_path = self._extract_dataset_paths(session, user_message)
        if not request.train_adata_file:
            request.train_adata_file = train_path
        if not request.test_adata_file and test_path:
            request.test_adata_file = test_path
        if not request.test_adata_file and request.run_evaluation:
            request.test_adata_file = getattr(self.registries.datasets.get("smith_st_processed"), "path", "")
        if not request.tasks:
            request.tasks = ["recon", "cls", "region", "pathology"]
        if request.panel_size <= 0:
            request.panel_size = 64
        if not request.species:
            request.species = (
                "homo_sapiens"
                if any(token in request.train_adata_file.lower() for token in ["sc_dataset_processed", "st_dataset_processed", "htapp", "human"])
                else "mus_musculus"
            )

    def _fallback_pipeline_request(self, session: AgentSession, user_message: str) -> PanelPipelineRequest:
        lower = user_message.lower()
        train_path, test_path = self._extract_dataset_paths(session, user_message)
        panel_size = self._extract_panel_size(user_message) or 64
        formal = not any(token in lower for token in ["smoke", "quick"])
        tasks = ["recon", "cls", "region", "pathology"]
        objective = "balanced"
        label = "pathology"
        if "pathology" in lower and "classification" in lower:
            tasks = ["cls", "pathology"]
            objective = "pathology_classification"
        elif "region" in lower and "classification" in lower:
            tasks = ["region"]
            objective = "region_classification"
            label = "region"
        elif "reconstruction" in lower or "recon" in lower:
            tasks = ["recon"]
            objective = "reconstruction"
        return PanelPipelineRequest(
            train_adata_file=train_path,
            test_adata_file=test_path or "",
            panel_size=panel_size,
            objective=objective,
            tasks=tasks,
            label=label,
            formal=formal,
            epoch=5 if formal else 1,
            run_selection=any(token in lower for token in ["run smith", "panel selection", "select panel", "generate panel", "design panel"]),
            run_feasibility=any(token in lower for token in ["feasibility", "probe", "odt", "oligominer", "probedealer"]),
            run_evaluation=any(token in lower for token in ["evaluate", "test", "transfer performance", "cross-dataset", "cross dataset"]),
            build_report=any(token in lower for token in ["report", "summarize", "summary", "plot", "umap"]),
            skip_odt=any(token in lower for token in ["skip odt", "without odt", "no odt"]),
            species="homo_sapiens"
            if any(token in train_path.lower() for token in ["sc_dataset_processed", "st_dataset_processed", "htapp", "human"])
            else "mus_musculus",
        )

    def _explicit_tool_plan(self, user_message: str) -> dict[str, Any] | None:
        if self.tool_registry is None:
            return None
        match = re.search(
            r"\b(?:run|call|execute)\s+(?:the\s+)?(?:tool\s+)?`?([a-zA-Z_][a-zA-Z0-9_]*)`?",
            user_message,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        tool_name = match.group(1)
        if tool_name not in self.tool_registry.tools:
            return None
        arguments: dict[str, Any] = {}
        json_match = re.search(r"\{.*\}", user_message, flags=re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict):
                    arguments = parsed
            except json.JSONDecodeError:
                arguments = {}
        return {
            "assistant_message": "",
            "tool_calls": [{"tool_name": tool_name, "arguments": arguments}],
            "state_updates": {},
            "done": True,
        }

    def _specialized_plan(self, session: AgentSession, request: PanelPipelineRequest) -> dict[str, Any] | None:
        if not (request.run_selection or request.run_feasibility or request.run_evaluation or request.build_report):
            return None

        state = session.state
        if request.run_selection and self._looks_like_imaging_st_target(request.train_adata_file):
            return {
                "assistant_message": (
                    "For prospective imaging-ST panel design, the training/source input should be paired or tissue-matched "
                    "scRNA-seq. I will not run panel selection on a MERFISH/Xenium/CosMx/MERSCOPE/seqFISH/osmFISH target "
                    "dataset before the panel is designed. Provide the paired scRNA-seq h5ad path or ask me to retrieve "
                    "a CELLxGENE scRNA source reference."
                ),
                "tool_calls": [],
                "state_updates": {},
                "done": True,
            }
        if request.run_selection and not state.get("smith_selection_manifest"):
            return {
                "assistant_message": "",
                "tool_calls": [
                    {
                        "tool_name": "run_smith_selection",
                        "arguments": {
                            "adata_file": request.train_adata_file,
                            "tasks": request.task_string,
                            "task_name": request.objective or "interactive_panel",
                            "panel_size": request.panel_size,
                            "epoch": request.epoch,
                            "record": 1,
                            "execute": True,
                            "extra_args": {"batch_size": 4096, "balance_mode": "off"},
                        },
                    }
                ],
                "state_updates": {},
                "done": False,
            }
        if request.run_feasibility and not state.get("probe_candidate_manifest"):
            return {
                "assistant_message": "",
                "tool_calls": [
                    {
                        "tool_name": "build_probe_candidate_manifest",
                        "arguments": {
                            "panel_path": self._resolve_last_panel(session),
                            "panel_size": request.panel_size,
                            "species": request.species,
                        },
                    }
                ],
                "state_updates": {},
                "done": False,
            }
        if request.run_feasibility and not ((request.skip_odt or state.get("odt_summary_tsv")) and state.get("oligominer_summary_tsv")):
            return {
                "assistant_message": "",
                "tool_calls": (
                    [{"tool_name": "run_oligominer_specificity_screen", "arguments": {"species": request.species}}]
                    if request.skip_odt
                    else [
                        {"tool_name": "run_odt_property_batches", "arguments": {}},
                        {"tool_name": "run_oligominer_specificity_screen", "arguments": {"species": request.species}},
                    ]
                ),
                "state_updates": {},
                "done": False,
            }
        if request.run_feasibility and not state.get("probedealer_summary_tsv"):
            return {
                "assistant_message": "",
                "tool_calls": [
                    {"tool_name": "run_probedealer_backend_screen", "arguments": {"species": request.species}},
                ],
                "state_updates": {},
                "done": False,
            }
        if request.run_feasibility and not state.get("feasibility_table"):
            return {
                "assistant_message": "",
                "tool_calls": [
                    {"tool_name": "run_three_backend_feasibility", "arguments": {"skip_property_gate": request.skip_odt}},
                ],
                "state_updates": {},
                "done": False,
            }
        if request.run_evaluation and not state.get("cross_dataset_evaluation"):
            return {
                "assistant_message": "",
                "tool_calls": [
                    {
                        "tool_name": "evaluate_cross_dataset_panel",
                        "arguments": {
                            "panel_path": self._resolve_last_panel(session),
                            "train_adata_file": request.train_adata_file,
                            "test_adata_file": request.test_adata_file,
                            "panel_size": request.panel_size,
                            "label": request.label,
                            "obsm_key": request.obsm_key,
                        },
                    }
                ],
                "state_updates": {},
                "done": False,
            }
        if request.build_report and not state.get("last_report"):
            return {
                "assistant_message": "",
                "tool_calls": [
                    {
                        "tool_name": "build_run_report",
                        "arguments": {
                            "train_adata_file": request.train_adata_file,
                            "test_adata_file": request.test_adata_file,
                            "panel_size": request.panel_size,
                        },
                    }
                ],
                "state_updates": {},
                "done": False,
            }
        if any([request.run_selection, request.run_feasibility, request.run_evaluation, request.build_report]):
            return {
                "assistant_message": self.summarize_session(session, cached=True),
                "tool_calls": [],
                "state_updates": {},
                "done": True,
            }
        return None

    @staticmethod
    def _emit_progress(callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
        if callback is not None:
            callback(payload)

    @staticmethod
    def _tool_start_message(tool_name: str) -> str:
        return f"Running {tool_name}"

    @staticmethod
    def _tool_end_message(tool_name: str, result: dict[str, Any]) -> str:
        if tool_name == "run_smith_selection":
            status = result.get("status", "done")
            return f"Finished {tool_name} ({status})"
        if tool_name == "build_probe_candidate_manifest":
            return "Finished build_probe_candidate_manifest"
        if tool_name == "score_reference_transferability":
            top = result.get("top_candidates", [])
            if top:
                return f"Finished score_reference_transferability (top: {top[0].get('dataset_id')}, score {top[0].get('score')})"
            return "Finished score_reference_transferability (no candidates)"
        if tool_name == "run_three_backend_feasibility":
            passing = result.get("passing_count")
            total = result.get("total_count")
            if passing is not None and total is not None:
                return f"Finished {tool_name} ({passing}/{total} passing)"
            return f"Finished {tool_name}"
        if tool_name == "evaluate_cross_dataset_panel":
            return f"Finished {tool_name} ({len(result.get('rows', []))} rows)"
        if tool_name == "build_run_report":
            return "Finished build_run_report"
        return f"Finished {tool_name}"

    @staticmethod
    def _format_tool_error(tool_name: str, exc: Exception) -> str:
        return f"Tool `{tool_name}` failed: {exc}"

    def _session_snapshot(self, session: AgentSession) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "active_dataset_id": session.active_dataset_id,
            "active_model_id": session.active_model_id,
            "state": session.state,
            "memory": session.memory,
        }

    @staticmethod
    def _apply_state_updates(session: AgentSession, updates: dict[str, Any]) -> None:
        if not isinstance(updates, dict):
            return
        state = session.state
        for key, value in updates.items():
            if key == "active_dataset_id" and value:
                session.active_dataset_id = str(value)
            elif key == "active_model_id" and value:
                session.active_model_id = str(value)
            else:
                state[key] = value

    def _resolve_dataset_path(self, session: AgentSession, identifier: str | None) -> str:
        dataset_id = identifier or session.active_dataset_id
        if not dataset_id:
            raise ValueError("No dataset is active.")
        mounted = session.state.get("mounted_datasets", {})
        if dataset_id in mounted:
            return str(mounted[dataset_id]["path"])
        resolved = self.registries.resolve_dataset_id(dataset_id)
        if not resolved:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        return self.registries.datasets[resolved].path

    def _extract_dataset_paths(self, session: AgentSession, user_message: str) -> tuple[str, str | None]:
        mentions = DATASET_MENTION_RE.findall(user_message)
        if len(mentions) >= 2:
            return str(Path(mentions[0]).resolve()), str(Path(mentions[1]).resolve())
        if len(mentions) == 1:
            mention_path = str(Path(mentions[0]).resolve())
            ordered_ids = session.state.get("last_dataset_mentions", [])
            if len(ordered_ids) >= 2:
                mounted = session.state.get("mounted_datasets", {})
                second = ordered_ids[1]
                if second in mounted:
                    return mention_path, str(mounted[second]["path"])
            return mention_path, None
        ordered_ids = session.state.get("last_dataset_mentions", [])
        mounted = session.state.get("mounted_datasets", {})
        if len(ordered_ids) >= 2:
            return str(mounted[ordered_ids[0]]["path"]), str(mounted[ordered_ids[1]]["path"])
        return self._resolve_dataset_path(session, None), None

    def _resolve_last_panel(self, session: AgentSession) -> str:
        manifest_path = session.state.get("smith_selection_manifest")
        if not manifest_path:
            raise ValueError("No previous SMITH selection manifest found in this session.")
        epoch_files = sorted(Path(manifest_path).parent.glob("epoch_*.csv"))
        if not epoch_files:
            raise ValueError(f"No ranked epoch_*.csv files found next to manifest: {manifest_path}")
        return str(epoch_files[-1])

    @staticmethod
    def _report_covers_current_state(session: AgentSession) -> bool:
        report = session.state.get("last_report")
        if not isinstance(report, dict):
            return False
        if session.state.get("cross_dataset_evaluation") and not report.get("evaluation_metrics"):
            return False
        if session.state.get("feasibility_summary") and not report.get("feasibility_summary"):
            return False
        return True

    @staticmethod
    def _extract_panel_size(user_message: str) -> int | None:
        lowered = user_message.lower()
        patterns = [
            r"(panel size|top)\s*(=|of)?\s*(\d+)",
            r"(\d+)\s*-\s*gene panel",
            r"(\d+)\s*gene panel",
            r"panel of\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            for group in reversed(match.groups()):
                if group and str(group).isdigit():
                    return int(group)
        return None

    @staticmethod
    def _is_valid_h5ad_path(value: Any) -> bool:
        text = str(value or "").strip()
        return text.endswith(".h5ad") and Path(text).exists()

    @staticmethod
    def _looks_like_imaging_st_target(value: Any) -> bool:
        text = str(value or "").lower()
        return any(token in text for token in ["merfish", "xenium", "cosmx", "merscope", "seqfish", "osmfish"])

    def _repair_tool_dependencies(
        self,
        session: AgentSession,
        user_message: str,
        decision: dict[str, Any],
        mode: ControlMode,
    ) -> dict[str, Any]:
        tool_calls = decision.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            return decision
        lower = user_message.lower()
        wants_evaluation = any(token in lower for token in ["evaluate", "test", "cross-dataset", "cross dataset"])
        wants_report = any(token in lower for token in ["report", "summarize", "summary", "plot", "umap", "figure", "performance"])
        wants_feasibility = any(
            token in lower
            for token in [
                "feasibility",
                "filter",
                "probe",
                "odt",
                "oligominer",
                "probedealer",
                "deployment",
            ]
        )
        wants_st = any(
            token in lower
            for token in [
                "st dataset",
                "st_dataset_processed",
                "smith_st_processed",
                " on st ",
            ]
        )
        state = session.state
        has_panel = bool(state.get("smith_selection_manifest"))
        has_eval = bool(state.get("cross_dataset_evaluation"))
        has_report = bool(state.get("last_report"))
        report_is_fresh = self._report_covers_current_state(session)
        has_probe_manifest = bool(state.get("probe_candidate_manifest")) and bool(state.get("probe_candidate_transcript_fasta"))
        has_odt = bool(state.get("odt_summary_tsv"))
        has_oligo = bool(state.get("oligominer_summary_tsv"))
        has_probedealer = bool(state.get("probedealer_summary_tsv"))
        has_feasibility = bool(state.get("feasibility_table"))
        train_path_from_message, test_path_from_message = self._extract_dataset_paths(session, user_message)
        pipeline_request = self._current_pipeline_request(session)
        skip_odt_requested = bool(pipeline_request.skip_odt) if pipeline_request else any(token in lower for token in ["skip odt", "without odt", "no odt"])
        repaired = []
        saw_run_smith = False
        saw_eval = False
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool_name", "")).strip()
            arguments = dict(call.get("arguments", {})) if isinstance(call.get("arguments", {}), dict) else {}
            if not tool_name:
                continue
            if tool_name == "resolve_dataset_context" and not arguments.get("dataset_path") and not arguments.get("dataset_id"):
                if session.active_dataset_id:
                    arguments["dataset_id"] = session.active_dataset_id
            elif tool_name == "query_cellxgene_metadata":
                query = dict(arguments.get("query", {})) if isinstance(arguments.get("query", {}), dict) else {}
                source_role = str(
                    query.get("query_role")
                    or query.get("dataset_role")
                    or query.get("reference_role")
                    or query.get("intended_use")
                    or ""
                ).lower()
                target_mentioned = any(
                    token in lower
                    for token in [
                        "merfish",
                        "xenium",
                        "cosmx",
                        "merscope",
                        "seqfish",
                        "osmfish",
                        "imaging-based",
                        "imaging based",
                    ]
                )
                panel_context = any(token in lower for token in ["panel", "probe", "panel design", "selection"])
                if query.get("panel_design_source") or "panel_design_source" in source_role or (target_mentioned and panel_context):
                    query["query_role"] = "panel_design_source"
                    query["assay_family"] = "single_cell_rna"
                    raw_excluded = query.get("exclude_assay_families", []) or []
                    excluded = {str(raw_excluded)} if isinstance(raw_excluded, str) else {str(item) for item in raw_excluded}
                    excluded.update({"imaging_spatial", "sequencing_spatial"})
                    query["exclude_assay_families"] = sorted(excluded)
                    if target_mentioned:
                        query.setdefault("target_assay_family", "imaging_spatial")
                    query.setdefault("target_data_policy", "held_out_imaging_spatial_target_not_used_for_source_retrieval")
                    arguments["query"] = query
                    arguments.setdefault("max_results", 10)
            elif tool_name == "score_reference_transferability":
                if not arguments.get("request"):
                    dataset_id = session.active_dataset_id
                    dataset = self.registries.datasets.get(dataset_id) if dataset_id in self.registries.datasets else None
                    arguments["request"] = {
                        "species": getattr(dataset, "species", ""),
                        "tissue": getattr(dataset, "tissue", ""),
                        "modality": getattr(dataset, "modality", ""),
                        "tasks": getattr(dataset, "tasks", []),
                    }
            elif tool_name == "run_smith_selection":
                saw_run_smith = True
                if has_panel and wants_evaluation and (wants_st or "selected panel" in lower):
                    tool_name = "evaluate_cross_dataset_panel"
                    arguments = {
                        "panel_path": self._resolve_last_panel(session),
                        "train_adata_file": train_path_from_message,
                        "test_adata_file": test_path_from_message or getattr(self.registries.datasets.get("smith_st_processed"), "path", ""),
                        "panel_size": self._extract_panel_size(user_message) or 16,
                        "label": "pathology",
                        "obsm_key": "X_pca",
                    }
                    saw_eval = True
                else:
                    if not self._is_valid_h5ad_path(arguments.get("adata_file")):
                        arguments["adata_file"] = train_path_from_message
                    arguments.setdefault("tasks", "recon,cls,region,pathology")
                    arguments.setdefault("task_name", "interactive_panel")
                    arguments.setdefault("panel_size", self._extract_panel_size(user_message) or 16)
                    arguments.setdefault("epoch", 1 if "smoke" in user_message.lower() or "quick" in user_message.lower() else 5)
                    arguments.setdefault("record", 1)
                    arguments.setdefault("execute", True)
                    extra_args = dict(arguments.get("extra_args", {}))
                    extra_args.setdefault("batch_size", 4096)
                    extra_args.setdefault("balance_mode", "off")
                    arguments["extra_args"] = extra_args
            elif tool_name == "evaluate_cross_dataset_panel":
                saw_eval = True
                arguments.setdefault("panel_path", self._resolve_last_panel(session))
                if not self._is_valid_h5ad_path(arguments.get("train_adata_file")):
                    arguments["train_adata_file"] = train_path_from_message
                test_dataset = self.registries.datasets.get("smith_st_processed")
                if not self._is_valid_h5ad_path(arguments.get("test_adata_file")):
                    arguments["test_adata_file"] = test_path_from_message or getattr(test_dataset, "path", "")
                arguments.setdefault("panel_size", self._extract_panel_size(user_message) or 16)
                arguments.setdefault("label", "pathology")
                arguments.setdefault("obsm_key", "X_pca")
            elif tool_name == "build_run_report":
                if has_report:
                    decision["tool_calls"] = []
                    if not decision.get("assistant_message"):
                        decision["assistant_message"] = self.summarize_session(session, cached=True)
                    decision["done"] = True
                    return decision
            elif tool_name == "build_probe_candidate_manifest":
                arguments.setdefault("panel_path", self._resolve_last_panel(session))
                arguments.setdefault("panel_size", self._extract_panel_size(user_message) or 64)
                if "homo_sapiens" not in str(arguments.get("species", "")).lower() and "mus_musculus" not in str(arguments.get("species", "")).lower():
                    train_path_lower = train_path_from_message.lower()
                    if any(token in train_path_lower for token in ["sc_dataset_processed", "st_dataset_processed", "htapp", "human"]):
                        arguments["species"] = "homo_sapiens"
                    else:
                        arguments["species"] = "mus_musculus"
            elif tool_name == "run_odt_property_screen":
                if not arguments.get("genes") and state.get("probe_candidate_manifest"):
                    import pandas as _pd
                    arguments["genes"] = _pd.read_csv(state["probe_candidate_manifest"], sep="\t")["gene_symbol"].astype(str).tolist()
                arguments.setdefault("species", "homo_sapiens" if "homo_sapiens" in lower or "human" in lower or "sc_dataset_processed" in train_path_from_message.lower() else "mus_musculus")
            elif tool_name == "run_odt_property_batches":
                arguments.setdefault("manifest_tsv", state.get("probe_candidate_manifest", ""))
                arguments.setdefault("species", "homo_sapiens" if "homo_sapiens" in lower or "human" in lower or "sc_dataset_processed" in train_path_from_message.lower() else "mus_musculus")
                arguments.setdefault("batch_size", 10)
                arguments.setdefault("max_workers", 8)
            elif tool_name == "run_oligominer_specificity_screen":
                arguments.setdefault("transcript_fasta", state.get("probe_candidate_transcript_fasta", ""))
                if "homo_sapiens" not in str(arguments.get("species", "")).lower() and "mus_musculus" not in str(arguments.get("species", "")).lower():
                    arguments["species"] = "homo_sapiens" if "homo_sapiens" in lower or "human" in lower or "sc_dataset_processed" in train_path_from_message.lower() else "mus_musculus"
            elif tool_name == "run_probedealer_backend_screen":
                arguments.setdefault("transcript_fasta", state.get("probe_candidate_transcript_fasta", ""))
                if "homo_sapiens" not in str(arguments.get("species", "")).lower() and "mus_musculus" not in str(arguments.get("species", "")).lower():
                    arguments["species"] = "homo_sapiens" if "homo_sapiens" in lower or "human" in lower or "sc_dataset_processed" in train_path_from_message.lower() else "mus_musculus"
            elif tool_name == "run_three_backend_feasibility":
                arguments.setdefault("manifest_tsv", state.get("probe_candidate_manifest", ""))
                arguments.setdefault("odt_summary_tsv", state.get("odt_summary_tsv", ""))
                arguments.setdefault("oligominer_summary_tsv", state.get("oligominer_summary_tsv", ""))
                arguments.setdefault("probedealer_summary_tsv", state.get("probedealer_summary_tsv", ""))
            repaired.append({"tool_name": tool_name, "arguments": arguments})

        if mode == ControlMode.REACT:
            decision["tool_calls"] = repaired
            return decision

        if has_panel and wants_feasibility and not has_probe_manifest:
            panel_size = pipeline_request.panel_size if pipeline_request else self._extract_panel_size(user_message) or 64
            species = (
                pipeline_request.species
                if pipeline_request
                else ("homo_sapiens" if "sc_dataset_processed" in train_path_from_message.lower() or "human" in lower else "mus_musculus")
            )
            repaired = [
                {
                    "tool_name": "build_probe_candidate_manifest",
                    "arguments": {
                        "panel_path": self._resolve_last_panel(session),
                        "panel_size": panel_size,
                        "species": species,
                    },
                }
            ]
        elif has_probe_manifest and wants_feasibility and not ((skip_odt_requested or has_odt) and has_oligo):
            species = (
                pipeline_request.species
                if pipeline_request
                else ("homo_sapiens" if "sc_dataset_processed" in train_path_from_message.lower() or "human" in lower else "mus_musculus")
            )
            repaired = []
            if not skip_odt_requested and not has_odt:
                repaired.append(
                    {
                        "tool_name": "run_odt_property_batches",
                        "arguments": {
                            "manifest_tsv": state["probe_candidate_manifest"],
                            "species": species,
                        },
                    }
                )
            if not has_oligo:
                repaired.append(
                    {
                        "tool_name": "run_oligominer_specificity_screen",
                        "arguments": {
                            "transcript_fasta": state["probe_candidate_transcript_fasta"],
                            "species": species,
                        },
                    }
                )
        elif has_probe_manifest and wants_feasibility and (skip_odt_requested or has_odt) and has_oligo and not has_probedealer:
            repaired = [
                {
                    "tool_name": "run_probedealer_backend_screen",
                    "arguments": {
                        "transcript_fasta": state["probe_candidate_transcript_fasta"],
                        "species": pipeline_request.species if pipeline_request else ("homo_sapiens" if "sc_dataset_processed" in train_path_from_message.lower() or "human" in lower else "mus_musculus"),
                    },
                }
            ]
        elif has_probe_manifest and wants_feasibility and (skip_odt_requested or has_odt) and has_oligo and has_probedealer and not has_feasibility:
            repaired = [
                {
                    "tool_name": "run_three_backend_feasibility",
                    "arguments": {
                        "manifest_tsv": state["probe_candidate_manifest"],
                        "odt_summary_tsv": "" if skip_odt_requested else state["odt_summary_tsv"],
                        "oligominer_summary_tsv": state["oligominer_summary_tsv"],
                        "probedealer_summary_tsv": state["probedealer_summary_tsv"],
                        "skip_property_gate": skip_odt_requested,
                    },
                }
            ]
        elif has_panel and wants_evaluation and not has_eval and not saw_eval:
            panel_size = pipeline_request.panel_size if pipeline_request else self._extract_panel_size(user_message) or 16
            label = pipeline_request.label if pipeline_request else "pathology"
            obsm_key = pipeline_request.obsm_key if pipeline_request else "X_pca"
            repaired = [
                {
                    "tool_name": "evaluate_cross_dataset_panel",
                    "arguments": {
                        "panel_path": self._resolve_last_panel(session),
                        "train_adata_file": train_path_from_message,
                        "test_adata_file": test_path_from_message or getattr(self.registries.datasets.get("smith_st_processed"), "path", ""),
                        "panel_size": panel_size,
                        "label": label,
                        "obsm_key": obsm_key,
                    },
                }
            ]
        elif has_report and report_is_fresh and wants_report and not (wants_evaluation and not has_eval):
            decision["tool_calls"] = []
            if not decision.get("assistant_message"):
                decision["assistant_message"] = self.summarize_session(session, cached=True)
            decision["done"] = True
            return decision
        elif has_eval and (saw_run_smith or saw_eval):
            decision["tool_calls"] = []
            if not decision.get("assistant_message"):
                decision["assistant_message"] = self.summarize_session(session, cached=True)
            decision["done"] = True
            return decision
        elif (has_eval or has_feasibility) and wants_report and (not has_report or not report_is_fresh):
            panel_size = pipeline_request.panel_size if pipeline_request else self._extract_panel_size(user_message) or 64
            repaired = [
                {
                    "tool_name": "build_run_report",
                    "arguments": {
                        "train_adata_file": train_path_from_message,
                        "test_adata_file": test_path_from_message or getattr(self.registries.datasets.get("smith_st_processed"), "path", ""),
                        "panel_size": panel_size,
                    },
                }
            ]
        decision["tool_calls"] = repaired
        return decision
