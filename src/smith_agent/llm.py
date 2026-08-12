from __future__ import annotations

import json
import re
from typing import Any

import requests

from smith_agent.config import LLMSettings
from smith_agent.schemas import PanelPipelineRequest


class LLMClient:
    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.last_trace: dict[str, Any] = {}

    def is_configured(self) -> bool:
        return self.settings.is_configured()

    def _chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        purpose: str = "chat",
    ) -> str:
        if not self.is_configured():
            raise RuntimeError("LLM client is not configured.")
        base_payload = {
            "model": self.settings.model,
            "messages": messages,
        }
        variants = [base_payload]
        if response_format is not None:
            variants = [{**base_payload, "response_format": response_format}, base_payload]

        last_error: Exception | None = None
        chosen_payload: dict[str, Any] | None = None
        response = None
        for variant in variants:
            payload = dict(variant)
            if self.settings.temperature is not None:
                payload["temperature"] = self.settings.temperature
            try:
                response = requests.post(
                    f"{self.settings.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.resolved_api_key()}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.settings.timeout_seconds,
                )
                response.raise_for_status()
                chosen_payload = payload
                break
            except requests.HTTPError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code not in {400, 422} or variant is variants[-1]:
                    raise RuntimeError(self._format_http_error(exc)) from exc
            except requests.RequestException as exc:
                raise RuntimeError(f"Planner request failed: {exc}") from exc

        if response is None:
            if last_error is not None:
                raise RuntimeError(self._format_http_error(last_error))
            raise RuntimeError("Planner request failed before receiving a response.")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("LLM response did not include any choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        text = str(content).strip()
        self.last_trace = {
            "purpose": purpose,
            "request": {
                "messages": messages,
                "response_format": (chosen_payload or {}).get("response_format"),
            },
            "response": {
                "raw_text": text,
                "message": message,
                "usage": data.get("usage", {}),
            },
        }
        return text

    def plan_interactive_turn(
        self,
        user_message: str,
        session_snapshot: dict[str, Any],
        tools: list[dict[str, Any]],
        datasets: list[dict[str, Any]],
        models: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        tool_results: list[dict[str, Any]] | None = None,
        recent_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        compact_snapshot = self._compact_session_snapshot(session_snapshot)
        messages = [
            {
                "role": "system",
                "content": self._planner_system_prompt(),
            },
            {
                "role": "user",
                "content": (
                    f"User message:\n{user_message}\n\n"
                    f"Session snapshot:\n{json.dumps(compact_snapshot, indent=2)}\n\n"
                    f"Recent history:\n{json.dumps((recent_history or [])[-4:], indent=2)}\n\n"
                    f"Available skills:\n{json.dumps(skills, indent=2)}\n\n"
                    f"Available tools:\n{json.dumps(tools, indent=2)}\n\n"
                    f"Available datasets:\n{json.dumps(datasets, indent=2)}\n\n"
                    f"Available models:\n{json.dumps(models, indent=2)}\n\n"
                    f"Latest tool results:\n{json.dumps((tool_results or [])[-4:], indent=2)}"
                ),
            },
        ]
        raw = self._chat(messages, response_format=self._planner_response_format(), purpose="plan_interactive_turn")
        return self._normalize_plan_output(raw)

    def react_interactive_turn(
        self,
        user_message: str,
        session_snapshot: dict[str, Any],
        tools: list[dict[str, Any]],
        datasets: list[dict[str, Any]],
        models: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        tool_results: list[dict[str, Any]] | None = None,
        recent_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        compact_snapshot = self._compact_session_snapshot(session_snapshot)
        messages = [
            {
                "role": "system",
                "content": self._react_system_prompt(),
            },
            {
                "role": "user",
                "content": (
                    f"User message:\n{user_message}\n\n"
                    f"Session snapshot:\n{json.dumps(compact_snapshot, indent=2)}\n\n"
                    f"Recent history:\n{json.dumps((recent_history or [])[-6:], indent=2)}\n\n"
                    f"Available skills:\n{json.dumps(skills, indent=2)}\n\n"
                    f"Available tools:\n{json.dumps(tools, indent=2)}\n\n"
                    f"Available datasets:\n{json.dumps(datasets, indent=2)}\n\n"
                    f"Available models:\n{json.dumps(models, indent=2)}\n\n"
                    f"Latest tool observations:\n{json.dumps((tool_results or [])[-4:], indent=2)}"
                ),
            },
        ]
        raw = self._chat(messages, response_format=self._planner_response_format(), purpose="react_interactive_turn")
        return self._normalize_plan_output(raw)

    def extract_panel_pipeline_request(
        self,
        user_message: str,
        session_snapshot: dict[str, Any],
        mounted_dataset_paths: list[str],
    ) -> PanelPipelineRequest:
        messages = [
            {
                "role": "system",
                "content": self._pipeline_request_system_prompt(),
            },
            {
                "role": "user",
                "content": (
                    f"User message:\n{user_message}\n\n"
                    f"Mounted dataset paths, in mention order:\n{json.dumps(mounted_dataset_paths, indent=2)}\n\n"
                    f"Session snapshot:\n{json.dumps(self._compact_session_snapshot(session_snapshot), indent=2)}"
                ),
            },
        ]
        raw = self._chat(
            messages,
            response_format=self._pipeline_request_response_format(),
            purpose="extract_panel_pipeline_request",
        )
        payload = self._extract_json_object(raw)
        if not isinstance(payload, dict):
            raise ValueError("Pipeline request extraction did not return a JSON object.")
        return self._panel_pipeline_request_from_payload(payload)

    def summarize_interactive_results(
        self,
        user_message: str,
        session_snapshot: dict[str, Any],
        tool_results: list[dict[str, Any]],
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You summarize an interactive panel-design session. "
                    "Be concise, concrete, and mention the most useful generated artifacts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original request:\n{user_message}\n\n"
                    f"Session snapshot:\n{json.dumps(session_snapshot, indent=2)}\n\n"
                    f"Recent tool results:\n{json.dumps(tool_results, indent=2)}"
                ),
            },
        ]
        return self._chat(messages, purpose="summarize_interactive_results")

    @staticmethod
    def _planner_system_prompt() -> str:
        return (
            "You are the planner/controller for smith-agent, an interactive panel-design agent. "
            "Use the skill registry as workflow guidance and the tool registry for execution. "
            "Prefer tools over assumptions. Work incrementally. "
            "Select at most 2 tools per turn. "
            "If more work is needed, return tool calls and set done to false. "
            "If the user can be answered now, set done to true. "
            "Return JSON only with exactly these keys: assistant_message, tool_calls, state_updates, done. "
            "tool_calls must be a list of objects with keys tool_name and arguments. "
            "state_updates must be a JSON object."
        )

    @staticmethod
    def _react_system_prompt() -> str:
        return (
            "You are the ReAct controller for smith-agent. "
            "Use tools to inspect, search, compare, diagnose, plot, run explicitly requested tools, "
            "and explain existing panel-design results. "
            "For panel design toward an imaging-based spatial transcriptomics target such as MERFISH, Xenium, CosMx, "
            "MERSCOPE, seqFISH, osmFISH, or STARmap, treat paired or tissue-matched scRNA-seq as the source input. "
            "Do not retrieve, materialize, or evaluate on target imaging-ST data before the panel has been designed unless "
            "the user explicitly provides a held-out test dataset path. "
            "When querying CELLxGENE for source data in this setting, call query_cellxgene_metadata with "
            "query_role='panel_design_source', assay_family='single_cell_rna', and "
            "exclude_assay_families=['imaging_spatial','sequencing_spatial']; if a paired CELLxGENE collection or target "
            "dataset id is known, use collection_id(s) and exclude_dataset_ids to retrieve only the paired scRNA source. "
            "Reason internally from observations, but do not expose a long chain of thought. "
            "Prefer concrete tool calls over guesses when session state or artifacts can answer the question. "
            "Do not start a full formal panel-design pipeline unless the user clearly asks to generate a new deliverable. "
            "Select at most 2 tools per turn, then re-plan after observing results. "
            "If a user explicitly asks to run a specific tool, call that tool with the best available arguments. "
            "If the existing session has enough evidence, answer directly and set done to true. "
            "Return JSON only with exactly these keys: assistant_message, tool_calls, state_updates, done. "
            "tool_calls must be a list of objects with keys tool_name and arguments. "
            "state_updates must be a JSON object."
        )

    @staticmethod
    def _pipeline_request_system_prompt() -> str:
        return (
            "You convert a free-text smith-agent pipeline request into structured JSON. "
            "Do not execute tools. Extract only user intent and parameters. "
            "Use mounted dataset paths in order: the first path is usually train_adata_file, the second is usually test_adata_file. "
            "For prospective imaging-ST panel design, train_adata_file must be the paired/tissue-matched scRNA-seq source. "
            "MERFISH/Xenium/CosMx/MERSCOPE/seqFISH/osmFISH target data are held out and must not be used as the training "
            "or retrieval input; only set test_adata_file when the user explicitly provides a held-out dataset path. "
            "Infer objective and tasks from scientific intent. "
            "Use these task tokens only: recon, cls, region, pathology. "
            "Examples: pathology classification -> tasks ['cls','pathology'], objective 'pathology_classification', label 'pathology'; "
            "region classification -> tasks ['region'], objective 'region_classification', label 'region'; "
            "reconstruction -> tasks ['recon'], objective 'reconstruction'; balanced/default -> tasks ['recon','cls','region','pathology']. "
            "Set booleans for whether the user asked to run selection, feasibility, evaluation, and report generation. "
            "Return JSON only."
        )

    @staticmethod
    def _pipeline_request_response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "panel_pipeline_request",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "train_adata_file": {"type": "string"},
                        "test_adata_file": {"type": "string"},
                        "panel_size": {"type": "integer"},
                        "objective": {"type": "string"},
                        "tasks": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["recon", "cls", "region", "pathology"]},
                        },
                        "label": {"type": "string"},
                        "obsm_key": {"type": "string"},
                        "species": {"type": "string"},
                        "formal": {"type": "boolean"},
                        "epoch": {"type": "integer"},
                        "run_selection": {"type": "boolean"},
                        "run_feasibility": {"type": "boolean"},
                        "run_evaluation": {"type": "boolean"},
                        "build_report": {"type": "boolean"},
                        "skip_odt": {"type": "boolean"},
                        "must_keep_genes": {"type": "array", "items": {"type": "string"}},
                        "forbidden_genes": {"type": "array", "items": {"type": "string"}},
                        "report_format": {"type": "string"},
                        "notes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "train_adata_file",
                        "test_adata_file",
                        "panel_size",
                        "objective",
                        "tasks",
                        "label",
                        "obsm_key",
                        "species",
                        "formal",
                        "epoch",
                        "run_selection",
                        "run_feasibility",
                        "run_evaluation",
                        "build_report",
                        "skip_odt",
                        "must_keep_genes",
                        "forbidden_genes",
                        "report_format",
                        "notes",
                    ],
                },
            },
        }

    @staticmethod
    def _panel_pipeline_request_from_payload(payload: dict[str, Any]) -> PanelPipelineRequest:
        tasks = [str(item) for item in payload.get("tasks", []) if str(item) in {"recon", "cls", "region", "pathology"}]
        if not tasks:
            tasks = ["recon", "cls", "region", "pathology"]
        panel_size = int(payload.get("panel_size", 64) or 64)
        formal = bool(payload.get("formal", True))
        epoch = int(payload.get("epoch", 5 if formal else 1) or (5 if formal else 1))
        if epoch <= 0:
            epoch = 5 if formal else 1
        return PanelPipelineRequest(
            train_adata_file=str(payload.get("train_adata_file", "")),
            test_adata_file=str(payload.get("test_adata_file", "")),
            panel_size=panel_size,
            objective=str(payload.get("objective", "balanced") or "balanced"),
            tasks=tasks,
            label=str(payload.get("label", "pathology") or "pathology"),
            obsm_key=str(payload.get("obsm_key", "X_pca") or "X_pca"),
            species=str(payload.get("species", "homo_sapiens") or "homo_sapiens"),
            formal=formal,
            epoch=epoch,
            run_selection=bool(payload.get("run_selection", True)),
            run_feasibility=bool(payload.get("run_feasibility", False)),
            run_evaluation=bool(payload.get("run_evaluation", False)),
            build_report=bool(payload.get("build_report", False)),
            skip_odt=bool(payload.get("skip_odt", False)),
            must_keep_genes=[str(item) for item in payload.get("must_keep_genes", []) if str(item).strip()],
            forbidden_genes=[str(item) for item in payload.get("forbidden_genes", []) if str(item).strip()],
            report_format=str(payload.get("report_format", "pdf") or "pdf"),
            notes=[str(item) for item in payload.get("notes", []) if str(item).strip()],
        )

    @staticmethod
    def _planner_response_format() -> dict[str, Any]:
        return {"type": "json_object"}

    @staticmethod
    def _compact_session_snapshot(session_snapshot: dict[str, Any]) -> dict[str, Any]:
        state = session_snapshot.get("state", {})
        return {
            "session_id": session_snapshot.get("session_id", ""),
            "active_dataset_id": session_snapshot.get("active_dataset_id", ""),
            "active_model_id": session_snapshot.get("active_model_id", ""),
            "state": {
                "context_summary": state.get("context_summary", ""),
                "reference_candidates": state.get("reference_candidates", ""),
                "smith_selection_manifest": state.get("smith_selection_manifest", ""),
                "probe_candidate_manifest": state.get("probe_candidate_manifest", ""),
                "feasibility_table": state.get("feasibility_table", ""),
                "feasibility_summary": state.get("feasibility_summary", {}),
                "cross_dataset_evaluation": state.get("cross_dataset_evaluation", ""),
                "last_report": state.get("last_report", {}),
                "last_search": state.get("last_search", {}),
                "decision_trace": state.get("decision_trace", ""),
                "mounted_datasets": state.get("mounted_datasets", {}),
            },
        }

    def _normalize_plan_output(self, raw: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = self._extract_json_object(raw)
        if not isinstance(payload, dict):
            raise ValueError("Planner output is not a JSON object.")
        return {
            "assistant_message": str(payload.get("assistant_message", "")),
            "tool_calls": self._normalize_tool_calls(payload.get("tool_calls", [])),
            "state_updates": payload.get("state_updates", {}) if isinstance(payload.get("state_updates", {}), dict) else {},
            "done": bool(payload.get("done", False)),
        }

    @staticmethod
    def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
        normalized = []
        if not isinstance(tool_calls, list):
            return normalized
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool_name", "")).strip()
            arguments = item.get("arguments", {})
            if not tool_name or not isinstance(arguments, dict):
                continue
            normalized.append({"tool_name": tool_name, "arguments": dict(arguments)})
        return normalized

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        return json.loads(match.group(0))

    @staticmethod
    def _format_http_error(error: requests.HTTPError) -> str:
        response = error.response
        if response is None:
            return f"Planner HTTP error: {error}"
        body = response.text.strip()
        if len(body) > 400:
            body = body[:400] + "..."
        return f"Planner HTTP {response.status_code}: {body or error}"


def fallback_interactive_plan(
    user_message: str,
    session_snapshot: dict[str, Any],
) -> dict[str, Any]:
    text = user_message.lower()
    state = session_snapshot.get("state", {})
    active_dataset_id = session_snapshot.get("active_dataset_id", "")
    if "inspect dataset" in text:
        return {
            "assistant_message": "",
            "tool_calls": [{"tool_name": "resolve_dataset_context", "arguments": {"dataset_id": active_dataset_id}}],
            "state_updates": {},
            "done": False,
        }
    if "search" in text or "find" in text or "look up" in text:
        return {
            "assistant_message": "",
            "tool_calls": [{"tool_name": "search_smith_agent", "arguments": {"query": user_message}}],
            "state_updates": {},
            "done": False,
        }
    if ("reference" in text or "参考" in text or "检索" in text) and (
        "retrieve" in text
        or "rank" in text
        or "recommend" in text
        or "推荐" in text
        or "打分" in text
        or "排序" in text
        or "检索" in text
    ):
        return {
            "assistant_message": "",
            "tool_calls": [{"tool_name": "score_reference_transferability", "arguments": {"request": {}}}],
            "state_updates": {},
            "done": False,
        }
    if any(token in text for token in ["feasibility", "probe", "oligominer", "probedealer", "odt"]) and (
        state.get("smith_selection_manifest") or "panel" in text or "smith" in text
    ):
        return {
            "assistant_message": "",
            "tool_calls": [{"tool_name": "build_probe_candidate_manifest", "arguments": {}}],
            "state_updates": {},
            "done": False,
        }
    if state.get("cross_dataset_evaluation") and ("report" in text or ("summary" in text and "performance" in text)):
        return {
            "assistant_message": "",
            "tool_calls": [{"tool_name": "build_run_report", "arguments": {}}],
            "state_updates": {},
            "done": False,
        }
    if state.get("context_summary") and ("umap" in text or ("plot" in text and "dataset" in text)):
        return {
            "assistant_message": "",
            "tool_calls": [{"tool_name": "plot_dataset_umap", "arguments": {"color": "pathology"}}],
            "state_updates": {},
            "done": False,
        }
    if (
        "run smith" in text
        or "select panel" in text
        or "generate panel" in text
        or "panel selection" in text
        or ("smith" in text and "panel" in text)
    ):
        panel_size = 64 if "64" in text else 16
        epoch = 1 if any(token in text for token in ["smoke", "quick"]) else 5
        return {
            "assistant_message": "",
            "tool_calls": [
                {
                    "tool_name": "run_smith_selection",
                    "arguments": {
                        "adata_file": "",
                        "tasks": "recon,cls,region,pathology",
                        "task_name": "interactive_panel",
                        "panel_size": panel_size,
                        "epoch": epoch,
                        "record": 1,
                        "execute": True,
                        "extra_args": {"batch_size": 4096, "balance_mode": "off"},
                    },
                }
            ],
            "state_updates": {},
            "done": False,
        }
    if "evaluate" in text and ("test" in text or "cross" in text):
        return {
            "assistant_message": "",
            "tool_calls": [
                {
                    "tool_name": "evaluate_cross_dataset_panel",
                    "arguments": {
                        "panel_path": "",
                        "train_adata_file": "",
                        "test_adata_file": "",
                        "panel_size": 16,
                        "label": "pathology",
                        "obsm_key": "X_pca",
                    },
                }
            ],
            "state_updates": {},
            "done": False,
        }
    if state.get("cross_dataset_evaluation") or state.get("smith_selection_manifest"):
        return {
            "assistant_message": "I already have saved results in this session. Use /summary or ask me to evaluate or inspect them.",
            "tool_calls": [],
            "state_updates": {},
            "done": True,
        }
    return {
        "assistant_message": "I could not infer the next workflow step. Ask me to inspect a dataset, retrieve references, run SMITH, or evaluate a panel.",
        "tool_calls": [],
        "state_updates": {},
        "done": True,
    }
