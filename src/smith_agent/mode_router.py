from __future__ import annotations

from enum import Enum


class ControlMode(str, Enum):
    PIPELINE = "pipeline"
    REACT = "react"


PIPELINE_VERBS = {
    "build",
    "create",
    "design",
    "evaluate",
    "export",
    "filter",
    "generate",
    "produce",
    "run",
    "select",
    "test",
}

PIPELINE_DELIVERABLES = {
    "evaluation",
    "feasibility",
    "filtered panel",
    "formal",
    "odt",
    "oligominer",
    "panel",
    "pdf",
    "probe",
    "probedealer",
    "report",
    "smith",
    "transfer",
}

ANALYSIS_CUES = {
    "analyze",
    "compare",
    "debug",
    "diagnose",
    "explain",
    "find",
    "inspect",
    "interpret",
    "look",
    "search",
    "show",
    "summarize",
    "why",
}


def route_control_mode(user_message: str) -> ControlMode:
    """Choose the top-level controller.

    Pipeline mode is reserved for standard panel-design deliverables.
    Everything else goes through ReAct, including search, inspection,
    explicit tool requests, and result interpretation.
    """

    text = f" {user_message.lower()} "
    if re_like_explicit_tool_request(text):
        return ControlMode.REACT
    has_deliverable = any(token in text for token in PIPELINE_DELIVERABLES)
    has_pipeline_verb = any(f" {verb} " in text for verb in PIPELINE_VERBS)

    if has_deliverable and has_pipeline_verb:
        return ControlMode.PIPELINE

    # Common imperative panel-design phrasing without a simple verb token.
    if "panel selection" in text or ("selected panel" in text and "evaluate" in text):
        return ControlMode.PIPELINE

    # Analysis cues are intentionally not special-cased into submodes.
    # They document why these requests remain under ReAct.
    if any(f" {cue} " in text for cue in ANALYSIS_CUES):
        return ControlMode.REACT

    return ControlMode.REACT


def re_like_explicit_tool_request(text: str) -> bool:
    return any(token in text for token in [" run tool ", " call tool ", " execute tool ", " /run-tool "])
