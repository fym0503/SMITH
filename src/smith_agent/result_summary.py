from __future__ import annotations

from typing import Any


def _format_metric_lines(report: dict[str, Any]) -> list[str]:
    metrics = report.get("evaluation_metrics", []) if isinstance(report, dict) else []
    lines: list[str] = []
    for row in metrics[:6]:
        if not isinstance(row, dict):
            continue
        evaluation = str(row.get("evaluation", "")).strip()
        metric = str(row.get("metric", "")).strip()
        label = str(row.get("label", "")).strip()
        value = row.get("value")
        try:
            rendered = f"{float(value):.4f}"
        except (TypeError, ValueError):
            continue
        suffix = f" ({label})" if label else ""
        lines.append(f"{evaluation} / {metric}{suffix}: {rendered}")
    return lines


def build_session_summary(session_snapshot: dict[str, Any]) -> str:
    state = session_snapshot.get("state", {})
    lines: list[str] = []
    if session_snapshot.get("active_dataset_id"):
        lines.append(f"Active dataset: {session_snapshot['active_dataset_id']}")
    if session_snapshot.get("active_model_id"):
        lines.append(f"Active model: {session_snapshot['active_model_id']}")
    if state.get("context_summary"):
        lines.append(f"Dataset context: {state['context_summary']}")
    if state.get("smith_selection_manifest"):
        lines.append(f"Last SMITH selection manifest: {state['smith_selection_manifest']}")
    if state.get("cross_dataset_evaluation"):
        lines.append(f"Last cross-dataset evaluation: {state['cross_dataset_evaluation']}")
    if state.get("reference_candidates"):
        lines.append(f"Reference ranking: {state['reference_candidates']}")
    if state.get("reference_selection"):
        lines.append(f"Reference selection: {state['reference_selection']}")
    if state.get("reference_score_matrix_plot"):
        lines.append(f"Reference score plot: {state['reference_score_matrix_plot']}")
    if state.get("cellxgene_reference_candidates"):
        lines.append(f"CELLxGENE candidates: {state['cellxgene_reference_candidates']}")
    if state.get("cellxgene_materialized_h5ad"):
        lines.append(f"CELLxGENE materialized h5ad: {state['cellxgene_materialized_h5ad']}")
    if state.get("decision_trace"):
        lines.append(f"Decision trace: {state['decision_trace']}")
    if state.get("feasibility_summary"):
        summary = state["feasibility_summary"]
        if isinstance(summary, dict):
            if summary.get("integration_summary_tsv"):
                lines.append(f"Feasibility summary: {summary['integration_summary_tsv']}")
            if summary.get("passing_count") is not None and summary.get("total_count") is not None:
                lines.append(f"Feasibility passing count: {summary['passing_count']}/{summary['total_count']}")
    if state.get("last_umap_plot"):
        lines.append(f"Last UMAP plot: {state['last_umap_plot']}")
    if state.get("last_evaluation_plot"):
        lines.append(f"Last evaluation plot: {state['last_evaluation_plot']}")
    if state.get("last_report"):
        report = state["last_report"]
        if isinstance(report, dict):
            if report.get("report_markdown"):
                lines.append(f"Last report markdown: {report['report_markdown']}")
            if report.get("report_html"):
                lines.append(f"Last report HTML: {report['report_html']}")
            if report.get("report_pdf"):
                lines.append(f"Last report PDF: {report['report_pdf']}")
            lines.extend(_format_metric_lines(report))
    if state.get("last_search"):
        query = state["last_search"].get("query", "")
        lines.append(f"Last search query: {query}")
    memory = session_snapshot.get("memory", {})
    if memory.get("last_control_mode"):
        lines.append(f"Last control mode: {memory['last_control_mode']}")
    if memory.get("last_planner_error"):
        lines.append(f"Last planner error: {memory['last_planner_error']}")
    if not lines:
        return "I do not have any completed smith-agent artifacts in this session yet."
    return "\n".join(lines)
