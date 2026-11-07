from pathlib import Path

from smith_agent.config import load_agent_config
from smith_agent.interactive import InteractiveSmithAgent
from smith_agent.mode_router import ControlMode
from smith_agent.schemas import PanelPipelineRequest


def test_skip_probe_screening_preserves_evaluation(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("SMITH_RUNTIME_ROOT", str(tmp_path))
    agent = InteractiveSmithAgent(load_agent_config(repo / "configs/agent/agent.yaml"))
    session = agent.create_session()
    session.state["smith_selection_manifest"] = "existing-panel.json"
    message = "Evaluate the selected panel. Skip all probe and feasibility screening."
    request = PanelPipelineRequest(
        train_adata_file="source.h5ad", test_adata_file="target.h5ad",
        run_evaluation=True, run_feasibility=False, panel_size=8,
    )
    session.memory["pipeline_request"] = {**request.to_dict(), "source_message": message}
    monkeypatch.setattr(agent, "_resolve_last_panel", lambda _: "rank.csv")
    monkeypatch.setattr(agent, "_extract_dataset_paths", lambda *_: ("source.h5ad", "target.h5ad"))
    decision = {"tool_calls": [{"tool_name": "evaluate_cross_dataset_panel", "arguments": {
        "panel_path": "rank.csv", "train_adata_file": "source.h5ad",
        "test_adata_file": "target.h5ad", "panel_size": 8,
    }}]}
    result = agent._repair_tool_dependencies(session, message, decision, ControlMode.PIPELINE)
    assert [call["tool_name"] for call in result["tool_calls"]] == ["evaluate_cross_dataset_panel"]


def test_requested_feasibility_still_inserts_prerequisite(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("SMITH_RUNTIME_ROOT", str(tmp_path))
    agent = InteractiveSmithAgent(load_agent_config(repo / "configs/agent/agent.yaml"))
    session = agent.create_session()
    session.state["smith_selection_manifest"] = "existing-panel.json"
    message = "Run probe feasibility screening for this panel."
    request = PanelPipelineRequest(train_adata_file="source.h5ad", run_feasibility=True, panel_size=8)
    session.memory["pipeline_request"] = {**request.to_dict(), "source_message": message}
    monkeypatch.setattr(agent, "_resolve_last_panel", lambda _: "rank.csv")
    monkeypatch.setattr(agent, "_extract_dataset_paths", lambda *_: ("source.h5ad", ""))
    result = agent._repair_tool_dependencies(session, message, {"tool_calls": []}, ControlMode.PIPELINE)
    assert result["tool_calls"][0]["tool_name"] == "build_probe_candidate_manifest"
