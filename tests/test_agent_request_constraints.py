from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from smith.datasets import get_dataset, get_prior_idx
from smith.eval import save_results
from smith_agent.adapters.smith_runner import SmithSelectionConfig, build_smith_command
from smith_agent.llm import LLMClient
from smith_agent.schemas import PanelPipelineRequest


def _args(adata_path, prior="0", forbidden="0", tasks="recon,cls"):
    return type("Args", (), {
        "adata_file": str(adata_path), "layer": "raw", "hvg": False,
        "forbidden_file": str(forbidden), "prior_file": str(prior),
        "panel_size": 3, "tasks": tasks, "balance_mode": "off", "seed": 7,
        "time_label": "development_time", "spatial_key": "spatial",
    })()


def test_pipeline_schema_preserves_time_coordinate_and_gene_constraints():
    payload = {
        "tasks": ["recon", "cls", "time", "standard_coordination"],
        "time_label": "embryo_time", "spatial_key": "X_spatial",
        "must_keep_genes": ["CSF1R"], "forbidden_genes": ["AQP4"],
        "obsm_key": "X_spatial",
    }
    request = LLMClient._panel_pipeline_request_from_payload(payload)
    assert request.tasks == payload["tasks"]
    assert request.time_label == "embryo_time"
    assert request.spatial_key == "X_spatial"
    assert request.obsm_key == "X_spatial"
    assert request.must_keep_genes == ["CSF1R"]
    assert request.forbidden_genes == ["AQP4"]
    assert request.task_string == "recon,cls,time,standard_coordination"


def test_pipeline_schema_does_not_turn_coordinates_into_region_classification():
    request = LLMClient._panel_pipeline_request_from_payload({
        "tasks": ["recon", "coordination"], "spatial_key": "spatial",
        "obsm_key": "spatial",
    })
    assert "coordination" in request.tasks
    assert "region" not in request.tasks
    assert request.obsm_key == "spatial"


def test_forbidden_genes_are_removed_and_prior_genes_retained(tmp_path):
    adata_path = tmp_path / "input.h5ad"
    ad.AnnData(np.arange(32, dtype=np.float32).reshape(8, 4),
               obs=pd.DataFrame({"celltype": ["a", "b"] * 4}, index=[f"c{i}" for i in range(8)]),
               var=pd.DataFrame(index=["A", "B", "C", "D"])).write_h5ad(adata_path)
    prior_path, forbidden_path = tmp_path / "prior.csv", tmp_path / "forbidden.csv"
    pd.DataFrame({"names": ["B"]}).to_csv(prior_path, index=False)
    pd.DataFrame({"names": ["D"]}).to_csv(forbidden_path, index=False)
    var, _, _ = get_dataset(_args(adata_path, prior_path, forbidden_path))
    assert "D" not in var.tolist()
    assert get_prior_idx(_args(adata_path, prior_path, forbidden_path), var) == [1]


def test_constraints_fail_explicitly_for_missing_or_over_budget_prior(tmp_path):
    adata_path = tmp_path / "input.h5ad"
    ad.AnnData(np.ones((8, 4), dtype=np.float32),
               obs=pd.DataFrame({"celltype": ["a", "b"] * 4}, index=[f"c{i}" for i in range(8)]),
               var=pd.DataFrame(index=["A", "B", "C", "D"])).write_h5ad(adata_path)
    prior, forbidden = tmp_path / "prior.csv", tmp_path / "forbidden.csv"
    pd.DataFrame({"names": ["D"]}).to_csv(prior, index=False)
    pd.DataFrame({"names": ["D"]}).to_csv(forbidden, index=False)
    with pytest.raises(ValueError, match="absent or excluded"):
        get_dataset(_args(adata_path, prior, forbidden))
    pd.DataFrame({"names": ["A", "B", "C", "D"]}).to_csv(prior, index=False)
    with pytest.raises(ValueError, match="exceed panel_size"):
        get_dataset(_args(adata_path, prior, "0"))


def test_smith_command_carries_time_spatial_and_constraint_files(tmp_path):
    config = SmithSelectionConfig(
        python_executable="python", smith_root=tmp_path,
        adata_file=tmp_path / "data.h5ad", saving_dir=tmp_path / "out",
        log_dir=tmp_path / "logs", tasks="recon,time,coordination",
        task_name="spatiotemporal", prior_file=tmp_path / "prior.csv",
        forbidden_file=tmp_path / "forbidden.csv",
        extra_args={"time_label": "embryo_time", "spatial_key": "X_spatial"},
    )
    command = build_smith_command(config)
    assert command[command.index("--time_label") + 1] == "embryo_time"
    assert command[command.index("--spatial_key") + 1] == "X_spatial"
    assert command[command.index("--prior_file") + 1] == str(config.prior_file)
    assert command[command.index("--forbidden_file") + 1] == str(config.forbidden_file)


def test_saved_ranking_places_required_genes_inside_panel_budget(tmp_path):
    class Gate:
        mu = torch.tensor([0.1, 0.9, 0.8, 0.7])
    prior_path = tmp_path / "prior.csv"
    pd.DataFrame({"names": ["A"]}).to_csv(prior_path, index=False)
    args = type("Args", (), {"prior_file": str(prior_path), "panel_size": 2,
                                "tasks": "recon,cls", "task_name": "prior_test", "seed": 1,
                                "log_dir": str(tmp_path / "logs"),
                                "saving_dir": str(tmp_path / "saving"), "val": False, "save": False})()
    save_results(0, np.array(["A", "B", "C", "D"]), {"rep": Gate()}, args, ["input", "recon"], [0.0], [])
    ranking = pd.read_csv(tmp_path / "saving/epoch_0.csv")["marker"].tolist()
    assert ranking[:2] == ["A", "B"]
