from importlib import resources
from pathlib import Path
import sys

from smith_agent.config import load_agent_config
from smith_agent.registry import load_registries


def test_core_packages_import():
    import smith  # noqa: F401
    import smith_agent  # noqa: F401
    from smith_agent import feasibility, probedealer  # noqa: F401


def test_agent_registry_loads_from_packaged_config():
    repo = Path(__file__).resolve().parents[1]
    config = load_agent_config(repo / "configs" / "agent" / "agent.yaml")
    registries = load_registries(config)
    assert "run_smith_selection" in registries.tools
    assert "smith_default" in registries.models
    assert len(registries.datasets) >= 1
    assert registries.models["smith_default"].entrypoint == "scripts/main.py"


def test_dataset_registry_uses_portable_paths():
    repo = Path(__file__).resolve().parents[1]
    config = load_agent_config(repo / "configs" / "agent" / "agent.yaml")
    registries = load_registries(config)
    for entry in registries.datasets.values():
        assert not entry.path.startswith("/workspace/fanyimin")
        assert entry.metadata.get("data_status") == "external_not_packaged"
        assert entry.metadata.get("original_local_path", "").startswith("/workspace/fanyimin/")

def test_packaged_resource_config_loads(monkeypatch, tmp_path):
    resource_root = Path(str(resources.files("smith_agent.resources")))
    config_path = resource_root / "configs" / "agent" / "agent.yaml"
    monkeypatch.setenv("SMITH_RUNTIME_ROOT", str(tmp_path))

    config = load_agent_config(config_path)
    registries = load_registries(config)

    assert config.repo_root == tmp_path.resolve()
    assert config.tools_dir == resource_root / "configs" / "agent" / "tools"
    assert config.external_roots["smith_unified_root"] == resource_root.resolve()
    assert "run_smith_selection" in registries.tools
    assert "smith_default" in registries.models


def test_model_entrypoint_resolves_from_packaged_resources(monkeypatch, tmp_path):
    from smith_agent.runtime import build_runtime
    from smith_agent.tools.defaults import _resolve_model_entrypoint

    resource_root = Path(str(resources.files("smith_agent.resources")))
    config_path = resource_root / "configs" / "agent" / "agent.yaml"
    monkeypatch.setenv("SMITH_RUNTIME_ROOT", str(tmp_path))

    runtime = build_runtime(config_path=config_path)
    entrypoint = _resolve_model_entrypoint(runtime, runtime.registries.models["smith_default"])

    assert entrypoint == resource_root / "scripts" / "main.py"
    assert entrypoint.exists()


def test_smith_selection_falls_back_to_current_python(monkeypatch, tmp_path):
    from smith_agent.runtime import build_runtime
    from smith_agent.tools import defaults

    repo = Path(__file__).resolve().parents[1]
    runtime = build_runtime(config_path=repo / "configs" / "agent" / "agent.yaml")
    captured = {}

    monkeypatch.setattr(defaults.shutil, "which", lambda _name: None)

    def fake_run(config, execute=False):
        captured["python_executable"] = config.python_executable
        return {"status": "planned", "manifest_path": str(tmp_path / "manifest.json")}

    monkeypatch.setattr(defaults, "run_smith_selection", fake_run)
    defaults._handle_run_smith_selection(
        runtime,
        {
            "adata_file": str(tmp_path / "input.h5ad"),
            "tasks": "recon",
            "task_name": "test",
            "execute": False,
        },
    )

    assert captured["python_executable"] == sys.executable
