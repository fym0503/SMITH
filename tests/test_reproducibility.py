from __future__ import annotations

import io
import tarfile
from dataclasses import replace
from pathlib import Path

import nbformat
import pytest
import yaml

from smith.reproducibility import check_case, load_cases, run_case
from scripts.download_tutorial_data import safe_extract


EXPECTED_CASES = {"01_wmb", "02_regulatory_activity", "03_ribomap_transfer", "05_agent"}


def test_public_case_registry_has_only_approved_chapters():
    cases = load_cases()
    assert set(cases) == EXPECTED_CASES
    assert [case.order for case in cases.values()] == [1, 2, 3, 5]
    assert all("fixture" not in str(item).lower() for case in cases.values() for item in case.inputs)
    assert not check_case(cases["01_wmb"])["ready"]
    assert all(not check_case(cases[case_id])["ready"] for case_id in EXPECTED_CASES - {"01_wmb"})


def test_data_root_drives_readiness(tmp_path: Path):
    original = load_cases()["02_regulatory_activity"]
    case = replace(original, inputs=tuple({"path": item["path"], "kind": "data"} for item in original.inputs))
    for item in case.inputs:
        path = tmp_path / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"h5ad-placeholder")
    status = check_case(case, data_root=tmp_path)
    assert status["ready"]
    assert all(item["kind"] == "data" for item in status["inputs"])


def test_wmb_and_missing_data_cannot_run(tmp_path: Path):
    for case_id in ("01_wmb", "02_regulatory_activity"):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            run_case(load_cases()[case_id], tmp_path / "out", data_root=tmp_path)


def test_data_manifest_has_sizes_checksums_and_unpublished_zenodo():
    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((root / "reproducibility" / "data_manifest.yaml").read_text())
    assert manifest["zenodo_record_url"] is None
    assert manifest["publication_status"] == "prepared_not_uploaded"
    assert set(manifest["cases"]) == EXPECTED_CASES - {"01_wmb"}
    for case in manifest["cases"].values():
        assert case["archive_url"] is None
        assert case["files"]
        for item in case["files"]:
            assert item["path"].endswith(".h5ad")
            assert item["bytes"] > 0
            assert len(item["sha256"]) == 64


def test_safe_extract_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        payload = b"unsafe"
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="escapes data root"):
        safe_extract(archive, tmp_path / "extract")


def test_notebooks_call_workflows_and_do_not_read_reference_outputs():
    root = Path(__file__).resolve().parents[1]
    sources = sorted(
        path for path in (root / "docs" / "source" / "tutorials" / "notebooks").glob("**/*_source.ipynb")
        if not path.name.startswith("._")
    )
    assert len(sources) == 3
    for path in sources:
        notebook = nbformat.read(path, as_version=4)
        text = "\n".join(str(cell.source) for cell in notebook.cells)
        assert "run_tutorial.py" in text
        assert "plot_figure" in text
        assert "run_manifest.json" in text
        assert "reference_outputs" not in text
        assert "verify_fixture" not in text
        assert "Provenance Analysis" not in text


def test_tutorial_sources_target_manuscript_panels():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "02_SMITH_Regulatory_Activity_source.ipynb": ("Regulatory activity and developmental identity in C. elegans", "plot_figure3.py"),
        "03_SMITH_RIBOMap_Transfer_source.ipynb": ("Cross-modality brain panel transfer to RIBOMap", "plot_figure4.py"),
        "05_SMITH_Agent_Evaluation_source.ipynb": ("Liver cell identity in MERFISH", "plot_figure6.py"),
    }
    sources = (
        path for path in (root / "docs" / "source" / "tutorials" / "notebooks").glob("**/*_source.ipynb")
        if not path.name.startswith("._")
    )
    for path in sources:
        notebook = nbformat.read(path, as_version=4)
        text = "\n".join(str(cell.source) for cell in notebook.cells)
        figure, plotter = expected[path.name]
        assert figure in text
        assert "Reproduce SMITH Figure" not in text
        assert plotter in text
        assert "quick hosted run" in text
        assert "--output-dir" in text
        assert "import pandas" not in text
        assert "pd.DataFrame" not in text
        assert "pd.read_csv" not in text
        assert "display(Image" in text


def test_executed_tutorials_include_rendered_figure_outputs():
    root = Path(__file__).resolve().parents[1]
    executed = sorted(
        path for path in (root / "docs" / "source" / "tutorials" / "notebooks").glob("**/*_executed.ipynb")
        if not path.name.startswith("._")
    )
    assert len(executed) == 3
    for path in executed:
        notebook = nbformat.read(path, as_version=4)
        image_outputs = [
            output
            for cell in notebook.cells
            for output in cell.get("outputs", [])
            if output.output_type == "display_data" and "image/png" in output.get("data", {})
        ]
        assert image_outputs, f"{path} has no rendered example figures"


def test_ribomap_plot_has_no_placeholder_panel():
    root = Path(__file__).resolve().parents[1]
    plotter = (root / "reproducibility" / "workflows" / "ribomap_transfer" / "plot_figure4.py").read_text()
    assert "Pathway enrichment" not in plotter
    assert "versioned Reactome/GO" not in plotter


def test_plotters_export_independent_manuscript_panels():
    root = Path(__file__).resolve().parents[1]
    notebook_builder = (root / "scripts" / "build_tutorial_notebooks.py").read_text()
    expected = {
        "regulatory_activity/plot_figure3.py": ["figure3_c", "figure3_d", "figure3_e", "figure3_f"],
        "ribomap_transfer/plot_figure4.py": [
            "figure4_c", "figure4_d", "figure4_e", "figure4_f", "figure4_g", "figure4_h"
        ],
        "agent/plot_figure6.py": ["figure6_c", "figure6_d"],
    }
    workflow_root = root / "reproducibility" / "workflows"
    for relative, stems in expected.items():
        text = (workflow_root / relative).read_text()
        assert "--output-dir" in text
        assert "--output-prefix" not in text
        for stem in stems:
            assert stem in text or stem in notebook_builder
