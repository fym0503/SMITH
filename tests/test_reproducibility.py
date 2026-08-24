from __future__ import annotations

import io
import tarfile
from dataclasses import replace
from pathlib import Path

import anndata as ad
import nbformat
import numpy as np
import pandas as pd
import pytest
import yaml

from smith.reproducibility import check_case, load_cases, run_case
from scripts.download_tutorial_data import safe_extract
from scripts.build_tutorial_archives import case_files
from reproducibility.workflows.ribomap_transfer.analysis import (
    bh_adjust,
    jaccard_similarity,
    ribomap_bias,
)
from reproducibility.workflows.regulatory_activity.analysis import paired_wilcoxon
from reproducibility.workflows.regulatory_activity.paper_analysis import (
    coactivity_reconstruction,
    module_miss_rate,
    tf_scrna_correlation,
)
from reproducibility.workflows.regulatory_activity.plot_figure3 import plot as plot_figure3


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
    paper_inputs = manifest["cases"]["02_regulatory_activity"]["paper_inputs"]
    assert paper_inputs["status"] == "prepared_pending_zenodo_upload"
    assert len(paper_inputs["files"]) == 3
    assert all(item["bytes"] > 0 and len(item["sha256"]) == 64 for item in paper_inputs["files"])
    assert len(case_files(manifest["cases"]["02_regulatory_activity"])) == (
        len(manifest["cases"]["02_regulatory_activity"]["files"]) + 3
    )


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
        "02_SMITH_Regulatory_Activity_source.ipynb": ("Regulatory programs and cross-modality transfer in C. elegans", "plot_figure3.py"),
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


def test_regulatory_tutorial_interleaves_code_and_figures():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(
        root / "docs/source/tutorials/notebooks/regulatory_section/02_SMITH_Regulatory_Activity_source.ipynb",
        as_version=4,
    )
    display_cells = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and "render_figure3_panel(" in str(cell.source)
        and not str(cell.source).lstrip().startswith("from pathlib")
    ]
    assert len(display_cells) == 10
    assert "Render the manuscript panels" not in "\n".join(str(cell.source) for cell in notebook.cells)
    for index in display_cells:
        assert notebook.cells[index - 1].cell_type == "markdown"
        assert str(notebook.cells[index - 1].source).lstrip().startswith("#")
        assert str(notebook.cells[index].source).count("render_figure3_panel(") == 1


def test_ribomap_plot_has_no_placeholder_panel():
    root = Path(__file__).resolve().parents[1]
    plotter = (root / "reproducibility" / "workflows" / "ribomap_transfer" / "plot_figure4.py").read_text()
    assert "Pathway enrichment" not in plotter
    assert "versioned Reactome/GO" not in plotter


def test_plotters_export_independent_manuscript_panels():
    root = Path(__file__).resolve().parents[1]
    notebook_builder = (root / "scripts" / "build_tutorial_notebooks.py").read_text()
    expected = {
        "regulatory_activity/plot_figure3.py": ["figure3_c", "figure3_d", "figure3_e", "figure3_f", "figure3_g", "figure3_h", "figure3_i", "figure3_j", "figure3_k"],
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


def test_ribomap_analysis_formulas_are_manuscript_defined():
    assert jaccard_similarity({"A", "B"}, {"B", "C"}) == pytest.approx(1 / 3)
    ribomap_values = np.array([1.0, 3.0, 9.0])
    starmap_values = np.array([1.0, 2.0, 8.0])
    expected_ribo = (np.log1p(ribomap_values) - np.mean(np.log1p(ribomap_values))) / np.std(np.log1p(ribomap_values))
    expected_star = (np.log1p(starmap_values) - np.mean(np.log1p(starmap_values))) / np.std(np.log1p(starmap_values))
    assert np.allclose(ribomap_bias(ribomap_values, starmap_values), expected_ribo - expected_star)
    assert np.allclose(bh_adjust([0.01, 0.04, 0.2]), [0.03, 0.06, 0.2])


def test_regulatory_paired_test_uses_split_as_pairing_unit():
    values = pd.DataFrame([
        {"dataset": "elegans_tf", "split": "split_1", "panel_size": 32, "method": "SMITH", "metric": 0.8},
        {"dataset": "elegans_tf", "split": "split_1", "panel_size": 32, "method": "PERSIST-class", "metric": 0.6},
        {"dataset": "elegans_tf", "split": "split_2", "panel_size": 32, "method": "SMITH", "metric": 0.9},
        {"dataset": "elegans_tf", "split": "split_2", "panel_size": 32, "method": "PERSIST-class", "metric": 0.7},
    ])
    result = paired_wilcoxon(values.rename(columns={"metric": "cell_type_accuracy"}), "cell_type_accuracy")
    assert len(result) == 1
    assert result.iloc[0]["n_pairs"] == 2


def test_regulatory_module_miss_rate_uses_selected_panel_only():
    modules = pd.DataFrame([
        {"module_id": "muscle_early", "gene_symbol": "MyoD"},
        {"module_id": "muscle_early", "gene_symbol": "HLH-1"},
        {"module_id": "neuron_late", "gene_symbol": "UNC-30"},
    ])
    assert module_miss_rate(["myod"], modules) == pytest.approx(0.5)


def test_regulatory_paper_analysis_reconstructs_coactivity_and_tf_transfer(tmp_path: Path):
    rng = np.random.default_rng(4)
    genes = ["TF1", "TF2", "TF3", "TF4"]
    lineages = [f"lineage_{index // 2}" for index in range(16)]
    train_obs = pd.DataFrame({
        "cell_type": ["muscle"] * 8 + ["neuron"] * 8,
        "cell_name": lineages,
        "random_precise_lineage": lineages,
    })
    test_obs = train_obs.copy()
    train = ad.AnnData(rng.normal(size=(16, 4)).astype("float32"), obs=train_obs, var=pd.DataFrame(index=genes))
    test = ad.AnnData(rng.normal(size=(16, 4)).astype("float32"), obs=test_obs, var=pd.DataFrame(index=genes))
    train_file, test_file = tmp_path / "train.h5ad", tmp_path / "test.h5ad"
    train.write_h5ad(train_file)
    test.write_h5ad(test_file)
    panel_file = tmp_path / "panel.tsv"
    pd.DataFrame({"gene_symbol": ["TF1", "TF2"]}).to_csv(panel_file, sep="\t", index=False)
    pair_file = tmp_path / "pairs.tsv"
    pd.DataFrame({"gene_a": ["TF1", "TF2"], "gene_b": ["TF3", "TF4"]}).to_csv(pair_file, sep="\t", index=False)
    coactivity_file = tmp_path / "coactivity.tsv"
    coactivity_reconstruction(
        train_file, test_file, panel_file, coactivity_file, pair_file=pair_file, method="ridge"
    )
    coactivity = pd.read_csv(coactivity_file, sep="\t")
    assert set(coactivity["lineage"]) == {"muscle", "neuron"}
    correlation_file = tmp_path / "correlation.tsv"
    tf_scrna_correlation(train_file, test_file, correlation_file, n_clusters=2)
    correlation = pd.read_csv(correlation_file, sep="\t").iloc[0]
    assert correlation["shared_genes"] == 4
    assert correlation["shared_lineages"] == 8
    assert (tmp_path / correlation["matrix_file"]).is_file()


def test_figure3_plotter_consumes_generated_analysis_outputs(tmp_path: Path):
    values = []
    for dataset, sizes in (("elegans_tf", (32, 64, 128)), ("elegans_mirna", (16, 24, 32))):
        for size in sizes:
            values.append({
                "dataset": dataset, "split": "split_1", "panel_size": size,
                "method": "SMITH", "cell_type_accuracy": 0.65,
                "developmental_time_pearson": 0.9,
            })
    values_path = tmp_path / "values.tsv"
    pd.DataFrame(values).to_csv(values_path, sep="\t", index=False)

    modules_path = tmp_path / "modules.tsv"
    pd.DataFrame([
        {"module_id": "muscle|M|early", "tissue": "muscle", "progenitor_lineage": "M",
         "temporal_module": "early", "gene_symbol": "TF1"},
        {"module_id": "neuron|AB|late", "tissue": "neuron", "progenitor_lineage": "AB",
         "temporal_module": "late", "gene_symbol": "TF2"},
    ]).to_csv(modules_path, sep="\t", index=False)
    coverage_path = tmp_path / "coverage.tsv"
    pd.DataFrame([
        {"method": "SMITH", "panel_size": size, "module_miss_rate": 0.4 - size / 200}
        for size in (16, 24, 32)
    ]).to_csv(coverage_path, sep="\t", index=False)
    coactivity_path = tmp_path / "coactivity.tsv"
    pd.DataFrame([
        {"method": method, "lineage": lineage, "pearson": 0.5}
        for method in ("SMITH", "PERSIST") for lineage in ("muscle", "neuron", "pharynx", "skin")
    ]).to_csv(coactivity_path, sep="\t", index=False)
    matrix_path = tmp_path / "figure3_j_correlation_matrices.npz"
    np.savez_compressed(matrix_path, genes=np.array(["TF1", "TF2"]), tf=np.eye(2), scrna=np.eye(2))
    correlation_path = tmp_path / "correlation.tsv"
    pd.DataFrame([{
        "mean_rowwise_pearson": 0.38, "matrix_file": matrix_path.name,
    }]).to_csv(correlation_path, sep="\t", index=False)
    transfer_path = tmp_path / "transfer.tsv"
    pd.DataFrame([
        {"method": method, "source_modality": source, "panel_size": size, "cell_type_accuracy": 0.6}
        for method in ("SMITH", "PERSIST-class") for source in ("TF-TF", "RNA-TF")
        for size in (32, 64, 128)
    ]).to_csv(transfer_path, sep="\t", index=False)

    outputs = plot_figure3(
        values_path, tmp_path / "figures", modules_path=modules_path,
        module_coverage_path=coverage_path, coactivity_path=coactivity_path,
        correlation_path=correlation_path, transfer_path=transfer_path,
    )
    assert {f"figure3_{letter}" for letter in "cdefghijk"}.issubset(outputs)
    assert all(Path(outputs[f"figure3_{letter}"]["png"]).is_file() for letter in "cdefghijk")
