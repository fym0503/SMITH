import anndata as ad
import numpy as np
import pandas as pd

from smith_agent.reporting.plots import plot_dataset_umap


def test_report_generates_missing_umap_without_modifying_input(tmp_path):
    rng = np.random.default_rng(42)
    values = rng.lognormal(size=(36, 12)).astype(np.float32)
    path = tmp_path / "real_format_no_embedding.h5ad"
    ad.AnnData(
        values,
        obs=pd.DataFrame({"celltype": ["A", "B", "C"] * 12}, index=[f"cell_{i}" for i in range(36)]),
    ).write_h5ad(path)
    output = tmp_path / "report/train_umap.png"
    plot_dataset_umap(path, output, color="celltype", max_cells=30, seed=9)
    assert output.is_file() and output.stat().st_size > 0
    assert output.with_suffix(".embedding.csv").is_file()
    import json
    metadata = json.loads(output.with_suffix(".embedding.json").read_text())
    assert metadata["generated"] is True
    assert metadata["n_plotted_cells"] == 30
    original = ad.read_h5ad(path)
    assert "X_umap" not in original.obsm


def test_report_embedding_caps_large_missing_embeddings(tmp_path):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(2100, 40)).astype(np.float32)
    path = tmp_path / "large_without_embedding.h5ad"
    ad.AnnData(values).write_h5ad(path)
    output = tmp_path / "large_umap.png"
    plot_dataset_umap(path, output, max_cells=50000)
    metadata = __import__("json").loads(output.with_suffix(".embedding.json").read_text())
    assert metadata["n_plotted_cells"] == 2000
    assert metadata["embedding_cell_limit"] == 2000
