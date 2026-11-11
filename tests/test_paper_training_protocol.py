"""Regression checks for the shared training and tutorial evaluation protocol."""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from reproducibility.workflows.ribomap_transfer.evaluate_outputs import evaluate_panel_loaded
from reproducibility.workflows.common import resolve_training_model, resolve_tutorial_profile


def test_task_profiles_share_implementation_and_keep_parameters():
    ribo = resolve_tutorial_profile("ribomap_profile")
    agent = resolve_tutorial_profile("Agent_profile")
    tf = resolve_tutorial_profile("TF_profile")
    assert ribo["training_model"]["training_backend"] == "smith"
    assert agent["training_model"]["training_backend"] == "smith"
    assert tf["training_model"]["training_backend"] == "smith"
    assert ribo["parameters"]["panel_sizes"] == [32, 64, 128]
    assert agent["parameters"]["training_seeds"] == [1, 2, 3, 4, 5]
    assert tf["parameters"]["training_seed"] == 8013
    assert tf["parameters"]["splits"] == [f"split_{i}" for i in range(1, 6)]
    assert tf["parameters"]["max_cells"] is None
    assert tf["parameters"]["epochs"] == 90
    assert tf["parameters"]["datasets"] == ["elegans_tf"]


def test_tf_reference_grid_and_rankings_cover_all_splits():
    root = Path(__file__).resolve().parents[1]
    metrics = pd.read_csv(root / "reproducibility/reference_outputs/tf_paper_metrics.tsv", sep="\t")
    panels = pd.read_csv(root / "reproducibility/reference_outputs/tf_paper_panels.tsv", sep="\t")
    expected = {(f"split_{i}", size) for i in range(1, 6) for size in (32, 64, 128)}
    assert set(zip(metrics["split"], metrics["panel_size"])) == expected
    assert len(metrics) == len(expected)
    for split, frame in panels.groupby("split"):
        assert frame["rank"].tolist() == list(range(1, 129))
        assert frame["gene_symbol"].nunique() == 128
    config = json.loads((root / "reproducibility/configs/tf_paper.json").read_text())
    assert config["tasks"] == "recon,cls,time"
    assert config["epoch"] == 90 and config["seed"] == 8013
    assert config["balance_mode"] == "off" and config["max_cells"] is None


def test_paper_evaluation_matches_original_unscaled_uniform_knn():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(90, 4)).astype(np.float32) * np.array([1, 50, 2, 0.1], dtype=np.float32)
    labels = np.where(x[:, 0] > 0, 'a', 'b')
    data = ad.AnnData(x, obs=pd.DataFrame({'celltype': labels}, index=[str(i) for i in range(90)]))
    data.var_names = ['A','B','C','D']
    train, test = train_test_split(np.arange(90), test_size=.2, random_state=2)
    expected = KNeighborsClassifier(n_neighbors=5).fit(x[train], labels[train]).predict(x[test])
    result, predictions = evaluate_panel_loaded(data, list(data.var_names), 4, 2,
                                                label_column='celltype', protocol='paper')
    assert predictions.cell_index.tolist() == list(data.obs_names[test])
    np.testing.assert_array_equal(predictions.prediction, expected)
    assert result['metrics']['accuracy'] == accuracy_score(labels[test], expected)


def test_profiles_share_the_same_trainer():
    from smith import train
    import runpy
    root = Path(__file__).resolve().parents[1]
    for path in (root / 'scripts/main.py', root / 'src/smith_agent/resources/scripts/main.py'):
        assert runpy.run_path(str(path))['main'] is train.main
    entries = {resolve_tutorial_profile(name)['training_model']['entrypoint']
               for name in ('TF_profile', 'ribomap_profile', 'Agent_profile')}
    assert entries == {root / 'scripts/main.py'}
    assert not list((root / 'reproducibility/backends').glob('*/main.py'))
