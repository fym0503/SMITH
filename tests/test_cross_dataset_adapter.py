import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from smith.eval import load_panel_genes
from smith_agent.adapters.smith_eval import evaluate_cross_dataset_panel


def test_cross_dataset_adapter_uses_source_fit_and_named_genes(tmp_path):
    train_x = np.array([[0, 100], [1, -100], [9, 100], [10, -100]], dtype=float)
    test_x = np.array([[8, 0.2], [-8, 9.8]], dtype=float)
    train = ad.AnnData(sparse.csr_matrix(train_x), var=pd.DataFrame(index=['G', 'UNUSED']))
    test = ad.AnnData(sparse.csr_matrix(test_x), var=pd.DataFrame(index=['UNUSED', 'G']))
    train.obs['label'] = ['low', 'low', 'high', 'high']
    test.obs['label'] = ['low', 'high']
    train.obs['time'] = train_x[:, 0]
    test.obs['time'] = test_x[:, 1]
    train.obsm['spatial'] = (2 * train_x[:, :1] + 3)
    test.obsm['spatial'] = (2 * test_x[:, 1:] + 3)
    train_path, test_path, panel = tmp_path/'train.h5ad', tmp_path/'test.h5ad', tmp_path/'panel.csv'
    train.write_h5ad(train_path); test.write_h5ad(test_path)
    pd.DataFrame({'marker': ['G']}).to_csv(panel, index=False)
    result = evaluate_cross_dataset_panel(
        tmp_path, panel, train_path, test_path, panel_size=1, label='label',
        obsm_key='spatial', time_label='time', n_neighbors=1,
    )
    np.testing.assert_allclose(result['value'], [1, 1, 1])
    # Reversing only target labels must change the score: no target-label fitting.
    test.obs['label'] = ['high', 'low']
    test.write_h5ad(test_path)
    reversed_result = evaluate_cross_dataset_panel(
        tmp_path, panel, train_path, test_path, panel_size=1,
        label='label', obsm_key=None, n_neighbors=1,
    )
    assert reversed_result.iloc[0]['value'] == 0


def test_panel_loader_rejects_short_or_duplicate_panel(tmp_path):
    panel = tmp_path/'panel.csv'
    pd.DataFrame({'marker': ['G', 'G']}).to_csv(panel, index=False)
    with pytest.raises(ValueError):
        load_panel_genes(panel, panel_size=2)
    with pytest.raises(ValueError):
        load_panel_genes(panel, panel_size=3)
