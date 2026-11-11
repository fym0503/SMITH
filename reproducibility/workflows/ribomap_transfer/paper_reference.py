"""Re-evaluate frozen manuscript SMITH panels independently of fresh training."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import anndata as ad
from .evaluate_outputs import evaluate_panel_loaded
from reproducibility.workflows.common import read_panel, write_json

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / 'reproducibility/reference_outputs'

def replay_paper_panels(data_root, output_dir, *, verify=True):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_file = Path(data_root) / 'ribomap_transfer/ribomap/mouse_brain_ribomap_rep2.h5ad'
    target = ad.read_h5ad(target_file)
    panel_manifest = pd.read_csv(REFERENCE / 'ribomap_paper_panels/panel_manifest.tsv', sep='\t')
    records = []
    for row in panel_manifest.itertuples():
        path = REFERENCE / row.panel_path
        with path.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != row.sha256:
                raise ValueError(f'Paper panel checksum mismatch: {path}')
        genes = read_panel(path, int(row.panel_size))
        for seed in range(5):
            result, _ = evaluate_panel_loaded(target, genes, int(row.panel_size), seed,
                                              label_column=row.label, protocol='paper')
            records.append(dict(source=row.source, label=row.label, panel_size=int(row.panel_size),
                                method='SMITH', evaluation_seed=seed, accuracy=result['metrics']['accuracy']))
    metrics = pd.DataFrame(records)
    summary = metrics.groupby(['source','label','panel_size'], as_index=False).agg(
        recomputed_mean=('accuracy','mean'), recomputed_sd=('accuracy','std'))
    expected = pd.read_csv(REFERENCE / 'ribomap_paper_accuracy.tsv', sep='\t').rename(columns={'paper_accuracy':'paper_mean'})
    validation = summary.merge(expected, on=['source','label','panel_size'], how='outer', validate='one_to_one', indicator=True)
    validation['absolute_difference'] = (validation.recomputed_mean - validation.paper_mean).abs()
    metrics.to_csv(out/'metrics.tsv', sep='\t', index=False)
    validation.to_csv(out/'validation.tsv', sep='\t', index=False)
    passed = bool(validation['_merge'].eq('both').all() and np.isfinite(validation.absolute_difference).all()
                  and validation.absolute_difference.le(0.0003).all() and len(validation)==12)
    write_json(out/'validation.json', {'status':'pass' if passed else 'fail', 'tolerance':0.0003,
                                      'max_absolute_difference':float(validation.absolute_difference.max()),
                                      'kind':'frozen_paper_panel_reevaluation', 'target':str(target_file)})
    if verify and not passed:
        raise ValueError(f'Paper-panel replay differs from the reference; inspect {out / "validation.tsv"}')
    return {'metrics':metrics,'summary':summary,'validation':validation}
