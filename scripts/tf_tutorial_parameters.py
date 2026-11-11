"""TF-only notebook configuration and five-split execution cells."""

def apply_tf_parameters(nb):
    nb.cells[0].source = '# Regulatory programs and cross-modality transfer in C. elegans\n\nUse `TF_profile` to select TF panels and evaluate cell identity and developmental time across five lineage splits. Each split trains one model for 90 epochs with seed 8013 and the full training set, then evaluates its top 32, 64 and 128 TFs.'
    nb.cells[1].source = '## Biological question\n\nWhich TFs preserve cell identity, developmental timing and co-activity in unseen cell lineages? Compare panels selected from TF activity with panels selected from a scRNA-seq reference.'
    nb.cells[7].source = '## Train SMITH and select TF panels\n\nTrain one model per lineage split using reconstruction, classification and time objectives. Extract three panel sizes from each learned ranking.'
    nb.cells[14].source = '## Regulatory reconstruction and modality transfer\n\n### TF co-activity reconstruction\n\nReconstruct held-out activity using the top 32 TFs from each split model. Summarize agreement for muscle, neuron, pharynx and skin across five splits.'
    nb.cells[18].source = '### Transfer from scRNA-seq to TF activity\n\nTrain the scRNA-seq model with its recorded transfer configuration, then evaluate its panels on each of the five TF splits.'
    setup=nb.cells[4].source
    setup=setup.replace('EPOCHS = int(os.environ.get("SMITH_TUTORIAL_EPOCHS", PROFILE_PARAMETERS["epochs"]))','EPOCHS = PROFILE_PARAMETERS["epochs"]')
    setup=setup.replace('MAX_CELLS = int(os.environ.get("SMITH_TUTORIAL_MAX_CELLS", str(PROFILE_PARAMETERS["max_cells"])))','MAX_CELLS = PROFILE_PARAMETERS["max_cells"]')
    setup += '''
import json
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 180
CONFIG = json.loads((ROOT / PROFILE["parameter_set"]).read_text())
SPLITS = PROFILE_PARAMETERS["splits"]
MODEL_KEYS = ("learning_rate", "dim", "rep_dim", "rep_hidden_dims", "head_hidden_dims", "dropout_rate", "lam", "sigma", "activation", "optimizer")
REUSE_EXISTING = os.environ.get("SMITH_TUTORIAL_REUSE_EXISTING", "0") == "1"
'''
    nb.cells[4].source=setup
    nb.cells[6].source='''relative_inputs = [f"regulatory_activity/elegans/splits/elegans_tf/{split}/{part}.h5ad" for split in SPLITS for part in ("train", "test")]
relative_inputs += ["regulatory_activity/elegans/annotations/tf_spatiotemporal_modules.tsv", "regulatory_activity/elegans/annotations/tf_regulatory_pairs.tsv", "regulatory_activity/elegans/reference/elegans_scrna.h5ad"]
input_paths = {name: DATA_ROOT / name for name in relative_inputs}
input_checksums = {name: sha256_file(path) for name, path in input_paths.items()}
loaded_splits = {}
for split in SPLITS:
    prefix = f"regulatory_activity/elegans/splits/elegans_tf/{split}"
    train = ad.read_h5ad(input_paths[prefix + "/train.h5ad"])
    test = ad.read_h5ad(input_paths[prefix + "/test.h5ad"])
    assert not set(train.obs["cell_name"].astype(str)) & set(test.obs["cell_name"].astype(str)), split
    loaded_splits[split] = (train, test)
tf_train, tf_test = loaded_splits[SPLITS[0]]
modules = pd.read_csv(input_paths[relative_inputs[-3]], sep="\\t")
tf_pairs = pd.read_csv(input_paths[relative_inputs[-2]], sep="\\t")
scrna = ad.read_h5ad(input_paths[relative_inputs[-1]])
'''
    nb.cells[8].source='''training_runs, panel_records = {}, []
for split, (train, test) in loaded_splits.items():
    result = run_smith(
        adata_file=input_paths[f"regulatory_activity/elegans/splits/elegans_tf/{split}/train.h5ad"],
        output_dir=CASE_OUTPUT / "runs" / split, tasks=CONFIG["tasks"],
        task_name=f"elegans_tf-{split}-panel128", panel_size=128,
        epochs=CONFIG["epoch"], device=DEVICE, seed=CONFIG["seed"], batch_size=CONFIG["batch_size"],
        time_label=CONFIG["time_label"], max_cells=CONFIG["max_cells"],
        sampling_strategy=CONFIG["sampling_strategy"], balance_mode=CONFIG["balance_mode"], balance_cap=CONFIG["balance_cap"],
        model_options={key: CONFIG[key] for key in MODEL_KEYS}, validation=True,
        save_model=True, tutorial_profile=TUTORIAL_PROFILE, force=not REUSE_EXISTING, include_in_memory=True,
    )
    training_runs[split] = result
    for size in PROFILE_PARAMETERS["benchmark_sizes"]["elegans_tf"]:
        genes = ranked_genes(result["ranking_frame"], size)
        panel_path = CASE_OUTPUT / "panels" / split / f"panel_{size}.tsv"
        write_panel_genes(genes, panel_path)
        panel_records.append({"dataset": "elegans_tf", "split": split, "training_seed": TRAINING_SEED,
                              "method": "SMITH", "panel_size": size, "panel_genes": genes, "panel_file": str(panel_path)})
'''
    nb.cells[10].source='''benchmark_rows, predictions = [], {}
for record in panel_records:
    split, size = record["split"], record["panel_size"]
    train, test = loaded_splits[split]
    metrics, cell_prediction, time_prediction = evaluate_loaded(
        train, test, record["panel_genes"], size, time_column="absolute_time", neighbors=5,
        output_dir=CASE_OUTPUT / "evaluations" / split / f"panel_{size}",
    )
    benchmark_rows.append({**{key: record[key] for key in ("dataset", "split", "training_seed", "method", "panel_size")}, **metrics["metrics"]})
    predictions[(split, size)] = {"cell_type": cell_prediction, "developmental_time": time_prediction}
benchmark_values = pd.DataFrame(benchmark_rows)
tf_values = benchmark_values
benchmark_values.to_csv(FIGURE_DATA / "figure3_c_d_values.tsv", sep="\\t", index=False)
reference = pd.read_csv(ROOT / "reproducibility/reference_outputs/tf_paper_metrics.tsv", sep="\\t")
comparison = benchmark_values.merge(reference, on=["dataset", "split", "panel_size"], validate="one_to_one")
comparison["accuracy_difference"] = comparison["cell_type_accuracy"] - comparison["celltype_knn_accuracy"]
comparison["time_difference"] = comparison["developmental_time_pearson"] - comparison["time_knn_pearson"]
comparison.to_csv(FIGURE_DATA / "figure3_training_comparison.tsv", sep="\\t", index=False)
display(comparison[["split", "panel_size", "cell_type_accuracy", "developmental_time_pearson", "accuracy_difference", "time_difference"]])
'''
    nb.cells[15].source='''coactivity_parts = []
for split, (train, test) in loaded_splits.items():
    run = training_runs[split]
    part = coactivity_from_objects(train, test, ranked_genes(run["ranking_frame"], 32), pairs=tf_pairs,
                                  checkpoint_file=run["checkpoint_file"], method="SMITH", seed=TRAINING_SEED)
    part["split"] = split
    coactivity_parts.append(part)
coactivity = pd.concat(coactivity_parts, ignore_index=True)
coactivity.to_csv(FIGURE_DATA / "figure3_i_coactivity.tsv", sep="\\t", index=False)
'''
    transfer=nb.cells[19].source
    a=transfer.index('counts = scrna_for_training.layers')
    b=transfer.index('transfer_rows = []')
    transfer=transfer[:a]+'''# Match the transfer run's lineage subset and normalization.
import scanpy as sc
shared_names = set(tf_train.obs["cell_name"].astype(str).str.strip()) | set(tf_test.obs["cell_name"].astype(str).str.strip())
scrna_for_training = scrna_for_training[scrna_for_training.obs["cell_name"].astype(str).str.strip().isin(shared_names)].copy()
if "counts" in scrna_for_training.layers:
    scrna_for_training.X = scrna_for_training.layers["counts"].copy()
sc.pp.normalize_total(scrna_for_training, target_sum=1e4)
sc.pp.log1p(scrna_for_training)
X = scrna_for_training.X.toarray() if sparse.issparse(scrna_for_training.X) else np.asarray(scrna_for_training.X, dtype=np.float32)
minimum = X.min(axis=0); span = X.max(axis=0) - minimum; span[span == 0] = 1.0
scrna_for_training.X = ((X - minimum) / span).astype(np.float32)
transfer_config = json.loads((ROOT / "reproducibility/configs/tf_transfer_paper.json").read_text())
transfer_run = run_smith(
    adata_file=None, adata=scrna_for_training, output_dir=CASE_OUTPUT / "runs/transfer_scRNA",
    tasks="recon,cls,time", task_name=transfer_config["task_name"], panel_size=128,
    epochs=transfer_config["epoch"], device=DEVICE, seed=transfer_config["seed"], batch_size=transfer_config["batch_size"],
    time_label="absolute_time", max_cells=None, sampling_strategy="random",
    balance_mode=transfer_config["balance_mode"], balance_cap=transfer_config["balance_cap"],
    model_options={key: transfer_config[key] for key in MODEL_KEYS},
    tutorial_profile=TUTORIAL_PROFILE, validation=True, force=not REUSE_EXISTING, include_in_memory=True,
)
transfer_rows = []
for split, (train, test) in loaded_splits.items():
    for size in PROFILE_PARAMETERS["transfer_panel_sizes"]:
        genes = ranked_genes(transfer_run["ranking_frame"], size)
        metrics, _, _ = evaluate_loaded(train, test, genes, size, time_column="absolute_time", neighbors=5,
                                        output_dir=CASE_OUTPUT / "evaluations/transfer" / split / f"panel_{size}")
        transfer_rows.append({"split": split, "method": "SMITH", "source_modality": "RNA-TF", "panel_size": size, **metrics["metrics"]})
transfer_rows.extend({**row, "source_modality": "TF-TF"} for row in benchmark_rows)
transfer = pd.DataFrame(transfer_rows)
transfer.to_csv(FIGURE_DATA / "figure3_k_transfer.tsv", sep="\\t", index=False)
'''
    nb.cells[19].source=transfer
    nb.cells[21].source=nb.cells[21].source.replace('    ("e", mirna_values, PANEL_SPECS["e"]),\n','').replace('    ("f", mirna_values, PANEL_SPECS["f"]),\n','')
    manifest=nb.cells[23].source.replace('for dataset_runs in training_runs.values() for run in dataset_runs.values()', 'for run in training_runs.values()').replace('figure3_c_f_values.tsv','figure3_c_d_values.tsv')
    nb.cells[23].source=manifest
    nb.cells[-1].source='## Training configuration\n\n`reproducibility/configs/tf_paper.json` records the TF training parameters, and `tf_transfer_paper.json` records the scRNA-seq transfer parameters. Change `SMITH_TUTORIAL_DEVICE` to choose the training device. Each run writes its configuration and input checksums to `run_manifest.json`.'
    return remove_coactivity_section(nb)


def remove_coactivity_section(nb):
    """Omit the separate reconstruction experiment from the TF tutorial."""
    retained = []
    for cell in nb.cells:
        source = cell.source
        if cell.cell_type == 'markdown' and '### TF co-activity reconstruction' in source:
            continue
        if cell.cell_type == 'code' and source.startswith('coactivity_parts ='):
            continue
        if '_draw_coactivity(axis, coactivity)' in source:
            start = source.index('figure, axis = plt.subplots', source.index('plt.close(figure)', source.index('_draw_module_schematic')))
            end = source.index('plt.close(figure)', start) + len('plt.close(figure)')
            source = source[:start] + source[end:]
            if cell.get('outputs'):
                images = [o for o in cell.outputs if 'image/png' in o.get('data', {})]
                if len(images) != 6:
                    raise ValueError('Expected six TF plots before removing reconstruction')
                cell.outputs = images[:3] + images[4:]
        source = source.replace(' and co-activity', '').replace('timing in', 'timing in')
        source = source.replace('cell identity, developmental timing in', 'cell identity and developmental timing in')
        source = source.replace('co-activity reconstruction and ', '')
        source = source.replace('spatiotemporal module annotations, TF pairs and', 'spatiotemporal module annotations and')
        source = source.replace('coactivity_from_objects, ', '').replace(', _draw_coactivity', '')
        source = source.replace(', "regulatory_activity/elegans/annotations/tf_regulatory_pairs.tsv"', '')
        source = '\n'.join(line for line in source.split('\n')
                           if not line.startswith('tf_pairs = ') and '"coactivity": str(' not in line)
        source = source.replace('input_paths[relative_inputs[-3]]', 'input_paths[relative_inputs[-2]]')
        cell.source = source
        retained.append(cell)
    nb.cells = retained
    return nb
