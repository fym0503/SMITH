"""Apply manuscript training/evaluation settings to tutorial notebook templates."""
from pathlib import Path
import nbformat

RIBO_RUN = '''import json
from reproducibility.workflows.common import resolve_tutorial_profile
PROFILE = resolve_tutorial_profile(TUTORIAL_PROFILE)
PROFILE_PARAMETERS = PROFILE["parameters"]
configs = json.loads((ROOT / PROFILE["parameter_set"]).read_text())
assert len(configs) == PROFILE_PARAMETERS["training_configurations"]
source_runs = {}
for config_id, config in configs.items():
    source_adata = deep_shared if config_id.startswith("deep_") else star_shared
    options = {key: config[key] for key in (
        "learning_rate", "dim", "rep_dim", "rep_hidden_dims", "head_hidden_dims",
        "dropout_rate", "lam", "sigma", "activation", "optimizer",
    )}
    source_runs[config_id] = run_smith(
        adata_file=None, adata=source_adata,
        output_dir=CASE_OUTPUT / "runs" / config_id,
        tasks=config["tasks"], task_name=config_id, panel_size=128,
        epochs=config["epochs"], device=DEVICE, seed=config["seed"],
        batch_size=config["batch_size"], max_cells=config["max_cells"],
        sampling_strategy=config["sampling_strategy"], balance_mode=config["balance_mode"],
        balance_cap=config["balance_cap"], model_options=options,
        validation=True, tutorial_profile=TUTORIAL_PROFILE, force=True, include_in_memory=True,
    )

# Freeze the paper's endpoint/configuration mapping before evaluation.
panel_records, metric_rows = [], []
for source in ("Deep-RIBOmap", "STARmap"):
    for label in ("celltype", "region"):
        for size in PROFILE_PARAMETERS["panel_sizes"]:
            config_id = ("deep_celltype" if size == 128 else "deep_region") if source == "Deep-RIBOmap" else "star_" + label
            trained = source_runs[config_id]
            genes = ranked_genes(trained["ranking_frame"], size)
            panel_path = CASE_OUTPUT / "panels" / source / label / f"panel_{size}.tsv"
            write_panel_genes(genes, panel_path)
            record = {"source": source, "method": "SMITH", "label": label,
                      "config_id": config_id, "training_seed": configs[config_id]["seed"],
                      "panel_size": size, "panel_genes": genes, "panel_file": str(panel_path)}
            panel_records.append(record)
            for seed in PROFILE_PARAMETERS["evaluation_seeds"]:
                result, prediction = evaluate_panel_loaded(
                    ribomap_target, genes, size, seed, label_column=label, protocol="paper",
                    output_dir=CASE_OUTPUT / "evaluations" / source / label / f"panel_{size}" / f"seed_{seed}",
                )
                metric_rows.append({**{key: record[key] for key in ("source", "method", "label", "config_id", "training_seed", "panel_size")},
                                    "evaluation_seed": seed, **result["metrics"]})
metrics = pd.DataFrame(metric_rows)
metrics.to_csv(FIGURE_DATA / "figure4_c_f_values.tsv", sep="\\t", index=False)
reference = pd.read_csv(ROOT / "reproducibility/reference_outputs/ribomap_paper_accuracy.tsv", sep="\\t")
current = metrics.groupby(["source", "label", "panel_size"], as_index=False)["accuracy"].mean()
comparison = current.merge(reference, on=["source", "label", "panel_size"], validate="one_to_one")
comparison["difference"] = comparison["accuracy"] - comparison["paper_accuracy"]
comparison.to_csv(FIGURE_DATA / "figure4_training_comparison.tsv", sep="\\t", index=False)
display(comparison)
'''
AGENT_LOAD = '''from scripts.agent_examples.formal_liver_merfish_multi_visium_smith import prepare_smith_input
from reproducibility.workflows.common import resolve_tutorial_profile
PROFILE = resolve_tutorial_profile(TUTORIAL_PROFILE)
PROFILE_PARAMETERS = PROFILE["parameters"]
reference_names = ["PSC011_C1_visium", "C73_B1_healthy_visium", "H35_sample1_visium",
                   "WSSS_F_IMMsp9838712_visium", "H35_sample2_visium"]
relative_inputs = ["agent/liver_merfish/adata_healthy_nucseq.h5ad", "agent/liver_merfish/adata_healthy_merfish.h5ad"] + [f"agent/references/{name}.h5ad" for name in reference_names]
paths = {name: DATA_ROOT / name for name in relative_inputs}
for path in paths.values():
    if not path.is_file():
        raise FileNotFoundError(path)
merfish = ad.read_h5ad(paths[relative_inputs[1]])
gene_universe = list(dict.fromkeys(str(gene).upper() for gene in merfish.var_names if str(gene).strip()))
source_preparation = prepare_smith_input(
    paths[relative_inputs[0]], CASE_OUTPUT / "prepared/source.h5ad", gene_universe,
        label_column=PROFILE_PARAMETERS["source_label"], max_cells=PROFILE_PARAMETERS["source_max_cells"],
        min_label_count=PROFILE_PARAMETERS["minimum_label_count"], seed=1,
)
source = ad.read_h5ad(source_preparation["output_h5ad"])
references = []
for name in reference_names:
    prepared = prepare_smith_input(
        paths[f"agent/references/{name}.h5ad"], CASE_OUTPUT / "prepared" / f"{name}.h5ad", gene_universe,
        label_column=PROFILE_PARAMETERS["reference_label"], require_spatial=True,
        max_cells=PROFILE_PARAMETERS["reference_max_cells"],
        min_label_count=PROFILE_PARAMETERS["minimum_label_count"], seed=1,
    )
    references.append((name, Path(prepared["output_h5ad"])))
# The whole MERFISH object is reserved for evaluation; its labels never tune panel weights.
merfish_evaluation = merfish
'''
AGENT_TRAIN = '''from concurrent.futures import ThreadPoolExecutor
training_seeds = PROFILE_PARAMETERS["training_seeds"]
epochs = PROFILE_PARAMETERS["epochs"]
batch_size = PROFILE_PARAMETERS["batch_size"]

def train_seed(seed):
    source_run = run_smith(
        adata_file=None, adata=source, output_dir=CASE_OUTPUT / "runs" / f"seed_{seed}" / "source_smith",
        tasks="recon,cls", task_name="source_scrna_smith", panel_size=PROFILE_PARAMETERS["candidate_rank_size"],
        epochs=epochs, device=DEVICE, seed=seed, batch_size=batch_size,
        sampling_strategy="celltype", layer="raw", validation=False, record_interval=50,
        tutorial_profile=TUTORIAL_PROFILE, force=True, include_in_memory=True,
    )
    reference_runs = []
    for name, reference in references:
        reference_runs.append(run_smith(
            adata_file=reference,
            output_dir=CASE_OUTPUT / "runs" / f"seed_{seed}" / "references" / name,
            tasks="recon,cls,standard_coordination", task_name=name, panel_size=PROFILE_PARAMETERS["candidate_rank_size"],
            epochs=epochs, device=DEVICE, seed=seed, batch_size=batch_size,
            sampling_strategy="celltype_spatial", layer="raw", validation=False, record_interval=50,
            tutorial_profile=TUTORIAL_PROFILE, force=True, include_in_memory=True,
        ))
    aggregation = aggregate_reference_panel_ranks_loaded(
        source_run["ranking_frame"], [run["ranking_frame"] for run in reference_runs],
        panel_size=PROFILE_PARAMETERS["candidate_rank_size"], source_weight=PROFILE_PARAMETERS["source_weight"],
        reference_weight=PROFILE_PARAMETERS["reference_weight"],
        min_reference_support=PROFILE_PARAMETERS["min_reference_support"],
        restrict_gene_symbols=gene_universe, gene_universe="source",
    )
    return seed, aggregation, {"seed": seed, "source": source_run, "references": reference_runs}

# Independent seeds can run concurrently when GPU memory permits.
workers = int(os.environ.get("SMITH_TUTORIAL_WORKERS", "1"))
with ThreadPoolExecutor(max_workers=workers) as pool:
    seed_results = list(pool.map(train_seed, training_seeds))
aggregations = {seed: aggregation for seed, aggregation, _ in seed_results}
training_runs = [run for _, _, run in seed_results]
'''
AGENT_EVAL = '''merfish_expression = mean_expression_loaded(merfish_evaluation)
rows = []
for seed, aggregation in aggregations.items():
    for size in PROFILE_PARAMETERS["panel_sizes"]:
        for panel_name, genes in (("snRNA-seq", aggregation["source_panel_genes"][:size]),
                                  ("snRNA-seq + 5 ST", aggregation["integrated_panel_genes"][:size])):
            write_panel_genes(genes, CASE_OUTPUT / "panels" / f"seed_{seed}" / f"{panel_name}_{size}.tsv")
            classification, _ = cell_type_evaluation_loaded(
                merfish_evaluation, genes, panel_size=size, label_column=PROFILE_PARAMETERS["merfish_label"],
                seed=PROFILE_PARAMETERS["evaluation_seed"],
                output_dir=CASE_OUTPUT / "evaluations" / f"seed_{seed}" / f"{panel_name}_{size}",
            )
            rows.append({"training_seed": seed, "panel": panel_name, "panel_size": size,
                         "cell_type_accuracy": classification["metrics"]["cell_type_accuracy"],
                         "mean_merfish_expression": float(np.mean([merfish_expression.get(gene, 0.0) for gene in genes])),
                         "panel_genes": genes})
results = pd.DataFrame(rows)
results.to_csv(FIGURE_DATA / "figure6_c_d_values.tsv", sep="\\t", index=False)
reference = pd.read_csv(ROOT / "reproducibility/reference_outputs/agent_multi_reference_metrics.tsv", sep="\\t")
reference = reference[reference["metric"] == "cell_type_accuracy"].copy()
reference["panel"] = reference["panel"].map({"source_smith": "snRNA-seq", "multi_visium_smith": "snRNA-seq + 5 ST"})
comparison = results.merge(reference[["training_seed", "panel_size", "panel", "value"]], on=["training_seed", "panel_size", "panel"], validate="one_to_one")
comparison["difference"] = comparison["cell_type_accuracy"] - comparison["value"]
comparison.to_csv(FIGURE_DATA / "figure6_training_comparison.tsv", sep="\\t", index=False)
display(comparison[["training_seed", "panel_size", "panel", "cell_type_accuracy", "value", "difference"]])
'''

def apply_paper_parameters(nb, case):
    if case == "tf":
        return nb
    selector = 'TUTORIAL_PROFILE = os.environ.get("SMITH_TUTORIAL_PROFILE", "ribomap_profile")'
    if case == 'ribomap':
        nb.cells[3].source += "\n" + selector
        nb.cells[0].source = '# Cross-modality brain panel transfer to RIBOMap\n\nTrain SMITH using the four recorded manuscript configurations, then evaluate cell-type and region prediction using the original five-split 5-NN protocol. Each endpoint uses its preselected manuscript configuration. The comparison table reports new-training results beside the archived paper means.'
        nb.cells[3].source = nb.cells[3].source.replace('EPOCHS = ', '# Per-run epochs are loaded from ribomap_paper.json.\nEPOCHS = ')
        nb.cells[4].source += '\n# Preserve source feature metadata and ordering, as in prepare_shared_gene_subset.py.\nfor original, name in ((deep_raw, "deep"), (star_raw, "star")):\n    original.var_names = original.var_names.astype(str).str.upper()\n    shared = [g for g in original.var_names if g in set(ribomap_target.var_names.astype(str).str.upper())]\n    if name == "deep": deep_shared = original[:, shared].copy()\n    else: star_shared = original[:, shared].copy()\n'
        nb.cells[6].source = RIBO_RUN
        nb.cells[7].source='### Deep-RIBOMap cell-type prediction'
        nb.cells[9].source='### Deep-RIBOMap region prediction'
        nb.cells[11].source='### STARmap cell-type prediction'
        nb.cells[13].source='### STARmap region prediction'
        nb.cells[15].source='### Panel overlap across source modalities\n\nCompare panel overlap within and across modalities using the saved SMITH, PERSIST-class and scGIST results.'
        nb.cells[16].source=nb.cells[16].source.replace('jaccard_from_panel_records(panel_records)', 'pd.read_csv(ROOT / "reproducibility/reference_outputs/ribomap_panel_overlap.tsv", sep="\\t")')
        nb.cells[17].source='### RIBOMap expression support\n\nCompare expression support for panels selected from Deep-RIBOmap and STARmap.'
        nb.cells[18].source=nb.cells[18].source.replace('row["training_seed"] == 1', 'row["label"] == "celltype"')
        nb.cells[20].source=nb.cells[20].source.replace('"epochs": EPOCHS, "device": DEVICE, "training_seeds": [1, 2]', '"tutorial_profile": TUTORIAL_PROFILE, "configs": configs, "device": DEVICE, "evaluation_seeds": PROFILE_PARAMETERS["evaluation_seeds"], "evaluation_protocol": PROFILE["evaluation_protocol"]')
        nb.cells[21].source='## Recorded configurations\n\n`reproducibility/configs/ribomap_paper.json` records the four training configurations used above. `figure4_training_comparison.tsv` compares the newly computed endpoint scores with the recorded paper means.'
    elif case == "agent":
        selector = 'TUTORIAL_PROFILE = os.environ.get("SMITH_TUTORIAL_PROFILE", "Agent_profile")'
        nb.cells[3].source += "\n" + selector
        nb.cells[0].source='# Liver panel design with five spatial references\n\nTrain source and reference SMITH models across five training seeds. Combine reference rankings with fixed 0.5/0.5 source/reference weights and evaluate the resulting panels in MERFISH.'
        nb.cells[1].source='## Input data\n\nUse the full Agent archive, which includes all five Visium references. For an existing complete download, set `SMITH_TUTORIAL_DATA` to its absolute directory before starting Jupyter.\n\n```bash\npython scripts/download_tutorial_data.py --case 05_agent --variant full --data-root data/tutorials\n```'
        nb.cells[4].source=AGENT_LOAD
        nb.cells[5].source='## Train the source and five reference models for each seed'
        nb.cells[6].source=AGENT_TRAIN
        nb.cells[7].source='## Fixed reference aggregation\n\nUse the manuscript source/reference weights (0.5/0.5) and require support from at least two references. These settings are fixed before evaluating MERFISH.'
        nb.cells[8].source='display(pd.DataFrame({"training_seed": list(aggregations), "source_weight": PROFILE_PARAMETERS["source_weight"], "reference_weight": PROFILE_PARAMETERS["reference_weight"], "n_references": len(references)}))'
        nb.cells[10].source=AGENT_EVAL
        def replace_cell(marker, source):
            index = next(i for i, cell in enumerate(nb.cells) if marker in cell.source)
            nb.cells[index].source = source

        replace_cell("Figure 6f", '### Figure 6f: Probe-screen pass rates\n\nShow the share of candidate genes that pass each probe-design filter.')
        replace_cell("agent_full_tool_pass_summary.tsv", '''probe_reference_root = ROOT / "reproducibility/reference_outputs"
gate_summary = pd.read_csv(probe_reference_root / "agent_full_tool_pass_summary.tsv", sep="\\t")
gate_summary.to_csv(FIGURE_DATA / "figure6_f_pass_rates_reference.tsv", sep="\\t", index=False)
gate_names = {"ODT_property_ge20": "ODT", "OligoMiner_geneaware_specific_ge10": "OligoMiner",
              "ProbeDealer_target_final_ge20": "ProbeDealer", "three_tool_feasibility": "Integrated"}
pass_rates = gate_summary[gate_summary["gate"].isin(gate_names)].copy()
pass_rates["tool"] = pass_rates["gate"].map(gate_names)
pass_rates["pass_percent"] = 100 * pass_rates["pass_count"] / pass_rates["total_count"]
pass_rates = pass_rates.set_index("tool").reindex(list(gate_names.values())).reset_index()
figure, axis = plt.subplots(figsize=(3.05, 3.0), facecolor="white")
axis.bar(pass_rates["tool"], pass_rates["pass_percent"], color=["#c8c9cb", "#c8c9cb", "#c8c9cb", "#1f77b4"], width=0.72)
for index, value in enumerate(pass_rates["pass_percent"]):
    axis.text(index, value + 1.5, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
axis.set_ylabel("Genes passing (%)")
axis.set_ylim(0, 108)
axis.set_yticks([0, 25, 50, 75, 100])
axis.tick_params(axis="x", labelrotation=28)
figure.text(0.015, 0.985, "f", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.21, right=0.98, bottom=0.27, top=0.95)
display(figure)
plt.close(figure)''')
        replace_cell("Figure 6g", '### Figure 6g: ProbeDealer off-target examples\n\nSeparate target-compatible probes, known off-targets and unresolved hits for representative genes.')
        replace_cell("agent_probedealer_offtarget_examples.tsv", '''probe_reference_root = ROOT / "reproducibility/reference_outputs"
examples = pd.read_csv(probe_reference_root / "agent_probedealer_offtarget_examples.tsv", sep="\\t").sort_values("example_order")
examples.to_csv(FIGURE_DATA / "figure6_g_offtarget_examples.tsv", sep="\\t", index=False)
figure, axis = plt.subplots(figsize=(5.2, 3.55), facecolor="white")
y = np.arange(len(examples))
target = examples["target_compatible"].to_numpy(float)
offtarget = examples["known_offtarget"].to_numpy(float)
unknown = examples["unknown"].to_numpy(float)
axis.barh(y, target, color="#1f77b4", height=0.82, label="Target")
axis.barh(y, offtarget, left=target, color="#a9c6e2", height=0.82, label="Known off-target")
axis.barh(y, unknown, left=target + offtarget, color="#929292", height=0.82, label="Unknown")
for yi, left, width, symbol in zip(y, target, offtarget, examples["offtarget_symbol"], strict=True):
    if symbol and width > 0:
        axis.text(left + width / 2, yi, symbol, ha="center", va="center", fontsize=7.5, fontstyle="italic")
axis.set_yticks(y)
axis.set_yticklabels(examples["gene_symbol"], fontstyle="italic")
axis.invert_yaxis()
axis.set_xlabel("Candidate probes")
axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=3, fontsize=8)
figure.text(0.015, 0.985, "g", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.17, right=0.99, top=0.98, bottom=0.27)
display(figure)
plt.close(figure)''')
        replace_cell('write_json(CASE_OUTPUT / "run_manifest.json"', '''write_json(CASE_OUTPUT / "run_manifest.json", {
    "workflow": "05_agent", "inputs": relative_inputs,
    "configuration": {"tutorial_profile": TUTORIAL_PROFILE, **PROFILE_PARAMETERS, "device": DEVICE, "validation": False},
    "outputs": {"metrics": str(FIGURE_DATA / "figure6_c_d_values.tsv"), "comparison": str(FIGURE_DATA / "figure6_training_comparison.tsv")},
    "probe_backend": {"status": "reference_output"},
})''')
    for cell in nb.cells:
        if cell.cell_type=='code':
            cell.outputs=[];cell.execution_count=None
    return nb
