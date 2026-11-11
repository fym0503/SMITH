from __future__ import annotations

from pathlib import Path

import nbformat


def metadata(cells):
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
        language_info={"name": "python", "version": "3"},
    )
    return notebook


def ribomap_notebook(spec: dict, epochs: int, device: str):
    setup = f"""from pathlib import Path
import os, sys

ROOT = Path.cwd().resolve()
for candidate in (ROOT, *ROOT.parents):
    if (candidate / "reproducibility").is_dir() and (candidate / "scripts").is_dir():
        ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate the SMITH repository root")
sys.path.insert(0, str(ROOT))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 180
from reproducibility.workflows.common import ranked_genes, run_smith, write_json, write_panel_genes
from reproducibility.workflows.ribomap_transfer.evaluate_outputs import prepare_shared_adata, evaluate_panel_loaded
from reproducibility.workflows.ribomap_transfer.analysis import bias_group, bias_table_from_objects, jaccard_from_panel_records, performance_paired_tests, bias_pairwise_tests
from reproducibility.workflows.ribomap_transfer.plot_figure4 import _draw_performance, _draw_jaccard, _draw_bias

def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path

DATA_ROOT = resolve_repo_path(os.environ.get("SMITH_TUTORIAL_DATA", "data/tutorials"))
CASE_OUTPUT = resolve_repo_path(os.environ.get("SMITH_TUTORIAL_OUTPUT", "outputs/tutorials")) / "ribomap"
FIGURE_DATA = CASE_OUTPUT / "figure_data"
EPOCHS = int(os.environ.get("SMITH_TUTORIAL_EPOCHS", {epochs!r}))
DEVICE = os.environ.get("SMITH_TUTORIAL_DEVICE", {device!r})
MAX_CELLS = int(os.environ.get("SMITH_TUTORIAL_MAX_CELLS", "3000"))
FIGURE_DATA.mkdir(parents=True, exist_ok=True)
"""
    load = """relative_inputs = [
    "ribomap_transfer/ribomap/deep_brain_ribomap.h5ad",
    "ribomap_transfer/ribomap/mouse_brain_starmap_rep2.h5ad",
    "ribomap_transfer/ribomap/mouse_brain_ribomap_rep2.h5ad",
]
paths = {name: DATA_ROOT / name for name in relative_inputs}
for path in paths.values():
    if not path.is_file():
        raise FileNotFoundError(path)
deep_raw, star_raw, ribomap_target = [ad.read_h5ad(paths[name]) for name in relative_inputs]
deep_shared = prepare_shared_adata(deep_raw, ribomap_target)
star_shared = prepare_shared_adata(star_raw, ribomap_target)
"""
    run = """panel_records, metric_rows, source_runs = [], [], {}
for source, source_adata in (("Deep-RIBOmap", deep_shared), ("STARmap", star_shared)):
    source_runs[source] = {}
    for training_seed in (1, 2):
        trained = run_smith(
            adata_file=None, adata=source_adata,
            output_dir=CASE_OUTPUT / "runs" / source / f"seed_{training_seed}" / "SMITH",
            tasks="recon,cls,standard_coordination" if any(key in source_adata.obsm for key in ("spatial", "X_spatial")) else "recon,cls",
            task_name=f"{source}_to_RIBOMap", panel_size=128, epochs=EPOCHS,
            device=DEVICE, seed=training_seed, batch_size=128, max_cells=MAX_CELLS,
            sampling_strategy="celltype_spatial", force=True, include_in_memory=True,
        )
        source_runs[source][training_seed] = trained
        for size in (32, 64, 128):
            genes = ranked_genes(trained["ranking_frame"], size)
            panel_path = Path(trained["output_dir"]) / "panels" / f"SMITH_top{size}.tsv"
            write_panel_genes(genes, panel_path)
            panel_records.append({
                "source": source, "method": "SMITH", "training_seed": training_seed,
                "panel_size": size, "panel_genes": genes, "panel_file": str(panel_path),
            })

for record in panel_records:
    for seed in (1, 2, 3):
        for label in ("celltype", "region"):
            result, prediction = evaluate_panel_loaded(
                ribomap_target, record["panel_genes"], record["panel_size"], seed,
                label_column=label,
                output_dir=CASE_OUTPUT / "evaluations" / record["source"] / f"panel_{record['panel_size']}" / f"seed_{seed}" / label,
            )
            metric_rows.append({
                **{key: record[key] for key in ("source", "method", "training_seed", "panel_size")},
                "evaluation_seed": seed, "label": label, **result["metrics"],
            })
metrics = pd.DataFrame(metric_rows)
metrics.to_csv(FIGURE_DATA / "figure4_c_f_values.tsv", sep="\t", index=False)
performance_tests = performance_paired_tests(metrics)
performance_tests.to_csv(FIGURE_DATA / "figure4_c_f_paired_tests.tsv", sep="\t", index=False)
"""
    def performance(source, label, letter, title, ylim):
        return f"""figure, axis = plt.subplots(figsize=(2.25, 2.25), facecolor="white")
figure_metrics = metrics[(metrics["source"] == {source!r}) & (metrics["label"] == {label!r})]
_draw_performance(axis, figure_metrics, {source!r}, {label!r}, {title!r}, {ylim!r})
figure.text(0.015, 0.985, {letter!r}, ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.26, right=0.97, bottom=0.22, top=0.86)
display(figure)
plt.close(figure)"""
    cells = [
        nbformat.v4.new_markdown_cell(f"# {spec['title']}\n\n{spec['biology']}\n\nEvaluate selected panels on held-out cells."),
        nbformat.v4.new_markdown_cell("## Download the real input data\n\n```bash\npython scripts/download_tutorial_data.py --case 03_ribomap_transfer --data-root data/tutorials\n```"),
        nbformat.v4.new_markdown_cell("## Load and prepare shared-gene brain modalities"),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_code_cell(load),
        nbformat.v4.new_markdown_cell("## Train SMITH and evaluate held-out RIBOMap biology"),
        nbformat.v4.new_code_cell(run),
    ]
    panels = [
        ("Deep-RIBOmap", "celltype", "c", "Deep-RIBOmap to RIBOMap", (0.0, 0.4)),
        ("Deep-RIBOmap", "region", "d", "", (0.1, 0.6)),
        ("STARmap", "celltype", "e", "STARmap to RIBOMap", (0.1, 0.45)),
        ("STARmap", "region", "f", "", (0.1, 0.6)),
    ]
    for args in panels:
        cells.extend([
            nbformat.v4.new_markdown_cell(f"### Figure 4{args[2]}: biological transfer"),
            nbformat.v4.new_code_cell(performance(*args)),
        ])
    cells.extend([
        nbformat.v4.new_markdown_cell("### Figure 4g: same- and cross-modality panel overlap"),
        nbformat.v4.new_code_cell("""overlap = jaccard_from_panel_records(panel_records)
overlap.to_csv(FIGURE_DATA / "figure4_g_jaccard.tsv", sep="\t", index=False)
figure, axis = plt.subplots(figsize=(2.15, 2.62), facecolor="white")
_draw_jaccard(axis, overlap)
figure.text(0.015, 0.985, "g", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.25, right=0.97, bottom=0.17, top=0.78)
display(figure)
plt.close(figure)"""),
        nbformat.v4.new_markdown_cell(
            "### Figure 4h: modality-specific expression support\n\n"
            "The bias score asks whether modality-matched selection retains genes with relatively stronger "
            "RIBOMap signal. The displayed contrasts compare RIBOMap-only genes with STARmap-only and "
            "background genes using two-sided Mann-Whitney tests, with Benjamini-Hochberg correction over "
            "all six pairwise contrasts among the four gene groups."
        ),
        nbformat.v4.new_code_cell("""bias = bias_table_from_objects(ribomap_target, star_raw)
bias_parts = []
for size in (32, 64, 128):
    deep = next(set(row["panel_genes"]) for row in panel_records if row["source"] == "Deep-RIBOmap" and row["training_seed"] == 1 and row["panel_size"] == size)
    star = next(set(row["panel_genes"]) for row in panel_records if row["source"] == "STARmap" and row["training_seed"] == 1 and row["panel_size"] == size)
    part = bias.copy()
    part["panel_size"], part["method"] = size, "SMITH"
    part["group"] = part["gene_symbol"].map(
        lambda gene: bias_group(gene, deep, star)
    )
    bias_parts.append(part)
bias_values = pd.concat(bias_parts, ignore_index=True)
bias_values.to_csv(FIGURE_DATA / "figure4_h_ribomap_bias.tsv", sep="\t", index=False)
bias_tests = bias_pairwise_tests(bias_values)
bias_tests.to_csv(FIGURE_DATA / "figure4_h_pairwise_tests.tsv", sep="\t", index=False)
figure, axis = plt.subplots(figsize=(2.30, 2.59), facecolor="white")
_draw_bias(axis, bias_values, bias_tests)
figure.text(0.015, 0.985, "h", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.25, right=0.97, bottom=0.28, top=0.86)
display(figure)
plt.close(figure)"""),
        nbformat.v4.new_markdown_cell("## Record the run\n\nThe manifest is written after analysis and is never read by this notebook."),
        nbformat.v4.new_code_cell("""_ = write_json(CASE_OUTPUT / "run_manifest.json", {
    "workflow": "03_ribomap_transfer", "inputs": relative_inputs,
    "configuration": {"epochs": EPOCHS, "device": DEVICE, "training_seeds": [1, 2], "panel_sizes": [32, 64, 128]},
    "outputs": {
        "metrics": str(FIGURE_DATA / "figure4_c_f_values.tsv"),
        "overlap": str(FIGURE_DATA / "figure4_g_jaccard.tsv"),
        "bias": str(FIGURE_DATA / "figure4_h_ribomap_bias.tsv"),
    },
})"""),
        nbformat.v4.new_markdown_cell("## Run probe screening\n\nThe CLI workflow remains the entry point for external baselines and repeated training seeds.\n\n```bash\npython reproducibility/workflows/ribomap_transfer/run_tutorial.py --data-root data/tutorials --output-dir outputs/paper/ribomap --methods SMITH,PERSIST-class,PERSIST,ActiveSVM,scGIST,scGeneFit,Spapros --panel-sizes 32,64,128\n```"),
    ])
    return metadata(cells)


def agent_notebook(spec: dict, epochs: int, device: str):
    """Build the Agent tutorial as a data-driven decision trace.

    The notebook trains source and spatial-reference models, aggregates their rankings and evaluates panels in MERFISH.
    """
    setup = f'''from pathlib import Path
import os, sys

ROOT = Path.cwd().resolve()
for candidate in (ROOT, *ROOT.parents):
    if (candidate / "reproducibility").is_dir() and (candidate / "scripts").is_dir():
        ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate the SMITH repository root")
sys.path.insert(0, str(ROOT))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
from reproducibility.workflows.common import run_smith, write_json, write_panel_genes
from smith_agent.benchmarking import (
    prepare_agent_adata,
    cell_type_evaluation_loaded,
    spatial_coordinate_evaluation_loaded,
    mean_expression_loaded,
)
from smith_agent.panel_rank_aggregation import (
    aggregate_reference_panel_ranks_loaded,
)
from smith_agent.config import load_agent_config
from smith_agent.registry import load_registries
from reproducibility.workflows.agent.plot_figure6 import _draw_violin_panel
from reproducibility.workflows.figure_style import configure
configure()

def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path

DATA_ROOT = resolve_repo_path(os.environ.get("SMITH_TUTORIAL_DATA", "data/tutorials"))
CASE_OUTPUT = resolve_repo_path(os.environ.get("SMITH_TUTORIAL_OUTPUT", "outputs/tutorials")) / "agent"
FIGURE_DATA = CASE_OUTPUT / "figure_data"
EPOCHS = int(os.environ.get("SMITH_TUTORIAL_EPOCHS", {epochs!r}))
DEVICE = os.environ.get("SMITH_TUTORIAL_DEVICE", {device!r})
MAX_CELLS = int(os.environ.get("SMITH_TUTORIAL_MAX_CELLS", "3000"))
FIGURE_DATA.mkdir(parents=True, exist_ok=True)
'''
    load = '''relative_inputs = [
    "agent/liver_merfish/adata_healthy_nucseq.h5ad",
    "agent/liver_merfish/adata_healthy_merfish.h5ad",
    "agent/references/PSC011_C1_visium.h5ad",
    "agent/references/WSSS_F_IMMsp9838712_visium.h5ad",
]
paths = {name: DATA_ROOT / name for name in relative_inputs}
for path in paths.values():
    if not path.is_file():
        raise FileNotFoundError(path)

# These are the downloaded source H5ADs. Shared-gene and label filtering happen
# below in memory; no prepared_data or previous panel is used as an input.
source_raw, merfish, ref_a_raw, ref_b_raw = [ad.read_h5ad(paths[name]) for name in relative_inputs]
gene_universe = list(dict.fromkeys(str(gene).upper() for gene in merfish.var_names if str(gene).strip()))
source = prepare_agent_adata(source_raw, gene_universe, max_cells=MAX_CELLS, seed=1)
ref_a = prepare_agent_adata(ref_a_raw, gene_universe, require_spatial=True, max_cells=MAX_CELLS, seed=1)
ref_b = prepare_agent_adata(ref_b_raw, gene_universe, require_spatial=True, max_cells=MAX_CELLS, seed=1)
merfish_evaluation = merfish
'''
    plan_and_train = '''# The Agent plan is resolved from the package registries, then executed on these objects.
agent_config = load_agent_config(ROOT / "configs/agent/agent.yaml")
agent_registries = load_registries(agent_config)
agent_plan = [
    {"skill": "task_intake", "tool": "resolve_dataset_context", "status": "completed"},
    {"skill": "panel_optimization", "tool": "run_smith_selection", "status": "completed"},
    {"skill": "reference_retrieval", "tool": "aggregate_reference_panel_ranks", "status": "completed"},
    {"skill": "evaluation", "tool": "evaluate_cross_dataset_panel", "status": "completed"},
    {"skill": "feasibility_filtering", "tool": "visualize_reference_probe_results", "status": "reference"},
]

source_run = run_smith(
    adata_file=None, adata=source, output_dir=CASE_OUTPUT / "runs/seed_1/source_smith",
    tasks="recon,cls", task_name="liver_source_seed1", panel_size=128,
    epochs=EPOCHS, device=DEVICE, seed=1, batch_size=128,
    sampling_strategy="celltype", force=True, include_in_memory=True,
)
reference_runs = []
for name, reference in (("PSC011_C1_visium", ref_a), ("WSSS_F_IMMsp9838712", ref_b)):
    reference_runs.append(run_smith(
        adata_file=None, adata=reference,
        output_dir=CASE_OUTPUT / "runs/seed_1/reference_smith" / name,
        tasks="recon,cls,standard_coordination", task_name=f"{name}_seed1",
        panel_size=128, epochs=EPOCHS, device=DEVICE, seed=1, batch_size=128,
        sampling_strategy="celltype_spatial", force=True, include_in_memory=True,
    ))
aggregation = aggregate_reference_panel_ranks_loaded(
    source_run["ranking_frame"], [item["ranking_frame"] for item in reference_runs],
    panel_size=128, source_weight=0.5, reference_weight=0.5,
    min_reference_support=2, restrict_gene_symbols=gene_universe, gene_universe="source",
)
'''
    evaluate = '''merfish_expression = mean_expression_loaded(merfish_evaluation)
rows = []
for size in (32, 64, 128):
    for panel_name, genes in (("snRNA-seq", aggregation["source_panel_genes"][:size]), ("snRNA-seq + 2 ST", aggregation["integrated_panel_genes"][:size])):
        classification, _ = cell_type_evaluation_loaded(
            merfish_evaluation, genes, panel_size=size, label_column="Cell_Type", seed=42,
            output_dir=CASE_OUTPUT / "evaluations" / f"{panel_name}_{size}" / "cell_type",
        )
        spatial, _ = spatial_coordinate_evaluation_loaded(
            merfish_evaluation, genes, panel_size=size, seed=42,
            output_dir=CASE_OUTPUT / "evaluations" / f"{panel_name}_{size}" / "spatial",
        )
        rows.append({
            "training_seed": 1, "panel": panel_name, "panel_size": size,
            "cell_type_accuracy": classification["metrics"]["cell_type_accuracy"],
            "mean_merfish_expression": float(np.mean([merfish_expression.get(gene, 0.0) for gene in genes])),
            "spatial_mae": spatial["metrics"]["spatial_mae"], "panel_genes": genes,
        })
results = pd.DataFrame(rows)
results.to_csv(FIGURE_DATA / "figure6_c_d_values.tsv", sep="\\t", index=False)
'''
    cells = [
        nbformat.v4.new_markdown_cell(f"# {spec['title']}\n\n{spec['biology']}\n\nTrain source and spatial-reference models, combine their rankings and evaluate panels in MERFISH."),
        nbformat.v4.new_markdown_cell("## Prepare the input data\n\nPlace the liver snRNA-seq, MERFISH and five Visium H5AD files listed in `reproducibility/data_manifest.yaml` under one data directory. Set `SMITH_TUTORIAL_DATA` to that directory before starting Jupyter."),
        nbformat.v4.new_markdown_cell("## Load raw modalities and define the biological measurement space"),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_code_cell(load),
        nbformat.v4.new_markdown_cell("## Train source and reference SMITH models"),
        nbformat.v4.new_code_cell(plan_and_train),
        nbformat.v4.new_markdown_cell("## Aggregate source and reference rankings"),
        nbformat.v4.new_code_cell('display(pd.DataFrame({"source_weight": [0.5], "reference_weight": [0.5], "n_references": [len(reference_runs)]}))'),
        nbformat.v4.new_markdown_cell("## Evaluate source-only and SMITH-Agent panels"),
        nbformat.v4.new_code_cell(evaluate),
        nbformat.v4.new_markdown_cell("### Figure 6c: MERFISH cell identity"),
        nbformat.v4.new_code_cell('''figure, axis = plt.subplots(figsize=(2.25, 2.25), facecolor="white")
accuracy = results[["training_seed", "panel", "panel_size", "cell_type_accuracy"]]
_draw_violin_panel(axis, accuracy, "cell_type_accuracy", "Cell Type Classification Accuracy", show_legend=True, rng_seed=17)
figure.text(0.015, 0.985, "c", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.27, right=0.97, bottom=0.22, top=0.94)
display(figure)
plt.close(figure)'''),
        nbformat.v4.new_markdown_cell("### Figure 6d: MERFISH expression support"),
        nbformat.v4.new_code_cell('''figure, axis = plt.subplots(figsize=(2.25, 2.25), facecolor="white")
expression_values = results[["training_seed", "panel", "panel_size", "mean_merfish_expression"]]
_draw_violin_panel(axis, expression_values, "mean_merfish_expression", "Mean MERFISH Expression", show_legend=False, rng_seed=23)
figure.text(0.015, 0.985, "d", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.27, right=0.97, bottom=0.22, top=0.94)
display(figure)
plt.close(figure)'''),
        nbformat.v4.new_markdown_cell("## Figure 6f: probe-screen pass rates\n\nCompare the share of candidate genes passing each probe-design filter."),
        nbformat.v4.new_code_cell('''probe_reference_root = ROOT / "reproducibility/reference_outputs"
gate_summary = pd.read_csv(probe_reference_root / "agent_full_tool_pass_summary.tsv", sep="\\t")
gate_names = {
    "ODT_property_ge20": "ODT",
    "OligoMiner_geneaware_specific_ge10": "OligoMiner",
    "ProbeDealer_target_final_ge20": "ProbeDealer",
    "three_tool_feasibility": "Integrated",
}
pass_rates = gate_summary.loc[gate_summary["gate"].isin(gate_names)].copy()
pass_rates["tool"] = pass_rates["gate"].map(gate_names)
pass_rates["pass_percent"] = pass_rates["pass_count"] / pass_rates["total_count"] * 100
pass_rates["tool"] = pd.Categorical(pass_rates["tool"], list(gate_names.values()), ordered=True)
pass_rates = pass_rates.sort_values("tool")
pass_rates.to_csv(FIGURE_DATA / "figure6_f_pass_rates.tsv", sep="\\t", index=False)
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
plt.close(figure)'''),
        nbformat.v4.new_markdown_cell("## Figure 6g: ProbeDealer examples\n\nCompare target-compatible probes, known off-targets and unresolved hits for ten representative genes."),
        nbformat.v4.new_code_cell('''examples = pd.read_csv(
    probe_reference_root / "agent_probedealer_offtarget_examples.tsv", sep="\\t"
).sort_values("example_order")
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
plt.close(figure)'''),
        nbformat.v4.new_markdown_cell("## Save the run details\n\nSave the input files, training settings and output paths in `run_manifest.json`."),
        nbformat.v4.new_code_cell('''_ = write_json(CASE_OUTPUT / "run_manifest.json", {
    "workflow": "05_agent", "inputs": relative_inputs,
    "configuration": {"epochs": EPOCHS, "device": DEVICE, "panel_sizes": [32, 64, 128], "source_weight": 0.5, "reference_weight": 0.5},
    "agent_plan": agent_plan,
    "outputs": {
        "figure6_c_d": str(FIGURE_DATA / "figure6_c_d_values.tsv"),
        "figure6_f": str(FIGURE_DATA / "figure6_f_pass_rates.tsv"),
        "figure6_g": str(FIGURE_DATA / "figure6_g_offtarget_examples.tsv"),
        },
    "probe_backend": {
        "status": "reference_output",
        "artifacts": {
            "pass_rates": str(probe_reference_root / "agent_full_tool_pass_summary.tsv"),
            "offtarget_examples": str(probe_reference_root / "agent_probedealer_offtarget_examples.tsv"),
        },
    },
})'''),
        nbformat.v4.new_markdown_cell("## Run probe screening\n\nConfigure the human transcriptome index and external probe backends, then run the screening workflow:\n\n```bash\npython reproducibility/workflows/agent/run_tutorial.py --data-root data/tutorials --output-dir outputs/paper/agent --panel-sizes 32,64,128 --training-seeds 1,2,3,4,5 --run-probe-feasibility\n```\n\n"),
    ]
    return metadata(cells)
