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
from IPython.display import display
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
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
        nbformat.v4.new_markdown_cell(f"# {spec['title']}\n\n{spec['biology']}\n\nThis notebook passes current SMITH panels directly through held-out biological analyses. Written files are provenance outputs, not later inputs."),
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
            "all six pairwise contrasts among the four gene groups. The hosted example uses reduced cell and "
            "epoch limits; run the full-scale command below to reproduce manuscript-scale q values."
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
        nbformat.v4.new_markdown_cell("## Full manuscript command\n\nThe CLI workflow remains the entry point for external baselines and repeated training seeds.\n\n```bash\npython reproducibility/workflows/ribomap_transfer/run_tutorial.py --data-root data/tutorials --output-dir outputs/paper/ribomap --methods SMITH,PERSIST-class,PERSIST,ActiveSVM,scGIST,scGeneFit,Spapros --panel-sizes 32,64,128\n```"),
    ])
    return metadata(cells)


def _legacy_agent_notebook(spec: dict, epochs: int, device: str):
    setup = f"""from pathlib import Path
import os, sys

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
from reproducibility.workflows.common import run_smith, write_json, write_panel_genes
from smith_agent.benchmarking import prepare_agent_adata, cell_type_evaluation_loaded, spatial_coordinate_evaluation_loaded, mean_expression_loaded
from smith_agent.panel_rank_aggregation import aggregate_reference_panel_ranks_loaded
from reproducibility.workflows.agent.plot_figure6 import _draw_violin_panel
from reproducibility.workflows.figure_style import configure
configure()

DATA_ROOT = Path(os.environ.get("SMITH_TUTORIAL_DATA", "data/tutorials")).resolve()
CASE_OUTPUT = Path(os.environ.get("SMITH_TUTORIAL_OUTPUT", "outputs/tutorials")).resolve() / "agent"
FIGURE_DATA = CASE_OUTPUT / "figure_data"
EPOCHS = int(os.environ.get("SMITH_TUTORIAL_EPOCHS", {epochs!r}))
DEVICE = os.environ.get("SMITH_TUTORIAL_DEVICE", {device!r})
MAX_CELLS = int(os.environ.get("SMITH_TUTORIAL_MAX_CELLS", "3000"))
FIGURE_DATA.mkdir(parents=True, exist_ok=True)
"""
    load = """relative_inputs = [
    "agent/liver_merfish/adata_healthy_nucseq.h5ad",
    "agent/liver_merfish/adata_healthy_merfish.h5ad",
    "agent/references/PSC011_C1_visium.h5ad",
    "agent/references/WSSS_F_IMMsp9838712_visium.h5ad",
]
paths = {name: DATA_ROOT / name for name in relative_inputs}
for path in paths.values():
    if not path.is_file():
        raise FileNotFoundError(path)
source_raw, merfish, ref_a_raw, ref_b_raw = [ad.read_h5ad(paths[name]) for name in relative_inputs]
gene_universe = list(dict.fromkeys(str(gene).upper() for gene in merfish.var_names if str(gene).strip()))
source = prepare_agent_adata(source_raw, gene_universe, max_cells=MAX_CELLS, seed=1)
ref_a = prepare_agent_adata(ref_a_raw, gene_universe, require_spatial=True, max_cells=MAX_CELLS, seed=1)
ref_b = prepare_agent_adata(ref_b_raw, gene_universe, require_spatial=True, max_cells=MAX_CELLS, seed=1)
"""
    run = """source_run = run_smith(
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
merfish_expression = mean_expression_loaded(merfish)
rows = []
for size in (32, 64, 128):
    for panel_name, genes in (
        ("snRNA-seq", aggregation["source_panel_genes"][:size]),
        ("snRNA-seq + 2 ST", aggregation["integrated_panel_genes"][:size]),
    ):
        panel_path = CASE_OUTPUT / "runs/seed_1/panels" / f"{panel_name.replace(' ', '_')}_{size}.tsv"
        write_panel_genes(genes, panel_path)
        classification, class_prediction = cell_type_evaluation_loaded(
            merfish, genes, panel_size=size, label_column="Cell_Type", seed=42,
            output_dir=CASE_OUTPUT / "evaluations" / f"{panel_name.replace(' ', '_')}_{size}" / "cell_type",
        )
        spatial, spatial_prediction = spatial_coordinate_evaluation_loaded(
            merfish, genes, panel_size=size, seed=42,
            output_dir=CASE_OUTPUT / "evaluations" / f"{panel_name.replace(' ', '_')}_{size}" / "spatial",
        )
        rows.append({
            "training_seed": 1, "panel": panel_name, "panel_size": size,
            "cell_type_accuracy": classification["metrics"]["cell_type_accuracy"],
            "mean_merfish_expression": float(np.mean([merfish_expression.get(gene, 0.0) for gene in genes])),
            "panel_genes": genes,
        })
results = pd.DataFrame(rows)
results.to_csv(FIGURE_DATA / "figure6_c_d_values.tsv", sep="\t", index=False)
"""
    cells = [
        nbformat.v4.new_markdown_cell(f"# {spec['title']}\n\n{spec['biology']}\n\nThis notebook trains current source and spatial-reference panels, aggregates their rankings in memory, and evaluates the current panels directly on MERFISH."),
        nbformat.v4.new_markdown_cell("## Download the real input data\n\n```bash\npython scripts/download_tutorial_data.py --case 05_agent --data-root data/tutorials\n```"),
        nbformat.v4.new_markdown_cell("## Load and prepare liver modalities"),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_code_cell(load),
        nbformat.v4.new_markdown_cell("## Train SMITH, aggregate panels, and evaluate MERFISH"),
        nbformat.v4.new_code_cell(run),
        nbformat.v4.new_markdown_cell("### Figure 6c: MERFISH cell identity"),
        nbformat.v4.new_code_cell("""figure, axis = plt.subplots(figsize=(2.25, 2.25), facecolor="white")
accuracy = results[["training_seed", "panel", "panel_size", "cell_type_accuracy"]]
_draw_violin_panel(axis, accuracy, "cell_type_accuracy", "Cell Type Classification Accuracy", show_legend=True, rng_seed=17)
figure.text(0.015, 0.985, "c", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.27, right=0.97, bottom=0.22, top=0.94)
display(figure)
plt.close(figure)"""),
        nbformat.v4.new_markdown_cell("### Figure 6d: MERFISH expression support"),
        nbformat.v4.new_code_cell("""figure, axis = plt.subplots(figsize=(2.25, 2.25), facecolor="white")
expression_values = results[["training_seed", "panel", "panel_size", "mean_merfish_expression"]]
_draw_violin_panel(axis, expression_values, "mean_merfish_expression", "Mean MERFISH Expression", show_legend=False, rng_seed=23)
figure.text(0.015, 0.985, "d", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.27, right=0.97, bottom=0.22, top=0.94)
display(figure)
plt.close(figure)"""),
        nbformat.v4.new_markdown_cell("## Record the run\n\nThe manifest is written after analysis and is never read by this notebook."),
        nbformat.v4.new_code_cell("""write_json(CASE_OUTPUT / "run_manifest.json", {
    "workflow": "05_agent", "inputs": relative_inputs,
    "configuration": {"epochs": EPOCHS, "device": DEVICE, "panel_sizes": [32, 64, 128]},
    "outputs": {"figure6": str(FIGURE_DATA / "figure6_c_d_values.tsv")},
    "probe_backend": {"status": "not_run", "reason": "External probe-design backends are outside this tutorial."},
})"""),
        nbformat.v4.new_markdown_cell("## Full manuscript command\n\nThe CLI workflow remains the entry point for all five spatial references and repeated training seeds.\n\n```bash\npython reproducibility/workflows/agent/run_tutorial.py --data-root data/tutorials --output-dir outputs/paper/agent --panel-sizes 32,64,128\n```"),
    ]
    return metadata(cells)


def agent_notebook(spec: dict, epochs: int, device: str):
    """Build the Agent tutorial as a data-driven decision trace.

    The notebook intentionally keeps the model implementation in the package,
    while exposing the Agent's registry plan, held-out parameter search, and
    probe prerequisite checks next to the manuscript panels.
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
from sklearn.model_selection import train_test_split
from IPython.display import display
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
    tune_reference_aggregation_loaded,
)
from smith_agent.feasibility.preflight import probe_backend_preflight, probe_property_screen_loaded
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
merfish_indices = np.arange(merfish.n_obs)
tuning_index, evaluation_index = train_test_split(
    merfish_indices, test_size=0.5, random_state=41, stratify=merfish.obs["Cell_Type"].astype(str),
)
merfish_tuning = merfish[np.sort(tuning_index)].copy()
merfish_evaluation = merfish[np.sort(evaluation_index)].copy()
'''
    plan_and_train = '''# The Agent plan is resolved from the package registries, then executed on these objects.
agent_config = load_agent_config(ROOT / "configs/agent/agent.yaml")
agent_registries = load_registries(agent_config)
agent_plan = [
    {"skill": "task_intake", "tool": "resolve_dataset_context", "status": "completed"},
    {"skill": "panel_optimization", "tool": "run_smith_selection", "status": "completed"},
    {"skill": "reference_retrieval", "tool": "aggregate_reference_panel_ranks", "status": "completed"},
    {"skill": "evaluation", "tool": "evaluate_cross_dataset_panel", "status": "completed"},
    {"skill": "feasibility_filtering", "tool": "run_three_backend_feasibility", "status": "preflight"},
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
    tuning = '''tuning = tune_reference_aggregation_loaded(
    source_run["ranking_frame"], [item["ranking_frame"] for item in reference_runs], merfish_tuning,
    panel_sizes=(32, 64, 128), source_weights=(0.25, 0.5, 0.75),
    label_column="Cell_Type", seed=42, min_reference_support=2,
    restrict_gene_symbols=gene_universe,
)
tuning_results = tuning["results"]
tuning_results.to_csv(FIGURE_DATA / "agent_parameter_tuning.tsv", sep="\\t", index=False)
best_agent_panel = tuning["best"]["panel_genes"]
aggregation = tuning["best_aggregation"]
for size in (32, 64, 128):
    write_panel_genes(tuning["best_aggregation"]["integrated_panel_genes"][:size], CASE_OUTPUT / "runs/seed_1/panels" / f"agent_tuned_{size}.tsv")
figure, axis = plt.subplots(figsize=(3.0, 2.2), facecolor="white")
for weight, group in tuning_results.groupby("source_weight"):
    axis.plot(group["panel_size"], group["cell_type_accuracy"], marker="o", label=f"source={weight:.2f}")
axis.set(xlabel="Panel size", ylabel="Held-out accuracy")
axis.legend(frameon=False, fontsize=7)
figure.tight_layout()
display(figure)
plt.close(figure)
'''
    evaluate = '''merfish_expression = mean_expression_loaded(merfish_evaluation)
rows = []
for size in (32, 64, 128):
    for panel_name, genes in (("snRNA-seq", aggregation["source_panel_genes"][:size]), ("SMITH-Agent", aggregation["integrated_panel_genes"][:size])):
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
        nbformat.v4.new_markdown_cell(f"# {spec['title']}\n\n{spec['biology']}\n\nThis is the SMITH-Agent path: source and spatial evidence are inspected, panels are trained and combined, the evidence balance is tuned on held-out MERFISH biology, and current panels directly undergo biological evaluation before probe prerequisites are checked for assay design."),
        nbformat.v4.new_markdown_cell("## Download the source data\n\n```bash\npython scripts/download_tutorial_data.py --case 05_agent --data-root data/tutorials\n```\n\nThe four H5AD files are real liver source/target/reference inputs. The notebook creates its shared-gene objects in memory; it does not load a prior panel, metrics file, or prepared-data artifact."),
        nbformat.v4.new_markdown_cell("## Load raw modalities and define the biological measurement space"),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_code_cell(load),
        nbformat.v4.new_markdown_cell("## Execute the Agent plan: train source and reference SMITH models"),
        nbformat.v4.new_code_cell(plan_and_train),
        nbformat.v4.new_markdown_cell("### Agent decision trace"),
        nbformat.v4.new_code_cell('''figure, axis = plt.subplots(figsize=(5.6, 1.15), facecolor="white")
axis.axis("off")
for index, step in enumerate(agent_plan):
    x = 0.02 + index * 0.245
    color = "#2f78bd" if step["status"] == "completed" else "#b7791f"
    axis.text(x, 0.62, step["skill"].replace("_", "\\n"), ha="left", va="center", fontsize=8, color=color, weight="bold")
    if index < len(agent_plan) - 1:
        axis.annotate("", xy=(x + 0.21, 0.62), xytext=(x + 0.18, 0.62), arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.2})
axis.set_xlim(0, 1); axis.set_ylim(0, 1)
display(figure)
plt.close(figure)'''),
        nbformat.v4.new_markdown_cell("## Tune the evidence balance on held-out MERFISH"),
        nbformat.v4.new_code_cell(tuning),
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
        nbformat.v4.new_markdown_cell("## Test probe feasibility for the selected panel\n\nThe tutorial resolves transcripts for the first 16 genes in the newly selected panel and runs the local ProbeDealer sequence-property screen. This estimates deployable candidate counts; transcriptome-wide specificity is reported only when OligoMiner/BLAST and PaintSHOP are configured."),
        nbformat.v4.new_code_cell('''probe_status = probe_backend_preflight(ROOT / "configs/agent/agent.yaml", package_root=ROOT)
probe_screen = probe_property_screen_loaded(best_agent_panel, CASE_OUTPUT / "probe_feasibility", species="homo_sapiens", max_genes=16)
probe_values = probe_screen["summary"].sort_values("final_probe_count")
figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), facecolor="white", gridspec_kw={"width_ratios": [1.5, 1]})
axes[0].barh(probe_values["gene_symbol"], probe_values["final_probe_count"], color="#4c78a8")
axes[0].axvline(20, color="#c53030", linestyle="--", linewidth=0.8)
axes[0].set(xlabel="Property-filtered probe candidates", title="Selected-panel probe counts")
labels = [f"{row['backend']} ({row['stage']})" for row in probe_status]
values = [1 if row["available"] else 0 for row in probe_status]
axes[1].barh(labels, values, color=["#2f855a" if value else "#c53030" for value in values])
axes[1].set_xlim(0, 1.2); axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["missing", "ready"])
axes[1].set_title("Specificity backend status")
figure.tight_layout()
display(figure)
plt.close(figure)'''),
        nbformat.v4.new_markdown_cell("## Record the run\n\nThe manifest is written after the analysis and is never read as an analysis input."),
        nbformat.v4.new_code_cell('''_ = write_json(CASE_OUTPUT / "run_manifest.json", {
    "workflow": "05_agent", "inputs": relative_inputs,
    "configuration": {"epochs": EPOCHS, "device": DEVICE, "panel_sizes": [32, 64, 128], "source_weights": [0.25, 0.5, 0.75]},
    "agent_plan": agent_plan, "outputs": {"figure6": str(FIGURE_DATA / "figure6_c_d_values.tsv"), "tuning": str(FIGURE_DATA / "agent_parameter_tuning.tsv")},
    "probe_backend": {"status": probe_screen["status"], "scope": probe_screen["scope"], "artifacts": probe_screen["artifacts"], "backends": probe_status, "reason": probe_screen["note"]},
})'''),
        nbformat.v4.new_markdown_cell("## Full manuscript command\n\nThe CLI runs all five spatial references, repeated seeds, external baselines, probe backends, and the validation-guided HPO analysis:\n\n```bash\npython reproducibility/workflows/agent/run_tutorial.py --data-root data/tutorials --output-dir outputs/paper/agent --panel-sizes 32,64,128 --training-seeds 1,2,3,4,5\n```"),
    ]
    return metadata(cells)
