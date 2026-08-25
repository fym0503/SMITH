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
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
from reproducibility.workflows.common import ranked_genes, run_smith, write_json, write_panel_genes
from reproducibility.workflows.ribomap_transfer.evaluate_outputs import prepare_shared_adata, evaluate_panel_loaded
from reproducibility.workflows.ribomap_transfer.analysis import bias_table_from_objects, jaccard_from_panel_records, performance_paired_tests, bias_pairwise_tests
from reproducibility.workflows.ribomap_transfer.plot_figure4 import _draw_performance, _draw_jaccard, _draw_bias

DATA_ROOT = Path(os.environ.get("SMITH_TUTORIAL_DATA", "data/tutorials")).resolve()
CASE_OUTPUT = Path(os.environ.get("SMITH_TUTORIAL_OUTPUT", "outputs/tutorials")).resolve() / "ribomap"
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
            force=True, include_in_memory=True,
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
        nbformat.v4.new_markdown_cell("### Figure 4h: RIBOMap-specific expression bias"),
        nbformat.v4.new_code_cell("""bias = bias_table_from_objects(ribomap_target, star_raw)
bias_parts = []
for size in (32, 64, 128):
    deep = next(set(row["panel_genes"]) for row in panel_records if row["source"] == "Deep-RIBOmap" and row["training_seed"] == 1 and row["panel_size"] == size)
    star = next(set(row["panel_genes"]) for row in panel_records if row["source"] == "STARmap" and row["training_seed"] == 1 and row["panel_size"] == size)
    part = bias.copy()
    part["panel_size"], part["method"] = size, "SMITH"
    part["group"] = part["gene_symbol"].map(
        lambda gene: "RIBOMap-only" if gene in deep - star else (
            "Shared" if gene in deep & star else (
                "STARmap-only" if gene in star - deep else "Background"
            )
        )
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
        nbformat.v4.new_code_cell("""write_json(CASE_OUTPUT / "run_manifest.json", {
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


def agent_notebook(spec: dict, epochs: int, device: str):
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
