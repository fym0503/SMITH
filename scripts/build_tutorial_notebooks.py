#!/usr/bin/env python3
"""Build biological-question notebooks and optionally execute them."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from memory_tutorial_notebooks import agent_notebook, ribomap_notebook


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "docs" / "source" / "tutorials" / "notebooks"


SPECS = {
    "02_regulatory_activity": {
        "case": "02_regulatory_activity",
        "folder": "regulatory_section", "stem": "02_SMITH_Regulatory_Activity",
        "title": "Regulatory programs and cross-modality transfer in C. elegans", "figure": "Figure 3c-k",
        "biology": "Which compact set of regulatory features is sufficient to preserve C. elegans cell identity and developmental progression? The TF and miRNA assays represent regulatory activity rather than a generic feature-selection benchmark: a useful panel should retain discrete lineage labels and the continuous developmental-time signal in held-out cells.",
        "data_role": "The train/test H5AD files contain lineage-aware TF or miRNA activity, cell-type labels, and absolute developmental time. The supplementary module and TF-pair annotations define the developmental programs and regulator relationships used in the manuscript analyses, while the scRNA H5AD supplies the independent reference for RNA-to-TF transfer. Training cells learn the activity representation; held-out cells test whether selected regulators still recover identity, age, and regulatory structure.",
        "model_role": "SMITH is trained on the training H5AD with reconstruction, cell-type classification, and developmental-time objectives. Its learned gene ranking is then truncated to the manuscript panel sizes; no packaged aggregate ranking is used.",
        "analysis_role": "Cell-type accuracy asks whether the panel preserves discrete lineage information, while developmental-time correlation asks whether it preserves the ordered embryonic trajectory. The paper-specific analyses then ask three biological follow-up questions: does the panel retain annotated developmental modules, can selected targets reconstruct TF co-activity in muscle, neuron, pharynx and skin, and does a panel selected from scRNA-seq retain cell identity when its genes are evaluated on held-out TF-activity lineages?",
        "workflow": "reproducibility/workflows/regulatory_activity/run_tutorial.py",
        "plotter": "reproducibility/workflows/regulatory_activity/plot_figure3.py", "output": "regulatory",
        "inputs": [
            "regulatory_activity/elegans/splits/elegans_tf/split_1/train.h5ad",
            "regulatory_activity/elegans/splits/elegans_tf/split_1/test.h5ad",
            "regulatory_activity/elegans/splits/elegans_mirna/split_1/train.h5ad",
            "regulatory_activity/elegans/splits/elegans_mirna/split_1/test.h5ad",
            "regulatory_activity/elegans/annotations/tf_spatiotemporal_modules.tsv",
            "regulatory_activity/elegans/annotations/tf_regulatory_pairs.tsv",
            "regulatory_activity/elegans/reference/elegans_scrna.h5ad",
        ],
        "arguments": ["--datasets", "elegans_tf,elegans_mirna", "--splits", "split_1", "--methods", "SMITH", "--seeds", "1", "--max-cells", "3000", "--paper-analyses", "--module-file", "regulatory_activity/elegans/annotations/tf_spatiotemporal_modules.tsv", "--regulatory-pair-file", "regulatory_activity/elegans/annotations/tf_regulatory_pairs.tsv", "--scrna-file", "regulatory_activity/elegans/reference/elegans_scrna.h5ad"],
        "plot_arguments": ["--values", "figure_data/figure3_c_f_values.tsv", "--modules", "regulatory_activity/elegans/annotations/tf_spatiotemporal_modules.tsv", "--module-coverage", "figure_data/figure3_h_module_miss_rate.tsv", "--coactivity", "figure_data/figure3_i_coactivity.tsv", "--correlation", "figure_data/figure3_j_tf_scrna_correlation.tsv", "--transfer", "figure_data/figure3_k_transfer.tsv"],
        "tables": ["figure_data/figure3_c_f_summary.tsv", "figure_data/figure3_c_f_paired_tests.tsv", "figure_data/figure3_h_module_miss_rate.tsv", "figure_data/figure3_i_coactivity.tsv", "figure_data/figure3_j_tf_scrna_correlation.tsv", "figure_data/figure3_k_transfer.tsv"],
        "figure_panels": [
            ("Figure 3c - TF cell-type accuracy", "figures/figure3_c.png", 430),
            ("Figure 3d - TF developmental-time correlation", "figures/figure3_d.png", 430),
            ("Figure 3e - miRNA cell-type accuracy", "figures/figure3_e.png", 430),
            ("Figure 3f - miRNA developmental-time correlation", "figures/figure3_f.png", 430),
            ("Figure 3g - annotated developmental modules", "figures/figure3_g.png", 500),
            ("Figure 3h - developmental module miss rate", "figures/figure3_h.png", 500),
            ("Figure 3i - TF co-activity reconstruction", "figures/figure3_i.png", 500),
            ("Figure 3j - scRNA/TF correlation structure", "figures/figure3_j.png", 500),
            ("Figure 3k - scRNA-to-TF panel transfer", "figures/figure3_k.png", 500),
            ("Shared method legend", "figures/figure3_method_legend.png", 900),
        ],
        "paper_command": "--splits split_1,split_2,split_3,split_4,split_5 --methods SMITH,PERSIST-class,PERSIST,ActiveSVM,scGIST,scGeneFit,Spapros --baseline-root external/SMITH_baselines/GPS_tools-main/baselines --baseline-python PERSIST=/opt/envs/persist/bin/python --baseline-python PERSIST-class=/opt/envs/persist/bin/python --baseline-python scGIST=/opt/envs/scgist/bin/python --epochs 200",
        "scope": "This executed page uses one real TF split, one real miRNA split and one scRNA-to-TF transfer split. The paper command above regenerates Figure 3c-k with all five lineage-aware splits and manuscript baselines. The module, TF-pair and scRNA inputs are versioned biological inputs; the workflow stops with an explicit error if they are absent.",
    },
    "03_ribomap_transfer": {
        "case": "03_ribomap_transfer",
        "folder": "ribomap_section", "stem": "03_SMITH_RIBOMap_Transfer",
        "title": "Cross-modality brain panel transfer to RIBOMap", "figure": "Figure 4c-h",
        "biology": "Can a compact gene panel transfer cell-type and brain-region biology from reference modalities into RIBOMap? Deep-RIBOmap and STARmap measure related but non-identical molecular views, so the biological question is whether the same panel captures stable tissue structure without erasing ribosome-associated expression differences.",
        "data_role": "The workflow starts from real Deep-RIBOmap, STARmap, and target RIBOMap H5AD files. Shared-gene preparation defines the common biological measurement space; target cell-type and region labels provide held-out transfer endpoints, while expression means support the RIBOMap-bias analysis.",
        "model_role": "For each source modality, SMITH is trained from the shared-gene H5AD with reconstruction and cell-type objectives, plus spatial coordination when coordinates are available. The ranking is converted into new source panels and evaluated on the target RIBOMap cells.",
        "analysis_role": "Cell-type and region accuracy test whether the panel transfers biological identity and anatomy. Same- versus cross-modality Jaccard measures which biological features are stable across references. RIBOMap bias tests whether panel membership follows translatome-specific abundance rather than only generic expression.",
        "workflow": "reproducibility/workflows/ribomap_transfer/run_tutorial.py",
        "plotter": "reproducibility/workflows/ribomap_transfer/plot_figure4.py", "output": "ribomap",
        "inputs": [
            "ribomap_transfer/ribomap/deep_brain_ribomap.h5ad",
            "ribomap_transfer/ribomap/mouse_brain_starmap_rep2.h5ad",
            "ribomap_transfer/ribomap/mouse_brain_ribomap_rep2.h5ad",
        ],
        "arguments": ["--methods", "SMITH", "--panel-sizes", "32,64,128", "--training-seeds", "1,2", "--evaluation-seeds", "1,2,3", "--max-cells", "3000"],
        "plot_arguments": ["--metrics", "figure_data/figure4_c_f_values.tsv", "--overlap", "figure_data/figure4_g_jaccard.tsv", "--bias", "figure_data/figure4_h_ribomap_bias.tsv", "--bias-tests", "figure_data/figure4_h_pairwise_tests.tsv"],
        "tables": ["figure_data/figure4_c_f_values.tsv", "figure_data/figure4_g_jaccard.tsv", "figure_data/figure4_h_ribomap_bias.tsv", "figure_data/figure4_h_pairwise_tests.tsv"],
        "figure_panels": [
            ("Figure 4c - Deep-RIBOmap cell-type transfer", "figures/figure4_c.png", 430),
            ("Figure 4d - Deep-RIBOmap region transfer", "figures/figure4_d.png", 430),
            ("Figure 4e - STARmap cell-type transfer", "figures/figure4_e.png", 430),
            ("Figure 4f - STARmap region transfer", "figures/figure4_f.png", 430),
            ("Figure 4g - same- versus cross-modality overlap", "figures/figure4_g.png", 430),
            ("Figure 4h - RIBOMap expression bias", "figures/figure4_h.png", 430),
            ("Shared method legend", "figures/figure4_method_legend.png", 900),
        ],
        "paper_command": "--methods SMITH,PERSIST-class,PERSIST,ActiveSVM,scGIST,scGeneFit,Spapros --baseline-root external/SMITH_baselines/GPS_tools-main/baselines --baseline-python PERSIST=/opt/envs/persist/bin/python --baseline-python PERSIST-class=/opt/envs/persist/bin/python --baseline-python scGIST=/opt/envs/scgist/bin/python --panel-sizes 32,64,128 --training-seeds 1,2,3,4,5 --evaluation-seeds 1,2,3,4,5 --epochs 200",
        "scope": "The workflow tests whether a panel selected in related brain modalities preserves cell-type and region biology in RIBOMap. Figure 4c-h provide the manuscript-aligned visual summaries. Figure 4i is deliberately omitted unless a versioned Reactome/GO snapshot is supplied. Figure 4j-n additionally require the manuscript clean-fusion aligned H5AD and are not replaced with unrelated summaries.",
    },
    "05_agent": {
        "case": "05_agent",
        "folder": "agent_section", "stem": "05_SMITH_Agent_Evaluation",
        "title": "Liver cell identity in MERFISH", "figure": "Figure 6c-d",
        "biology": "Which genes should be measured in a spatial liver assay so that cell identities and their expression programs remain interpretable? The source snRNA-seq provides broad cell-state coverage, while spatial references add tissue context; the experiment compares a source-only panel with a panel informed by both kinds of biological evidence before reading out MERFISH.",
        "data_role": "The inputs are healthy-liver snRNA-seq, MERFISH, and spatial-reference H5AD files. MERFISH is the held-out assay used for evaluation; the snRNA-seq and spatial references are training views that contribute complementary cell-state and tissue-location information.",
        "model_role": "SMITH is trained separately on the source and each spatial reference after restricting them to the MERFISH gene universe. Source and reference rankings are aggregated into source-only and multi-reference panels, which are newly written for each seed and panel size.",
        "analysis_role": "MERFISH cell-type accuracy asks whether the selected genes retain cellular identity in the assay that will be measured. Mean MERFISH expression asks whether the panel is supported by detectable biology. The paired comparison tests whether spatial references add biological information beyond the source transcriptome.",
        "workflow": "reproducibility/workflows/agent/run_tutorial.py",
        "plotter": "reproducibility/workflows/agent/plot_figure6.py", "output": "agent",
        "inputs": [
            "agent/liver_merfish/adata_healthy_nucseq.h5ad", "agent/liver_merfish/adata_healthy_merfish.h5ad",
            "agent/references/PSC011_C1_visium.h5ad", "agent/references/WSSS_F_IMMsp9838712_visium.h5ad",
        ],
        "arguments": ["--reference", "references/PSC011_C1_visium.h5ad", "--reference", "references/WSSS_F_IMMsp9838712_visium.h5ad", "--panel-sizes", "32,64,128", "--training-seeds", "1,2", "--max-cells", "3000"],
        "plot_arguments": ["--accuracy", "figure_data/figure6_c_cell_type_accuracy.tsv", "--expression", "figure_data/figure6_d_merfish_expression.tsv"],
        "tables": ["figure_data/figure6_c_cell_type_accuracy.tsv", "figure_data/figure6_d_merfish_expression.tsv"],
        "figure_panels": [
            ("Figure 6c - MERFISH cell-type accuracy", "figures/figure6_c.png", 430),
            ("Figure 6d - MERFISH expression support", "figures/figure6_d.png", 430),
        ],
        "paper_command": "--panel-sizes 32,64,128 --training-seeds 1,2,3,4,5 --epochs 200 (omit --reference to use all five manifest-listed defaults)",
        "scope": "This hosted run uses two real spatial references and two training seeds. The manuscript Figure 6c-d command uses five retrieved liver references and five training seeds. Figure 6e-j requires external probe-design backends and the validation-guided HPO run; this notebook does not fabricate those panels.",
    },
}


def build_regulatory_notebook(spec: dict, epochs: int, device: str) -> nbformat.NotebookNode:
    """Build Figure 3 as a single in-memory data-to-result analysis."""
    github = (
        "https://github.com/fym0503/SMITH/blob/main/docs/source/tutorials/notebooks/"
        f"{spec['folder']}/{spec['stem']}_source.ipynb"
    )
    setup = f'''from pathlib import Path
import hashlib, os, sys
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from IPython.display import display

from reproducibility.workflows.common import ranked_genes, run_smith, write_json, write_panel_genes
from reproducibility.workflows.figure_style import configure
from reproducibility.workflows.regulatory_activity.analysis import statistical_analysis
from reproducibility.workflows.regulatory_activity.evaluate_outputs import evaluate_loaded
from reproducibility.workflows.regulatory_activity.paper_analysis import (
    coactivity_from_objects, module_coverage, tf_scrna_correlation_from_objects,
)
from reproducibility.workflows.regulatory_activity.plot_figure3 import (
    PANEL_SPECS, _draw_bar_panel, _draw_module_coverage,
    _draw_module_schematic, _draw_transfer,
)

def find_repository(start):
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "reproducibility").exists():
            return candidate
    raise RuntimeError("Run this notebook inside a SMITH repository checkout.")

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

ROOT = find_repository(Path.cwd().resolve())
sys.path.insert(0, str(ROOT / "src"))
DATA_ROOT = Path(os.environ.get("SMITH_TUTORIAL_DATA", "data/tutorials")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("SMITH_TUTORIAL_OUTPUT", "outputs/tutorials")).expanduser().resolve()
CASE_OUTPUT = OUTPUT_ROOT / "regulatory"
FIGURE_DATA = CASE_OUTPUT / "figure_data"
EPOCHS = int(os.environ.get("SMITH_TUTORIAL_EPOCHS", {epochs!r}))
DEVICE = os.environ.get("SMITH_TUTORIAL_DEVICE", {device!r})
MAX_CELLS = int(os.environ.get("SMITH_TUTORIAL_MAX_CELLS", "3000"))
BATCH_SIZE = int(os.environ.get("SMITH_TUTORIAL_BATCH_SIZE", "1024"))
FIGURE_DATA.mkdir(parents=True, exist_ok=True)
configure()
'''
    inspect = f'''relative_inputs = {spec['inputs']!r}
input_paths = {{name: DATA_ROOT / name for name in relative_inputs}}
for name, path in input_paths.items():
    if not path.is_file():
        raise FileNotFoundError(f"Missing {{path}}. Run scripts/download_tutorial_data.py first.")
input_checksums = {{name: sha256_file(path) for name, path in input_paths.items()}}

tf_train = ad.read_h5ad(input_paths[relative_inputs[0]])
tf_test = ad.read_h5ad(input_paths[relative_inputs[1]])
mirna_train = ad.read_h5ad(input_paths[relative_inputs[2]])
mirna_test = ad.read_h5ad(input_paths[relative_inputs[3]])
modules = pd.read_csv(input_paths[relative_inputs[4]], sep="\\t")
tf_pairs = pd.read_csv(input_paths[relative_inputs[5]], sep="\\t")
scrna = ad.read_h5ad(input_paths[relative_inputs[6]])

if "cell_name" in tf_train.obs and "cell_name" in tf_test.obs:
    leaked = set(tf_train.obs["cell_name"].astype(str)) & set(tf_test.obs["cell_name"].astype(str))
    if leaked:
        raise ValueError(f"Lineage-aware split contains {{len(leaked)}} shared cell identifiers")
'''
    training = '''benchmark_sizes = {"elegans_tf": (32, 64, 128), "elegans_mirna": (16, 24, 32)}
training_sizes = {"elegans_tf": (16, 24, 32, 64, 128), "elegans_mirna": (16, 24, 32)}
loaded_data = {
    "elegans_tf": (tf_train, tf_test),
    "elegans_mirna": (mirna_train, mirna_test),
}
training_runs = {"elegans_tf": {}, "elegans_mirna": {}}
panel_records = []

for dataset, sizes in training_sizes.items():
    train_adata, _ = loaded_data[dataset]
    for panel_size in sizes:
        run_dir = CASE_OUTPUT / "runs" / dataset / "split_1" / "seed_1" / f"panel_{panel_size}" / "SMITH"
        result = run_smith(
            adata_file=input_paths[
                "regulatory_activity/elegans/splits/" + dataset + "/split_1/train.h5ad"
            ],
            output_dir=run_dir,
            tasks="recon,cls,standard_coordination,time" if dataset == "elegans_tf" else "recon,cls,time",
            task_name=f"{dataset}_split_1_panel{panel_size}_seed1",
            panel_size=panel_size, epochs=EPOCHS, device=DEVICE, seed=1,
            batch_size=BATCH_SIZE, time_label="absolute_time", max_cells=MAX_CELLS,
            sampling_strategy="celltype",
            save_model=dataset == "elegans_tf" and panel_size == 32,
            force=True, include_in_memory=True,
        )
        panel_path = run_dir.parent / "panels" / f"SMITH_top{panel_size}.tsv"
        write_panel_genes(result["panel_genes"], panel_path)
        training_runs[dataset][panel_size] = result
        panel_records.append({
            "dataset": dataset, "split": "split_1", "training_seed": 1,
            "method": "SMITH", "panel_size": panel_size,
            "panel_genes": result["panel_genes"], "panel_file": str(panel_path),
        })
'''
    evaluation = '''benchmark_rows = []
predictions = {}
for dataset, sizes in benchmark_sizes.items():
    train_adata, test_adata = loaded_data[dataset]
    for panel_size in sizes:
        run = training_runs[dataset][panel_size]
        evaluation_dir = Path(run["output_dir"]) / "evaluation"
        metrics, celltype_prediction, time_prediction = evaluate_loaded(
            train_adata, test_adata, run["panel_genes"], panel_size,
            time_column="absolute_time", neighbors=5, output_dir=evaluation_dir,
        )
        benchmark_rows.append({
            "dataset": dataset, "split": "split_1", "training_seed": 1,
            "method": "SMITH", "panel_size": panel_size, **metrics["metrics"],
        })
        predictions[(dataset, panel_size)] = {
            "cell_type": celltype_prediction, "developmental_time": time_prediction,
        }

benchmark_values = pd.json_normalize(benchmark_rows)
benchmark_values.to_csv(FIGURE_DATA / "figure3_c_f_values.tsv", sep="\\t", index=False)
paired_tests = statistical_analysis(benchmark_values)
paired_tests.to_csv(FIGURE_DATA / "figure3_c_f_paired_tests.tsv", sep="\\t", index=False)
tf_values = benchmark_values.loc[benchmark_values["dataset"] == "elegans_tf"]
mirna_values = benchmark_values.loc[benchmark_values["dataset"] == "elegans_mirna"]
'''

    def bar_cell(variable: str, panel: str) -> str:
        return f'''figure, axis = plt.subplots(figsize=(2.35, 2.10), facecolor="white")
_draw_bar_panel(axis, {variable}, PANEL_SPECS["{panel}"])
figure.text(0.015, 0.985, "{panel}", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.25, right=0.97, bottom=0.22, top=0.86)
display(figure)
plt.close(figure)'''

    cells = [
        nbformat.v4.new_markdown_cell(
            f"# {spec['title']}\n\n"
            "This tutorial starts from the real activity and atlas inputs, trains SMITH, and passes the "
            "newly generated panels through every downstream analysis in memory. Files written under the "
            "output directory are provenance artifacts, not inputs to later notebook cells. "
            f"[Open the editable source notebook on GitHub]({github})."
        ),
        nbformat.v4.new_markdown_cell(
            f"## Biological question\n\n{spec['biology']}\n\n**Biological endpoints:** {spec['analysis_role']}"
        ),
        nbformat.v4.new_markdown_cell(
            "## Download the real input data\n\n"
            "```bash\npython scripts/download_tutorial_data.py \\\n+  --case 02_regulatory_activity \\\n+  --data-root data/tutorials\n```\n\n"
            "Read the Docs displays this executed notebook; it does not train the model during the documentation build."
        ),
        nbformat.v4.new_markdown_cell("## Configuration"),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_markdown_cell(f"## Load the activity atlases and biological annotations\n\n{spec['data_role']}"),
        nbformat.v4.new_code_cell(inspect),
        nbformat.v4.new_markdown_cell(
            f"## Train SMITH and generate panels\n\n{spec['model_role']} The returned ranking and panel genes remain in memory; their files are written only so the run can be audited."
        ),
        nbformat.v4.new_code_cell(training),
        nbformat.v4.new_markdown_cell(
            "## Evaluate held-out cell identity and developmental time\n\n"
            "The classifier and time regressor operate directly on each newly selected panel. Their predictions are retained in memory and also saved for provenance."
        ),
        nbformat.v4.new_code_cell(evaluation),
        nbformat.v4.new_markdown_cell(
            "### TF activity retains lineage identity\n\nCell-type accuracy measures whether the compact TF panel separates held-out cell identities."
        ),
        nbformat.v4.new_code_cell(bar_cell("tf_values", "c")),
        nbformat.v4.new_markdown_cell(
            "### TF activity retains developmental order\n\nPearson correlation measures whether the same panel preserves the continuous developmental trajectory."
        ),
        nbformat.v4.new_code_cell(bar_cell("tf_values", "d")),
        nbformat.v4.new_markdown_cell(
            "### miRNA activity retains lineage identity\n\nThe analysis asks whether a smaller post-transcriptional panel still resolves held-out cell types."
        ),
        nbformat.v4.new_code_cell(bar_cell("mirna_values", "e")),
        nbformat.v4.new_markdown_cell(
            "### miRNA activity retains developmental order\n\nThe temporal endpoint tests whether selected miRNA activities follow embryonic progression."
        ),
        nbformat.v4.new_code_cell(bar_cell("mirna_values", "f")),
        nbformat.v4.new_markdown_cell(
            "## Developmental regulatory programs\n\n### Spatiotemporal TF modules\n\n"
            "The atlas groups regulators by tissue and developmental phase, defining the programs whose representation is tested next."
        ),
        nbformat.v4.new_code_cell('''figure, axis = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
_draw_module_schematic(axis, modules)
figure.text(0.015, 0.985, "g", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
display(figure)
plt.close(figure)'''),
        nbformat.v4.new_markdown_cell(
            "### Coverage of developmental modules\n\n"
            "For each current TF panel, the miss rate is the fraction of annotated modules containing no selected TF."
        ),
        nbformat.v4.new_code_cell('''coverage_panels = pd.json_normalize([
    row for row in panel_records
    if row["dataset"] == "elegans_tf" and row["panel_size"] in (16, 24, 32)
])
coverage_panels["panel_genes"] = [
    training_runs["elegans_tf"][int(size)]["panel_genes"]
    for size in coverage_panels["panel_size"]
]
coverage = module_coverage(coverage_panels, modules)
coverage.to_csv(FIGURE_DATA / "figure3_h_module_miss_rate.tsv", sep="\\t", index=False)

figure, axis = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
_draw_module_coverage(axis, coverage)
figure.text(0.015, 0.985, "h", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
display(figure)
plt.close(figure)'''),
        nbformat.v4.new_markdown_cell(
            "## Regulatory reconstruction and modality transfer\n\n### Reconstruction of TF co-activity\n\n"
            "The reconstruction head learned during the current 32-TF training run predicts held-out activity. Atlas TF pairs then quantify agreement within four lineages."
        ),
        nbformat.v4.new_code_cell('''tf32 = training_runs["elegans_tf"][32]
coactivity = coactivity_from_objects(
    tf_train, tf_test, tf32["panel_genes"], pairs=tf_pairs,
    checkpoint_file=tf32["checkpoint_file"], method="SMITH", seed=1,
)
coactivity.to_csv(FIGURE_DATA / "figure3_i_coactivity.tsv", sep="\\t", index=False)

lineages = ["muscle", "neuron", "pharynx", "skin"]
means = [coactivity.loc[coactivity["lineage"] == lineage, "pearson"].mean() for lineage in lineages]
errors = [coactivity.loc[coactivity["lineage"] == lineage, "pearson"].sem() if sum(coactivity["lineage"] == lineage) > 1 else 0.0 for lineage in lineages]
figure, axis = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
axis.bar(np.arange(len(lineages)), means, yerr=errors, capsize=1.5, color="#2f75b5", edgecolor="black", linewidth=0.4)
axis.set(title="TF co-activity reconstruction", ylabel="Pearson agreement")
axis.set_xticks(np.arange(len(lineages)), [name.title() for name in lineages], rotation=25, ha="right")
axis.set_ylim(-1, 1)
figure.text(0.015, 0.985, "i", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
display(figure)
plt.close(figure)'''),
        nbformat.v4.new_markdown_cell(
            "### Conservation of regulatory structure across modalities\n\n"
            "Shared TFs are aggregated over matched lineages in scRNA-seq and TF activity. Biclustering exposes whether the two modalities retain similar regulatory blocks."
        ),
        nbformat.v4.new_code_cell('''tf_combined = ad.concat([tf_train, tf_test], axis=0, join="inner", merge="same", index_unique=None)
tf_combined.var_names = tf_train.var_names.copy()
correlation, correlation_matrices = tf_scrna_correlation_from_objects(scrna, tf_combined)
correlation.to_csv(FIGURE_DATA / "figure3_j_tf_scrna_correlation.tsv", sep="\\t", index=False)
np.savez_compressed(FIGURE_DATA / "figure3_j_correlation_matrices.npz", **correlation_matrices)

figure, axes = plt.subplots(1, 2, figsize=(4.7, 2.25), facecolor="white")
for axis, key, title in zip(axes, ("scrna", "tf"), ("scRNA-seq", "TF activity")):
    image = axis.imshow(correlation_matrices[key], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    image.set_rasterized(True)
    axis.set_title(title, fontsize=8.5, pad=4)
    axis.set_xticks([]); axis.set_yticks([])
axes[1].text(0.98, 0.02, f"corr = {float(correlation.iloc[0]['mean_rowwise_pearson']):.2f}", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
figure.text(0.008, 0.985, "j", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.86, wspace=0.08)
display(figure)
plt.close(figure)'''),
        nbformat.v4.new_markdown_cell(
            "### Transfer from scRNA-seq into TF activity\n\n"
            "A new panel is now learned from the loaded scRNA-seq reference, then evaluated directly on the held-out TF-activity split."
        ),
        nbformat.v4.new_code_cell('''source_names = scrna.var["gene_short_name"].astype(str).str.upper() if "gene_short_name" in scrna.var else scrna.var_names.astype(str).str.upper()
scrna_for_training = scrna.copy()
scrna_for_training.var_names = pd.Index(source_names)
scrna_for_training.var_names_make_unique()
if "gene_short_name" in scrna_for_training.var:
    scrna_for_training.var = scrna_for_training.var.rename(columns={"gene_short_name": "gene_short_name_orig"})
if "cell_type" not in scrna_for_training.obs and "cell.type" in scrna_for_training.obs:
    scrna_for_training.obs["cell_type"] = scrna_for_training.obs["cell.type"].astype(str)
if "absolute_time" not in scrna_for_training.obs and "embryo.time" in scrna_for_training.obs:
    scrna_for_training.obs["absolute_time"] = pd.to_numeric(scrna_for_training.obs["embryo.time"], errors="coerce")
valid_labels = scrna_for_training.obs["cell_type"].astype(str).str.strip()
valid_time = pd.to_numeric(scrna_for_training.obs["absolute_time"], errors="coerce")
keep_cells = valid_labels.ne("") & ~valid_labels.str.lower().isin({"nan", "none"}) & valid_time.notna()
scrna_for_training = scrna_for_training[keep_cells.to_numpy()].copy()
scrna_for_training.obs["cell_type"] = valid_labels.loc[keep_cells].to_numpy()
scrna_for_training.obs["absolute_time"] = valid_time.loc[keep_cells].to_numpy()
shared_tf = sorted(set(scrna_for_training.var_names) & set(tf_test.var_names.astype(str).str.upper()))
scrna_for_training = scrna_for_training[:, scrna_for_training.var_names.isin(shared_tf)].copy()
counts = scrna_for_training.layers["counts"] if "counts" in scrna_for_training.layers else scrna_for_training.X
counts = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
totals = counts.sum(axis=1, keepdims=True); totals[totals == 0] = 1.0
normalized = np.log1p(counts.astype(np.float32) * (1e4 / totals))
minimum = normalized.min(axis=0); span = normalized.max(axis=0) - minimum; span[span == 0] = 1.0
scrna_for_training.X = ((normalized - minimum) / span).astype(np.float32)

transfer_run = run_smith(
    adata_file=None, adata=scrna_for_training,
    output_dir=CASE_OUTPUT / "runs/transfer_scRNA/seed_1/SMITH",
    tasks="recon,cls,time", task_name="elegans_scrna_to_tf_seed1",
    panel_size=128, epochs=EPOCHS, device=DEVICE, seed=1,
    batch_size=BATCH_SIZE, time_label="absolute_time", max_cells=MAX_CELLS,
    sampling_strategy="celltype", force=True, include_in_memory=True,
)
transfer_rows = []
for panel_size in (32, 64, 128):
    panel_genes = ranked_genes(transfer_run["ranking_frame"], panel_size)
    panel_path = Path(transfer_run["output_dir"]) / "panels" / f"RNA_TF_top{panel_size}.tsv"
    write_panel_genes(panel_genes, panel_path)
    metrics, celltype_prediction, time_prediction = evaluate_loaded(
        tf_train, tf_test, panel_genes, panel_size, time_column="absolute_time", neighbors=5,
        output_dir=Path(transfer_run["output_dir"]) / "evaluation" / f"top{panel_size}",
    )
    transfer_rows.append({
        "split": "split_1", "training_seed": 1, "method": "SMITH",
        "source_modality": "RNA-TF", "panel_size": panel_size, **metrics["metrics"],
    })
for row in benchmark_rows:
    if row["dataset"] == "elegans_tf":
        transfer_rows.append({**row, "source_modality": "TF-TF"})
transfer = pd.json_normalize(transfer_rows)
transfer.to_csv(FIGURE_DATA / "figure3_k_transfer.tsv", sep="\\t", index=False)

figure, axis = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
_draw_transfer(axis, transfer)
figure.text(0.015, 0.985, "k", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
display(figure)
plt.close(figure)'''),
        nbformat.v4.new_markdown_cell(
            "## Record the run\n\nThe manifest records the inputs and artifacts after all analyses finish. It is never read as an analysis input."
        ),
        nbformat.v4.new_code_cell('''def serializable_run(run):
    return {key: value for key, value in run.items() if key not in {"ranking_frame", "panel_genes"}}

write_json(CASE_OUTPUT / "run_manifest.json", {
    "case": "02_regulatory_activity",
    "inputs": [{"path": name, "sha256": digest} for name, digest in input_checksums.items()],
    "configuration": {"epochs": EPOCHS, "device": DEVICE, "max_cells": MAX_CELLS, "seed": 1},
    "training_runs": [
        serializable_run(run)
        for dataset_runs in training_runs.values() for run in dataset_runs.values()
    ] + [serializable_run(transfer_run)],
    "outputs": {
        "benchmark": str(FIGURE_DATA / "figure3_c_f_values.tsv"),
        "module_coverage": str(FIGURE_DATA / "figure3_h_module_miss_rate.tsv"),
        "coactivity": str(FIGURE_DATA / "figure3_i_coactivity.tsv"),
        "tf_scrna_correlation": str(FIGURE_DATA / "figure3_j_tf_scrna_correlation.tsv"),
        "transfer": str(FIGURE_DATA / "figure3_k_transfer.tsv"),
    },
})'''),
        nbformat.v4.new_markdown_cell(
            "## Full manuscript command\n\nThe command-line workflow remains the entry point for all five splits and manuscript baselines:\n\n"
            "```bash\npython reproducibility/workflows/regulatory_activity/run_tutorial.py \\\n+  --data-root data/tutorials \\\n+  --output-dir outputs/paper/regulatory \\\n+  --datasets elegans_tf,elegans_mirna \\\n+  --splits split_1,split_2,split_3,split_4,split_5 \\\n+  --methods SMITH,PERSIST-class,PERSIST,ActiveSVM,scGIST,scGeneFit,Spapros \\\n+  --paper-analyses --epochs 200\n```\n\n"
            "The executed tutorial uses one real lineage split and one seed so that the complete data-to-result path remains practical to rerun."
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
        language_info={"name": "python", "version": "3"},
    )
    return notebook


def build_notebook(spec: dict, epochs: int, device: str) -> nbformat.NotebookNode:
    if spec["case"] == "02_regulatory_activity":
        return build_regulatory_notebook(spec, epochs, device)
    if spec["case"] == "03_ribomap_transfer":
        return ribomap_notebook(spec, epochs, device)
    if spec["case"] == "05_agent":
        return agent_notebook(spec, epochs, device)
    if spec["case"] == "03_ribomap_transfer":
        return build_ribomap_notebook(spec, epochs, device)
    if spec["case"] == "05_agent":
        return build_agent_notebook(spec, epochs, device)
    github = f"https://github.com/fym0503/SMITH/blob/main/docs/source/tutorials/notebooks/{spec['folder']}/{spec['stem']}_source.ipynb"
    setup = f'''from pathlib import Path
import hashlib, json, os, subprocess, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

def find_repository(start):
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "reproducibility").exists():
            return candidate
    raise RuntimeError("Run this notebook inside a SMITH repository checkout.")

ROOT = find_repository(Path.cwd().resolve())
DATA_ROOT = Path(os.environ.get("SMITH_TUTORIAL_DATA", "data/tutorials")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("SMITH_TUTORIAL_OUTPUT", "outputs/tutorials")).expanduser().resolve()
CASE_OUTPUT = OUTPUT_ROOT / {spec['output']!r}
EPOCHS = int(os.environ.get("SMITH_TUTORIAL_EPOCHS", {epochs!r}))
DEVICE = os.environ.get("SMITH_TUTORIAL_DEVICE", {device!r})
REUSE_EXISTING = os.environ.get("SMITH_TUTORIAL_REUSE_EXISTING", "0") == "1"
WORKFLOW_ENV = os.environ.copy()
WORKFLOW_ENV["PYTHONPATH"] = os.pathsep.join(
    [str(ROOT), str(ROOT / "src"), WORKFLOW_ENV.get("PYTHONPATH", "")]
)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
'''
    if spec["case"] == "02_regulatory_activity":
        setup += '''

from reproducibility.workflows.figure_style import configure
from reproducibility.workflows.regulatory_activity.plot_figure3 import (
    PANEL_SPECS, _draw_bar_panel, _draw_coactivity, _draw_module_coverage,
    _draw_module_schematic, _draw_transfer, _plot_tf_correlation,
)
configure()
'''
    inspect = f'''inputs = {spec['inputs']!r}
input_checksums = {{}}
for relative in inputs:
    path = DATA_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing {{path}}. Run scripts/download_tutorial_data.py first.")
    input_checksums[relative] = sha256_file(path)
'''
    command = f'''command = [sys.executable, str(ROOT / {spec['workflow']!r}), "--data-root", str(DATA_ROOT), "--output-dir", str(CASE_OUTPUT), "--device", DEVICE, "--epochs", str(EPOCHS)] + {spec['arguments']!r}
if not REUSE_EXISTING:
    command.append("--force")
completed = subprocess.run(
    command, cwd=ROOT, env=WORKFLOW_ENV, text=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
if completed.returncode:
    print(completed.stdout)
    raise subprocess.CalledProcessError(completed.returncode, command)
manifest = json.loads((CASE_OUTPUT / "run_manifest.json").read_text())
if not manifest.get("training_runs"):
    raise RuntimeError("The workflow did not record any SMITH training runs.")
for relative in {spec['tables']!r}:
    if not (CASE_OUTPUT / relative).is_file():
        raise FileNotFoundError(CASE_OUTPUT / relative)
'''
    if spec["case"] == "02_regulatory_activity":
        analysis = '''import warnings

from reproducibility.workflows.regulatory_activity.analysis import write_statistical_analysis
from reproducibility.workflows.regulatory_activity.evaluate_outputs import evaluate
from reproducibility.workflows.regulatory_activity.paper_analysis import (
    coactivity_reconstruction,
    tf_scrna_correlation,
    write_module_coverage,
)

panel_file = CASE_OUTPUT / "runs/elegans_tf/split_1/seed_1/panel_32/panels/SMITH_top32.tsv"
recheck_dir = CASE_OUTPUT / "notebook_recheck" / "elegans_tf"
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    evaluate(
        DATA_ROOT / "regulatory_activity/elegans/splits/elegans_tf/split_1/train.h5ad",
        DATA_ROOT / "regulatory_activity/elegans/splits/elegans_tf/split_1/test.h5ad",
        panel_file, recheck_dir, 32, neighbors=5,
    )
write_statistical_analysis(CASE_OUTPUT / "figure_data/figure3_c_f_values.tsv", CASE_OUTPUT / "figure_data")
paper_outputs = manifest["outputs"]
required_paper_outputs = ("module_coverage", "coactivity", "tf_scrna_correlation", "transfer")
if any(key not in paper_outputs or not Path(paper_outputs[key]).is_file() for key in required_paper_outputs):
    raise FileNotFoundError("The paper-specific Figure 3g-k outputs were not generated")
if not (recheck_dir / "cell_type_predictions.tsv").is_file() or not (recheck_dir / "developmental_time_predictions.tsv").is_file():
    raise FileNotFoundError("Held-out prediction files were not generated")
'''
    elif spec["case"] == "03_ribomap_transfer":
        analysis = '''from reproducibility.workflows.ribomap_transfer.analysis import write_statistical_analysis
from reproducibility.workflows.ribomap_transfer.evaluate_outputs import evaluate_panel

panel_file = CASE_OUTPUT / "runs/Deep-RIBOmap/seed_1/panels/SMITH_top32.tsv"
recheck_dir = CASE_OUTPUT / "notebook_recheck" / "Deep-RIBOmap_celltype"
evaluate_panel(
    DATA_ROOT / "ribomap_transfer/ribomap/mouse_brain_ribomap_rep2.h5ad",
    panel_file, 32, 1, "celltype", recheck_dir, 5,
)
write_statistical_analysis(
    CASE_OUTPUT / "figure_data/figure4_c_f_values.tsv",
    CASE_OUTPUT / "figure_data/figure4_h_ribomap_bias.tsv",
    CASE_OUTPUT / "figure_data",
)
if not (recheck_dir / "predictions.tsv").is_file():
    raise FileNotFoundError("Held-out RIBOMap predictions were not generated")
'''
    else:
        analysis = '''if not list(CASE_OUTPUT.glob("evaluations/**/metrics.json")):
    raise FileNotFoundError("No generated evaluation files found")
'''
    plot_args = []
    for index in range(0, len(spec["plot_arguments"]), 2):
        relative = spec["plot_arguments"][index + 1]
        root_name = "DATA_ROOT" if relative.startswith("regulatory_activity/") else "CASE_OUTPUT"
        plot_args.extend([spec["plot_arguments"][index], f"str({root_name} / {relative!r})"])
    plot_pairs = ", ".join([repr(plot_args[i]) if i % 2 == 0 else plot_args[i] for i in range(len(plot_args))])
    plotting = f'''figure_dir = CASE_OUTPUT / "figures"
plot_command = [sys.executable, str(ROOT / {spec['plotter']!r}), {plot_pairs}, "--output-dir", str(figure_dir)]
subprocess.run(
    plot_command, cwd=ROOT, env=WORKFLOW_ENV, check=True, text=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
for heading, relative, width in {spec['figure_panels']!r}:
    display(Markdown(f"### {{heading}}"))
    display(Image(filename=str(CASE_OUTPUT / relative), width=width))
'''
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"}, language_info={"name": "python", "version": "3"})
    opening_cells = [
        nbformat.v4.new_markdown_cell(f"# {spec['title']}\n\nThis notebook follows the biological workflow from real input data to a trained model, a newly selected panel, and manuscript-matched biological analyses. It does not read packaged aggregate results as tutorial inputs. [Open the editable source notebook on GitHub]({github})."),
        nbformat.v4.new_markdown_cell(f"## Biological question\n\n{spec['biology']}\n\n**How to read the endpoint:** {spec['analysis_role']}"),
        nbformat.v4.new_markdown_cell(f"## Step 0: Download the real input data\n\nDownload the versioned Zenodo archive and verify its checksums before training:\n\n```bash\npython scripts/download_tutorial_data.py \\\n  --case {spec['case']} \\\n  --data-root data/tutorials\n```\n\nThe notebook is pre-executed for documentation. Read the Docs does not download large data or train SMITH during documentation builds." + ("\n\nThe developmental-module and TF-pair tables in the archive are normalized from the source atlas Supplementary Table 5. Their preparation can be audited independently with:\n\n```bash\npython scripts/prepare_elegans_atlas_annotations.py --data-root data/tutorials\n```" if spec["case"] == "02_regulatory_activity" else "")),
        nbformat.v4.new_markdown_cell("## Configuration"), nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_markdown_cell(f"## Step 1: Inspect the biological input data\n\n{spec['data_role']}"), nbformat.v4.new_code_cell(inspect),
        nbformat.v4.new_markdown_cell(f"## Step 2: Train SMITH and select a panel\n\n{spec['model_role']}\n\nThe command below starts from the H5AD inputs above and writes a fresh model ranking, panel, evaluation, and run manifest."), nbformat.v4.new_code_cell(command),
    ]
    closing_cell = nbformat.v4.new_markdown_cell(
        f"## Full manuscript command\n\nAppend the following paper-scale arguments to the workflow command:\n\n```text\n{spec['paper_command']}\n```\n\n{spec['scope']} This quick hosted run is intentionally smaller than the paper-scale comparison, but it starts from the same real inputs and executes the same prediction and analysis functions."
    )
    if spec["case"] == "02_regulatory_activity":
        tf_recheck = '''import warnings
from reproducibility.workflows.regulatory_activity.evaluate_outputs import evaluate

panel_file = CASE_OUTPUT / "runs/elegans_tf/split_1/seed_1/panel_32/panels/SMITH_top32.tsv"
recheck_dir = CASE_OUTPUT / "notebook_recheck/elegans_tf"
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    _ = evaluate(
        DATA_ROOT / "regulatory_activity/elegans/splits/elegans_tf/split_1/train.h5ad",
        DATA_ROOT / "regulatory_activity/elegans/splits/elegans_tf/split_1/test.h5ad",
        panel_file, recheck_dir, 32, neighbors=5,
    )
tf_values = pd.read_csv(CASE_OUTPUT / "figure_data/figure3_c_f_values.tsv", sep="\t")
figure, axis = plt.subplots(figsize=(2.35, 2.10), facecolor="white")
_draw_bar_panel(axis, tf_values, PANEL_SPECS["c"])
figure.text(0.015, 0.985, "c", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.25, right=0.97, bottom=0.22, top=0.86)
display(figure)
plt.close(figure)'''
        time_analysis = '''from reproducibility.workflows.regulatory_activity.analysis import write_statistical_analysis

_ = write_statistical_analysis(
    CASE_OUTPUT / "figure_data/figure3_c_f_values.tsv",
    CASE_OUTPUT / "figure_data",
)
tf_values = pd.read_csv(CASE_OUTPUT / "figure_data/figure3_c_f_values.tsv", sep="\t")
figure, axis = plt.subplots(figsize=(2.35, 2.10), facecolor="white")
_draw_bar_panel(axis, tf_values, PANEL_SPECS["d"])
figure.text(0.015, 0.985, "d", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.25, right=0.97, bottom=0.22, top=0.86)
display(figure)
plt.close(figure)'''
        mirna_recheck = '''import warnings
from reproducibility.workflows.regulatory_activity.evaluate_outputs import evaluate

panel_file = CASE_OUTPUT / "runs/elegans_mirna/split_1/seed_1/panel_32/panels/SMITH_top32.tsv"
recheck_dir = CASE_OUTPUT / "notebook_recheck/elegans_mirna"
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    _ = evaluate(
        DATA_ROOT / "regulatory_activity/elegans/splits/elegans_mirna/split_1/train.h5ad",
        DATA_ROOT / "regulatory_activity/elegans/splits/elegans_mirna/split_1/test.h5ad",
        panel_file, recheck_dir, 32, neighbors=5,
    )
mirna_values = pd.read_csv(CASE_OUTPUT / "figure_data/figure3_c_f_values.tsv", sep="\t")
figure, axis = plt.subplots(figsize=(2.35, 2.10), facecolor="white")
_draw_bar_panel(axis, mirna_values, PANEL_SPECS["e"])
figure.text(0.015, 0.985, "e", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.25, right=0.97, bottom=0.22, top=0.86)
display(figure)
plt.close(figure)'''
        notebook.cells = opening_cells + [
            nbformat.v4.new_markdown_cell(
                "## Preserving cell identity and developmental progression\n\n"
                "### TF activity retains lineage identity\n\n"
                "The selected TF panel is evaluated on held-out lineages with the same 5-nearest-neighbour classifier used for the manuscript comparison. Higher accuracy means that a compact regulatory panel still separates cell identities."
            ),
            nbformat.v4.new_code_cell(tf_recheck),
            nbformat.v4.new_markdown_cell(
                "### TF activity retains developmental order\n\n"
                "Developmental time is predicted independently from the same held-out TF cells. Pearson correlation measures whether the panel preserves the continuous temporal trajectory rather than only discrete labels."
            ),
            nbformat.v4.new_code_cell(time_analysis),
            nbformat.v4.new_markdown_cell(
                "### miRNA activity retains lineage identity\n\n"
                "The same held-out analysis is repeated in the miRNA activity space, where smaller panel sizes test whether post-transcriptional regulation contains sufficient lineage information."
            ),
            nbformat.v4.new_code_cell(mirna_recheck),
            nbformat.v4.new_markdown_cell(
                "### miRNA activity retains developmental order\n\n"
                "The temporal endpoint asks whether the compact miRNA panel follows embryonic progression across held-out cells."
            ),
            nbformat.v4.new_code_cell('''mirna_values = pd.read_csv(CASE_OUTPUT / "figure_data/figure3_c_f_values.tsv", sep="\\t")
figure, axis = plt.subplots(figsize=(2.35, 2.10), facecolor="white")
_draw_bar_panel(axis, mirna_values, PANEL_SPECS["f"])
figure.text(0.015, 0.985, "f", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.25, right=0.97, bottom=0.22, top=0.86)
display(figure)
plt.close(figure)'''),
            nbformat.v4.new_markdown_cell(
                "## Developmental regulatory programs\n\n"
                "### Spatiotemporal TF modules\n\n"
                "The atlas annotation organizes regulators by tissue system and temporal module. This establishes the biological programs whose coverage is tested in the next analysis."
            ),
            nbformat.v4.new_code_cell('''modules = pd.read_csv(DATA_ROOT / "regulatory_activity/elegans/annotations/tf_spatiotemporal_modules.tsv", sep="\\t")
figure, axis = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
_draw_module_schematic(axis, modules)
figure.text(0.015, 0.985, "g", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
display(figure)
plt.close(figure)'''),
            nbformat.v4.new_markdown_cell(
                "### Coverage of developmental modules\n\n"
                "For each newly selected panel, the miss rate is the fraction of annotated modules with no selected TF. A lower value therefore indicates broader coverage of known developmental programs."
            ),
            nbformat.v4.new_code_cell('''module_coverage = pd.read_csv(CASE_OUTPUT / "figure_data/figure3_h_module_miss_rate.tsv", sep="\\t")
figure, axis = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
_draw_module_coverage(axis, module_coverage)
figure.text(0.015, 0.985, "h", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
display(figure)
plt.close(figure)'''),
            nbformat.v4.new_markdown_cell(
                "## Regulatory reconstruction and modality transfer\n\n"
                "### Reconstruction of TF co-activity\n\n"
                "The trained reconstruction head predicts held-out TF activity from the selected regulators. Agreement is measured within muscle, neuronal, pharyngeal and skin lineages using atlas-defined TF pairs."
            ),
            nbformat.v4.new_code_cell('''coactivity = pd.read_csv(CASE_OUTPUT / "figure_data/figure3_i_coactivity.tsv", sep="\\t")
lineages = ["muscle", "neuron", "pharynx", "skin"]
methods = [name for name in ("SMITH", "PERSIST") if name in set(coactivity["method"])]
colors = {"SMITH": "#2f75b5", "PERSIST": "#f2b134"}
x = np.arange(len(lineages))
bar_width = 0.72 / len(methods)

figure, axis = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
for method_index, method in enumerate(methods):
    means, errors = [], []
    for lineage in lineages:
        values = coactivity.loc[
            (coactivity["method"] == method) & (coactivity["lineage"] == lineage),
            "pearson",
        ].dropna()
        means.append(values.mean())
        errors.append(values.sem() if len(values) > 1 else 0.0)
    positions = x - 0.36 + bar_width / 2 + method_index * bar_width
    axis.bar(
        positions, means, bar_width, yerr=errors, capsize=1.5,
        color=colors[method], edgecolor="black", linewidth=0.4, label=method,
    )

axis.set(title="TF co-activity reconstruction", ylabel="Pearson agreement")
axis.set_xticks(x, [name.title() for name in lineages], rotation=25, ha="right")
axis.set_ylim(-1, 1)
axis.legend(frameon=False, fontsize=5.5)
figure.text(0.015, 0.985, "i", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
display(figure)
plt.close(figure)'''),
            nbformat.v4.new_markdown_cell(
                "### Conservation of regulatory structure across modalities\n\n"
                "Shared TFs are aggregated over matched lineages in scRNA-seq and TF-activity data, then biclustered using the TF-activity correlation matrix. Similar block structure indicates that inferred regulatory activity preserves transcriptomic organization."
            ),
            nbformat.v4.new_code_cell('''correlation = pd.read_csv(CASE_OUTPUT / "figure_data/figure3_j_tf_scrna_correlation.tsv", sep="\\t")
figure = _plot_tf_correlation(correlation, CASE_OUTPUT / "figure_data/figure3_j_tf_scrna_correlation.tsv")
figure.text(0.008, 0.985, "j", ha="left", va="top", fontsize=10, weight="bold")
display(figure)
plt.close(figure)'''),
            nbformat.v4.new_markdown_cell(
                "### Transfer from scRNA-seq into TF activity\n\n"
                "Panels selected from scRNA-seq are evaluated on held-out TF-activity lineages. The comparison with TF-selected panels tests whether gene choice transfers across molecular representations without losing cell identity."
            ),
            nbformat.v4.new_code_cell('''transfer = pd.read_csv(CASE_OUTPUT / "figure_data/figure3_k_transfer.tsv", sep="\\t")
figure, axis = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
_draw_transfer(axis, transfer)
figure.text(0.015, 0.985, "k", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
display(figure)
plt.close(figure)'''),
            closing_cell,
        ]
    else:
        notebook.cells = opening_cells + [
            nbformat.v4.new_markdown_cell(f"## Step 3: Evaluate held-out biology\n\n{spec['analysis_role']}\n\nThe next cell recomputes one held-out prediction from the newly selected panel and writes truth/prediction files. The workflow also records split-level metrics and statistical metadata. No aggregate reference output is read."), nbformat.v4.new_code_cell(analysis),
            nbformat.v4.new_markdown_cell("## Step 4: Render the manuscript panels\n\nOnly the manuscript figure images are rendered below. Intermediate tables and logs remain in the output directory but are not displayed."), nbformat.v4.new_code_cell(plotting),
            closing_cell,
        ]
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only", choices=sorted(SPECS))
    parser.add_argument("--data-root", default="data/tutorials")
    parser.add_argument("--output-root", default="outputs/tutorials")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="Reuse model outputs from a prior clean run while rebuilding downstream analyses.",
    )
    args = parser.parse_args()
    selected = {args.only: SPECS[args.only]} if args.only else SPECS
    old_environment = os.environ.copy()
    if args.execute:
        os.environ.update({
            "SMITH_TUTORIAL_DATA": str(Path(args.data_root).expanduser().resolve()),
            "SMITH_TUTORIAL_OUTPUT": str(Path(args.output_root).expanduser().resolve()),
            "SMITH_TUTORIAL_EPOCHS": str(args.epochs), "SMITH_TUTORIAL_DEVICE": args.device,
            "SMITH_TUTORIAL_REUSE_EXISTING": "1" if args.reuse_existing else "0",
        })
    try:
        for spec in selected.values():
            folder = NOTEBOOK_ROOT / spec["folder"]; folder.mkdir(parents=True, exist_ok=True)
            source = folder / f"{spec['stem']}_source.ipynb"
            executed = folder / f"{spec['stem']}_executed.ipynb"
            notebook = build_notebook(spec, args.epochs, args.device)
            nbformat.write(notebook, source)
            print(f"wrote {source.relative_to(ROOT)}")
            if args.execute:
                result = NotebookClient(nbformat.from_dict(notebook), timeout=None, kernel_name="python3", allow_errors=False).execute(cwd=str(ROOT))
                nbformat.write(result, executed)
                print(f"wrote {executed.relative_to(ROOT)}")
            elif not executed.exists():
                raise FileNotFoundError(f"Missing executed notebook {executed}; generate it on a data host with --execute.")
    finally:
        os.environ.clear(); os.environ.update(old_environment)


if __name__ == "__main__":
    main()
