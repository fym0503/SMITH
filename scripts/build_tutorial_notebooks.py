#!/usr/bin/env python3
"""Build biological-question notebooks and optionally execute them."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from memory_tutorial_notebooks import agent_notebook, ribomap_notebook
from paper_tutorial_parameters import apply_paper_parameters
from tutorial_reader_copy import apply_reader_copy
from tf_tutorial_parameters import apply_tf_parameters


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "docs" / "source" / "tutorials" / "notebooks"


SPECS = {
    "02_regulatory_activity": {
        "case": "02_regulatory_activity",
        "folder": "regulatory_section",
        "stem": "02_SMITH_Regulatory_Activity",
        "title": "Regulatory programs and cross-modality transfer in C. elegans",
        "biology": "Which TF panels preserve cell identity and developmental timing in held-out lineages?",
        "data_role": "Five lineage-separated TF train/test splits, atlas module annotations and a scRNA-seq reference support the evaluation.",
        "model_role": "Train one TF panel ranking per lineage split with reconstruction, classification and developmental-time objectives.",
        "analysis_role": "Evaluate cell identity and developmental timing, then compare TF correlation structure and scRNA-to-TF transfer.",
        "inputs": [
            f"regulatory_activity/elegans/splits/elegans_tf/split_{split}/{part}.h5ad"
            for split in range(1, 6) for part in ("train", "test")
        ] + [
            "regulatory_activity/elegans/annotations/tf_spatiotemporal_modules.tsv",
            "regulatory_activity/elegans/reference/elegans_scrna.h5ad",
        ],
    },
    "03_ribomap_transfer": {
        "case": "03_ribomap_transfer",
        "folder": "ribomap_section",
        "stem": "03_SMITH_RIBOMap_Transfer",
        "title": "Cross-modality brain panel transfer to RIBOMap",
        "biology": "Can compact panels selected from transcriptomic and translatomic references recover cell-type and brain-region information in RIBOMap?",
    },
    "05_agent": {
        "case": "05_agent",
        "folder": "agent_section",
        "stem": "05_SMITH_Agent_Evaluation",
        "title": "Liver panel design with five spatial references",
        "biology": "How do spatial references change the genes selected for a liver MERFISH panel?",
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

from reproducibility.workflows.common import ranked_genes, resolve_tutorial_profile, run_smith, write_json, write_panel_genes
from reproducibility.workflows.figure_style import configure
from reproducibility.workflows.regulatory_activity.analysis import statistical_analysis
from reproducibility.workflows.regulatory_activity.evaluate_outputs import evaluate_loaded
from reproducibility.workflows.regulatory_activity.paper_analysis import (
    coactivity_from_objects, tf_scrna_correlation_from_objects,
)
from reproducibility.workflows.regulatory_activity.plot_figure3 import (
            PANEL_SPECS, _draw_bar_panel, _draw_coactivity,
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
DEVICE = os.environ.get("SMITH_TUTORIAL_DEVICE", {device!r})
TUTORIAL_PROFILE = os.environ.get("SMITH_TUTORIAL_PROFILE", "TF_profile")
PROFILE = resolve_tutorial_profile(TUTORIAL_PROFILE)
PROFILE_PARAMETERS = PROFILE["parameters"]
EPOCHS = int(os.environ.get("SMITH_TUTORIAL_EPOCHS", PROFILE_PARAMETERS["epochs"]))
MAX_CELLS = int(os.environ.get("SMITH_TUTORIAL_MAX_CELLS", str(PROFILE_PARAMETERS["max_cells"])))
BATCH_SIZE = int(os.environ.get("SMITH_TUTORIAL_BATCH_SIZE", str(PROFILE_PARAMETERS["batch_size"])))
TRAINING_SEED = int(PROFILE_PARAMETERS["training_seed"])
FIGURE_DATA.mkdir(parents=True, exist_ok=True)
configure()
'''
    inspect = f'''relative_inputs = {spec['inputs']!r}
input_paths = {{name: DATA_ROOT / name for name in relative_inputs}}
for name, path in input_paths.items():
    if not path.is_file():
        raise FileNotFoundError(f"Missing {{path}}. Place the licensed input files listed in reproducibility/data_manifest.yaml under DATA_ROOT.")
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
    training = '''benchmark_sizes = {key: tuple(value) for key, value in PROFILE_PARAMETERS["benchmark_sizes"].items()}
training_sizes = {key: tuple(value) for key, value in PROFILE_PARAMETERS["training_sizes"].items()}
loaded_data = {
    "elegans_tf": (tf_train, tf_test),
    "elegans_mirna": (mirna_train, mirna_test),
}
training_sizes = {key: value for key, value in training_sizes.items() if key in PROFILE_PARAMETERS["datasets"]}
benchmark_sizes = {key: value for key, value in benchmark_sizes.items() if key in PROFILE_PARAMETERS["datasets"]}
training_runs = {"elegans_tf": {}, "elegans_mirna": {}}
panel_records = []

for dataset, sizes in training_sizes.items():
    train_adata, _ = loaded_data[dataset]
    for panel_size in sizes:
        run_dir = CASE_OUTPUT / "runs" / dataset / "split_1" / f"seed_{TRAINING_SEED}" / f"panel_{panel_size}" / "SMITH"
        result = run_smith(
            adata_file=input_paths[
                "regulatory_activity/elegans/splits/" + dataset + "/split_1/train.h5ad"
            ],
            output_dir=run_dir,
            tasks="recon,cls,standard_coordination,time" if dataset == "elegans_tf" else "recon,cls,time",
            task_name=f"{dataset}_split_1_panel{panel_size}_seed{TRAINING_SEED}",
            panel_size=panel_size, epochs=EPOCHS, device=DEVICE, seed=TRAINING_SEED,
            batch_size=BATCH_SIZE, time_label=PROFILE_PARAMETERS["time_label"], max_cells=MAX_CELLS,
            sampling_strategy="celltype",
            save_model=dataset == "elegans_tf" and panel_size == 32,
            tutorial_profile=TUTORIAL_PROFILE, force=True, include_in_memory=True,
        )
        panel_path = run_dir.parent / "panels" / f"SMITH_top{panel_size}.tsv"
        write_panel_genes(result["panel_genes"], panel_path)
        training_runs[dataset][panel_size] = result
        panel_records.append({
            "dataset": dataset, "split": "split_1", "training_seed": TRAINING_SEED,
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
            time_column=PROFILE_PARAMETERS["time_label"], neighbors=5, output_dir=evaluation_dir,
        )
        benchmark_rows.append({
            "dataset": dataset, "split": "split_1", "training_seed": TRAINING_SEED,
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

    cells = [
        nbformat.v4.new_markdown_cell(
            f"# {spec['title']}\n\n"
            "Train TF panels across five lineage splits and evaluate cell identity, developmental timing and cross-modality transfer. "
            f"[Open the editable source notebook on GitHub]({github})."
        ),
        nbformat.v4.new_markdown_cell(
            f"## Biological question\n\n{spec['biology']}\n\n**Biological endpoints:** {spec['analysis_role']}"
        ),
        nbformat.v4.new_markdown_cell(
            "## Prepare the input data\n\n"
            "Place the TF split files and scRNA-seq reference listed in `reproducibility/data_manifest.yaml` under one data directory, then set `SMITH_TUTORIAL_DATA` to that directory before starting Jupyter. The input manifest includes file sizes and SHA-256 checksums."
        ),
        nbformat.v4.new_markdown_cell("## Configuration"),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_markdown_cell(f"## Load the activity atlases and biological annotations\n\n{spec['data_role']}"),
        nbformat.v4.new_code_cell(inspect),
        nbformat.v4.new_markdown_cell(
            f"## Train SMITH and generate panels\n\n{spec['model_role']} Save the ranking and panel genes with the run parameters."
        ),
        nbformat.v4.new_code_cell(training),
        nbformat.v4.new_markdown_cell(
            "## Evaluate held-out cell identity and developmental time\n\n"
            "The classifier and time regressor operate directly on each newly selected panel. Their predictions are retained in memory and also saved for provenance."
        ),
        nbformat.v4.new_code_cell(evaluation),
        nbformat.v4.new_markdown_cell(
            "### Cell identity and developmental-time metrics\n\nCompare cell-type accuracy and developmental-time correlation across panel sizes."
        ),
        nbformat.v4.new_markdown_cell(
            "## Developmental regulatory programs\n\n### Spatiotemporal TF modules\n\n"
            "The atlas groups regulators by tissue and developmental phase, defining the programs whose representation is tested next."
        ),
        nbformat.v4.new_code_cell('''module_annotation = modules.copy()
module_annotation["module_id"] = module_annotation["module_id"].astype(str)
module_annotation.to_csv(FIGURE_DATA / "figure3_g_module_annotations.tsv", sep="\\t", index=False)
'''),
        nbformat.v4.new_markdown_cell(
            "## Regulatory structure and modality transfer\n\n"
            "Compare TF correlation structure across molecular modalities, then evaluate panels selected from scRNA-seq on held-out TF activity."
        ),
        nbformat.v4.new_code_cell('''tf32 = training_runs["elegans_tf"][32]
coactivity = coactivity_from_objects(
    tf_train, tf_test, tf32["panel_genes"], pairs=tf_pairs,
    checkpoint_file=tf32["checkpoint_file"], method="SMITH", seed=TRAINING_SEED,
)
coactivity.to_csv(FIGURE_DATA / "figure3_i_coactivity.tsv", sep="\\t", index=False)
'''),
        nbformat.v4.new_markdown_cell(
            "### Conservation of regulatory structure across modalities\n\n"
            "Shared TFs are aggregated over matched lineages in scRNA-seq and TF activity. Biclustering exposes whether the two modalities retain similar regulatory blocks."
        ),
        nbformat.v4.new_code_cell('''tf_combined = ad.concat([tf_train, tf_test], axis=0, join="inner", merge="same", index_unique=None)
tf_combined.var_names = tf_train.var_names.copy()
correlation, correlation_matrices = tf_scrna_correlation_from_objects(scrna, tf_combined)
correlation.to_csv(FIGURE_DATA / "figure3_j_tf_scrna_correlation.tsv", sep="\\t", index=False)
np.savez_compressed(FIGURE_DATA / "figure3_j_correlation_matrices.npz", **correlation_matrices)
'''),
        nbformat.v4.new_markdown_cell(
            "### Transfer from scRNA-seq into TF activity\n\n"
            "A new panel is now learned from the loaded scRNA-seq reference, then evaluated directly on the held-out TF-activity split. Compare five-split SMITH results for panels selected from scRNA-seq and TF activity."
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
    output_dir=CASE_OUTPUT / f"runs/transfer_scRNA/seed_{TRAINING_SEED}/SMITH",
    tasks="recon,cls,time", task_name=f"elegans_scrna_to_tf_seed{TRAINING_SEED}",
    panel_size=128, epochs=EPOCHS, device=DEVICE, seed=TRAINING_SEED,
    batch_size=BATCH_SIZE, time_label="absolute_time", max_cells=MAX_CELLS,
    sampling_strategy="celltype", tutorial_profile=TUTORIAL_PROFILE,
    force=True, include_in_memory=True,
)
transfer_rows = []
for panel_size in PROFILE_PARAMETERS["transfer_panel_sizes"]:
    panel_genes = ranked_genes(transfer_run["ranking_frame"], panel_size)
    panel_path = Path(transfer_run["output_dir"]) / "panels" / f"RNA_TF_top{panel_size}.tsv"
    write_panel_genes(panel_genes, panel_path)
    metrics, celltype_prediction, time_prediction = evaluate_loaded(
        tf_train, tf_test, panel_genes, panel_size, time_column="absolute_time", neighbors=5,
        output_dir=Path(transfer_run["output_dir"]) / "evaluation" / f"top{panel_size}",
    )
    transfer_rows.append({
    "split": "split_1", "training_seed": TRAINING_SEED, "method": "SMITH",
        "source_modality": "RNA-TF", "panel_size": panel_size, **metrics["metrics"],
    })
for row in benchmark_rows:
    if row["dataset"] == "elegans_tf":
        transfer_rows.append({**row, "source_modality": "TF-TF"})
transfer = pd.json_normalize(transfer_rows)
transfer.to_csv(FIGURE_DATA / "figure3_k_transfer.tsv", sep="\\t", index=False)
'''),
        nbformat.v4.new_markdown_cell(
            "## Render tutorial-scale Figure 3 panels\n\n"
            "Render the prediction, module, correlation and transfer results from this analysis."
        ),
        nbformat.v4.new_code_cell('''
for letter, values, spec in (
    ("c", tf_values, PANEL_SPECS["c"]),
    ("d", tf_values, PANEL_SPECS["d"]),
    ("e", mirna_values, PANEL_SPECS["e"]),
    ("f", mirna_values, PANEL_SPECS["f"]),
):
    figure, axis = plt.subplots(figsize=(2.35, 2.10), facecolor="white")
    _draw_bar_panel(axis, values, spec)
    figure.text(0.015, 0.985, letter, ha="left", va="top", fontsize=10, weight="bold")
    figure.subplots_adjust(left=0.25, right=0.97, bottom=0.22, top=0.86)
    display(figure)
    plt.close(figure)

figure, axis = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
_draw_module_schematic(axis, modules)
figure.text(0.015, 0.985, "g", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
display(figure)
plt.close(figure)


figure, axis = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
_draw_coactivity(axis, coactivity)
figure.text(0.015, 0.985, "i", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
display(figure)
plt.close(figure)

figure, axes = plt.subplots(1, 2, figsize=(4.7, 2.25), facecolor="white")
for axis, key, title in zip(axes, ("scrna", "tf"), ("scRNA-seq", "TF activity")):
    image = axis.imshow(correlation_matrices[key], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    image.set_rasterized(True)
    axis.set_title(title, fontsize=8.5, pad=4)
    axis.set_xticks([])
    axis.set_yticks([])
axes[1].text(0.98, 0.02, f"corr = {float(correlation.iloc[0]['mean_rowwise_pearson']):.2f}",
             transform=axes[1].transAxes, ha="right", va="bottom", fontsize=7,
             bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
figure.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.86, wspace=0.08)
figure.text(0.008, 0.985, "j", ha="left", va="top", fontsize=10, weight="bold")
display(figure)
plt.close(figure)

figure, axis = plt.subplots(figsize=(2.55, 2.25), facecolor="white")
_draw_transfer(axis, transfer)
figure.text(0.015, 0.985, "k", ha="left", va="top", fontsize=10, weight="bold")
figure.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.85)
display(figure)
plt.close(figure)
'''),
        nbformat.v4.new_markdown_cell(
            "## Record the run\n\nThe manifest records the inputs and artifacts after all analyses finish. It is never read as an analysis input."
        ),
        nbformat.v4.new_code_cell('''def serializable_run(run):
    return {key: value for key, value in run.items() if key not in {"ranking_frame", "panel_genes"}}

write_json(CASE_OUTPUT / "run_manifest.json", {
    "case": "02_regulatory_activity",
    "inputs": [{"path": name, "sha256": digest} for name, digest in input_checksums.items()],
    "configuration": {"tutorial_profile": TUTORIAL_PROFILE, "profile_parameters": PROFILE_PARAMETERS,
                       "epochs": EPOCHS, "device": DEVICE, "max_cells": MAX_CELLS, "seed": TRAINING_SEED},
    "training_runs": [
        serializable_run(run)
        for dataset_runs in training_runs.values() for run in dataset_runs.values()
    ] + [serializable_run(transfer_run)],
    "outputs": {
        "benchmark": str(FIGURE_DATA / "figure3_c_f_values.tsv"),
        "coactivity": str(FIGURE_DATA / "figure3_i_coactivity.tsv"),
        "tf_scrna_correlation": str(FIGURE_DATA / "figure3_j_tf_scrna_correlation.tsv"),
        "transfer": str(FIGURE_DATA / "figure3_k_transfer.tsv"),
    },
})'''),
        nbformat.v4.new_markdown_cell(
            "## Full manuscript command\n\nThe command-line workflow remains the entry point for all five splits and manuscript baselines:\n\n"
            "```bash\npython reproducibility/workflows/regulatory_activity/run_tutorial.py \\\n  --data-root data/tutorials \\\n  --output-dir outputs/paper/regulatory \\\n  --datasets elegans_tf,elegans_mirna \\\n  --splits split_1,split_2,split_3,split_4,split_5 \\\n  --methods SMITH,PERSIST-class,PERSIST,ActiveSVM,scGIST,scGeneFit,Spapros \\\n  --paper-analyses --epochs 150 --seeds 42 --batch-size 1024\n```\n\n"
            "This command runs five lineage splits with the listed comparison methods."
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
        language_info={"name": "python", "version": "3"},
    )
    return notebook


def build_notebook(spec: dict, epochs: int, device: str) -> nbformat.NotebookNode:
    """Build one of the three maintained biological tutorial notebooks."""
    if spec["case"] == "02_regulatory_activity":
        notebook = apply_tf_parameters(
            apply_paper_parameters(build_regulatory_notebook(spec, epochs, device), "tf")
        )
        return apply_reader_copy(notebook, "tf")
    if spec["case"] == "03_ribomap_transfer":
        return apply_reader_copy(
            apply_paper_parameters(ribomap_notebook(spec, epochs, device), "ribomap"), "ribomap"
        )
    if spec["case"] == "05_agent":
        return apply_reader_copy(
            apply_paper_parameters(agent_notebook(spec, epochs, device), "agent"), "agent"
        )
    raise KeyError(f"Unknown tutorial case: {spec['case']}")

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
