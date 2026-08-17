#!/usr/bin/env python3
"""Build manuscript-figure reproduction notebooks and optionally execute them."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "docs" / "source" / "tutorials" / "notebooks"


SPECS = {
    "02_regulatory_activity": {
        "case": "02_regulatory_activity",
        "folder": "regulatory_section", "stem": "02_SMITH_Regulatory_Activity",
        "title": "Reproduce SMITH Figure 3c-f", "figure": "Figure 3c-f",
        "biology": "Which compact set of regulatory features is sufficient to preserve C. elegans cell identity and developmental progression? The TF and miRNA assays represent regulatory activity rather than a generic feature-selection benchmark: a useful panel should retain discrete lineage labels and the continuous developmental-time signal in held-out cells.",
        "data_role": "The train/test H5AD files contain lineage-aware TF or miRNA activity, cell-type labels, and absolute developmental time. Training cells are used to learn the activity representation; held-out cells test whether the selected regulators still recover biological identity and age.",
        "model_role": "SMITH is trained on the training H5AD with reconstruction, cell-type classification, and developmental-time objectives. Its learned gene ranking is then truncated to the manuscript panel sizes; no packaged aggregate ranking is used.",
        "analysis_role": "Cell-type accuracy asks whether the panel preserves discrete lineage information. Developmental-time Pearson correlation asks whether it preserves the ordered trajectory between stages. Together they distinguish a panel that merely separates classes from one that also captures progression.",
        "workflow": "reproducibility/workflows/regulatory_activity/run_tutorial.py",
        "plotter": "reproducibility/workflows/regulatory_activity/plot_figure3.py", "output": "regulatory",
        "inputs": [
            "regulatory_activity/elegans/splits/elegans_tf/split_1/train.h5ad",
            "regulatory_activity/elegans/splits/elegans_tf/split_1/test.h5ad",
            "regulatory_activity/elegans/splits/elegans_mirna/split_1/train.h5ad",
            "regulatory_activity/elegans/splits/elegans_mirna/split_1/test.h5ad",
        ],
        "arguments": ["--datasets", "elegans_tf,elegans_mirna", "--splits", "split_1", "--methods", "SMITH", "--seeds", "1", "--max-cells", "3000"],
        "plot_arguments": ["--values", "figure_data/figure3_c_f_values.tsv"],
        "tables": ["figure_data/figure3_c_f_summary.tsv"],
        "figure_panels": [
            ("Figure 3c - TF cell-type accuracy", "figures/figure3_c.png", 430),
            ("Figure 3d - TF developmental-time correlation", "figures/figure3_d.png", 430),
            ("Figure 3e - miRNA cell-type accuracy", "figures/figure3_e.png", 430),
            ("Figure 3f - miRNA developmental-time correlation", "figures/figure3_f.png", 430),
            ("Shared method legend", "figures/figure3_method_legend.png", 900),
        ],
        "paper_command": "--splits split_1,split_2,split_3,split_4,split_5 --methods SMITH,PERSIST-class,PERSIST,ActiveSVM,scGIST,scGeneFit,Spapros --baseline-root external/SMITH_baselines/GPS_tools-main/baselines --baseline-python PERSIST=/opt/envs/persist/bin/python --baseline-python PERSIST-class=/opt/envs/persist/bin/python --baseline-python scGIST=/opt/envs/scgist/bin/python --epochs 200",
        "scope": "This executed page uses one real TF split, one real miRNA split and SMITH only to keep the hosted example tractable. The paper command above regenerates Figure 3c-f with all five lineage-aware splits and manuscript baselines. Figure 3h-k additionally require versioned module, TF-pair and scRNA-to-TF transfer inputs and are not represented by substitute plots.",
    },
    "03_ribomap_transfer": {
        "case": "03_ribomap_transfer",
        "folder": "ribomap_section", "stem": "03_SMITH_RIBOMap_Transfer",
        "title": "Reproduce SMITH Figure 4c-h", "figure": "Figure 4c-h",
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
        "plot_arguments": ["--metrics", "figure_data/figure4_c_f_values.tsv", "--overlap", "figure_data/figure4_g_jaccard.tsv", "--bias", "figure_data/figure4_h_ribomap_bias.tsv"],
        "tables": ["figure_data/figure4_c_f_values.tsv", "figure_data/figure4_g_jaccard.tsv", "figure_data/figure4_h_ribomap_bias.tsv"],
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
        "scope": "The workflow reproduces the quantitative logic and layout of Figure 4c-h from newly selected Deep-RIBOmap and STARmap panels. Figure 4i is deliberately omitted unless a versioned Reactome/GO snapshot is supplied. Figure 4j-n additionally require the manuscript clean-fusion aligned H5AD and are not replaced with unrelated summaries.",
    },
    "05_agent": {
        "case": "05_agent",
        "folder": "agent_section", "stem": "05_SMITH_Agent_Evaluation",
        "title": "Reproduce SMITH Figure 6c-d", "figure": "Figure 6c-d",
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


def build_notebook(spec: dict, epochs: int, device: str) -> nbformat.NotebookNode:
    github = f"https://github.com/fym0503/SMITH/blob/main/docs/source/tutorials/notebooks/{spec['folder']}/{spec['stem']}_source.ipynb"
    setup = f'''from pathlib import Path
import hashlib, json, os, subprocess, sys
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

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
'''
    inspect = f'''inputs = {spec['inputs']!r}
input_checksums = {{}}
for relative in inputs:
    path = DATA_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing {{path}}. Run scripts/download_tutorial_data.py first.")
    input_checksums[relative] = sha256_file(path)
'''
    command = f'''command = [sys.executable, str(ROOT / {spec['workflow']!r}), "--data-root", str(DATA_ROOT), "--output-dir", str(CASE_OUTPUT), "--device", DEVICE, "--epochs", str(EPOCHS)] + {spec['arguments']!r} + ["--force"]
completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
    plot_args = []
    for index in range(0, len(spec["plot_arguments"]), 2):
        plot_args.extend([spec["plot_arguments"][index], f"str(CASE_OUTPUT / {spec['plot_arguments'][index + 1]!r})"])
    plot_pairs = ", ".join([repr(plot_args[i]) if i % 2 == 0 else plot_args[i] for i in range(len(plot_args))])
    plotting = f'''figure_dir = CASE_OUTPUT / "figures"
plot_command = [sys.executable, str(ROOT / {spec['plotter']!r}), {plot_pairs}, "--output-dir", str(figure_dir)]
subprocess.run(plot_command, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for heading, relative, width in {spec['figure_panels']!r}:
    display(Markdown(f"### {{heading}}"))
    display(Image(filename=str(CASE_OUTPUT / relative), width=width))
'''
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"}, language_info={"name": "python", "version": "3"})
    notebook.cells = [
        nbformat.v4.new_markdown_cell(f"# {spec['title']}\n\nThis notebook follows the biological workflow from real input data to a trained model, a newly selected panel, and manuscript-matched biological analyses. It does not read packaged aggregate results as tutorial inputs. [Open the editable source notebook on GitHub]({github})."),
        nbformat.v4.new_markdown_cell(f"## Biological question\n\n{spec['biology']}\n\n**How to read the endpoint:** {spec['analysis_role']}"),
        nbformat.v4.new_markdown_cell(f"## Step 0: Download the real input data\n\nDownload the versioned files and verify their checksums before training:\n\n```bash\npython scripts/download_tutorial_data.py \\\n  --case {spec['case']} \\\n  --data-root data/tutorials\n```\n\nThe hosted notebook was executed against the same staged files on pc157; Read the Docs does not download or train during its documentation build."),
        nbformat.v4.new_markdown_cell("## Configuration"), nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_markdown_cell(f"## Step 1: Inspect the biological input data\n\n{spec['data_role']}"), nbformat.v4.new_code_cell(inspect),
        nbformat.v4.new_markdown_cell(f"## Step 2: Train SMITH and select a panel\n\n{spec['model_role']}\n\nThe command below starts from the H5AD inputs above and writes a fresh model ranking, panel, evaluation, and run manifest."), nbformat.v4.new_code_cell(command),
        nbformat.v4.new_markdown_cell(f"## Step 3: Visualize the biological analysis\n\n{spec['analysis_role']}\n\nOnly the manuscript panels are rendered below. Intermediate tables remain in the output directory for reproducibility but are not printed in this notebook. The quick hosted run uses only the methods/repeats executed above; use the full command to regenerate the complete multi-method comparison."), nbformat.v4.new_code_cell(plotting),
        nbformat.v4.new_markdown_cell(f"## Full manuscript command\n\nAppend the following paper-scale arguments to the workflow command:\n\n```text\n{spec['paper_command']}\n```\n\n{spec['scope']}"),
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
    args = parser.parse_args()
    selected = {args.only: SPECS[args.only]} if args.only else SPECS
    old_environment = os.environ.copy()
    if args.execute:
        os.environ.update({
            "SMITH_TUTORIAL_DATA": str(Path(args.data_root).expanduser().resolve()),
            "SMITH_TUTORIAL_OUTPUT": str(Path(args.output_root).expanduser().resolve()),
            "SMITH_TUTORIAL_EPOCHS": str(args.epochs), "SMITH_TUTORIAL_DEVICE": args.device,
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
