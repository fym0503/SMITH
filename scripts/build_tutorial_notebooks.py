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
        "folder": "regulatory_section", "stem": "02_SMITH_Regulatory_Activity",
        "title": "Reproduce SMITH Figure 3c-f", "figure": "Figure 3c-f",
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
        "figure_prefix": "figures/figure3_c_f_reproduced",
        "paper_command": "--splits split_1,split_2,split_3,split_4,split_5 --methods SMITH,PERSIST-class,PERSIST,ActiveSVM,scGIST,scGeneFit,Spapros --baseline-root external/SMITH_baselines/GPS_tools-main/baselines --baseline-python PERSIST=/opt/envs/persist/bin/python --baseline-python PERSIST-class=/opt/envs/persist/bin/python --baseline-python scGIST=/opt/envs/scgist/bin/python --epochs 200",
        "scope": "This executed page uses one real TF split, one real miRNA split and SMITH only to keep the hosted example tractable. The paper command above regenerates Figure 3c-f with all five lineage-aware splits and manuscript baselines. Figure 3h-k additionally require versioned module, TF-pair and scRNA-to-TF transfer inputs and are not represented by substitute plots.",
    },
    "03_ribomap_transfer": {
        "folder": "ribomap_section", "stem": "03_SMITH_RIBOMap_Transfer",
        "title": "Reproduce SMITH Figure 4c-h", "figure": "Figure 4c-h",
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
        "figure_prefix": "figures/figure4_c_h_reproduced",
        "paper_command": "--methods SMITH,PERSIST-class,PERSIST,ActiveSVM,scGIST,scGeneFit,Spapros --baseline-root external/SMITH_baselines/GPS_tools-main/baselines --baseline-python PERSIST=/opt/envs/persist/bin/python --baseline-python PERSIST-class=/opt/envs/persist/bin/python --baseline-python scGIST=/opt/envs/scgist/bin/python --panel-sizes 32,64,128 --training-seeds 1,2,3,4,5 --evaluation-seeds 1,2,3,4,5 --epochs 200",
        "scope": "The workflow reproduces the quantitative logic and layout of Figure 4c-h from newly selected Deep-RIBOmap and STARmap panels. Figure 4i is deliberately omitted unless a versioned Reactome/GO snapshot is supplied. Figure 4j-n additionally require the manuscript clean-fusion aligned H5AD and are not replaced with unrelated summaries.",
    },
    "05_agent": {
        "folder": "agent_section", "stem": "05_SMITH_Agent_Evaluation",
        "title": "Reproduce SMITH Figure 6c-d", "figure": "Figure 6c-d",
        "workflow": "reproducibility/workflows/agent/run_tutorial.py",
        "plotter": "reproducibility/workflows/agent/plot_figure6.py", "output": "agent",
        "inputs": [
            "agent/liver_merfish/adata_healthy_nucseq.h5ad", "agent/liver_merfish/adata_healthy_merfish.h5ad",
            "agent/references/PSC011_C1_visium.h5ad", "agent/references/WSSS_F_IMMsp9838712_visium.h5ad",
        ],
        "arguments": ["--reference", "references/PSC011_C1_visium.h5ad", "--reference", "references/WSSS_F_IMMsp9838712_visium.h5ad", "--panel-sizes", "32,64,128", "--training-seeds", "1,2", "--max-cells", "3000"],
        "plot_arguments": ["--accuracy", "figure_data/figure6_c_cell_type_accuracy.tsv", "--expression", "figure_data/figure6_d_merfish_expression.tsv"],
        "tables": ["figure_data/figure6_c_cell_type_accuracy.tsv", "figure_data/figure6_d_merfish_expression.tsv"],
        "figure_prefix": "figures/figure6_c_d_reproduced",
        "paper_command": "--panel-sizes 32,64,128 --training-seeds 1,2,3,4,5 --epochs 200 (omit --reference to use all five manifest-listed defaults)",
        "scope": "This hosted run uses two real spatial references and two training seeds. The manuscript Figure 6c-d command uses five retrieved liver references and five training seeds. Figure 6e-j requires external probe-design backends and the validation-guided HPO run; this notebook does not fabricate those panels.",
    },
}


def build_notebook(spec: dict, epochs: int, device: str) -> nbformat.NotebookNode:
    github = f"https://github.com/fym0503/SMITH/blob/main/docs/source/tutorials/notebooks/{spec['folder']}/{spec['stem']}_source.ipynb"
    setup = f'''from pathlib import Path
import hashlib, json, os, subprocess, sys
import pandas as pd
from IPython.display import Image, display

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
rows = []
for relative in inputs:
    path = DATA_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing {{path}}. Run scripts/download_tutorial_data.py first.")
    rows.append({{"file": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}})
display(pd.DataFrame(rows))
'''
    command = f'''command = [sys.executable, str(ROOT / {spec['workflow']!r}), "--data-root", str(DATA_ROOT), "--output-dir", str(CASE_OUTPUT), "--device", DEVICE, "--epochs", str(EPOCHS)] + {spec['arguments']!r} + ["--force"]
display_command = ["python", {spec['workflow']!r}, "--data-root", "data/tutorials", "--output-dir", "outputs/tutorials/{spec['output']}", "--device", DEVICE, "--epochs", str(EPOCHS)] + {spec['arguments']!r} + ["--force"]
print(" ".join(display_command))
completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
if completed.returncode:
    print(completed.stdout)
    raise subprocess.CalledProcessError(completed.returncode, command)
manifest = json.loads((CASE_OUTPUT / "run_manifest.json").read_text())
print("Generated", manifest["manuscript_figure"], "data from fresh workflow outputs.")
'''
    tables = f'''for relative in {spec['tables']!r}:
    path = CASE_OUTPUT / relative
    print(relative)
    display(pd.read_csv(path, sep="\\t").head(20))
'''
    plot_args = []
    for index in range(0, len(spec["plot_arguments"]), 2):
        plot_args.extend([spec["plot_arguments"][index], f"str(CASE_OUTPUT / {spec['plot_arguments'][index + 1]!r})"])
    plot_pairs = ", ".join([repr(plot_args[i]) if i % 2 == 0 else plot_args[i] for i in range(len(plot_args))])
    plotting = f'''figure_prefix = CASE_OUTPUT / {spec['figure_prefix']!r}
plot_command = [sys.executable, str(ROOT / {spec['plotter']!r}), {plot_pairs}, "--output-prefix", str(figure_prefix)]
subprocess.run(plot_command, cwd=ROOT, check=True)
display(Image(filename=str(figure_prefix.with_suffix(".png"))))
print("Editable exports:", figure_prefix.with_suffix(".pdf"), figure_prefix.with_suffix(".svg"))
'''
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"}, language_info={"name": "python", "version": "3"})
    notebook.cells = [
        nbformat.v4.new_markdown_cell(f"# {spec['title']}\n\nThis notebook regenerates manuscript-matched data panels from real H5AD inputs; it does not read the packaged reference-output tables. [Open the editable source notebook on GitHub]({github})."),
        nbformat.v4.new_markdown_cell("## Configure real inputs and fresh outputs"), nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_markdown_cell("## Verify input files"), nbformat.v4.new_code_cell(inspect),
        nbformat.v4.new_markdown_cell(f"## Run the workflow for {spec['figure']}"), nbformat.v4.new_code_cell(command),
        nbformat.v4.new_markdown_cell("## Inspect newly generated figure data"), nbformat.v4.new_code_cell(tables),
        nbformat.v4.new_markdown_cell("## Draw panels from this run\n\nThe quick hosted run below has the manuscript axes and panel definitions, but only the methods/repeats actually executed above. Use the full command to regenerate the complete multi-method manuscript comparison."), nbformat.v4.new_code_cell(plotting),
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
