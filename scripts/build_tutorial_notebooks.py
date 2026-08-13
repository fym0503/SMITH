#!/usr/bin/env python3
"""Build source notebooks and, only when requested, run real tutorial workflows."""

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
        "folder": "regulatory_section",
        "stem": "02_SMITH_Regulatory_Activity",
        "title": "Run the regulatory-activity workflow",
        "inputs": [
            "regulatory_activity/elegans/splits/elegans_tf/split_1/train.h5ad",
            "regulatory_activity/elegans/splits/elegans_tf/split_1/test.h5ad",
        ],
        "workflow": "reproducibility/workflows/regulatory_activity/run_tutorial.py",
        "output": "regulatory",
        "arguments": ["--dataset", "elegans_tf", "--split", "split_1", "--panel-size", "32", "--max-cells", "3000"],
        "panel": "smith/panel_top32.csv",
        "ranking_glob": "smith/ranking/epoch_*.csv",
        "metrics": "evaluation/metrics.tsv",
        "plot_index": "metric",
        "plot_group": None,
        "scope": "The notebook uses one real TF split and tutorial-scale epochs. The manuscript result repeats the workflow across TF/miRNA splits, seeds, panel sizes and baselines.",
    },
    "03_ribomap_transfer": {
        "folder": "ribomap_section",
        "stem": "03_SMITH_RIBOMap_Transfer",
        "title": "Run RIBOMap transfer",
        "inputs": [
            "ribomap_transfer/ribomap/deep_brain_ribomap.h5ad",
            "ribomap_transfer/ribomap/mouse_brain_ribomap_rep1.h5ad",
            "ribomap_transfer/ribomap/mouse_brain_ribomap_rep2.h5ad",
        ],
        "workflow": "reproducibility/workflows/ribomap_transfer/run_tutorial.py",
        "output": "ribomap",
        "arguments": ["--panel-size", "64", "--max-cells", "3000"],
        "panel": "smith/panel_top64.csv",
        "ranking_glob": "smith/ranking/epoch_*.csv",
        "metrics": "evaluation/metrics.tsv",
        "plot_index": "dataset",
        "plot_group": "method",
        "scope": "This tutorial recomputes shared genes, SMITH and a variance baseline for deep-brain to mouse-brain transfer. The manuscript adds STARmap directions, more baselines and five seeds.",
    },
    "05_agent": {
        "folder": "agent_section",
        "stem": "05_SMITH_Agent_Evaluation",
        "title": "Run SMITH-Agent panel evaluation",
        "inputs": [
            "agent/liver_merfish/adata_healthy_nucseq.h5ad",
            "agent/liver_merfish/adata_healthy_merfish.h5ad",
            "agent/references/PSC011_C1_visium.h5ad",
            "agent/references/WSSS_F_IMMsp9838712_visium.h5ad",
        ],
        "workflow": "reproducibility/workflows/agent/run_tutorial.py",
        "output": "agent",
        "arguments": ["--panel-size", "64", "--max-cells", "3000"],
        "panel": "aggregation/integrated_top_64_panel.tsv",
        "ranking_glob": "aggregation/integrated_panel_rank.tsv",
        "metrics": "evaluation/metrics.tsv",
        "plot_index": "metric",
        "plot_group": "panel",
        "metric_filter": ["cell_type_accuracy", "spatial_pearson"],
        "scope": "The tutorial trains a source panel and two real spatial-reference panels before rank aggregation. The paper-scale run uses five references and more seeds. Probe filtering is a separate backend-dependent stage and no pass rate is fabricated here.",
    },
}


def build_notebook(spec: dict, epochs: int, device: str) -> nbformat.NotebookNode:
    github = f"https://github.com/fym0503/SMITH/blob/main/docs/source/tutorials/notebooks/{spec['folder']}/{spec['stem']}_source.ipynb"
    setup = f'''from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

def find_repository(start):
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "reproducibility").exists():
            return candidate
    raise RuntimeError("Run this notebook inside a SMITH repository checkout.")

ROOT = find_repository(Path.cwd().resolve())
DATA_ROOT = Path(os.environ.get("SMITH_TUTORIAL_DATA", "data/tutorials")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("SMITH_TUTORIAL_OUTPUT", "outputs/tutorials")).expanduser().resolve()
CASE_OUTPUT = OUTPUT_ROOT / {spec['output']!r}
EPOCHS = int(os.environ.get("SMITH_TUTORIAL_EPOCHS", {str(epochs)!r}))
DEVICE = os.environ.get("SMITH_TUTORIAL_DEVICE", {device!r})
print("Input and output roots are configured. Set SMITH_TUTORIAL_DATA/OUTPUT to override them.")

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
'''
    inspect = f'''inputs = {[str(value) for value in spec['inputs']]!r}
input_rows = []
for relative in inputs:
    path = DATA_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing {{path}}. Run scripts/download_tutorial_data.py first.")
    digest = sha256_file(path)
    input_rows.append({{"file": relative, "bytes": path.stat().st_size, "sha256": digest}})
display(pd.DataFrame(input_rows))
'''
    command = f'''command = [
    sys.executable, str(ROOT / {spec['workflow']!r}),
    "--data-root", str(DATA_ROOT),
    "--output-dir", str(CASE_OUTPUT),
    "--device", DEVICE,
    "--epochs", str(EPOCHS),
    "--seed", "42",
] + {spec['arguments']!r}
display_command = [
    "python", {spec['workflow']!r}, "--data-root", "data/tutorials",
    "--output-dir", "outputs/tutorials/{spec['output']}", "--device", DEVICE,
    "--epochs", str(EPOCHS), "--seed", "42",
] + {spec['arguments']!r}
print(" ".join(display_command))
completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
if completed.returncode:
    print(completed.stdout)
    raise subprocess.CalledProcessError(completed.returncode, command)
manifest = json.loads((CASE_OUTPUT / "run_manifest.json").read_text())
print("Completed:", manifest["workflow"], "->", "outputs/tutorials/{spec['output']}/run_manifest.json")
'''
    analyze = f'''panel_path = CASE_OUTPUT / {spec['panel']!r}
ranking_path = sorted(CASE_OUTPUT.glob({spec['ranking_glob']!r}))[-1]
metrics_path = CASE_OUTPUT / {spec['metrics']!r}
panel = pd.read_csv(panel_path, sep="\\t" if panel_path.suffix == ".tsv" else ",")
ranking = pd.read_csv(ranking_path, sep="\\t" if ranking_path.suffix == ".tsv" else ",")
metrics = pd.read_csv(metrics_path, sep="\\t")
print("Generated panel:", panel_path.relative_to(CASE_OUTPUT))
display(panel.head(15))
print("Generated ranking:", ranking_path.relative_to(CASE_OUTPUT))
display(ranking.head(10))
display(metrics)
'''
    metric_filter = spec.get("metric_filter")
    filter_line = f'metrics = metrics[metrics["metric"].isin({metric_filter!r})].copy()\n' if metric_filter else ""
    if spec["plot_group"]:
        plot = f'''plot_data = metrics.pivot_table(index={spec['plot_index']!r}, columns={spec['plot_group']!r}, values="value", aggfunc="mean")
ax = plot_data.plot(kind="bar", figsize=(10, 4.5))
ax.set_ylabel("value")
ax.set_title({spec['title']!r})
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()
'''
    else:
        plot = f'''ax = metrics.set_index("metric")["value"].plot(kind="bar", figsize=(9, 4.5), color="#2f6690")
ax.set_ylabel("value")
ax.set_title({spec['title']!r})
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()
'''
    plot = filter_line + plot
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"}, language_info={"name": "python", "version": "3"})
    notebook.cells = [
        nbformat.v4.new_markdown_cell(f"# {spec['title']}\n\nThis notebook starts from real H5AD inputs and writes a fresh ranking, panel, evaluation and run manifest. [Open the editable source notebook on GitHub]({github})."),
        nbformat.v4.new_markdown_cell("## Set data and output locations"),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_markdown_cell("## Check real input files"),
        nbformat.v4.new_code_cell(inspect),
        nbformat.v4.new_markdown_cell("## Run the workflow"),
        nbformat.v4.new_code_cell(command),
        nbformat.v4.new_markdown_cell("## Inspect newly generated outputs"),
        nbformat.v4.new_code_cell(analyze),
        nbformat.v4.new_markdown_cell("## Analyze this run"),
        nbformat.v4.new_code_cell(plot),
        nbformat.v4.new_markdown_cell("## Tutorial run versus manuscript run\n\n" + spec["scope"]),
    ]
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run workflows and write executed notebooks.")
    parser.add_argument("--only", choices=sorted(SPECS))
    parser.add_argument("--data-root", default="data/tutorials")
    parser.add_argument("--output-root", default="outputs/tutorials")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.execute and not Path(args.data_root).expanduser().exists():
        raise FileNotFoundError(f"--data-root does not exist: {args.data_root}")
    selected = {args.only: SPECS[args.only]} if args.only else SPECS
    old_environment = os.environ.copy()
    if args.execute:
        os.environ["SMITH_TUTORIAL_DATA"] = str(Path(args.data_root).expanduser().resolve())
        os.environ["SMITH_TUTORIAL_OUTPUT"] = str(Path(args.output_root).expanduser().resolve())
        os.environ["SMITH_TUTORIAL_EPOCHS"] = str(args.epochs)
        os.environ["SMITH_TUTORIAL_DEVICE"] = args.device
    try:
        for case_id, spec in selected.items():
            folder = NOTEBOOK_ROOT / spec["folder"]
            folder.mkdir(parents=True, exist_ok=True)
            source_path = folder / f"{spec['stem']}_source.ipynb"
            executed_path = folder / f"{spec['stem']}_executed.ipynb"
            notebook = build_notebook(spec, args.epochs, args.device)
            nbformat.write(notebook, source_path)
            print(f"wrote {source_path.relative_to(ROOT)}")
            if args.execute:
                executed = NotebookClient(nbformat.from_dict(notebook), timeout=None, kernel_name="python3", allow_errors=False).execute(cwd=str(ROOT))
                nbformat.write(executed, executed_path)
                print(f"wrote {executed_path.relative_to(ROOT)}")
            elif not executed_path.exists():
                raise FileNotFoundError(
                    f"Missing executed notebook {executed_path}. Generate it on a data host with --execute; "
                    "CI and Read the Docs must not manufacture unexecuted copies."
                )
    finally:
        os.environ.clear()
        os.environ.update(old_environment)


if __name__ == "__main__":
    main()
