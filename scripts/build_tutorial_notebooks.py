#!/usr/bin/env python3
"""Build and optionally execute the manuscript-section tutorial notebooks."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "docs" / "source" / "tutorials" / "notebooks"


SETUP = '''from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


def find_repository(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "reproducibility").exists():
            return candidate
    raise RuntimeError("Run this notebook from inside a SMITH repository checkout.")


ROOT = find_repository(Path.cwd().resolve())
sys.path.insert(0, str(ROOT / "src"))

from smith.reproducibility import check_case, load_cases, run_case

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 30)
print(f"Repository: {ROOT}")
'''


RUN_CASE = '''case = load_cases()[CASE_ID]
status = check_case(case)
if status["inputs"]:
    display(pd.DataFrame(status["inputs"])[["path", "exists", "sha256_ok"]])
else:
    print("This tutorial creates its deterministic input during execution.")
assert status["ready"], "The pinned tutorial inputs are missing or have changed."

output_dir = ROOT / "outputs" / "notebooks" / CASE_ID
result = run_case(case, output_dir)
print(f"Summary written to: {result['summary_json']}")
result
'''


NOTEBOOKS = {
    "01_wmb": {
        "folder": "wmb_section",
        "stem": "01_SMITH_WMB_Panel_Selection",
        "title": "SMITH whole-mouse-brain panel selection",
        "intro": """This notebook runs a compact, deterministic version of the multi-objective panel-selection analysis underlying the whole-mouse-brain Results section. It creates an annotated expression object, trains the stochastic-gate selector on five objectives, and exports an eight-target ranking.

The compact input is designed for a runnable tutorial. It demonstrates the analysis path but does not claim to regenerate donor-level Figure 2 values.""",
        "analysis": '''selected = pd.DataFrame({
    "target": result["selected_targets"],
    "rank_score": np.arange(result["selected_panel_size"], 0, -1),
})
display(selected)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(selected["target"][::-1], selected["rank_score"][::-1], color="#3264a8")
ax.set(xlabel="Selection rank score", ylabel="Target", title="Compact SMITH target panel")
fig.tight_layout()
plt.show()
''',
        "limits": """## What this reproduces

This notebook reproduces the package path from annotated observations through multi-objective target ranking. The complete manuscript workflow additionally requires the WMB references, donor-aware splits, baseline panels, repeated seeds, transfer evaluation, ablations, and runtime benchmarking listed in `reproducibility/manifests/01_wmb.yaml`.""",
    },
    "02_regulatory_activity": {
        "folder": "regulatory_section",
        "stem": "02_SMITH_Regulatory_Activity",
        "title": "SMITH regulatory-activity preservation",
        "intro": """This notebook recomputes a representative analysis for the regulatory-activity Results section. It selects the strongest validated SMITH configuration for every dataset and panel size using the pinned five-run summary, then compares preservation of cell identity and developmental time.""",
        "analysis": '''best = pd.DataFrame(result["best_smith_by_dataset_and_panel"])
display(best.sort_values(["dataset", "panel_size"]))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for dataset, group in best.groupby("dataset"):
    ax.scatter(group["celltype_accuracy"], group["time_pearson"], s=65, label=dataset)
    for row in group.itertuples():
        ax.annotate(str(row.panel_size), (row.celltype_accuracy, row.time_pearson),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
ax.set(xlabel="Cell-type kNN accuracy", ylabel="Developmental-time Pearson r",
       title="Validated SMITH configurations")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
''',
        "limits": """## What this reproduces

The table and plot recompute the joint cell-identity/developmental-time comparison from pinned aggregate results. Full Figure 3 regeneration additionally needs lineage-aware splits, TF/miRNA activity inference, baseline training, module coverage, co-activity reconstruction, and scRNA-to-TF transfer.""",
    },
    "03_ribomap_transfer": {
        "folder": "ribomap_section",
        "stem": "03_SMITH_RIBOMap_Transfer",
        "title": "SMITH RIBOMap and STARmap transfer",
        "intro": """This notebook extracts the SMITH transfer results for modality-matched RIBOMap references and cross-modality STARmap references from a pinned five-seed benchmark table. It preserves the reported uncertainty and ranking rather than copying values into prose.""",
        "analysis": '''transfer = pd.DataFrame(result["smith_transfer_metrics"])
display(transfer)

labels = transfer["dataset"] + "\\n" + transfer["label"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(np.arange(len(transfer)), transfer["value_mean"],
       yerr=transfer["value_std"], capsize=4, color="#6c4aa5")
ax.set_xticks(np.arange(len(transfer)), labels, rotation=30, ha="right")
ax.set(ylabel="Accuracy", title="SMITH spatial-transfer performance (mean ± SD)")
fig.tight_layout()
plt.show()
''',
        "limits": """## What this reproduces

This notebook recomputes the manuscript-facing SMITH accuracy summary and its uncertainty. Complete Figure 4 regeneration also requires raw spatial objects, baseline selection, pathway enrichment, panel-overlap analysis, and aligned clean-fusion experiments.""",
    },
    "04_inhouse_disease": {
        "folder": "disease_section",
        "stem": "04_SMITH_InHouse_Disease_Transfer",
        "title": "SMITH human disease transfer robustness",
        "intro": """Participant-level human brain data are controlled access. This notebook therefore performs an auditable analysis of the released de-identified aggregate table: it compares improvements in rank correlation and top-64 panel overlap across cohort transfers.""",
        "analysis": '''robustness = pd.DataFrame(result["transfer_robustness"])
display(robustness)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(robustness["comparison"], robustness["delta_spearman_mean"], color="#218c74")
axes[0].set(ylabel="Δ Spearman correlation", title="Expression-rank transfer")
axes[1].bar(robustness["comparison"], robustness["delta_top64_mean"], color="#cc8e35")
axes[1].set(ylabel="Δ top-64 overlap", title="Panel-overlap transfer")
for ax in axes:
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0, color="black", linewidth=0.8)
fig.tight_layout()
plt.show()
''',
        "limits": """## Data boundary

The notebook reproduces only analyses supported by the de-identified aggregate release. Participant-level panels, UMAPs, imputation results, and spatial gene examples require authorized inputs; this is recorded explicitly in `reproducibility/manifests/04_inhouse_disease.yaml`.""",
    },
    "05_agent": {
        "folder": "agent_section",
        "stem": "05_SMITH_Agent_Evaluation",
        "title": "SMITH-Agent evaluation and probe feasibility",
        "intro": """This notebook covers two auditable SMITH-Agent stages: multi-reference panel evaluation on a locked spatial test set and integrated probe-feasibility filtering. It recomputes group statistics and feasibility pass rates from pinned tables.""",
        "analysis": '''accuracy = pd.DataFrame(result["multi_reference_accuracy"])
pass_rates = pd.Series(result["feasibility_pass_rates"], name="pass_rate")
display(accuracy)
display(pass_rates.to_frame())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for panel, group in accuracy.groupby("panel"):
    axes[0].errorbar(group["panel_size"], group["mean"], yerr=group["std"],
                     marker="o", capsize=3, label=panel)
axes[0].set(xlabel="Panel size", ylabel="Cell-type accuracy", title="Multi-reference evaluation")
axes[0].legend(frameon=False, fontsize=8)
axes[1].bar(pass_rates.index, pass_rates.values, color="#b33939")
axes[1].set(ylim=(0, 1), ylabel="Fraction passing", title="Probe-feasibility gates")
axes[1].tick_params(axis="x", rotation=35)
fig.tight_layout()
plt.show()
''',
        "limits": """## What this reproduces

The notebook recomputes multi-reference accuracy and feasibility summaries. Full Figure 6 also requires public-reference retrieval, repeated SMITH runs, a locked MERFISH test, ODT, BLAST indexes, ProbeDealer resources, and Pareto-guided hyperparameter search.""",
    },
}


def build_notebook(case_id: str, spec: dict[str, str]) -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update(
        kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
        language_info={"name": "python", "version": "3"},
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(f"# {spec['title']}\n\n{spec['intro']}"),
        nbformat.v4.new_markdown_cell(
            "[Open the editable source notebook on GitHub]("
            f"https://github.com/fym0503/SMITH/blob/main/docs/source/tutorials/notebooks/"
            f"{spec['folder']}/{spec['stem']}_source.ipynb)"
        ),
        nbformat.v4.new_markdown_cell(
            "## Setup\n\nRun this notebook from a cloned SMITH repository with "
            "`pip install -e '.[notebooks]'`. All inputs are checksum-validated before analysis."
        ),
        nbformat.v4.new_code_cell(SETUP),
        nbformat.v4.new_code_cell(f'CASE_ID = "{case_id}"\n' + RUN_CASE),
        nbformat.v4.new_markdown_cell("## Analysis"),
        nbformat.v4.new_code_cell(spec["analysis"]),
        nbformat.v4.new_markdown_cell(spec["limits"]),
    ]
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="also write executed notebooks for the docs")
    parser.add_argument("--only", choices=sorted(NOTEBOOKS), help="build one notebook")
    args = parser.parse_args()

    selected = {args.only: NOTEBOOKS[args.only]} if args.only else NOTEBOOKS
    generated_paths: list[Path] = []
    for case_id, spec in selected.items():
        folder = NOTEBOOK_ROOT / spec["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        source_path = folder / f"{spec['stem']}_source.ipynb"
        executed_path = folder / f"{spec['stem']}_executed.ipynb"
        notebook = build_notebook(case_id, spec)
        nbformat.write(notebook, source_path)
        generated_paths.append(source_path)
        print(f"wrote {source_path.relative_to(ROOT)}")
        if args.execute:
            executed = NotebookClient(
                nbformat.from_dict(notebook), timeout=600, kernel_name="python3", allow_errors=False
            ).execute(cwd=str(ROOT))
            nbformat.write(executed, executed_path)
            generated_paths.append(executed_path)
            print(f"wrote {executed_path.relative_to(ROOT)}")

    for path in generated_paths:
        notebook = nbformat.read(path, as_version=4)
        if path.name.endswith("_source.ipynb"):
            assert all(cell.get("execution_count") is None for cell in notebook.cells if cell.cell_type == "code")
            assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
        else:
            code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
            assert code_cells and all(cell.get("execution_count") is not None for cell in code_cells)
            assert any(cell.get("outputs") for cell in code_cells)


if __name__ == "__main__":
    main()
