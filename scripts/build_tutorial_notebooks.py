#!/usr/bin/env python3
"""Build executable notebooks from real paper-derived result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "docs" / "source" / "tutorials" / "notebooks"

SETUP = '''from pathlib import Path
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

def find_repository(start):
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "reproducibility").exists():
            return candidate
    raise RuntimeError("Run this notebook from inside a SMITH repository checkout.")

ROOT = find_repository(Path.cwd().resolve())
plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 40)

def verify_fixture(relative_path, expected_sha256):
    path = ROOT / "reproducibility" / relative_path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected_sha256, f"Checksum mismatch for {path}: {digest}"
    print(f"Verified {relative_path} ({digest[:12]}...)")
    return path
'''

NOTEBOOKS = {
    "02_regulatory_activity": {
        "folder": "regulatory_section", "stem": "02_SMITH_Regulatory_Activity",
        "title": "SMITH regulatory-activity preservation",
        "intro": "This notebook reads the exact five-run summary produced by the C. elegans regulatory-activity workflow. It compares SMITH and baselines across the reported cell-identity and developmental-time metrics.",
        "analysis": '''path = verify_fixture("fixtures/elegans_five_run_results_summary.csv", "0a6f4d3fbc75057b58f5dcd9271dbf0f426ba93f4b97274d59fdc0d6b152d01d")
df = pd.read_csv(path)
summary = (df.groupby(["dataset", "method"], as_index=False)
             .agg(celltype_accuracy=("celltype_knn_accuracy_mean", "max"),
                  time_pearson=("time_knn_pearson_mean", "max")))
smith = summary[summary["method"].str.startswith("SMITH")].sort_values(["dataset", "celltype_accuracy"], ascending=[True, False])
display(smith)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for dataset, group in summary.groupby("dataset"):
    axes[0].plot(group["method"], group["celltype_accuracy"], marker="o", label=dataset)
    axes[1].plot(group["method"], group["time_pearson"], marker="o", label=dataset)
for ax, ylabel, title in zip(axes, ["Cell-type kNN accuracy", "Developmental-time Pearson r"], ["Identity preservation", "Time preservation"]):
    ax.set_ylabel(ylabel); ax.set_title(title); ax.tick_params(axis="x", rotation=65); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); plt.show()
''',
        "limits": "The bundled CSV is a real completed five-run output. Re-running Figure 3 from raw atlas data still requires the source datasets, activity inference, donor-aware splits and baseline workflows. The exact generation scripts are archived under `reproducibility/workflows/regulatory_activity/`.",
    },
    "03_ribomap_transfer": {
        "folder": "ribomap_section", "stem": "03_SMITH_RIBOMap_Transfer",
        "title": "SMITH RIBOMap and STARmap transfer",
        "intro": "This notebook reads the real benchmark table used for the RIBOMap transfer analysis and visualizes SMITH against the recorded baselines for each dataset and metric.",
        "analysis": '''path = verify_fixture("fixtures/ribomap_benchmark_methods_summary.csv", "ea91769109a2aa473707f1adacc7c2471f80b969daf721cfcc31df2781544e4e")
df = pd.read_csv(path)
accuracy = df[df["metric"].eq("accuracy")].copy()
smith = accuracy[accuracy["method"].eq("SMITH")].sort_values(["dataset", "label"])
display(smith[["dataset", "label", "panel_size", "value_mean", "value_std", "rank"]])
fig, ax = plt.subplots(figsize=(10, 4.5))
for method, group in accuracy.groupby("method"):
    means = group.groupby("dataset")["value_mean"].mean()
    ax.plot(means.index, means.values, marker="o", label=method)
ax.set_ylabel("Mean accuracy"); ax.set_title("RIBOMap/STARmap transfer benchmark")
ax.tick_params(axis="x", rotation=35); ax.legend(frameon=False, ncol=2, fontsize=8)
fig.tight_layout(); plt.show()
''',
        "limits": "This is the real completed benchmark summary, not hand-entered tutorial data. Full Figure 4 regeneration additionally requires raw spatial objects and the archived workflows under `reproducibility/workflows/ribomap_transfer/`.",
    },
    "04_inhouse_disease": {
        "folder": "disease_section", "stem": "04_SMITH_InHouse_Disease_Transfer",
        "title": "SMITH human disease transfer robustness",
        "intro": "Participant-level human brain data remain controlled access. This notebook uses the real released per-seed metrics and checks that their aggregate summary agrees with the published de-identified table.",
        "analysis": '''summary_path = verify_fixture("fixtures/inhouse_transfer_summary.csv", "2adf61357af7215d93ea48be992817fa74cb4daac942c02a26649034ad568f20")
seed_path = verify_fixture("fixtures/inhouse_transfer_per_seed_metrics.csv", "843de9833c00489e27e0f83d05421fb4c49beab566936af544b63871ef50acf5")
summary = pd.read_csv(summary_path)
per_seed = pd.read_csv(seed_path)
recomputed = (per_seed.groupby("comparison", as_index=False)
              .agg(n_seeds=("seed", "nunique"), delta_spearman_mean=("delta_spearman", "mean"), delta_top64_mean=("delta_top64", "mean")))
check = summary[["comparison", "n_seeds", "delta_spearman_mean", "delta_top64_mean"]].merge(recomputed, on="comparison", suffixes=("_released", "_recomputed"))
check["spearman_match"] = np.isclose(check["delta_spearman_mean_released"], check["delta_spearman_mean_recomputed"])
check["top64_match"] = np.isclose(check["delta_top64_mean_released"], check["delta_top64_mean_recomputed"])
assert check["spearman_match"].all() and check["top64_match"].all()
display(check)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for column, ax, title in [("delta_spearman", axes[0], "Rank-correlation gain"), ("delta_top64", axes[1], "Top-64 overlap gain")]:
    groups = per_seed.groupby("comparison")[column]
    names = list(groups.groups)
    ax.boxplot([groups.get_group(name) for name in names], showmeans=True)
    ax.set_xticks(np.arange(1, len(names) + 1), names)
    ax.axhline(0, color="black", linewidth=0.8); ax.set_title(title); ax.tick_params(axis="x", rotation=55)
fig.tight_layout(); plt.show()
''',
        "limits": "The real per-seed table contains de-identified rank metrics only. Participant-level panels, UMAPs and spatial examples require authorized data access. The source summarization script is archived under `reproducibility/workflows/inhouse_disease/`.",
    },
    "05_agent": {
        "folder": "agent_section", "stem": "05_SMITH_Agent_Evaluation",
        "title": "SMITH-Agent evaluation and probe feasibility",
        "intro": "This notebook combines the real five-seed MERFISH evaluation output with the full 12,160-gene three-tool feasibility summary from the Agent workflow.",
        "analysis": '''metrics_path = verify_fixture("fixtures/agent_multi_reference_metrics.tsv", "113eb3ca6329a3aaadb9c0612b1cf25a8c8da87a0778dc5d687f7161558912ab")
feasibility_path = verify_fixture("fixtures/agent_full_tool_pass_summary.tsv", "38bc0c51d8650037fe90e74acb13938265b0b71600cc4711d2361f84d55ed90c")
metrics = pd.read_csv(metrics_path, sep="\\t")
accuracy = metrics[metrics["metric"].eq("cell_type_accuracy")]
display(accuracy.groupby(["panel_size", "panel"], as_index=False)["value"].agg(["mean", "std"]).reset_index())
gates = pd.read_csv(feasibility_path, sep="\\t")
gates["pass_rate"] = gates["pass_count"] / gates["total_count"]
display(gates)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for panel, group in accuracy.groupby("panel"):
    grouped = group.groupby("panel_size")["value"].agg(["mean", "std"])
    axes[0].errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], marker="o", capsize=3, label=panel)
axes[0].set(xlabel="Panel size", ylabel="Cell-type accuracy", title="Five-seed MERFISH evaluation"); axes[0].legend(frameon=False)
axes[1].bar(gates["gate"], gates["pass_rate"], color="#2f6690")
axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("Fraction of 12,160 genes passing"); axes[1].tick_params(axis="x", rotation=60)
fig.tight_layout(); plt.show()
''',
        "limits": "These are completed Agent outputs. Full Figure 6 regeneration still needs reference retrieval, locked test objects and the external ODT/OligoMiner/ProbeDealer resources. The source scripts are archived under `reproducibility/workflows/agent/`.",
    },
}


def build_notebook(case_id, spec):
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"}, language_info={"name": "python", "version": "3"})
    notebook.cells = [
        nbformat.v4.new_markdown_cell(f"# {spec['title']}\n\n{spec['intro']}"),
        nbformat.v4.new_markdown_cell("[Open the editable source notebook on GitHub](https://github.com/fym0503/SMITH/blob/main/docs/source/tutorials/notebooks/" + f"{spec['folder']}/{spec['stem']}_source.ipynb)"),
        nbformat.v4.new_markdown_cell("## Provenance\n\nThe code cell verifies the SHA-256 of every bundled result table. These files are copied from completed paper-workspace runs; they are not synthetic replacements for the manuscript outputs."),
        nbformat.v4.new_code_cell(SETUP),
        nbformat.v4.new_markdown_cell("## Analysis"),
        nbformat.v4.new_code_cell(spec["analysis"]),
        nbformat.v4.new_markdown_cell("## Scope\n\n" + spec["limits"]),
    ]
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only", choices=sorted(NOTEBOOKS))
    args = parser.parse_args()
    selected = {args.only: NOTEBOOKS[args.only]} if args.only else NOTEBOOKS
    generated = []
    for case_id, spec in selected.items():
        folder = NOTEBOOK_ROOT / spec["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        source_path = folder / f"{spec['stem']}_source.ipynb"
        executed_path = folder / f"{spec['stem']}_executed.ipynb"
        notebook = build_notebook(case_id, spec)
        nbformat.write(notebook, source_path); generated.append(source_path)
        print(f"wrote {source_path.relative_to(ROOT)}")
        if args.execute:
            executed = NotebookClient(nbformat.from_dict(notebook), timeout=600, kernel_name="python3", allow_errors=False).execute(cwd=str(ROOT))
            nbformat.write(executed, executed_path); generated.append(executed_path)
            print(f"wrote {executed_path.relative_to(ROOT)}")
    for path in generated:
        notebook = nbformat.read(path, as_version=4)
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if path.name.endswith("_source.ipynb"):
            assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in code)
        else:
            assert code and all(cell.get("execution_count") is not None for cell in code)


if __name__ == "__main__":
    main()
