#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd


WORKSPACE_ROOT = Path("/workspace/fanyimin/SMITH_unified")
CODE_ROOT = WORKSPACE_ROOT / "code"
DATA_ROOT = WORKSPACE_ROOT / "data" / "SMITH_data_ribomap" / "data"
OUTPUT_ROOT = WORKSPACE_ROOT / "outputs" / "cross_study" / "deep_mouse_ribomap"
COMMON_PY = Path("/workspace/fanyimin/.conda/envs/smith-common/bin/python")
SCGIST_PY = Path("/workspace/fanyimin/.conda/envs/smith-scgist/bin/python")
SMITH_ROOT = CODE_ROOT / "SMITH_tool-main"
BASELINE_ROOT = CODE_ROOT / "SMITH_baselines"
DATA_PREP_ROOT = CODE_ROOT / "data_prep"

PANEL_SIZE = 32

SOURCE_SPECS = {
    "deep_to_mouse": {
        "source": DATA_ROOT / "deep_brain_ribomap.h5ad",
        "shared_with": DATA_ROOT / "mouse_brain_ribomap_rep1.h5ad",
        "targets": [
            DATA_ROOT / "mouse_brain_ribomap_rep1.h5ad",
            DATA_ROOT / "mouse_brain_ribomap_rep2.h5ad",
        ],
    },
    "mouse_rep1_to_deep": {
        "source": DATA_ROOT / "mouse_brain_ribomap_rep1.h5ad",
        "shared_with": DATA_ROOT / "deep_brain_ribomap.h5ad",
        "targets": [DATA_ROOT / "deep_brain_ribomap.h5ad"],
    },
    "mouse_rep2_to_deep": {
        "source": DATA_ROOT / "mouse_brain_ribomap_rep2.h5ad",
        "shared_with": DATA_ROOT / "deep_brain_ribomap.h5ad",
        "targets": [DATA_ROOT / "deep_brain_ribomap.h5ad"],
    },
}


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None, log_file: Path | None = None) -> None:
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as handle:
            subprocess.run(cmd, check=True, env=env, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, env=env, cwd=cwd)


def smith_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SMITH_ROOT)
    return env


def prepare_shared_subsets(direction: str, spec: dict[str, object]) -> Path:
    prep_dir = OUTPUT_ROOT / "prepared_data" / direction
    subset_file = prep_dir / "source_shared.h5ad"
    genes_file = prep_dir / "shared_genes.txt"
    if subset_file.exists() and genes_file.exists():
        return subset_file
    prep_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        str(DATA_PREP_ROOT / "prepare_shared_gene_subset.py"),
        "--source",
        str(spec["source"]),
        "--target",
        str(spec["shared_with"]),
        "--output",
        str(subset_file),
        "--genes-output",
        str(genes_file),
    ]
    run(cmd)
    return subset_file


def evaluate_panel(panel_csv: Path, targets: list[Path], output_csv: Path) -> pd.DataFrame:
    cmd = [
        str(COMMON_PY),
        str(BASELINE_ROOT / "scripts" / "eval_knn_panels.py"),
        "--panel_csv",
        str(panel_csv),
        "--label",
        "celltype",
        "--output",
        str(output_csv),
        *[str(t) for t in targets],
    ]
    run(cmd, env=smith_env())
    return pd.read_csv(output_csv)


def run_scgenefit(source_h5ad: Path, out_dir: Path) -> Path:
    panel_csv = out_dir / f"marker_{PANEL_SIZE}.csv"
    if panel_csv.exists():
        return panel_csv
    cmd = [
        str(COMMON_PY),
        str(BASELINE_ROOT / "GPS_tools-main" / "baselines" / "scGeneFit" / "run.py"),
        "--adata",
        str(source_h5ad),
        "--label",
        "celltype",
        "--num_markers",
        str(PANEL_SIZE),
        "--output",
        str(out_dir),
    ]
    run(cmd, log_file=out_dir / "train.log")
    return panel_csv


def run_activesvm(source_h5ad: Path, out_dir: Path, max_iter: int) -> Path:
    panel_csv = out_dir / f"marker_{PANEL_SIZE}.csv"
    if panel_csv.exists():
        return panel_csv
    cmd = [
        str(COMMON_PY),
        str(BASELINE_ROOT / "GPS_tools-main" / "baselines" / "activeSVM" / "run.py"),
        "--adata",
        str(source_h5ad),
        "--label",
        "celltype",
        "--num_markers",
        str(PANEL_SIZE),
        "--num_samples",
        "2000",
        "--max_iter",
        str(max_iter),
        "--output",
        str(out_dir),
    ]
    run(cmd, log_file=out_dir / "train.log")
    return panel_csv


def run_persist_sup(source_h5ad: Path, out_dir: Path, max_epochs: int) -> Path:
    panel_csv = out_dir / f"marker_{PANEL_SIZE}.csv"
    if panel_csv.exists():
        return panel_csv
    cmd = [
        str(COMMON_PY),
        str(BASELINE_ROOT / "GPS_tools-main" / "baselines" / "persist" / "run_sup.py"),
        "--adata",
        str(source_h5ad),
        "--label",
        "celltype",
        "--num_markers",
        str(PANEL_SIZE),
        "--max_epochs",
        str(max_epochs),
        "--output",
        str(out_dir),
    ]
    run(cmd, log_file=out_dir / "train.log")
    return panel_csv


def run_spapros(source_h5ad: Path, out_dir: Path) -> Path:
    panel_csv = out_dir / f"marker_{PANEL_SIZE}.csv"
    if panel_csv.exists():
        return panel_csv
    cmd = [
        str(COMMON_PY),
        str(BASELINE_ROOT / "GPS_tools-main" / "baselines" / "spapros" / "run.py"),
        "--adata",
        str(source_h5ad),
        "--label",
        "celltype",
        "--num_markers",
        str(PANEL_SIZE),
        "--output",
        str(out_dir),
    ]
    run(cmd, log_file=out_dir / "train.log")
    return panel_csv


def run_scgist(source_h5ad: Path, out_dir: Path, epochs: int) -> Path:
    panel_csv = out_dir / f"marker_{PANEL_SIZE}.csv"
    if panel_csv.exists():
        return panel_csv
    cmd = [
        str(SCGIST_PY),
        str(BASELINE_ROOT / "GPS_tools-main" / "baselines" / "scGIST" / "run.py"),
        "--adata",
        str(source_h5ad),
        "--label",
        "celltype",
        "--num_markers",
        str(PANEL_SIZE),
        "--epochs",
        str(epochs),
        "--output",
        str(out_dir),
    ]
    run(cmd, log_file=out_dir / "train.log")
    return panel_csv


def latest_epoch_csv(index_dir: Path) -> Path:
    epoch_files = sorted(index_dir.glob("epoch_*.csv"))
    if not epoch_files:
        raise FileNotFoundError(f"No epoch CSV found in {index_dir}")
    return max(epoch_files, key=lambda p: int(p.stem.split("_", 1)[1]))


def run_smith(source_h5ad: Path, out_dir: Path, epochs: int) -> Path:
    panel_csv = out_dir / f"panel_top{PANEL_SIZE}.csv"
    if panel_csv.exists():
        return panel_csv
    index_dir = out_dir / "index"
    log_dir = out_dir / "log"
    index_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(COMMON_PY),
        str(SMITH_ROOT / "main.py"),
        "--adata_file",
        str(source_h5ad),
        "--saving_dir",
        str(index_dir),
        "--log_dir",
        str(log_dir),
        "--tasks",
        "recon,cls,coordination",
        "--task_name",
        out_dir.name,
        "--layer",
        "raw",
        "--learning_rate",
        "4e-05",
        "--batch_size",
        "1024",
        "--dim",
        "32",
        "--rep_dim",
        "64",
        "--rep_hidden_dims",
        "16",
        "--head_hidden_dims",
        "64,32",
        "--panel_size",
        str(PANEL_SIZE),
        "--dropout_rate",
        "0.22",
        "--lam",
        "0.42",
        "--sigma",
        "0.62",
        "--activation",
        "selu",
        "--epoch",
        str(epochs),
        "--record",
        str(epochs),
        "--optimizer",
        "Adam",
        "--device",
        "cpu",
        "--seed",
        "3400",
        "--val",
    ]
    run(cmd, env=smith_env(), cwd=SMITH_ROOT, log_file=out_dir / "train.log")
    ranking = pd.read_csv(latest_epoch_csv(index_dir))
    ranking.head(PANEL_SIZE).to_csv(panel_csv, index=False)
    return panel_csv


METHOD_RUNNERS = {
    "scGeneFit": lambda source, out_dir, args: run_scgenefit(source, out_dir),
    "activeSVM": lambda source, out_dir, args: run_activesvm(source, out_dir, args.active_svm_max_iter),
    "persist_sup": lambda source, out_dir, args: run_persist_sup(source, out_dir, args.persist_epochs),
    "spapros": lambda source, out_dir, args: run_spapros(source, out_dir),
    "scGIST": lambda source, out_dir, args: run_scgist(source, out_dir, args.scgist_epochs),
    "SMITH": lambda source, out_dir, args: run_smith(source, out_dir, args.smith_epochs),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deep<->mouse ribomap cross-study panel selection.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["scGeneFit", "activeSVM", "persist_sup", "SMITH"],
        choices=sorted(METHOD_RUNNERS.keys()),
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=list(SOURCE_SPECS.keys()),
        choices=sorted(SOURCE_SPECS.keys()),
    )
    parser.add_argument("--active-svm-max-iter", type=int, default=50)
    parser.add_argument("--persist-epochs", type=int, default=30)
    parser.add_argument("--scgist-epochs", type=int, default=30)
    parser.add_argument("--smith-epochs", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    manifest_rows = []
    for direction in args.directions:
        spec = SOURCE_SPECS[direction]
        shared_source = prepare_shared_subsets(direction, spec)
        manifest_rows.append(
            {
                "direction": direction,
                "source": spec["source"].name,
                "shared_with": spec["shared_with"].name,
                "shared_source_h5ad": str(shared_source),
                "targets": ",".join(p.name for p in spec["targets"]),
            }
        )
        for method in args.methods:
            method_dir = OUTPUT_ROOT / method / direction
            if args.force and method_dir.exists():
                subprocess.run(["rm", "-rf", str(method_dir)], check=True)
            method_dir.mkdir(parents=True, exist_ok=True)
            panel_csv = METHOD_RUNNERS[method](shared_source, method_dir, args)
            eval_csv = method_dir / "knn_eval.csv"
            eval_df = evaluate_panel(panel_csv, spec["targets"], eval_csv)
            for _, row in eval_df.iterrows():
                all_rows.append(
                    {
                        "method": method,
                        "direction": direction,
                        "source_dataset": spec["source"].name,
                        "target_dataset": row["dataset"],
                        "panel_csv": str(panel_csv),
                        "n_panel_genes_present": int(row["n_panel_genes_present"]),
                        "knn_accuracy": float(row["knn_accuracy"]),
                        "balanced_accuracy": float(row["balanced_accuracy"]),
                        "macro_f1": float(row["macro_f1"]),
                    }
                )

    pd.DataFrame(manifest_rows).to_csv(OUTPUT_ROOT / "manifest.tsv", sep="\t", index=False)
    summary = pd.DataFrame(all_rows).sort_values(["method", "direction", "target_dataset"])
    summary.to_csv(OUTPUT_ROOT / "summary.csv", index=False)
    (OUTPUT_ROOT / "run_config.json").write_text(
        json.dumps(
            {
                "panel_size": PANEL_SIZE,
                "methods": args.methods,
                "active_svm_max_iter": args.active_svm_max_iter,
                "persist_epochs": args.persist_epochs,
                "scgist_epochs": args.scgist_epochs,
                "smith_epochs": args.smith_epochs,
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
