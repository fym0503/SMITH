#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path

import pandas as pd


WORKSPACE_ROOT = Path("/workspace/fanyimin/SMITH_unified")
CODE_ROOT = WORKSPACE_ROOT / "code"
DATA_ROOT = WORKSPACE_ROOT / "data" / "SMITH_data_ribomap" / "data"
COMMON_PY = Path("/workspace/fanyimin/.conda/envs/smith-common/bin/python")
DYFLOW_PY = Path("/workspace/fanyimin/.conda/envs/dynflow/bin/python")
SCGIST_GPU_ENV = Path("/workspace/fanyimin/.conda/envs/smith-scgist-gpu")
SMITH_ROOT = CODE_ROOT / "SMITH_tool-main"
BASELINE_ROOT = CODE_ROOT / "SMITH_baselines"

SOURCE_SPECS = {
    "deep_starmap_to_ribomap": {
        "source": DATA_ROOT / "deep_brain_starmap.h5ad",
        "targets": [DATA_ROOT / "deep_brain_ribomap.h5ad"],
    },
    "mouse_starmap_to_ribomap_rep1": {
        "source": DATA_ROOT / "mouse_brain_starmap_rep2.h5ad",
        "targets": [DATA_ROOT / "mouse_brain_ribomap_rep1.h5ad"],
    },
    "mouse_starmap_to_ribomap_rep2": {
        "source": DATA_ROOT / "mouse_brain_starmap_rep2.h5ad",
        "targets": [DATA_ROOT / "mouse_brain_ribomap_rep2.h5ad"],
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
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return env


def scgist_env() -> dict[str, str]:
    env = os.environ.copy()
    cuda_libs = subprocess.check_output(
        [
            "bash",
            "-lc",
            "find /workspace/fanyimin/.conda/envs/smith-scgist-gpu/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | paste -sd: -",
        ],
        text=True,
    ).strip()
    if cuda_libs:
        env["LD_LIBRARY_PATH"] = f"{cuda_libs}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    return env


def run_scgenefit(source_h5ad: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    cmd = [
        str(COMMON_PY),
        str(BASELINE_ROOT / "GPS_tools-main" / "baselines" / "scGeneFit" / "run.py"),
        "--adata",
        str(source_h5ad),
        "--label",
        "celltype",
        "--num_markers",
        str(args.panel_size),
        "--output",
        str(out_dir),
    ]
    run(cmd, log_file=out_dir / "train.log")
    return out_dir / f"marker_{args.panel_size}.csv"


def run_activesvm(source_h5ad: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    cmd = [
        str(COMMON_PY),
        str(BASELINE_ROOT / "GPS_tools-main" / "baselines" / "activeSVM" / "run.py"),
        "--adata",
        str(source_h5ad),
        "--label",
        "celltype",
        "--num_markers",
        str(args.panel_size),
        "--num_samples",
        str(args.active_svm_num_samples),
        "--max_iter",
        str(args.active_svm_max_iter),
        "--output",
        str(out_dir),
    ]
    run(cmd, log_file=out_dir / "train.log")
    return out_dir / f"marker_{args.panel_size}.csv"


def run_persist_sup(source_h5ad: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    cmd = [
        str(DYFLOW_PY),
        str(BASELINE_ROOT / "GPS_tools-main" / "baselines" / "persist" / "run_sup.py"),
        "--adata",
        str(source_h5ad),
        "--label",
        "celltype",
        "--num_markers",
        str(args.panel_size),
        "--max_epochs",
        str(args.persist_epochs),
        "--output",
        str(out_dir),
    ]
    run(cmd, log_file=out_dir / "train.log")
    return out_dir / f"marker_{args.panel_size}.csv"


def run_persist_unsup(source_h5ad: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    cmd = [
        str(DYFLOW_PY),
        str(BASELINE_ROOT / "GPS_tools-main" / "baselines" / "persist" / "run_unsup.py"),
        "--adata",
        str(source_h5ad),
        "--num_markers",
        str(args.panel_size),
        "--max_epochs",
        str(args.persist_epochs),
        "--output",
        str(out_dir),
    ]
    run(cmd, log_file=out_dir / "train.log")
    return out_dir / f"marker_{args.panel_size}.csv"


def run_spapros(source_h5ad: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    cmd = [
        str(COMMON_PY),
        str(BASELINE_ROOT / "GPS_tools-main" / "baselines" / "spapros" / "run.py"),
        "--adata",
        str(source_h5ad),
        "--label",
        "celltype",
        "--num_markers",
        str(args.panel_size),
        "--output",
        str(out_dir),
    ]
    run(cmd, log_file=out_dir / "train.log")
    return out_dir / f"marker_{args.panel_size}.csv"


def run_scgist(source_h5ad: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    cmd = [
        str(SCGIST_GPU_ENV / "bin" / "python"),
        str(BASELINE_ROOT / "GPS_tools-main" / "baselines" / "scGIST" / "run.py"),
        "--adata",
        str(source_h5ad),
        "--label",
        "celltype",
        "--num_markers",
        str(args.panel_size),
        "--epochs",
        str(args.scgist_epochs),
        "--output",
        str(out_dir),
    ]
    run(cmd, env=scgist_env(), log_file=out_dir / "train.log")
    return out_dir / f"marker_{args.panel_size}.csv"


def latest_epoch_csv(index_dir: Path) -> Path:
    epoch_files = sorted(index_dir.glob("epoch_*.csv"))
    if not epoch_files:
        raise FileNotFoundError(f"No epoch CSV found in {index_dir}")
    return max(epoch_files, key=lambda p: int(p.stem.split("_", 1)[1]))


def run_smith(source_h5ad: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    panel_csv = out_dir / f"marker_{args.panel_size}.csv"
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
        str(args.smith_tasks),
        "--task_name",
        out_dir.name,
        "--layer",
        "raw",
        "--learning_rate",
        str(args.smith_learning_rate),
        "--batch_size",
        str(args.smith_batch_size),
        "--dim",
        str(args.smith_dim),
        "--rep_dim",
        str(args.smith_rep_dim),
        "--rep_hidden_dims",
        str(args.smith_rep_hidden_dims),
        "--head_hidden_dims",
        str(args.smith_head_hidden_dims),
        "--panel_size",
        str(args.panel_size),
        "--dropout_rate",
        str(args.smith_dropout_rate),
        "--lam",
        str(args.smith_lam),
        "--sigma",
        str(args.smith_sigma),
        "--activation",
        str(args.smith_activation),
        "--epoch",
        str(args.smith_epochs),
        "--record",
        str(args.smith_epochs),
        "--optimizer",
        str(args.smith_optimizer),
        "--device",
        "cuda:0",
        "--seed",
        str(args.smith_seed),
        "--val",
        "--balance_mode",
        str(args.smith_balance_mode),
        "--balance_cap",
        str(args.smith_balance_cap),
        "--max_cells",
        str(args.smith_max_cells),
        "--sampling_strategy",
        str(args.smith_sampling_strategy),
    ]
    run(cmd, env=smith_env(), cwd=SMITH_ROOT, log_file=out_dir / "train.log")
    ranking = pd.read_csv(latest_epoch_csv(index_dir))
    ranking.head(args.panel_size).to_csv(panel_csv, index=False)
    return panel_csv


METHOD_RUNNERS = {
    "scGeneFit": run_scgenefit,
    "activeSVM": run_activesvm,
    "persist_sup": run_persist_sup,
    "persist_unsup": run_persist_unsup,
    "spapros": run_spapros,
    "scGIST": run_scgist,
    "SMITH": run_smith,
}


def append_completed(completed_path: Path, rows: list[dict[str, object]]) -> None:
    df_new = pd.DataFrame(rows)
    if completed_path.exists():
        df_old = pd.read_csv(completed_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["method", "direction", "source_dataset"], keep="last")
    else:
        df = df_new
    df = df.sort_values(["method", "direction", "source_dataset"])
    df.to_csv(completed_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run starmap-to-ribomap baseline panel selection without evaluation.")
    parser.add_argument("--panel-size", type=int, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["scGeneFit", "activeSVM", "persist_sup", "persist_unsup", "spapros", "scGIST", "SMITH"],
        choices=sorted(METHOD_RUNNERS.keys()),
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=list(SOURCE_SPECS.keys()),
        choices=sorted(SOURCE_SPECS.keys()),
    )
    parser.add_argument("--active-svm-num-samples", type=int, default=3600)
    parser.add_argument("--active-svm-max-iter", type=int, default=200)
    parser.add_argument("--persist-epochs", type=int, default=200)
    parser.add_argument("--scgist-epochs", type=int, default=200)
    parser.add_argument("--smith-epochs", type=int, default=180)
    parser.add_argument("--smith-tasks", type=str, default="recon,cls,coordination")
    parser.add_argument("--smith-balance-mode", type=str, default="mean", choices=["off", "mean", "capped"])
    parser.add_argument("--smith-balance-cap", type=int, default=500)
    parser.add_argument("--smith-max-cells", type=int, default=40000)
    parser.add_argument("--smith-sampling-strategy", type=str, default="random", choices=["random", "celltype", "spatial", "celltype_spatial"])
    parser.add_argument("--smith-learning-rate", type=float, default=4e-5)
    parser.add_argument("--smith-batch-size", type=int, default=1024)
    parser.add_argument("--smith-dim", type=int, default=32)
    parser.add_argument("--smith-rep-dim", type=int, default=64)
    parser.add_argument("--smith-rep-hidden-dims", type=str, default="16")
    parser.add_argument("--smith-head-hidden-dims", type=str, default="64,32")
    parser.add_argument("--smith-dropout-rate", type=float, default=0.22)
    parser.add_argument("--smith-lam", type=float, default=0.42)
    parser.add_argument("--smith-sigma", type=float, default=0.62)
    parser.add_argument("--smith-activation", type=str, default="selu")
    parser.add_argument("--smith-optimizer", type=str, default="Adam")
    parser.add_argument("--smith-seed", type=int, default=3400)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(
        json.dumps(
            {
                "panel_size": args.panel_size,
                "methods": args.methods,
                "directions": args.directions,
                "active_svm_num_samples": args.active_svm_num_samples,
                "active_svm_max_iter": args.active_svm_max_iter,
                "persist_epochs": args.persist_epochs,
                "scgist_epochs": args.scgist_epochs,
                "smith_epochs": args.smith_epochs,
                "smith_tasks": args.smith_tasks,
                "smith_balance_mode": args.smith_balance_mode,
                "smith_balance_cap": args.smith_balance_cap,
                "smith_max_cells": args.smith_max_cells,
                "smith_sampling_strategy": args.smith_sampling_strategy,
                "smith_learning_rate": args.smith_learning_rate,
                "smith_batch_size": args.smith_batch_size,
                "smith_dim": args.smith_dim,
                "smith_rep_dim": args.smith_rep_dim,
                "smith_rep_hidden_dims": args.smith_rep_hidden_dims,
                "smith_head_hidden_dims": args.smith_head_hidden_dims,
                "smith_dropout_rate": args.smith_dropout_rate,
                "smith_lam": args.smith_lam,
                "smith_sigma": args.smith_sigma,
                "smith_activation": args.smith_activation,
                "smith_optimizer": args.smith_optimizer,
                "smith_seed": args.smith_seed,
                "envs": {
                    "common": str(COMMON_PY),
                    "persist": str(DYFLOW_PY),
                    "scgist_gpu": str(SCGIST_GPU_ENV / "bin" / "python"),
                    "smith_gpu0": str(COMMON_PY),
                },
            },
            indent=2,
        )
    )
    failures_path = output_root / "failures.jsonl"
    failures_path.write_text("")
    completed_path = output_root / "completed.csv"

    manifest_rows = []
    completed_rows = []
    for direction in args.directions:
        spec = SOURCE_SPECS[direction]
        manifest_rows.append(
            {
                "direction": direction,
                "source": spec["source"].name,
                "targets": ",".join(p.name for p in spec["targets"]),
            }
        )
        for method in args.methods:
            out_dir = output_root / method / direction
            if args.force and out_dir.exists():
                subprocess.run(["rm", "-rf", str(out_dir)], check=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            panel_csv = out_dir / f"marker_{args.panel_size}.csv"
            try:
                if not panel_csv.exists():
                    panel_csv = METHOD_RUNNERS[method](spec["source"], out_dir, args)
                completed_rows.append(
                    {
                        "method": method,
                        "direction": direction,
                        "source_dataset": spec["source"].name,
                        "panel_csv": str(panel_csv),
                    }
                )
                append_completed(completed_path, [completed_rows[-1]])
            except Exception as exc:
                payload = {
                    "method": method,
                    "direction": direction,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                with failures_path.open("a") as handle:
                    handle.write(json.dumps(payload) + "\n")

    pd.DataFrame(manifest_rows).to_csv(output_root / "manifest.tsv", sep="\t", index=False)
    if completed_rows:
        print(pd.DataFrame(completed_rows).to_string(index=False))
    else:
        print("No completed runs yet. Check failures.jsonl.")


if __name__ == "__main__":
    main()
