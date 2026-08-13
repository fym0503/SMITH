#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import pandas as pd


WORKSPACE_ROOT = Path("/workspace/fanyimin/SMITH_unified")
SMITH_ROOT = WORKSPACE_ROOT / "code" / "SMITH_tool-main"
PYTHON_BIN = Path("/workspace/fanyimin/.conda/envs/smith-common/bin/python")

if str(SMITH_ROOT) not in sys.path:
    sys.path.insert(0, str(SMITH_ROOT))

from Smith.eval import (
    evaluate_knn_overlap,
    evaluate_split_reconstruction,
    evaluate_split_knn_classification,
    evaluate_split_obsm_regression,
    evaluate_split_time_knn_regression,
    load_panel_genes,
)


DATASET_CONFIG = {
    "elegans_tf": {
        "split_root": WORKSPACE_ROOT / "data" / "SMITH_data_elegans" / "data" / "cell_name_splits" / "elegans_tf",
        "tasks": "recon,cls,coordination,time",
        "time_label": "absolute_time",
        "label": "cell_type",
        "spatial_obsm_key": "spatial",
        "default_panel_size": 32,
    },
    "elegans_mirna": {
        "split_root": WORKSPACE_ROOT / "data" / "SMITH_data_elegans" / "data" / "cell_name_splits" / "elegans_mirna",
        "tasks": "recon,cls,time",
        "time_label": "absolute_time",
        "label": "cell_type",
        "spatial_obsm_key": None,
        "default_panel_size": 32,
    },
}


def effective_dataset_config(dataset: str, disable_spatial: bool) -> dict[str, object]:
    cfg = copy.deepcopy(DATASET_CONFIG[dataset])
    if disable_spatial and dataset == "elegans_tf":
        cfg["tasks"] = "recon,cls,time"
        cfg["spatial_obsm_key"] = None
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate SMITH on elegans fixed splits with multitask objectives."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG), required=True)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--split", default="split_1")
    parser.add_argument("--panel-size", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--record", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--rep-dim", type=int, default=32)
    parser.add_argument("--rep-hidden-dims", default="32")
    parser.add_argument("--head-hidden-dims", default="32")
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--balance-mode", default="mean", choices=["off", "mean", "capped"])
    parser.add_argument("--balance-cap", type=int, default=500)
    parser.add_argument("--hurdle", action="store_true")
    parser.add_argument("--disable-spatial", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "outputs" / "smith",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.config_json is not None:
        config = json.loads(args.config_json.read_text())
        config_fields = {
            "panel_size": "panel_size",
            "epoch": "epoch",
            "record": "record",
            "seed": "seed",
            "learning_rate": "learning_rate",
            "batch_size": "batch_size",
            "dim": "dim",
            "rep_dim": "rep_dim",
            "rep_hidden_dims": "rep_hidden_dims",
            "head_hidden_dims": "head_hidden_dims",
            "dropout_rate": "dropout_rate",
            "lam": "lam",
            "sigma": "sigma",
            "activation": "activation",
            "optimizer": "optimizer",
            "balance_mode": "balance_mode",
            "balance_cap": "balance_cap",
            "hurdle": "hurdle",
            "disable_spatial": "disable_spatial",
        }
        for key, attr in config_fields.items():
            if key in config:
                setattr(args, attr, config[key])

    return args


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def latest_ranking_file(saving_dir: Path) -> Path:
    candidates = sorted(saving_dir.glob("epoch_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No ranking files found in {saving_dir}")
    return max(candidates, key=lambda p: int(p.stem.split("_", 1)[1]))


def build_train_command(
    args: argparse.Namespace,
    cfg: dict[str, object],
    train_h5ad: Path,
    out_dir: Path,
    task_name: str,
) -> list[str]:
    saving_dir = out_dir / "training" / f"index{args.seed}"
    log_dir = out_dir / "training" / f"log{args.seed}"
    saving_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return [
        str(PYTHON_BIN),
        str(SMITH_ROOT / "main.py"),
        "--adata_file",
        str(train_h5ad),
        "--saving_dir",
        str(saving_dir),
        "--log_dir",
        str(log_dir),
        "--tasks",
        str(cfg["tasks"]),
        "--task_name",
        task_name,
        "--layer",
        "raw",
        "--learning_rate",
        str(args.learning_rate),
        "--batch_size",
        str(args.batch_size),
        "--dim",
        str(args.dim),
        "--rep_dim",
        str(args.rep_dim),
        "--rep_hidden_dims",
        str(args.rep_hidden_dims),
        "--head_hidden_dims",
        str(args.head_hidden_dims),
        "--panel_size",
        str(args.panel_size),
        "--dropout_rate",
        str(args.dropout_rate),
        "--lam",
        str(args.lam),
        "--sigma",
        str(args.sigma),
        "--activation",
        str(args.activation),
        "--epoch",
        str(args.epoch),
        "--record",
        str(args.record),
        "--optimizer",
        str(args.optimizer),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--val",
        "--balance_mode",
        str(args.balance_mode),
        "--balance_cap",
        str(args.balance_cap),
        "--time_label",
        str(cfg["time_label"]),
        *(["--hurdle"] if args.hurdle else []),
    ]


def evaluate_fixed_split(
    args: argparse.Namespace,
    cfg: dict[str, object],
    train_h5ad: Path,
    test_h5ad: Path,
    panel_csv: Path,
) -> dict[str, object]:
    label = str(cfg["label"])
    time_label = str(cfg["time_label"])
    spatial_obsm_key = cfg["spatial_obsm_key"]

    panel_genes = load_panel_genes(panel_csv, panel_size=args.panel_size)
    train_adata = ad.read_h5ad(train_h5ad)
    test_adata = ad.read_h5ad(test_h5ad)
    train_adata.var_names = train_adata.var_names.str.upper()
    test_adata.var_names = test_adata.var_names.str.upper()

    celltype_knn_accuracy, _ = evaluate_split_knn_classification(
        train_adata,
        test_adata,
        panel_genes,
        label_name=label,
        n_neighbors=5,
    )
    reconstruction_ev, reconstruction_pearson = evaluate_split_reconstruction(
        train_adata,
        test_adata,
        panel_genes,
    )
    time_knn_pearson = evaluate_split_time_knn_regression(
        train_adata,
        test_adata,
        panel_genes,
        time_label=time_label,
        n_neighbors=5,
    )
    spatial_pearson = None
    if spatial_obsm_key is not None:
        _, spatial_pearson = evaluate_split_obsm_regression(
            train_adata,
            test_adata,
            panel_genes,
            obsm_key=spatial_obsm_key,
        )

    overlap_df = evaluate_knn_overlap(
        test_adata,
        panel_genes,
        ks=(5, 10, 20),
        max_cells=5000,
        random_state=args.seed,
    )
    avg_row = overlap_df.loc[overlap_df["k"] < 0]
    overlap_avg = float(avg_row["mean_overlap_fraction"].iloc[0]) if not avg_row.empty else None

    return {
        "dataset": args.dataset,
        "split": args.split,
        "panel_size": args.panel_size,
        "celltype_knn_accuracy": float(celltype_knn_accuracy),
        "reconstruction_explained_variance": None if reconstruction_ev is None else float(reconstruction_ev),
        "reconstruction_pearson": None if reconstruction_pearson is None else float(reconstruction_pearson),
        "time_knn_pearson": float(time_knn_pearson),
        "spatial_pearson": None if spatial_pearson is None else float(spatial_pearson),
        "knn_overlap_mean_avg": overlap_avg,
    }


def main() -> int:
    args = parse_args()
    cfg = effective_dataset_config(args.dataset, args.disable_spatial)
    if args.panel_size is None:
        args.panel_size = int(cfg["default_panel_size"])

    split_dir = cfg["split_root"] / args.split
    train_h5ad = split_dir / "train.h5ad"
    test_h5ad = split_dir / "test.h5ad"
    if not train_h5ad.exists() or not test_h5ad.exists():
        raise FileNotFoundError(f"Missing fixed split files under {split_dir}")

    out_dir = args.output_root / args.dataset / args.split / f"panel_{args.panel_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / "score.json"
    if score_path.exists() and not args.force:
        print(f"Skipping existing run: {out_dir}")
        return 0

    task_name = f"{args.dataset}-{args.split}-panel{args.panel_size}"
    command = build_train_command(args, cfg, train_h5ad, out_dir, task_name)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SMITH_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    train_log = out_dir / "training" / "train_command.log"
    with train_log.open("w") as handle:
        subprocess.run(
            command,
            cwd=SMITH_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )

    saving_dir = out_dir / "training" / f"index{args.seed}"
    ranking_file = latest_ranking_file(saving_dir)
    panel_csv = out_dir / f"panel_top{args.panel_size}.csv"
    pd.read_csv(ranking_file).head(args.panel_size).to_csv(panel_csv, index=False)

    fixed_eval = evaluate_fixed_split(args, cfg, train_h5ad, test_h5ad, panel_csv)
    pd.DataFrame([fixed_eval]).to_csv(out_dir / "fixed_split_eval.csv", index=False)

    score = {
        "dataset": args.dataset,
        "split": args.split,
        "panel_size": args.panel_size,
        "tasks": cfg["tasks"],
        "disable_spatial": bool(args.disable_spatial),
        "ranking_file": str(ranking_file),
        "panel_csv": str(panel_csv),
        "train_h5ad": str(train_h5ad),
        "test_h5ad": str(test_h5ad),
        **fixed_eval,
    }
    write_json(score_path, score)
    print(json.dumps(score, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
