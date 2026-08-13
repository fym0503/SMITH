#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from reproducibility.workflows.common import run_smith, sha256, write_json
from reproducibility.workflows.regulatory_activity.evaluate_outputs import evaluate


def run(args: argparse.Namespace) -> dict:
    case_root = Path(args.data_root).resolve() / "regulatory_activity" / "elegans" / "splits" / args.dataset / args.split
    train_file = case_root / "train.h5ad"
    test_file = case_root / "test.h5ad"
    missing = [str(path) for path in (train_file, test_file) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing tutorial input files: " + ", ".join(missing))

    output_dir = Path(args.output_dir).resolve()
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training = run_smith(
        adata_file=train_file,
        output_dir=output_dir / "smith",
        tasks="recon,cls,time",
        task_name=f"{args.dataset}_{args.split}",
        panel_size=args.panel_size,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        time_label=args.time_column,
        max_cells=args.max_cells,
        sampling_strategy="celltype",
        force=args.force,
    )
    panel_file = Path(training.get("panel_csv") or (output_dir / "smith" / f"panel_top{args.panel_size}.csv"))
    evaluation = evaluate(
        train_file=train_file,
        test_file=test_file,
        panel_file=panel_file,
        output_dir=output_dir / "evaluation",
        panel_size=args.panel_size,
        time_column=args.time_column,
        neighbors=args.neighbors,
    )
    manifest = {
        "workflow": "02_regulatory_activity",
        "configuration": vars(args),
        "inputs": [
            {"path": str(train_file), "bytes": train_file.stat().st_size, "sha256": sha256(train_file)},
            {"path": str(test_file), "bytes": test_file.stat().st_size, "sha256": sha256(test_file)},
        ],
        "training": training,
        "evaluation": evaluation,
        "outputs": {
            "ranking": training["ranking_csv"],
            "panel": str(panel_file),
            "metrics_json": str(output_dir / "evaluation" / "metrics.json"),
            "metrics_tsv": str(output_dir / "evaluation" / "metrics.tsv"),
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SMITH from a real C. elegans train/test split.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=["elegans_tf", "elegans_mirna"], default="elegans_tf")
    parser.add_argument("--split", default="split_1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--time-column", default="absolute_time")
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
