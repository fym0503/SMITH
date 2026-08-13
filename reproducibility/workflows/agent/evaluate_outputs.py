#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from smith_agent.benchmarking import evaluate_panel_cell_type_classification, evaluate_panel_coordinate_regression
from reproducibility.workflows.common import write_json


def evaluate(
    merfish_file: str | Path,
    panels: list[tuple[str, str | Path]],
    output_dir: str | Path,
    panel_size: int,
    label_column: str,
    seed: int,
) -> dict:
    output_dir = Path(output_dir)
    rows = []
    details = {}
    for panel_name, panel_file in panels:
        panel_dir = output_dir / panel_name
        classification = evaluate_panel_cell_type_classification(
            adata_file=merfish_file,
            panel_path=panel_file,
            output_dir=panel_dir / "cell_type",
            panel_size=panel_size,
            label_column=label_column,
            seed=seed,
        )
        spatial = evaluate_panel_coordinate_regression(
            adata_file=merfish_file,
            panel_path=panel_file,
            output_dir=panel_dir / "spatial",
            panel_size=panel_size,
            seed=seed,
        )
        details[panel_name] = {
            "panel_file": str(Path(panel_file).resolve()),
            "classification": classification.to_dict(),
            "spatial": spatial.to_dict(),
        }
        rows.extend({"panel": panel_name, "metric": key, "value": value} for key, value in classification.metrics.items())
        rows.extend({"panel": panel_name, "metric": key, "value": value} for key, value in spatial.metrics.items())
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.tsv"
    pd.DataFrame(rows).to_csv(metrics_path, sep="\t", index=False)
    payload = {"merfish_file": str(Path(merfish_file).resolve()), "panels": details, "metrics_tsv": str(metrics_path)}
    write_json(output_dir / "metrics.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate newly generated source and multi-reference panels on MERFISH.")
    parser.add_argument("--merfish-file", required=True)
    parser.add_argument("--source-panel", required=True)
    parser.add_argument("--integrated-panel", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--panel-size", type=int, default=64)
    parser.add_argument("--label-column", default="Cell_Type")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = evaluate(
        args.merfish_file,
        [("source", args.source_panel), ("multi_reference", args.integrated_panel)],
        args.output_dir,
        args.panel_size,
        args.label_column,
        args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
