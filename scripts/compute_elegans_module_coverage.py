#!/usr/bin/env python3
"""Compute Figure 3h module miss rates from generated panel artifacts.

This is an audit/export command for a completed manuscript-scale run. The
hosted notebook computes coverage from in-memory panels; this command is useful
when the panels were generated remotely and only their provenance artifacts
were synchronized locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reproducibility.workflows.common import parse_int_list
from reproducibility.workflows.regulatory_activity.paper_analysis import write_module_coverage


DEFAULT_METHODS = ("SMITH", "PERSIST-class", "PERSIST", "ActiveSVM", "scGIST", "scGeneFit", "Spapros")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, required=True, help="generated_panels.tsv from run_tutorial.py")
    parser.add_argument("--module-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--splits", default="split_1,split_2,split_3,split_4,split_5")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--panel-sizes", default="16,24,32")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    if not args.panels.is_file():
        raise FileNotFoundError(args.panels)
    if not args.module_file.is_file():
        raise FileNotFoundError(args.module_file)
    panels = pd.read_csv(args.panels, sep="\t")
    required = {"method", "split", "training_seed", "panel_size", "panel_file"}
    missing = required - set(panels.columns)
    if missing:
        raise ValueError(f"Panel manifest lacks required columns: {sorted(missing)}")
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    seeds = tuple(parse_int_list(args.seeds))
    sizes = tuple(parse_int_list(args.panel_sizes))
    result = write_module_coverage(
        panels,
        args.module_file,
        args.output,
        expected_methods=methods,
        expected_splits=splits,
        expected_panel_sizes=sizes,
        expected_training_seeds=seeds,
        require_complete=not args.allow_incomplete,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
