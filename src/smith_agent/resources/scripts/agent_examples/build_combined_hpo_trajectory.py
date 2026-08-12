#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCREEN_DIR = PROJECT_ROOT / "outputs/liver_pareto_hpo_screen8_epoch15_extended"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/liver_pareto_hpo_screen8_epoch15_extended/combined_sum"


def _load_hpo_module():
    path = PROJECT_ROOT / "scripts/run_liver_pareto_hpo.py"
    spec = importlib.util.spec_from_file_location("run_liver_pareto_hpo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified combined-sum HPO trajectories from a screen run.")
    parser.add_argument("--screen-dir", default=str(DEFAULT_SCREEN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--panel-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-min-label-count", type=int, default=5)
    parser.add_argument("--merfish-max-cells", type=int, default=0)
    parser.add_argument("--class-weight", default="none")
    return parser.parse_args()


def _best_rows_by_order(trials: pd.DataFrame, order_col: str) -> pd.DataFrame:
    all_steps = sorted(trials[order_col].dropna().astype(int).unique())
    trials = trials[trials["status"].isin(["completed", "skipped_existing"])].copy()
    trials = trials.dropna(subset=["visium_mean_cell_type_accuracy", "visium_mean_spatial_pearson"])
    trials["visium_combined_sum"] = (
        trials["visium_mean_cell_type_accuracy"].astype(float)
        + trials["visium_mean_spatial_pearson"].astype(float)
    )
    rows: list[dict] = []
    for step in all_steps:
        subset = trials[trials[order_col].astype(int) <= step].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["visium_combined_sum", "visium_mean_cell_type_accuracy", "visium_mean_spatial_pearson", "trial_index"],
            ascending=[False, False, False, True],
        )
        selected = subset.iloc[0].to_dict()
        rows.append(
            {
                order_col: int(step),
                "selected_trial_id": selected["trial_id"],
                "selected_trial_index": int(selected["trial_index"]),
                "config_key": selected["config_key"],
                "panel_path": selected["panel_path"],
                "visium_mean_cell_type_accuracy": float(selected["visium_mean_cell_type_accuracy"]),
                "visium_mean_spatial_pearson": float(selected["visium_mean_spatial_pearson"]),
                "visium_combined_sum": float(selected["visium_combined_sum"]),
            }
        )
    return pd.DataFrame(rows)


def _attach_merfish_metrics(table: pd.DataFrame, merfish: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    merged = table.drop(columns=["merfish_cell_type_accuracy", "merfish_spatial_pearson"], errors="ignore").merge(
        merfish,
        left_on="selected_trial_id",
        right_on="trial_id",
        how="left",
    )
    merged = merged.drop(columns=["trial_id"], errors="ignore")
    merged["merfish_combined_sum"] = (
        merged["merfish_cell_type_accuracy"].astype(float) + merged["merfish_spatial_pearson"].astype(float)
    )
    merged["merfish_cell_type_accuracy_best_so_far"] = merged["merfish_cell_type_accuracy"].cummax()
    merged["merfish_spatial_pearson_best_so_far"] = merged["merfish_spatial_pearson"].cummax()
    merged["merfish_combined_sum_best_so_far"] = merged["merfish_combined_sum"].cummax()
    return merged


def main() -> int:
    args = parse_args()
    hpo = _load_hpo_module()
    screen_dir = Path(args.screen_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = pd.read_csv(screen_dir / "hpo_trials.tsv", sep="\t")
    by_trial = _best_rows_by_order(trials, "trial_index")
    by_iteration = _best_rows_by_order(trials, "iteration")

    selected_trial_ids = sorted(
        set(by_trial["selected_trial_id"].astype(str)) | set(by_iteration["selected_trial_id"].astype(str))
    )
    trial_lookup = trials.assign(_trial_id=trials["trial_id"].astype(str)).set_index("_trial_id")
    eval_rows: list[dict] = []
    for trial_id in selected_trial_ids:
        row = trial_lookup.loc[trial_id]
        dataset_rows = hpo.evaluate_panel_on_dataset(
            adata_file=hpo.DEFAULT_MERFISH,
            dataset_id="locked_merfish",
            dataset_role="merfish_locked_test",
            panel_path=row["panel_path"],
            output_dir=output_dir / "evaluation" / trial_id / "merfish_locked",
            panel_size=args.panel_size,
            label_column="Cell_Type",
            seed=args.seed,
            test_size=args.test_size,
            max_cells=args.merfish_max_cells,
            min_label_count=args.eval_min_label_count,
            class_weight=hpo._class_weight(args.class_weight),
            subset_cache_dir=output_dir / "evaluation" / "_shared_subsets",
        )
        for item in dataset_rows:
            item.update(
                {
                    "trial_id": trial_id,
                    "trial_index": int(row["trial_index"]),
                    "iteration": int(row["iteration"]),
                    "config_key": row["config_key"],
                }
            )
        eval_rows.extend(dataset_rows)

    eval_df = pd.DataFrame(eval_rows)
    if not eval_df.empty:
        merfish = (
            eval_df.pivot_table(index="trial_id", columns="metric", values="value", aggfunc="first")
            .reset_index()
            .rename(
                columns={
                    "cell_type_accuracy": "merfish_cell_type_accuracy",
                    "spatial_pearson": "merfish_spatial_pearson",
                }
            )
        )
        by_trial = _attach_merfish_metrics(by_trial, merfish)
        by_iteration = _attach_merfish_metrics(by_iteration, merfish)

    by_trial.to_csv(output_dir / "combined_by_trial_count.tsv", sep="\t", index=False)
    by_iteration.to_csv(output_dir / "combined_by_iteration.tsv", sep="\t", index=False)
    eval_df.to_csv(output_dir / "combined_merfish_evaluation_long.tsv", sep="\t", index=False)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
