#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_SCREEN_DIR = PROJECT_ROOT / "outputs/liver_pareto_hpo_screen8_epoch15_extended"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/liver_pareto_hpo_screen8_epoch15_exhaustive"


def _load_hpo_module():
    path = PROJECT_ROOT / "scripts/run_liver_pareto_hpo.py"
    spec = importlib.util.spec_from_file_location("run_liver_pareto_hpo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finish the finite liver HPO grid from an existing screen run.")
    parser.add_argument("--base-screen-dir", default=str(DEFAULT_BASE_SCREEN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size-for-iteration", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _grid_configs(hpo: Any) -> list[Any]:
    configs = []
    for tasks in hpo.TASK_GRID:
        for dropout_rate, lam, sigma, learning_rate in itertools.product(
            hpo.FLOAT_GRIDS["dropout_rate"],
            hpo.FLOAT_GRIDS["lam"],
            hpo.FLOAT_GRIDS["sigma"],
            hpo.FLOAT_GRIDS["learning_rate"],
        ):
            configs.append(
                hpo.HPOConfig(
                    tasks=tasks,
                    dropout_rate=dropout_rate,
                    lam=lam,
                    sigma=sigma,
                    learning_rate=learning_rate,
                )
            )
    return configs


def _write_checkpoint(
    *,
    output_dir: Path,
    trial_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    pd.DataFrame(trial_rows).to_csv(output_dir / "hpo_trials.tsv", sep="\t", index=False)
    pd.DataFrame(eval_rows).to_csv(output_dir / "hpo_evaluation_long.tsv", sep="\t", index=False)
    (output_dir / "hpo_exhaustive_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    hpo = _load_hpo_module()
    base_dir = Path(args.base_screen_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = _load_manifest(base_dir / "hpo_search_manifest.json")
    params = base_manifest["parameters"]
    references = hpo.references_for_full_from_screen(visium_reference_args=None, screen_manifest=base_manifest)
    base_trials = pd.read_csv(base_dir / "hpo_trials.tsv", sep="\t")
    base_eval = pd.read_csv(base_dir / "hpo_evaluation_long.tsv", sep="\t")
    base_keys = set(base_trials["config_key"].astype(str))

    output_trials_path = output_dir / "hpo_trials.tsv"
    output_eval_path = output_dir / "hpo_evaluation_long.tsv"
    if output_trials_path.exists() and not args.force:
        trials = pd.read_csv(output_trials_path, sep="\t")
        eval_df = pd.read_csv(output_eval_path, sep="\t") if output_eval_path.exists() else base_eval.copy()
    else:
        trials = base_trials.copy()
        eval_df = base_eval.copy()

    trial_rows = trials.to_dict(orient="records")
    eval_rows = eval_df.to_dict(orient="records")
    seen_keys = set(trials["config_key"].astype(str))
    trial_counter = int(pd.to_numeric(trials["trial_index"], errors="coerce").max())
    base_max_iteration = int(pd.to_numeric(base_trials["iteration"], errors="coerce").max())
    all_configs = _grid_configs(hpo)
    missing = [config for config in all_configs if config.key not in seen_keys]

    source_h5ad = base_dir / "prepared/source_hpo_input.h5ad"
    if not source_h5ad.exists():
        raise FileNotFoundError(f"Expected prepared source input from base screen: {source_h5ad}")

    shared_subset_cache = base_dir / "evaluation" / "_shared_subsets"
    manifest = {
        "algorithm": "Exhaustive completion of the finite SMITH HPO grid after the Visium-guided screen.",
        "base_screen_dir": str(base_dir),
        "base_n_trials": int(base_trials.shape[0]),
        "grid_size": len(all_configs),
        "n_seen_at_start": int(len(seen_keys)),
        "n_missing_at_start": int(len(missing)),
        "parameters": {
            "base_screen_parameters": params,
            "output_dir": str(output_dir),
            "batch_size_for_iteration": args.batch_size_for_iteration,
            "force": bool(args.force),
        },
        "leakage_policy": base_manifest.get("leakage_policy", {}),
        "shared_subset_cache": str(shared_subset_cache),
    }
    _write_checkpoint(output_dir=output_dir, trial_rows=trial_rows, eval_rows=eval_rows, manifest=manifest)

    new_counter = 0
    for config in missing:
        trial_counter += 1
        new_counter += 1
        trial_id = f"trial_{trial_counter:03d}"
        iteration = base_max_iteration + 1 + (new_counter - 1) // max(1, int(args.batch_size_for_iteration))
        run = hpo.run_smith_config(
            config=config,
            trial_id=trial_id,
            source_h5ad=source_h5ad,
            output_dir=output_dir / "smith_runs",
            panel_size=int(params["panel_size"]),
            epoch=int(params["epoch"]),
            record=int(params["record"]),
            device=str(params["device"]),
            seed=int(params["seed"]),
            batch_size=int(params["batch_size"]),
            max_cells=int(params["smith_max_cells"]),
            python_executable=str(params["python_executable"]),
            force=args.force,
        )
        row: dict[str, Any] = {
            "trial_index": trial_counter,
            "trial_id": trial_id,
            "iteration": iteration,
            "parent_trial_id": "",
            "mutation": "exhaustive_grid_completion",
            "config_key": config.key,
            "tasks": config.task_string,
            "dropout_rate": config.dropout_rate,
            "lam": config.lam,
            "sigma": config.sigma,
            "learning_rate": config.learning_rate,
            "status": run["status"],
            "run_dir": run.get("run_dir", ""),
            "rank_csv": run.get("rank_csv", ""),
            "panel_path": "",
            "stderr_tail": run.get("stderr_tail", ""),
        }
        if run["status"] in {"completed", "skipped_existing"}:
            panel_path = hpo.write_panel_from_rank(
                run["rank_csv"],
                output_dir / "panels" / f"{trial_id}_top_{int(params['panel_size'])}_panel.tsv",
                int(params["panel_size"]),
            )
            row["panel_path"] = panel_path
            trial_eval_rows: list[dict[str, Any]] = []
            for ref_id, ref_path in references:
                dataset_rows = hpo.evaluate_panel_on_dataset(
                    adata_file=ref_path,
                    dataset_id=ref_id,
                    dataset_role="visium_validation",
                    panel_path=panel_path,
                    output_dir=output_dir / "evaluation" / trial_id / "visium",
                    panel_size=int(params["panel_size"]),
                    label_column=str(params["visium_label_column"]),
                    seed=int(params["seed"]),
                    test_size=float(params["test_size"]),
                    max_cells=int(params["eval_max_cells"]),
                    min_label_count=int(params["eval_min_label_count"]),
                    class_weight=hpo._class_weight(str(params["class_weight"])),
                    subset_cache_dir=shared_subset_cache,
                )
                for item in dataset_rows:
                    item.update({"trial_id": trial_id, "iteration": iteration, "config_key": config.key})
                trial_eval_rows.extend(dataset_rows)
            eval_rows.extend(trial_eval_rows)
            row.update(hpo.aggregate_visium_metrics(trial_eval_rows))
        trial_rows.append(row)
        manifest["n_trials_written"] = len(trial_rows)
        manifest["n_new_trials_written"] = new_counter
        _write_checkpoint(output_dir=output_dir, trial_rows=trial_rows, eval_rows=eval_rows, manifest=manifest)

    final_trials = pd.DataFrame(trial_rows)
    manifest["n_trials_written"] = int(final_trials.shape[0])
    manifest["n_new_trials_written"] = int(new_counter)
    manifest["n_missing_after_completion"] = int(len([config for config in all_configs if config.key not in set(final_trials["config_key"].astype(str))]))
    _write_checkpoint(output_dir=output_dir, trial_rows=trial_rows, eval_rows=eval_rows, manifest=manifest)
    print(json.dumps({"trials_tsv": str(output_dir / "hpo_trials.tsv"), "manifest_json": str(output_dir / "hpo_exhaustive_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
