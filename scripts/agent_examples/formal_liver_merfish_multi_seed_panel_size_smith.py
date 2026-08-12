from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from formal_liver_merfish_multi_visium_smith import (  # noqa: E402
    DEFAULT_MERFISH,
    DEFAULT_SOURCE,
    _copy_rank_csv_as_tsv,
    _load_merfish_gene_universe,
    _safe_id,
    parse_reference_args,
    prepare_smith_input,
    run_smith,
)
from smith_agent.benchmarking import evaluate_panel_cell_type_classification  # noqa: E402
from smith_agent.panel_rank_aggregation import aggregate_reference_panel_ranks  # noqa: E402
from smith_agent.utils import ensure_dir, write_json  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_5seed_panels"


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"Expected a comma-separated list of positive integers, got `{raw}`.")
    return values


def _write_top_panel(rank_tsv: str | Path, output_tsv: str | Path, panel_size: int, rank_column: str = "") -> str:
    rank_tsv = Path(rank_tsv)
    df = pd.read_csv(rank_tsv, sep="\t")
    if rank_column and rank_column in df.columns:
        df = df.sort_values(rank_column, ascending=True)
    gene_column = next((col for col in df.columns if str(col).lower() in {"gene_symbol", "gene", "marker"}), df.columns[0])
    out = pd.DataFrame({"gene_symbol": df[gene_column].astype(str).str.upper().head(panel_size)})
    out.insert(0, "rank", range(1, out.shape[0] + 1))
    out["panel_size"] = int(panel_size)
    output_tsv = Path(output_tsv)
    ensure_dir(output_tsv.parent)
    out.to_csv(output_tsv, sep="\t", index=False)
    return str(output_tsv)


def _metric_rows(
    *,
    training_seed: int,
    panel_size: int,
    panel: str,
    result: Any,
) -> list[dict[str, Any]]:
    payload = result.to_dict()
    rows = []
    for metric, value in payload["metrics"].items():
        rows.append(
            {
                "training_seed": int(training_seed),
                "panel_size": int(panel_size),
                "panel": panel,
                "metric": metric,
                "value": float(value),
                "n_shared_genes": len(payload["shared_genes"]),
                "n_test_cells": int(payload["test_cells"]),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-adata", default=str(DEFAULT_SOURCE))
    parser.add_argument("--source-label-column", default="Cell_Type_final")
    parser.add_argument("--merfish-adata", default=str(DEFAULT_MERFISH))
    parser.add_argument("--visium-reference", action="append", help="Reference as name=/path/to/file.h5ad.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--panel-sizes", default="32,64,128")
    parser.add_argument("--training-seeds", default="1,2,3,4,5")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--record", type=int, default=50)
    parser.add_argument("--source-max-cells", type=int, default=30000)
    parser.add_argument("--visium-max-cells", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-label-count", type=int, default=5)
    parser.add_argument("--source-weight", type=float, default=0.50)
    parser.add_argument("--reference-weight", type=float, default=0.50)
    parser.add_argument("--min-reference-support", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    panel_sizes = sorted(_parse_int_list(args.panel_sizes))
    max_panel_size = max(panel_sizes)
    training_seeds = _parse_int_list(args.training_seeds)
    references = parse_reference_args(args.visium_reference)
    gene_universe = _load_merfish_gene_universe(args.merfish_adata)

    prepared_dir = ensure_dir(output_dir / "prepared")
    source_prepared = prepare_smith_input(
        args.source_adata,
        prepared_dir / "source_scrna_smith_input.h5ad",
        gene_universe,
        label_column=str(args.source_label_column),
        require_spatial=False,
        drop_unknown_labels=True,
        min_label_count=args.min_label_count,
        max_cells=args.source_max_cells,
        seed=training_seeds[0],
    )
    prepared_refs: list[tuple[str, dict[str, Any]]] = []
    for name, path in references:
        prepared = prepare_smith_input(
            path,
            prepared_dir / f"{_safe_id(name)}_smith_input.h5ad",
            gene_universe,
            label_column="cell_type",
            require_spatial=True,
            drop_unknown_labels=True,
            min_label_count=args.min_label_count,
            max_cells=args.visium_max_cells if args.visium_max_cells > 0 else None,
            seed=training_seeds[0],
        )
        prepared_refs.append((name, prepared))

    all_metric_rows: list[dict[str, Any]] = []
    seed_results: list[dict[str, Any]] = []
    for seed in training_seeds:
        seed_dir = ensure_dir(output_dir / f"seed_{seed}")
        ranks_dir = ensure_dir(seed_dir / "smith_ranks")
        source_run = run_smith(
            name="source_scrna_smith",
            adata_file=Path(source_prepared["output_h5ad"]),
            output_dir=seed_dir / "smith_runs",
            tasks="recon,cls",
            panel_size=max_panel_size,
            epoch=args.epoch,
            record=args.record,
            device=args.device,
            seed=seed,
            batch_size=args.batch_size,
            balance_mode="capped",
            balance_cap=500,
            max_cells=None,
            sampling_strategy="celltype",
            force=args.force,
        )
        source_rank_tsv = _copy_rank_csv_as_tsv(
            source_run["rank_csv"],
            ranks_dir / "source_scrna_smith_rank.tsv",
            "source_scrna_smith",
        )

        reference_rank_tsvs: list[str] = []
        reference_runs: list[dict[str, Any]] = []
        reference_ids: list[str] = []
        for name, prepared in prepared_refs:
            run = run_smith(
                name=name,
                adata_file=Path(prepared["output_h5ad"]),
                output_dir=seed_dir / "smith_runs",
                tasks="recon,cls,standard_coordination",
                panel_size=max_panel_size,
                epoch=args.epoch,
                record=args.record,
                device=args.device,
                seed=seed,
                batch_size=args.batch_size,
                balance_mode="capped",
                balance_cap=500,
                max_cells=None,
                sampling_strategy="celltype_spatial",
                force=args.force,
            )
            rank_tsv = _copy_rank_csv_as_tsv(
                run["rank_csv"],
                ranks_dir / f"{_safe_id(name)}_smith_rank.tsv",
                name,
            )
            reference_rank_tsvs.append(rank_tsv)
            reference_runs.append({**run, "prepared": prepared, "rank_tsv": rank_tsv})
            reference_ids.append(name)

        aggregation = aggregate_reference_panel_ranks(
            output_dir=seed_dir / "formal_rank_aggregation",
            source_rank_file=source_rank_tsv,
            reference_rank_files=reference_rank_tsvs,
            reference_ids=reference_ids,
            panel_size=max_panel_size,
            source_weight=args.source_weight,
            reference_weight=args.reference_weight,
            min_reference_support=args.min_reference_support,
            gene_universe="source",
            restrict_gene_symbols=gene_universe,
        )

        panel_paths: dict[str, dict[int, str]] = {"source_smith": {}, "multi_visium_smith": {}}
        for panel_size in panel_sizes:
            panel_paths["source_smith"][panel_size] = _write_top_panel(
                aggregation["source_rank_tsv"],
                seed_dir / "panels" / f"source_top_{panel_size}_panel.tsv",
                panel_size,
                rank_column="rank",
            )
            panel_paths["multi_visium_smith"][panel_size] = _write_top_panel(
                aggregation["integrated_rank_tsv"],
                seed_dir / "panels" / f"multi_visium_top_{panel_size}_panel.tsv",
                panel_size,
                rank_column="integrated_rank",
            )

        for panel_size in panel_sizes:
            for panel_name, panel_path in (
                ("source_smith", panel_paths["source_smith"][panel_size]),
                ("multi_visium_smith", panel_paths["multi_visium_smith"][panel_size]),
            ):
                result = evaluate_panel_cell_type_classification(
                    adata_file=args.merfish_adata,
                    panel_path=panel_path,
                    output_dir=seed_dir / "evaluation" / panel_name / f"panel_{panel_size}",
                    panel_size=panel_size,
                    label_column="Cell_Type",
                    seed=args.eval_seed,
                )
                all_metric_rows.extend(
                    _metric_rows(
                        training_seed=seed,
                        panel_size=panel_size,
                        panel=panel_name,
                        result=result,
                    )
                )

        seed_result = {
            "training_seed": seed,
            "source_run": source_run,
            "reference_runs": reference_runs,
            "source_rank_tsv": source_rank_tsv,
            "reference_rank_tsvs": reference_rank_tsvs,
            "aggregation": aggregation,
            "panel_paths": panel_paths,
        }
        seed_results.append(seed_result)
        write_json(seed_dir / "seed_result.json", seed_result)

        summary_df = pd.DataFrame(all_metric_rows)
        summary_df.to_csv(output_dir / "multi_seed_panel_size_metrics.tsv", sep="\t", index=False)

    summary_df = pd.DataFrame(all_metric_rows)
    metric_summary = (
        summary_df.groupby(["panel", "panel_size", "metric"], as_index=False)["value"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    summary_tsv = output_dir / "multi_seed_panel_size_metrics.tsv"
    metric_summary_tsv = output_dir / "multi_seed_panel_size_metric_summary.tsv"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    metric_summary.to_csv(metric_summary_tsv, sep="\t", index=False)
    payload = {
        "source_adata": str(args.source_adata),
        "merfish_adata": str(args.merfish_adata),
        "panel_sizes": panel_sizes,
        "training_seeds": training_seeds,
        "eval_seed": int(args.eval_seed),
        "max_panel_size": int(max_panel_size),
        "source_prepared": source_prepared,
        "prepared_refs": prepared_refs,
        "seed_results": seed_results,
        "metrics_tsv": str(summary_tsv),
        "metric_summary_tsv": str(metric_summary_tsv),
    }
    write_json(output_dir / "multi_seed_panel_size_result.json", payload)
    print(json.dumps({"metrics_tsv": str(summary_tsv), "metric_summary_tsv": str(metric_summary_tsv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
