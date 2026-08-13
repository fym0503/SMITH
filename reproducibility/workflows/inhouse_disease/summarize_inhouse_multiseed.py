#!/usr/bin/env python3
"""Summarize five-seed in-house transfer rankings as CSV tables."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import numpy as np


COMPARISONS = (
    ("AD-HC / X5704", "AD", "X4870NOHC", "X5704ADHC"),
    ("AD-HC / X5789", "AD", "X4870NOHC", "X5789ADHC"),
    ("AD-FC / X5665", "AD", "X4996NOFC", "X5665ADFC"),
    ("PD-FC / X5215", "PD", "X4996NOFC", "X5215PDFC"),
)
EXAMPLE_GENES = ("MCAM", "CD74", "OLIG2", "MOG")
METRICS = (
    "sc_spearman",
    "aligned_spearman",
    "delta_spearman",
    "sc_mae",
    "aligned_mae",
    "delta_mae",
    "top32_sc",
    "top32_aligned",
    "delta_top32",
    "top64_sc",
    "top64_aligned",
    "delta_top64",
    "top128_sc",
    "top128_aligned",
    "delta_top128",
    "top64_gained",
    "top64_lost",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/fanyimin/smith_inhouse_transfer_corrected_20260730"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=range(5))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_ranking(path: Path) -> list[str]:
    genes = [str(gene).upper() for gene in ast.literal_eval(path.read_text())]
    if len(genes) != 293 or len(set(genes)) != 293:
        raise ValueError(f"Expected 293 unique genes in {path}, found {len(genes)}")
    return genes


def rank_path(root: Path, cohort: str, method: str, sample: str, seed: int) -> Path:
    selection = "selection" if cohort == "AD" else "selection_pd"
    if method == "sc":
        cohort_name = "allAD" if cohort == "AD" else "allPD"
        relative = f"{selection}/sc_{cohort_name}/seed{seed}/index/epoch:999.txt"
    elif method == "aligned":
        relative = f"{selection}/{sample}/seed{seed}/index/epoch:999.txt"
    elif method == "true_st":
        relative = f"{selection}/direct_ST/{sample}/seed{seed}/index/epoch:999.txt"
    else:
        raise ValueError(method)
    return root / relative


def positions(genes: list[str]) -> dict[str, int]:
    return {gene: index + 1 for index, gene in enumerate(genes)}


def calculate_metrics(
    comparison: str,
    cohort: str,
    source: str,
    target: str,
    seed: int,
    sc_genes: list[str],
    aligned_genes: list[str],
    true_genes: list[str],
) -> dict[str, object]:
    universe = set(sc_genes)
    if set(aligned_genes) != universe or set(true_genes) != universe:
        raise ValueError(f"Gene universe mismatch for {comparison}, seed {seed}")

    sc_pos = positions(sc_genes)
    aligned_pos = positions(aligned_genes)
    true_pos = positions(true_genes)
    ordered = sorted(universe)
    true_vector = np.asarray([true_pos[gene] for gene in ordered], dtype=float)
    sc_vector = np.asarray([sc_pos[gene] for gene in ordered], dtype=float)
    aligned_vector = np.asarray([aligned_pos[gene] for gene in ordered], dtype=float)

    row: dict[str, object] = {
        "comparison": comparison,
        "cohort": cohort,
        "source_st": source,
        "target_st": target,
        "seed": seed,
        "n_genes": len(ordered),
        "sc_spearman": float(np.corrcoef(sc_vector, true_vector)[0, 1]),
        "aligned_spearman": float(np.corrcoef(aligned_vector, true_vector)[0, 1]),
        "sc_mae": float(np.mean(np.abs(sc_vector - true_vector))),
        "aligned_mae": float(np.mean(np.abs(aligned_vector - true_vector))),
    }
    row["delta_spearman"] = row["aligned_spearman"] - row["sc_spearman"]
    row["delta_mae"] = row["aligned_mae"] - row["sc_mae"]

    for k in (32, 64, 128):
        truth = set(true_genes[:k])
        sc_top = set(sc_genes[:k])
        aligned_top = set(aligned_genes[:k])
        row[f"top{k}_sc"] = len(sc_top & truth)
        row[f"top{k}_aligned"] = len(aligned_top & truth)
        row[f"delta_top{k}"] = row[f"top{k}_aligned"] - row[f"top{k}_sc"]
        if k == 64:
            row["top64_gained"] = len((aligned_top & truth) - sc_top)
            row["top64_lost"] = len((sc_top & truth) - aligned_top)
    return row


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def main() -> None:
    args = parse_args()
    per_seed: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    missing: list[Path] = []

    for comparison, cohort, source, target in COMPARISONS:
        for seed in args.seeds:
            paths = {
                "sc": rank_path(args.root, cohort, "sc", "", seed),
                "aligned": rank_path(args.root, cohort, "aligned", source, seed),
                "true_st": rank_path(args.root, cohort, "true_st", target, seed),
            }
            absent = [path for path in paths.values() if not path.exists()]
            if absent:
                missing.extend(absent)
                if args.allow_incomplete:
                    continue
                raise FileNotFoundError("\n".join(map(str, absent)))

            rankings = {name: read_ranking(path) for name, path in paths.items()}
            row = calculate_metrics(
                comparison,
                cohort,
                source,
                target,
                seed,
                rankings["sc"],
                rankings["aligned"],
                rankings["true_st"],
            )
            per_seed.append(row)

            if source == "X4870NOHC":
                rank_maps = {name: positions(genes) for name, genes in rankings.items()}
                for gene in EXAMPLE_GENES:
                    examples.append(
                        {
                            "comparison": comparison,
                            "source_st": source,
                            "target_st": target,
                            "seed": seed,
                            "gene": gene,
                            "sc_rank": rank_maps["sc"][gene],
                            "aligned_rank": rank_maps["aligned"][gene],
                            "true_st_rank": rank_maps["true_st"][gene],
                            "sc_abs_error": abs(rank_maps["sc"][gene] - rank_maps["true_st"][gene]),
                            "aligned_abs_error": abs(rank_maps["aligned"][gene] - rank_maps["true_st"][gene]),
                        }
                    )

    output = args.root / "qc" / "multiseed_rank_robustness"
    base_fields = ["comparison", "cohort", "source_st", "target_st", "seed", "n_genes"]
    write_csv(output / "per_seed_metrics.csv", per_seed, base_fields + list(METRICS))
    if examples:
        write_csv(
            output / "example_gene_ranks.csv",
            examples,
            [
                "comparison",
                "source_st",
                "target_st",
                "seed",
                "gene",
                "sc_rank",
                "aligned_rank",
                "true_st_rank",
                "sc_abs_error",
                "aligned_abs_error",
            ],
        )

    summary: list[dict[str, object]] = []
    for comparison, cohort, source, target in COMPARISONS:
        group = [row for row in per_seed if row["comparison"] == comparison]
        if not group:
            continue
        summary_row: dict[str, object] = {
            "comparison": comparison,
            "cohort": cohort,
            "source_st": source,
            "target_st": target,
            "n_seeds": len(group),
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in group], dtype=float)
            summary_row[f"{metric}_mean"] = float(values.mean())
            summary_row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        summary_row["spearman_improved_seeds"] = sum(row["delta_spearman"] > 0 for row in group)
        summary_row["mae_improved_seeds"] = sum(row["delta_mae"] < 0 for row in group)
        summary_row["top64_improved_seeds"] = sum(row["delta_top64"] > 0 for row in group)
        summary.append(summary_row)

    summary_fields = ["comparison", "cohort", "source_st", "target_st", "n_seeds"]
    for metric in METRICS:
        summary_fields.extend((f"{metric}_mean", f"{metric}_sd"))
    summary_fields.extend(("spearman_improved_seeds", "mae_improved_seeds", "top64_improved_seeds"))
    write_csv(output / "summary_metrics.csv", summary, summary_fields)

    print(f"Wrote {len(per_seed)} comparison-seed rows to {output}")
    if missing:
        print(f"Skipped {len(set(missing))} missing rank files")


if __name__ == "__main__":
    main()
