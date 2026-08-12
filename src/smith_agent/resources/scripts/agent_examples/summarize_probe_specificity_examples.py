#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRNA_RISK = PROJECT_ROOT / "outputs/scrna_probedealer_risk_scan_top1024/probe_risk_summary.tsv"
DEFAULT_MERFISH_FAILED = (
    PROJECT_ROOT / "outputs/agent_feasibility_filtering_merfish317/failed_probe_annotation_summary.tsv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/agent_feasibility_probe_examples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ProbeDealer cross-gene risk and annotation-rescue examples."
    )
    parser.add_argument("--scrna-risk", default=str(DEFAULT_SCRNA_RISK))
    parser.add_argument("--merfish-failed", default=str(DEFAULT_MERFISH_FAILED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-target-probes", type=int, default=20)
    parser.add_argument("--min-different-symbol-probes", type=int, default=20)
    parser.add_argument("--min-clean-different-symbol-probes", type=int, default=20)
    return parser.parse_args()


def parse_top_symbols(value: Any, target_symbol: str, max_items: int = 4) -> str:
    if pd.isna(value):
        return ""
    parsed: list[tuple[str, int]] = []
    for item in str(value).split(";"):
        if not item:
            continue
        if ":" not in item:
            continue
        symbol, count = item.rsplit(":", 1)
        symbol = symbol.strip()
        if not symbol or symbol == "?" or symbol == target_symbol:
            continue
        try:
            parsed.append((symbol, int(float(count))))
        except ValueError:
            continue
    parsed = sorted(parsed, key=lambda x: x[1], reverse=True)
    return ";".join(f"{symbol}:{count}" for symbol, count in parsed[:max_items])


def as_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(float(value))


def add_common_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        "initial_probe_count",
        "symbol_target_only_probe_count_known",
        "probes_with_known_different_symbol",
        "clean_different_symbol_probe_count",
        "unknown_symbol_probe_count",
        "no_hit_probe_count",
        "different_symbol_fraction_known",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["known_classified_probe_count"] = (
        df["symbol_target_only_probe_count_known"].fillna(0)
        + df["probes_with_known_different_symbol"].fillna(0)
    )
    df["target_probe_fraction_known"] = (
        df["symbol_target_only_probe_count_known"] / df["known_classified_probe_count"]
    ).where(df["known_classified_probe_count"] > 0)
    df["unknown_probe_fraction_initial"] = (
        df["unknown_symbol_probe_count"] / df["initial_probe_count"]
    ).where(df["initial_probe_count"] > 0)
    df["clean_offtarget_fraction_initial"] = (
        df["clean_different_symbol_probe_count"] / df["initial_probe_count"]
    ).where(df["initial_probe_count"] > 0)
    df["top_known_cross_symbols"] = [
        parse_top_symbols(value, target)
        for value, target in zip(df["top_cross_symbols"], df["gene_symbol"], strict=False)
    ]
    return df


def classify_scrna_examples(
    df: pd.DataFrame,
    min_target_probes: int,
    min_different_symbol_probes: int,
    min_clean_different_symbol_probes: int,
) -> pd.DataFrame:
    df = add_common_metrics(df)
    df["source_universe"] = "paired_snRNA_rank_top1024"
    df["example_class"] = "other"
    clean_risk = (
        df["resolved"].fillna(False).astype(bool)
        & (df["symbol_target_only_probe_count_known"].fillna(0) < min_target_probes)
        & (df["clean_different_symbol_probe_count"].fillna(0) >= min_clean_different_symbol_probes)
    )
    mixed_risk = (
        df["resolved"].fillna(False).astype(bool)
        & (df["symbol_target_only_probe_count_known"].fillna(0) < min_target_probes)
        & (df["probes_with_known_different_symbol"].fillna(0) >= min_different_symbol_probes)
        & ~clean_risk
    )
    annotation_rescue = (
        df["resolved"].fillna(False).astype(bool)
        & df["geneid_probedealer_fail"].fillna(False).astype(bool)
        & (df["symbol_target_only_probe_count_known"].fillna(0) >= min_target_probes)
    )
    df.loc[clean_risk, "example_class"] = "clean_cross_gene_risk"
    df.loc[mixed_risk, "example_class"] = "mixed_or_annotation_limited_cross_gene_risk"
    df.loc[annotation_rescue, "example_class"] = "gene_symbol_annotation_rescue"
    return df


def load_merfish_failed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.rename(
        columns={
            "symbol_target_only_count_known": "symbol_target_only_probe_count_known",
            "different_symbol_probe_count_known": "probes_with_known_different_symbol",
            "geneid_target_final": "geneid_target_only_probe_count",
        }
    )
    df["rank"] = pd.NA
    df["gene_id"] = pd.NA
    df["sequence_length"] = pd.NA
    df["clean_different_symbol_probe_count"] = pd.NA
    df["resolved"] = True
    df["geneid_probedealer_fail"] = df["geneid_target_only_probe_count"] < 20
    df["symbol_known_probedealer_fail"] = df["symbol_target_only_probe_count_known"] < 20
    return df


def classify_merfish_examples(df: pd.DataFrame, min_target_probes: int) -> pd.DataFrame:
    df = add_common_metrics(df)
    df["source_universe"] = "liver_merfish_317_gene_universe"
    df["example_class"] = "probe_or_annotation_failure"
    annotation_rescue = (
        df["geneid_target_only_probe_count"].fillna(0) < min_target_probes
    ) & (df["symbol_target_only_probe_count_known"].fillna(0) >= min_target_probes)
    cross_gene_risk = (
        df["symbol_target_only_probe_count_known"].fillna(0) < min_target_probes
    ) & (df["probes_with_known_different_symbol"].fillna(0) >= min_target_probes)
    unknown_limited = (
        df["symbol_target_only_probe_count_known"].fillna(0) < min_target_probes
    ) & (df["unknown_symbol_probe_count"].fillna(0) >= min_target_probes)
    df.loc[annotation_rescue, "example_class"] = "gene_symbol_annotation_rescue"
    df.loc[cross_gene_risk, "example_class"] = "cross_gene_specificity_risk"
    df.loc[unknown_limited & ~cross_gene_risk, "example_class"] = "annotation_limited_failure"
    return df


def select_manuscript_examples(scrna: pd.DataFrame, merfish: pd.DataFrame) -> pd.DataFrame:
    clean_scrna_all = scrna[scrna["example_class"] == "clean_cross_gene_risk"].copy()
    clean_priority = [
        "NDUFB2",
        "SRSF9",
        "PYCARD",
        "PTPRCAP",
        "SELENOH",
        "CCDC107",
        "SERF2",
        "ATP5PF",
        "EIF2S2",
        "PCBP2",
    ]
    curated_clean = clean_scrna_all[clean_scrna_all["gene_symbol"].isin(clean_priority)].copy()
    curated_clean["priority"] = curated_clean["gene_symbol"].map(
        {gene: i for i, gene in enumerate(clean_priority)}
    ).fillna(99)
    curated_clean = curated_clean.sort_values(["priority"]).head(8)
    fill_clean = clean_scrna_all.sort_values(
        [
            "clean_different_symbol_probe_count",
            "probes_with_known_different_symbol",
            "unknown_symbol_probe_count",
            "rank",
        ],
        ascending=[False, False, True, True],
    )
    clean_scrna = pd.concat([curated_clean, fill_clean], ignore_index=True, sort=False)
    clean_scrna = clean_scrna.drop_duplicates("gene_symbol").head(8)

    merfish_priority = merfish[
        merfish["example_class"].isin(["cross_gene_specificity_risk", "gene_symbol_annotation_rescue"])
    ].copy()
    merfish_priority["priority"] = merfish_priority["gene_symbol"].map(
        {
            "IGF1": 1,
            "ARHGEF26": 2,
            "CTSB": 3,
            "TCF7L1": 4,
        }
    ).fillna(99)
    merfish_priority = merfish_priority.sort_values(["priority", "probes_with_known_different_symbol"])

    selected = pd.concat(
        [
            merfish_priority.dropna(axis=1, how="all"),
            clean_scrna.dropna(axis=1, how="all"),
        ],
        ignore_index=True,
        sort=False,
    )
    selected = selected.drop_duplicates(["gene_symbol", "source_universe"], keep="first")
    return selected


def compact_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_universe",
        "example_class",
        "rank",
        "gene_symbol",
        "transcript_id",
        "initial_probe_count",
        "symbol_target_only_probe_count_known",
        "probes_with_known_different_symbol",
        "clean_different_symbol_probe_count",
        "unknown_symbol_probe_count",
        "no_hit_probe_count",
        "different_symbol_fraction_known",
        "target_probe_fraction_known",
        "unknown_probe_fraction_initial",
        "clean_offtarget_fraction_initial",
        "top_known_cross_symbols",
        "top_cross_symbols",
        "geneid_target_only_probe_count",
    ]
    return df[[col for col in columns if col in df.columns]].copy()


def write_summary(
    output_dir: Path,
    scrna: pd.DataFrame,
    merfish: pd.DataFrame,
    selected: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    clean_scrna = scrna[scrna["example_class"] == "clean_cross_gene_risk"]
    mixed_scrna = scrna[scrna["example_class"] == "mixed_or_annotation_limited_cross_gene_risk"]
    rescue_scrna = scrna[scrna["example_class"] == "gene_symbol_annotation_rescue"]
    rescue_merfish = merfish[merfish["example_class"] == "gene_symbol_annotation_rescue"]
    merfish_cross = merfish[merfish["example_class"] == "cross_gene_specificity_risk"]
    merfish_annotation = merfish[merfish["example_class"] == "annotation_limited_failure"]

    summary = {
        "inputs": {
            "scrna_risk_tsv": str(Path(args.scrna_risk).resolve()),
            "merfish_failed_tsv": str(Path(args.merfish_failed).resolve()),
        },
        "thresholds": {
            "min_target_probes": args.min_target_probes,
            "min_different_symbol_probes": args.min_different_symbol_probes,
            "min_clean_different_symbol_probes": args.min_clean_different_symbol_probes,
        },
        "scrna_top1024": {
            "n_total": int(len(scrna)),
            "n_resolved": int(scrna["resolved"].fillna(False).sum()),
            "n_symbol_level_probedealer_fail": int(scrna["symbol_known_probedealer_fail"].fillna(False).sum()),
            "n_clean_cross_gene_risk": int(len(clean_scrna)),
            "n_mixed_cross_gene_risk": int(len(mixed_scrna)),
            "n_gene_symbol_annotation_rescue": int(len(rescue_scrna)),
        },
        "merfish_317_failed_probe_cases": {
            "n_failed_probe_cases": int(len(merfish)),
            "n_cross_gene_specificity_risk": int(len(merfish_cross)),
            "n_gene_symbol_annotation_rescue": int(len(rescue_merfish)),
            "n_annotation_limited_failure": int(len(merfish_annotation)),
        },
        "selected_example_genes": selected["gene_symbol"].astype(str).tolist(),
    }
    (output_dir / "probe_specificity_example_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Probe specificity example summary",
        "",
        "This summary reuses existing ProbeDealer BLAST outputs and does not rerun probe design.",
        "",
        "## Key counts",
        "",
        f"- scRNA ranked top-1024 genes: {summary['scrna_top1024']['n_resolved']} resolved of {summary['scrna_top1024']['n_total']}.",
        f"- scRNA symbol-level ProbeDealer failures: {summary['scrna_top1024']['n_symbol_level_probedealer_fail']}.",
        f"- scRNA clean cross-gene risk examples: {summary['scrna_top1024']['n_clean_cross_gene_risk']}.",
        f"- MERFISH failed probe cases with true cross-gene risk: {summary['merfish_317_failed_probe_cases']['n_cross_gene_specificity_risk']}.",
        f"- MERFISH annotation-rescue cases: {summary['merfish_317_failed_probe_cases']['n_gene_symbol_annotation_rescue']}.",
        "",
        "## Recommended examples",
        "",
    ]
    for row in compact_columns(selected).itertuples(index=False):
        target = as_int(getattr(row, "symbol_target_only_probe_count_known", 0))
        diff = as_int(getattr(row, "probes_with_known_different_symbol", 0))
        unknown = as_int(getattr(row, "unknown_symbol_probe_count", 0))
        cross = getattr(row, "top_known_cross_symbols", "")
        lines.append(
            f"- {row.gene_symbol}: {row.example_class}; target-only={target}, known off-target={diff}, unknown={unknown}, top known cross={cross or 'none'}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use IGF1-like/cross-gene examples to show why sequence-level specificity filtering is not redundant with expression-based panel selection.",
            "- Use gene-symbol annotation-rescue examples to show why ProbeDealer output should be interpreted at the assay target-symbol level, not only at raw Ensembl gene-ID level.",
            "- Unknown-symbol-heavy cases are useful for supplement or manual-review discussion, but they are weaker as main figure examples.",
        ]
    )
    (output_dir / "probe_specificity_example_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scrna = pd.read_csv(args.scrna_risk, sep="\t")
    scrna = classify_scrna_examples(
        scrna,
        min_target_probes=args.min_target_probes,
        min_different_symbol_probes=args.min_different_symbol_probes,
        min_clean_different_symbol_probes=args.min_clean_different_symbol_probes,
    )
    merfish = load_merfish_failed(Path(args.merfish_failed))
    merfish = classify_merfish_examples(merfish, min_target_probes=args.min_target_probes)
    selected = select_manuscript_examples(scrna, merfish)

    compact_columns(scrna).to_csv(output_dir / "scrna_probe_specificity_examples_all.tsv", sep="\t", index=False)
    compact_columns(
        scrna[scrna["example_class"] == "clean_cross_gene_risk"].sort_values(
            [
                "clean_different_symbol_probe_count",
                "probes_with_known_different_symbol",
                "unknown_symbol_probe_count",
                "rank",
            ],
            ascending=[False, False, True, True],
        )
    ).to_csv(output_dir / "scrna_clean_cross_gene_risk_examples.tsv", sep="\t", index=False)
    compact_columns(merfish).to_csv(output_dir / "merfish_failed_probe_examples_classified.tsv", sep="\t", index=False)
    compact_columns(selected).to_csv(output_dir / "manuscript_probe_specificity_examples.tsv", sep="\t", index=False)
    write_summary(output_dir, scrna, merfish, selected, args)
    print(f"Wrote probe specificity example summary to {output_dir}")


if __name__ == "__main__":
    main()
