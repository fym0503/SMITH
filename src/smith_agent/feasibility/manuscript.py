"""In-memory analysis helpers for the manuscript's Agent feasibility panels.

The backend runners may materialize audit files, but these functions consume the
freshly returned tables and never use a reference output as an analysis input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .integration import IntegrationThresholds


MANUSCRIPT_FIGURE6G_GENES = (
    "JPX",
    "RGPD5",
    "RGPD2",
    "TMEM140",
    "IGF1",
    "UTP14C",
    "SERF2",
    "NDUFB2",
    "SRSF9",
    "PTPRCAP",
)


def ranked_gene_universe(
    ranking: pd.DataFrame,
    *,
    gene_column: str = "gene_symbol",
    max_genes: int | None = None,
) -> pd.DataFrame:
    """Return a stable, duplicate-free ranked gene universe from a SMITH ranking."""
    if gene_column not in ranking.columns:
        gene_column = next(
            (
                column
                for column in ranking.columns
                if str(column).lower() in {"marker", "gene", "gene_symbol", "target", "feature_name", "gene_name"}
            ),
            "",
        )
        if not gene_column:
            raise KeyError("Ranking must contain a gene-symbol column.")
    frame = ranking.copy()
    frame[gene_column] = frame[gene_column].astype(str).str.strip()
    frame = frame.loc[frame[gene_column].ne("") & frame[gene_column].ne("nan")]
    if "rank" in frame.columns:
        frame = frame.sort_values("rank", kind="stable")
    frame = frame.drop_duplicates(gene_column, keep="first").reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1)) if "rank" not in frame else None
    if max_genes is not None and max_genes > 0:
        frame = frame.head(max_genes).copy()
    frame["rank"] = np.arange(1, len(frame) + 1)
    if gene_column != "gene_symbol":
        frame = frame.rename(columns={gene_column: "gene_symbol"})
    return frame[["rank", "gene_symbol"] + [c for c in frame.columns if c not in {"rank", "gene_symbol"}]]


def _bool_gate(frame: pd.DataFrame, name: str, count_column: str, threshold: int) -> pd.Series:
    if name in frame.columns:
        return frame[name].fillna(False).astype(bool)
    if count_column not in frame.columns:
        raise KeyError(f"Feasibility table needs `{name}` or `{count_column}`.")
    return pd.to_numeric(frame[count_column], errors="coerce").fillna(0).ge(threshold)


def manuscript_pass_rates(
    feasibility: pd.DataFrame,
    *,
    thresholds: IntegrationThresholds = IntegrationThresholds(),
) -> pd.DataFrame:
    """Compute the four Figure 6f pass-rate records from a fresh table."""
    frame = feasibility.copy()
    if frame.empty:
        raise ValueError("Feasibility table is empty.")
    transcript = frame.get("transcript_resolved")
    if transcript is None:
        transcript = frame.get("pass_transcript")
    if transcript is None:
        transcript = frame.get("transcript_id", pd.Series(index=frame.index, dtype=object)).notna()
    transcript = transcript.fillna(False).astype(bool)
    gates = {
        "ODT": _bool_gate(frame, "pass_odt_property_20", "odt_property_probe_count", thresholds.min_property_probes),
        "OligoMiner": _bool_gate(
            frame,
            "pass_oligominer_geneaware_10",
            "oligominer_geneaware_specific_probe_count",
            thresholds.min_specific_probes,
        ),
        "ProbeDealer": _bool_gate(
            frame,
            "pass_probedealer_target_20",
            "probedealer_target_final_probe_count",
            thresholds.min_deployment_probes,
        ),
    }
    gates["Integrated"] = transcript & gates["ODT"] & gates["OligoMiner"] & gates["ProbeDealer"]
    rows = []
    total = int(len(frame))
    for tool in ("ODT", "OligoMiner", "ProbeDealer", "Integrated"):
        passed = int(gates[tool].sum())
        rows.append(
            {
                "tool": tool,
                "pass_count": passed,
                "total_count": total,
                "pass_fraction": passed / total,
                "pass_percent": passed / total * 100.0,
            }
        )
    return pd.DataFrame(rows)


def _parse_cross_symbol(value: Any, target: str) -> str:
    if pd.isna(value):
        return ""
    for item in str(value).split(";"):
        if ":" not in item:
            continue
        symbol, _count = item.rsplit(":", 1)
        symbol = symbol.strip()
        if symbol and symbol != "?" and symbol.upper() != target.upper():
            return symbol
    return ""


def manuscript_offtarget_examples(
    risk: pd.DataFrame,
    *,
    genes: Iterable[str] = MANUSCRIPT_FIGURE6G_GENES,
    strict: bool = True,
) -> pd.DataFrame:
    """Build the mutually exclusive stacked-bar records for Figure 6g."""
    if "gene_symbol" not in risk.columns:
        raise KeyError("ProbeDealer risk table must contain `gene_symbol`.")
    frame = risk.copy()
    frame["gene_symbol"] = frame["gene_symbol"].astype(str)
    wanted = [str(gene) for gene in genes]
    missing = [gene for gene in wanted if gene not in set(frame["gene_symbol"])]
    if missing and strict:
        raise ValueError("Figure 6g genes are missing from the current ranked universe: " + ", ".join(missing))
    selected = frame.set_index("gene_symbol").reindex(wanted).dropna(how="all").reset_index()
    if selected.empty:
        raise ValueError("No Figure 6g representative genes were found.")

    def numeric(*columns: str) -> pd.Series:
        for column in columns:
            if column in selected.columns:
                return pd.to_numeric(selected[column], errors="coerce").fillna(0)
        return pd.Series(0.0, index=selected.index)

    target = numeric("symbol_target_only_probe_count_known", "probedealer_target_final_probe_count")
    off = numeric("clean_different_symbol_probe_count", "probes_with_known_different_symbol", "probedealer_known_offtarget_probe_count")
    unknown = numeric("unknown_symbol_probe_count", "probedealer_unknown_symbol_probe_count") + numeric("no_hit_probe_count", "probedealer_no_hit_probe_count")
    selected["target_compatible"] = target
    selected["known_offtarget"] = off
    selected["unknown"] = unknown
    if "initial_probe_count" in selected.columns:
        initial = pd.to_numeric(selected["initial_probe_count"], errors="coerce")
        observed = selected["target_compatible"] + selected["known_offtarget"] + selected["unknown"]
        invalid = initial.notna() & initial.ne(observed)
        if invalid.any():
            bad = selected.loc[invalid, "gene_symbol"].tolist()
            raise ValueError("Figure 6g probe classes do not conserve initial counts for: " + ", ".join(bad))
    selected["total_probe_count"] = selected["target_compatible"] + selected["known_offtarget"] + selected["unknown"]
    cross_values = selected["top_cross_symbols"] if "top_cross_symbols" in selected.columns else pd.Series("", index=selected.index)
    selected["offtarget_symbol"] = [
        _parse_cross_symbol(value, gene)
        for value, gene in zip(cross_values, selected["gene_symbol"], strict=True)
    ]
    selected["example_order"] = np.arange(len(selected))
    return selected[["example_order", "gene_symbol", "target_compatible", "known_offtarget", "unknown", "total_probe_count", "offtarget_symbol"]]


def write_probe_audit_tables(
    output_dir: str | Path,
    feasibility: pd.DataFrame,
    risk: pd.DataFrame,
    pass_rates: pd.DataFrame,
    examples: pd.DataFrame,
) -> dict[str, str]:
    """Persist current-run artifacts without making them notebook inputs."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "feasibility": root / "three_tool_feasibility_table.tsv",
        "risk": root / "probe_risk_summary.tsv",
        "pass_rates": root / "figure6_f_pass_rates.tsv",
        "examples": root / "figure6_g_offtarget_examples.tsv",
    }
    feasibility.to_csv(paths["feasibility"], sep="\t", index=False)
    risk.to_csv(paths["risk"], sep="\t", index=False)
    pass_rates.to_csv(paths["pass_rates"], sep="\t", index=False)
    examples.to_csv(paths["examples"], sep="\t", index=False)
    return {key: str(path) for key, path in paths.items()}
