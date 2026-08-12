#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SMITH_ROOT = Path(os.environ.get("SMITH_ROOT", str(REPO_ROOT / "scripts")))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smith_agent.benchmarking import (  # noqa: E402
    evaluate_panel_cell_type_classification,
    evaluate_panel_coordinate_regression,
)
from smith_agent.utils import ensure_dir, write_json  # noqa: E402


DEFAULT_SOURCE = REPO_ROOT / "data/liver_merfish/adata_healthy_nucseq.h5ad"
DEFAULT_MERFISH = REPO_ROOT / "data/liver_merfish/adata_healthy_merfish.h5ad"
DEFAULT_VISIUM_REFERENCES = [
    (
        "PSC011_C1_visium",
        REPO_ROOT
        / "outputs/visium5_pilot/materialized/0cc59004-2b35-4767-8278-83e097ef32d1/0cc59004-2b35-4767-8278-83e097ef32d1.h5ad",
    ),
    (
        "C73_B1_healthy_visium",
        REPO_ROOT
        / "outputs/visium5_pilot/materialized/1fa9f77e-b79e-4ca6-b294-dcb9f47e5825/1fa9f77e-b79e-4ca6-b294-dcb9f47e5825.h5ad",
    ),
    (
        "H35_sample1_visium",
        REPO_ROOT
        / "outputs/visium5_pilot/materialized/266885f6-34bb-4ace-971d-fe18a2de27d4/266885f6-34bb-4ace-971d-fe18a2de27d4.h5ad",
    ),
    (
        "WSSS_F_IMMsp9838712_visium",
        REPO_ROOT
        / "outputs/visium5_pilot/materialized/2d7de988-0325-4c19-9eb8-808bcd1f594d/2d7de988-0325-4c19-9eb8-808bcd1f594d.h5ad",
    ),
    (
        "H35_sample2_visium",
        REPO_ROOT
        / "outputs/visium5_pilot/materialized/31dd9980-8c9b-4b9a-83ea-114e5fbdddee/31dd9980-8c9b-4b9a-83ea-114e5fbdddee.h5ad",
    ),
]

OBJECTIVES = ("visium_mean_cell_type_accuracy", "visium_mean_spatial_pearson")
TASK_GRID = (("cls",), ("recon", "cls"), ("recon",))
FLOAT_GRIDS = {
    "dropout_rate": (0.1, 0.2, 0.3),
    "lam": (0.25, 0.5, 1.0),
    "sigma": (0.3, 0.5, 0.8),
    "learning_rate": (0.0005, 0.001, 0.002),
}
UNKNOWN_LABELS = {"", "nan", "none", "unknown", "unassigned", "na"}


@dataclass(frozen=True)
class HPOConfig:
    tasks: tuple[str, ...] = ("recon", "cls")
    dropout_rate: float = 0.2
    lam: float = 0.5
    sigma: float = 0.5
    learning_rate: float = 0.001

    @property
    def key(self) -> str:
        task_text = "-".join(self.tasks)
        return (
            f"tasks={task_text}|dropout={self.dropout_rate:g}|lam={self.lam:g}|"
            f"sigma={self.sigma:g}|lr={self.learning_rate:g}"
        )

    @property
    def safe_id(self) -> str:
        return _safe_id(self.key.replace("|", "__").replace("=", "-"))

    @property
    def task_string(self) -> str:
        return ",".join(self.tasks)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _clean_gene_symbol(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text.upper()


def _load_merfish_gene_universe(path: str | Path) -> list[str]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        raw_genes = [_clean_gene_symbol(item) for item in adata.var_names.astype(str).tolist()]
    finally:
        adata.file.close()
    genes: list[str] = []
    seen: set[str] = set()
    for gene in raw_genes:
        if gene and gene not in seen:
            genes.append(gene)
            seen.add(gene)
    return genes


def _gene_symbols(adata: ad.AnnData) -> pd.Series:
    for column in ("feature_name", "gene_symbol", "gene_symbols", "gene_name", "gene_short_name", "symbol"):
        if column in adata.var.columns:
            return pd.Series(adata.var[column].astype(str).map(_clean_gene_symbol).to_numpy(), index=adata.var_names)
    return pd.Series(pd.Index(adata.var_names).astype(str).map(_clean_gene_symbol).to_numpy(), index=adata.var_names)


def _load_source_visium_shared_gene_universe(source: str | Path, references: list[tuple[str, Path]]) -> list[str]:
    source_adata = ad.read_h5ad(source, backed="r")
    try:
        shared = set(_gene_symbols(source_adata).astype(str))
    finally:
        source_adata.file.close()
    for _, path in references:
        ref = ad.read_h5ad(path, backed="r")
        try:
            shared &= set(_gene_symbols(ref).astype(str))
        finally:
            ref.file.close()
    return sorted(gene for gene in shared if gene)


def _select_gene_positions(adata: ad.AnnData, gene_universe: list[str]) -> tuple[list[str], list[int]]:
    symbols = _gene_symbols(adata)
    first_position: dict[str, int] = {}
    for idx, gene in enumerate(symbols.tolist()):
        if gene and gene not in first_position:
            first_position[gene] = idx
    genes: list[str] = []
    positions: list[int] = []
    for gene in gene_universe:
        clean = _clean_gene_symbol(gene)
        if clean in first_position:
            genes.append(clean)
            positions.append(first_position[clean])
    if not genes:
        raise ValueError("No source genes overlap the requested gene universe.")
    return genes, positions


def prepare_source_input(
    input_h5ad: str | Path,
    output_h5ad: str | Path,
    gene_universe: list[str],
    *,
    label_column: str,
    max_cells: int,
    seed: int,
) -> dict[str, Any]:
    input_h5ad = Path(input_h5ad)
    output_h5ad = Path(output_h5ad)
    ensure_dir(output_h5ad.parent)
    if output_h5ad.exists():
        return {"input_h5ad": str(input_h5ad), "output_h5ad": str(output_h5ad), "status": "existing"}

    source = ad.read_h5ad(input_h5ad)
    try:
        genes, positions = _select_gene_positions(source, gene_universe)
        obs_indices = np.arange(source.n_obs)
        if max_cells > 0 and obs_indices.size > max_cells:
            rng = np.random.default_rng(seed)
            labels = source.obs[label_column].astype(str).to_numpy() if label_column in source.obs else None
            if labels is None:
                obs_indices = np.sort(rng.choice(obs_indices, size=max_cells, replace=False))
            else:
                sampled: list[np.ndarray] = []
                counts = pd.Series(labels).value_counts()
                for label, count in counts.items():
                    label_positions = np.flatnonzero(labels == label)
                    n_take = max(1, int(round(max_cells * int(count) / source.n_obs)))
                    n_take = min(n_take, label_positions.size)
                    sampled.append(rng.choice(label_positions, size=n_take, replace=False))
                obs_indices = np.sort(np.concatenate(sampled))
        prepared = source[obs_indices, positions].copy()
    finally:
        del source

    if sparse.issparse(prepared.X):
        prepared.X = prepared.X.tocsr().astype(np.float32)
    else:
        prepared.X = np.asarray(prepared.X, dtype=np.float32)
    prepared.var_names = genes
    prepared.var = pd.DataFrame(index=pd.Index(genes, name=None))
    prepared.var["feature_name"] = genes
    if label_column in prepared.obs:
        prepared.obs["cell_type"] = prepared.obs[label_column].astype(str)
    prepared.write_h5ad(output_h5ad)
    return {
        "input_h5ad": str(input_h5ad),
        "output_h5ad": str(output_h5ad),
        "status": "created",
        "n_obs": int(prepared.n_obs),
        "n_vars": int(prepared.n_vars),
    }


def _latest_epoch_csv(saving_dir: Path) -> Path | None:
    epoch_files = list(saving_dir.glob("epoch_*.csv"))
    if not epoch_files:
        return None

    def epoch_number(path: Path) -> int:
        match = re.search(r"epoch_(\d+)\.csv$", path.name)
        return int(match.group(1)) if match else -1

    return max(epoch_files, key=epoch_number)


def _smith_command(
    *,
    python_executable: str,
    adata_file: Path,
    saving_dir: Path,
    log_dir: Path,
    config: HPOConfig,
    panel_size: int,
    epoch: int,
    record: int,
    device: str,
    seed: int,
    batch_size: int,
    max_cells: int,
) -> list[str]:
    command = [
        python_executable,
        str(SMITH_ROOT / "main.py"),
        "--adata_file",
        str(adata_file.resolve()),
        "--saving_dir",
        str(saving_dir.resolve()),
        "--log_dir",
        str(log_dir.resolve()),
        "--tasks",
        config.task_string,
        "--task_name",
        config.safe_id,
        "--panel_size",
        str(panel_size),
        "--epoch",
        str(epoch),
        "--record",
        str(record),
        "--device",
        device,
        "--seed",
        str(seed),
        "--learning_rate",
        str(config.learning_rate),
        "--batch_size",
        str(batch_size),
        "--dim",
        "32",
        "--rep_dim",
        "32",
        "--rep_hidden_dims",
        "32",
        "--head_hidden_dims",
        "",
        "--dropout_rate",
        str(config.dropout_rate),
        "--lam",
        str(config.lam),
        "--sigma",
        str(config.sigma),
        "--activation",
        "tanh",
        "--optimizer",
        "Adam",
        "--balance_mode",
        "capped",
        "--balance_cap",
        "500",
        "--sampling_strategy",
        "celltype",
    ]
    if max_cells > 0:
        command.extend(["--max_cells", str(max_cells)])
    return command


def run_smith_config(
    *,
    config: HPOConfig,
    trial_id: str,
    source_h5ad: str | Path,
    output_dir: str | Path,
    panel_size: int,
    epoch: int,
    record: int,
    device: str,
    seed: int,
    batch_size: int,
    max_cells: int,
    python_executable: str,
    force: bool,
) -> dict[str, Any]:
    run_dir = ensure_dir(Path(output_dir) / trial_id)
    saving_dir = ensure_dir(run_dir / "saving")
    log_dir = ensure_dir(run_dir / "logs")
    rank_csv = _latest_epoch_csv(saving_dir)
    if rank_csv is not None and not force:
        return {
            "status": "skipped_existing",
            "rank_csv": str(rank_csv),
            "run_dir": str(run_dir),
        }

    command = _smith_command(
        python_executable=python_executable,
        adata_file=Path(source_h5ad),
        saving_dir=saving_dir,
        log_dir=log_dir,
        config=config,
        panel_size=panel_size,
        epoch=epoch,
        record=record,
        device=device,
        seed=seed,
        batch_size=batch_size,
        max_cells=max_cells,
    )
    write_json(
        saving_dir / "smith_run_manifest.json",
        {
            "trial_id": trial_id,
            "config": asdict(config),
            "command": command,
            "panel_size": panel_size,
            "epoch": epoch,
            "record": record,
        },
    )
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, cwd=str(SMITH_ROOT), stdout=stdout, stderr=stderr, text=True)
    rank_csv = _latest_epoch_csv(saving_dir)
    if proc.returncode != 0 or rank_csv is None:
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        return {
            "status": "failed",
            "run_dir": str(run_dir),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "stderr_tail": stderr_text[-2000:],
        }
    return {
        "status": "completed",
        "rank_csv": str(rank_csv),
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def write_panel_from_rank(rank_csv: str | Path, output_tsv: str | Path, panel_size: int) -> str:
    df = pd.read_csv(rank_csv)
    gene_col = next((col for col in df.columns if str(col).lower() in {"marker", "gene", "gene_symbol"}), df.columns[0])
    genes = []
    seen: set[str] = set()
    for gene in df[gene_col].astype(str):
        cleaned = _clean_gene_symbol(gene)
        if cleaned and cleaned not in seen:
            genes.append(cleaned)
            seen.add(cleaned)
        if len(genes) >= panel_size:
            break
    panel = pd.DataFrame({"rank": np.arange(1, len(genes) + 1), "gene_symbol": genes})
    panel["panel_size"] = panel_size
    output_tsv = Path(output_tsv)
    ensure_dir(output_tsv.parent)
    panel.to_csv(output_tsv, sep="\t", index=False)
    return str(output_tsv)


def evaluate_panel_on_dataset(
    *,
    adata_file: str | Path,
    dataset_id: str,
    dataset_role: str,
    panel_path: str | Path,
    output_dir: str | Path,
    panel_size: int,
    label_column: str,
    seed: int,
    test_size: float,
    max_cells: int,
    min_label_count: int,
    class_weight: str | None,
    subset_cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    adata_path = Path(adata_file)
    eval_path = adata_path
    temp_path: Path | None = None
    if max_cells > 0 or min_label_count > 1:
        adata = ad.read_h5ad(adata_path)
        try:
            mask = np.ones(adata.n_obs, dtype=bool)
            if label_column in adata.obs and min_label_count > 1:
                labels = adata.obs[label_column].astype(str)
                valid = labels.notna() & ~labels.str.strip().str.lower().isin(UNKNOWN_LABELS)
                counts = labels[valid].value_counts()
                keep_labels = set(counts[counts >= int(min_label_count)].index.astype(str))
                mask &= valid.to_numpy(dtype=bool) & labels.isin(keep_labels).to_numpy(dtype=bool)
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                raise ValueError(f"No cells remain after label filtering for {dataset_id}.")
            if max_cells > 0 and indices.size > max_cells:
                if label_column in adata.obs:
                    indices = label_aware_sample_indices(
                        labels=adata.obs[label_column].astype(str),
                        candidate_indices=indices,
                        max_cells=max_cells,
                        min_label_count=min_label_count,
                        test_size=test_size,
                        seed=seed,
                    )
                else:
                    rng = np.random.default_rng(seed)
                    indices = np.sort(rng.choice(indices, size=max_cells, replace=False))
            if indices.size != adata.n_obs:
                max_tag = f"max{max_cells}" if max_cells > 0 else "all"
                cache_root = Path(subset_cache_dir) if subset_cache_dir is not None else Path(output_dir) / "_subsets"
                temp_path = ensure_dir(cache_root) / (
                    f"{_safe_id(dataset_id)}_{_safe_id(label_column)}_{max_tag}_"
                    f"minlabel{min_label_count}_seed{seed}.h5ad"
                )
                if not temp_path.exists():
                    subset = adata[indices, :].copy()
                    subset.write_h5ad(temp_path)
                eval_path = temp_path
        finally:
            del adata

    coord = evaluate_panel_coordinate_regression(
        adata_file=eval_path,
        panel_path=panel_path,
        output_dir=Path(output_dir) / _safe_id(dataset_id) / "coordinate",
        panel_size=panel_size,
        test_size=test_size,
        seed=seed,
    ).to_dict()
    cell = evaluate_panel_cell_type_classification(
        adata_file=eval_path,
        panel_path=panel_path,
        output_dir=Path(output_dir) / _safe_id(dataset_id) / "cell_type",
        panel_size=panel_size,
        label_column=label_column,
        test_size=test_size,
        seed=seed,
        class_weight=class_weight,
    ).to_dict()
    rows = [
        {
            "dataset_id": dataset_id,
            "dataset_role": dataset_role,
            "metric": "cell_type_accuracy",
            "value": float(cell["metrics"]["cell_type_accuracy"]),
            "n_shared_genes": len(cell["shared_genes"]),
            "train_cells": int(cell["train_cells"]),
            "test_cells": int(cell["test_cells"]),
            "label_column": label_column,
        },
        {
            "dataset_id": dataset_id,
            "dataset_role": dataset_role,
            "metric": "spatial_pearson",
            "value": float(coord["metrics"]["spatial_pearson"]),
            "n_shared_genes": len(coord["shared_genes"]),
            "train_cells": int(coord["train_cells"]),
            "test_cells": int(coord["test_cells"]),
            "label_column": "",
        },
    ]
    return rows


def aggregate_visium_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    df = pd.DataFrame(rows)
    visium = df[df["dataset_role"] == "visium_validation"]
    values: dict[str, float] = {}
    for metric in ("cell_type_accuracy", "spatial_pearson"):
        subset = visium[visium["metric"] == metric]
        values[f"visium_mean_{metric}"] = float(subset["value"].mean()) if not subset.empty else float("nan")
        values[f"visium_min_{metric}"] = float(subset["value"].min()) if not subset.empty else float("nan")
    return values


def label_aware_sample_indices(
    *,
    labels: pd.Series,
    candidate_indices: np.ndarray,
    max_cells: int,
    min_label_count: int,
    test_size: float,
    seed: int,
) -> np.ndarray:
    if max_cells <= 0 or candidate_indices.size <= max_cells:
        return np.sort(candidate_indices)

    min_per_label = max(2, int(min_label_count))
    candidate_labels = labels.iloc[candidate_indices].astype(str).to_numpy()
    counts = pd.Series(candidate_labels).value_counts()
    eligible = counts[counts >= min_per_label]
    if eligible.shape[0] < 2:
        return np.sort(candidate_indices[:max_cells])

    max_classes_by_total = max(2, max_cells // min_per_label)
    max_classes_by_test = max(2, int(np.floor(max_cells * float(test_size))))
    max_classes_by_train = max(2, int(np.floor(max_cells * (1.0 - float(test_size)))))
    n_labels = min(eligible.shape[0], max_classes_by_total, max_classes_by_test, max_classes_by_train)
    selected_counts = eligible.sort_values(ascending=False).iloc[:n_labels]

    base = pd.Series(min_per_label, index=selected_counts.index, dtype=int)
    capacity = (selected_counts - base).clip(lower=0).astype(int)
    remaining = int(max_cells - base.sum())
    quota = base.copy()
    if remaining > 0 and capacity.sum() > 0:
        raw = capacity / float(capacity.sum()) * remaining
        add = np.floor(raw).astype(int).clip(upper=capacity)
        quota += add
        leftover = int(max_cells - quota.sum())
        remainders = (raw - add).sort_values(ascending=False)
        for label in remainders.index:
            if leftover <= 0:
                break
            room = int(selected_counts[label] - quota[label])
            if room <= 0:
                continue
            quota[label] += 1
            leftover -= 1

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label, n_take in quota.items():
        label_positions = candidate_indices[candidate_labels == label]
        selected.append(rng.choice(label_positions, size=int(n_take), replace=False))
    return np.sort(np.concatenate(selected))


def _grid_neighbors(value: float, grid: tuple[float, ...]) -> list[float]:
    values = list(grid)
    idx = values.index(value)
    out = []
    if idx > 0:
        out.append(values[idx - 1])
    if idx < len(values) - 1:
        out.append(values[idx + 1])
    return out


def generate_neighbors(config: HPOConfig) -> list[tuple[HPOConfig, str]]:
    neighbors: list[tuple[HPOConfig, str]] = []
    task_values = list(TASK_GRID)
    task_idx = task_values.index(config.tasks)
    for idx in (task_idx - 1, task_idx + 1):
        if 0 <= idx < len(task_values):
            child = HPOConfig(
                tasks=task_values[idx],
                dropout_rate=config.dropout_rate,
                lam=config.lam,
                sigma=config.sigma,
                learning_rate=config.learning_rate,
            )
            neighbors.append((child, f"tasks:{config.task_string}->{child.task_string}"))
    for field, grid in FLOAT_GRIDS.items():
        current = getattr(config, field)
        for value in _grid_neighbors(current, grid):
            payload = asdict(config)
            payload[field] = value
            child = HPOConfig(**payload)
            neighbors.append((child, f"{field}:{current:g}->{value:g}"))
    return neighbors


def pareto_archive(df: pd.DataFrame, objective_columns: tuple[str, ...] = OBJECTIVES) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    candidates = df[df["status"].isin(["completed", "skipped_existing"])].copy()
    candidates = candidates.dropna(subset=list(objective_columns))
    if candidates.empty:
        return candidates
    dominated = np.zeros(candidates.shape[0], dtype=bool)
    values = candidates.loc[:, list(objective_columns)].to_numpy(dtype=float)
    for i in range(values.shape[0]):
        if dominated[i]:
            continue
        for j in range(values.shape[0]):
            if i == j:
                continue
            no_worse = np.all(values[j] >= values[i])
            strictly_better = np.any(values[j] > values[i])
            if no_worse and strictly_better:
                dominated[i] = True
                break
    archive = candidates.loc[~dominated].copy()
    return archive.sort_values(list(objective_columns), ascending=[False] * len(objective_columns)).reset_index(drop=True)


def select_objective_trajectory(
    trials: pd.DataFrame,
    archive_history: pd.DataFrame,
    objective_metric: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_col = f"visium_mean_{objective_metric}"
    other_col = "visium_mean_spatial_pearson" if objective_metric == "cell_type_accuracy" else "visium_mean_cell_type_accuracy"
    for iteration in sorted(archive_history["iteration"].unique()):
        trial_ids = archive_history.loc[archive_history["iteration"] <= iteration, "trial_id"].astype(str).tolist()
        subset = trials[trials["trial_id"].isin(trial_ids)].copy()
        subset = subset.dropna(subset=[metric_col])
        if subset.empty:
            continue
        subset = subset.sort_values([metric_col, other_col, "trial_index"], ascending=[False, False, True])
        selected = subset.iloc[0].to_dict()
        rows.append(
            {
                "iteration": int(iteration),
                "objective_metric": objective_metric,
                "selected_trial_id": selected["trial_id"],
                "visium_value": float(selected[metric_col]),
                "visium_other_objective_value": float(selected[other_col]),
                "panel_path": selected.get("panel_path", ""),
                "config_key": selected.get("config_key", ""),
            }
        )
    return pd.DataFrame(rows)


def frontier_best_row(
    archive: pd.DataFrame,
    iteration: int,
    n_new_trials: int,
    stop_reason: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "iteration": int(iteration),
        "frontier_size": int(archive.shape[0]),
        "n_new_trials": int(n_new_trials),
        "stop_reason": stop_reason,
    }
    for objective in OBJECTIVES:
        if archive.empty or objective not in archive:
            row[f"best_{objective}"] = float("nan")
        else:
            row[f"best_{objective}"] = float(archive[objective].max())
    return row


def select_candidate_neighbors(
    *,
    parent_rows: list[dict[str, Any]],
    configs_by_trial: dict[str, HPOConfig],
    seen_config_keys: set[str],
    expanded_keys: set[str],
    max_new_trials: int,
) -> tuple[list[tuple[HPOConfig, str, str]], set[str]]:
    candidates: list[tuple[HPOConfig, str, str]] = []
    scheduled_keys: set[str] = set()
    newly_expanded_keys: set[str] = set()
    if max_new_trials <= 0:
        return candidates, newly_expanded_keys

    for parent in parent_rows:
        if len(candidates) >= max_new_trials:
            break
        parent_id = str(parent["trial_id"])
        parent_config = configs_by_trial[parent_id]
        if parent_config.key in expanded_keys:
            continue
        parent_candidates: list[tuple[HPOConfig, str, str]] = []
        for child, mutation in generate_neighbors(parent_config):
            if child.key in seen_config_keys or child.key in scheduled_keys:
                continue
            parent_candidates.append((child, parent_id, mutation))
        if not parent_candidates:
            newly_expanded_keys.add(parent_config.key)
            continue

        remaining_slots = max_new_trials - len(candidates)
        scheduled = parent_candidates[:remaining_slots]
        candidates.extend(scheduled)
        scheduled_keys.update(child.key for child, _, _ in scheduled)
        if len(parent_candidates) <= remaining_slots:
            newly_expanded_keys.add(parent_config.key)
    return candidates, newly_expanded_keys


def full_rerun_selected_configs(
    *,
    output_dir: Path,
    selected_trial_ids: list[str],
    trajectory: pd.DataFrame,
    trials_df: pd.DataFrame,
    configs_by_trial: dict[str, HPOConfig],
    source_adata: str | Path,
    source_label_column: str,
    gene_universe: list[str],
    references: list[tuple[str, Path]],
    merfish_adata: str | Path,
    merfish_label_column: str,
    visium_label_column: str,
    panel_size: int,
    epoch: int,
    record: int,
    source_max_cells: int,
    smith_max_cells: int,
    eval_max_cells: int,
    merfish_max_cells: int,
    eval_min_label_count: int,
    test_size: float,
    seed: int,
    batch_size: int,
    device: str,
    python_executable: str,
    class_weight: str | None,
    force: bool,
) -> dict[str, str]:
    full_dir = ensure_dir(output_dir)
    full_source = prepare_source_input(
        source_adata,
        full_dir / "prepared/source_full_input.h5ad",
        gene_universe,
        label_column=source_label_column,
        max_cells=source_max_cells,
        seed=seed,
    )
    full_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    panel_by_original: dict[str, str] = {}
    for index, original_trial_id in enumerate(selected_trial_ids, start=1):
        config = configs_by_trial[original_trial_id]
        full_trial_id = f"full_{index:03d}_from_{original_trial_id}"
        run = run_smith_config(
            config=config,
            trial_id=full_trial_id,
            source_h5ad=full_source["output_h5ad"],
            output_dir=full_dir / "smith_runs",
            panel_size=panel_size,
            epoch=epoch,
            record=record,
            device=device,
            seed=seed,
            batch_size=batch_size,
            max_cells=smith_max_cells,
            python_executable=python_executable,
            force=force,
        )
        row: dict[str, Any] = {
            "full_trial_id": full_trial_id,
            "original_trial_id": original_trial_id,
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
            panel_path = write_panel_from_rank(
                run["rank_csv"],
                full_dir / "panels" / f"{full_trial_id}_top_{panel_size}_panel.tsv",
                panel_size,
            )
            row["panel_path"] = panel_path
            panel_by_original[original_trial_id] = panel_path
            trial_eval_rows: list[dict[str, Any]] = []
            for ref_id, ref_path in references:
                dataset_rows = evaluate_panel_on_dataset(
                    adata_file=ref_path,
                    dataset_id=ref_id,
                    dataset_role="visium_validation_full",
                    panel_path=panel_path,
                    output_dir=full_dir / "evaluation" / full_trial_id / "visium",
                    panel_size=panel_size,
                    label_column=visium_label_column,
                    seed=seed,
                    test_size=test_size,
                    max_cells=eval_max_cells,
                    min_label_count=eval_min_label_count,
                    class_weight=class_weight,
                    subset_cache_dir=full_dir / "evaluation" / "_shared_subsets",
                )
                for item in dataset_rows:
                    item.update(
                        {
                            "full_trial_id": full_trial_id,
                            "original_trial_id": original_trial_id,
                            "config_key": config.key,
                        }
                    )
                trial_eval_rows.extend(dataset_rows)
            merfish_rows = evaluate_panel_on_dataset(
                adata_file=merfish_adata,
                dataset_id="locked_merfish",
                dataset_role="merfish_locked_test_full",
                panel_path=panel_path,
                output_dir=full_dir / "evaluation" / full_trial_id / "merfish_locked",
                panel_size=panel_size,
                label_column=merfish_label_column,
                seed=seed,
                test_size=test_size,
                max_cells=merfish_max_cells,
                min_label_count=eval_min_label_count,
                class_weight=class_weight,
                subset_cache_dir=full_dir / "evaluation" / "_shared_subsets",
            )
            for item in merfish_rows:
                item.update(
                    {
                        "full_trial_id": full_trial_id,
                        "original_trial_id": original_trial_id,
                        "config_key": config.key,
                    }
                )
            trial_eval_rows.extend(merfish_rows)
            eval_rows.extend(trial_eval_rows)
            visium_for_agg = [
                {**item, "dataset_role": "visium_validation"}
                for item in trial_eval_rows
                if item["dataset_role"] == "visium_validation_full"
            ]
            row.update({f"full_{key}": value for key, value in aggregate_visium_metrics(visium_for_agg).items()})
        full_rows.append(row)

    full_trials = pd.DataFrame(full_rows)
    full_eval = pd.DataFrame(eval_rows)
    full_trajectory = trajectory.copy()
    if not full_eval.empty and not full_trajectory.empty:
        visium_wide = (
            full_eval[full_eval["dataset_role"] == "visium_validation_full"]
            .pivot_table(index="original_trial_id", columns="metric", values="value", aggfunc="mean")
            .reset_index()
            .rename(
                columns={
                    "cell_type_accuracy": "full_visium_cell_type_accuracy",
                    "spatial_pearson": "full_visium_spatial_pearson",
                }
            )
        )
        merfish_wide = (
            full_eval[full_eval["dataset_role"] == "merfish_locked_test_full"]
            .pivot_table(index="original_trial_id", columns="metric", values="value", aggfunc="first")
            .reset_index()
            .rename(
                columns={
                    "cell_type_accuracy": "full_merfish_cell_type_accuracy",
                    "spatial_pearson": "full_merfish_spatial_pearson",
                }
            )
        )
        full_trajectory = full_trajectory.merge(
            visium_wide,
            left_on="selected_trial_id",
            right_on="original_trial_id",
            how="left",
        ).drop(columns=["original_trial_id"], errors="ignore")
        full_trajectory = full_trajectory.merge(
            merfish_wide,
            left_on="selected_trial_id",
            right_on="original_trial_id",
            how="left",
        ).drop(columns=["original_trial_id"], errors="ignore")
        full_trajectory["full_visium_value"] = np.where(
            full_trajectory["objective_metric"] == "cell_type_accuracy",
            full_trajectory["full_visium_cell_type_accuracy"],
            full_trajectory["full_visium_spatial_pearson"],
        )
        full_trajectory["full_merfish_value"] = np.where(
            full_trajectory["objective_metric"] == "cell_type_accuracy",
            full_trajectory["full_merfish_cell_type_accuracy"],
            full_trajectory["full_merfish_spatial_pearson"],
        )

    trials_tsv = full_dir / "full_rerun_trials.tsv"
    evaluation_tsv = full_dir / "full_rerun_evaluation_long.tsv"
    trajectory_tsv = full_dir / "full_rerun_objective_trajectory.tsv"
    full_trials.to_csv(trials_tsv, sep="\t", index=False)
    full_eval.to_csv(evaluation_tsv, sep="\t", index=False)
    full_trajectory.to_csv(trajectory_tsv, sep="\t", index=False)
    return {
        "full_trials_tsv": str(trials_tsv),
        "full_evaluation_long_tsv": str(evaluation_tsv),
        "full_trajectory_tsv": str(trajectory_tsv),
    }


def parse_reference_args(items: list[str] | None) -> list[tuple[str, Path]]:
    if not items:
        return [(name, path) for name, path in DEFAULT_VISIUM_REFERENCES]
    refs: list[tuple[str, Path]] = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).stem
        refs.append((_safe_id(name), Path(path)))
    return refs


def _class_weight(value: str) -> str | None:
    lowered = value.strip().lower()
    if lowered in {"", "none", "null", "off"}:
        return None
    return lowered


def _screen_param(screen_manifest: dict[str, Any], key: str, fallback: Any) -> Any:
    params = screen_manifest.get("parameters", {})
    if isinstance(params, dict) and key in params:
        value = params[key]
        if value is not None and value != "":
            return value
    return fallback


def _references_from_paths(paths: list[str]) -> list[tuple[str, Path]]:
    refs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for index, path_text in enumerate(paths, start=1):
        path = Path(path_text)
        name = _safe_id(path.stem) or f"visium_{index}"
        if name in seen:
            name = f"{name}_{index}"
        seen.add(name)
        refs.append((name, path))
    return refs


def references_for_full_from_screen(
    *,
    visium_reference_args: list[str] | None,
    screen_manifest: dict[str, Any],
) -> list[tuple[str, Path]]:
    if visium_reference_args:
        return parse_reference_args(visium_reference_args)
    screen_reference_args = _screen_param(screen_manifest, "visium_reference", None)
    if screen_reference_args:
        return parse_reference_args(screen_reference_args)
    leakage_policy = screen_manifest.get("leakage_policy", {})
    if isinstance(leakage_policy, dict):
        validation_paths = leakage_policy.get("validation_for_search", [])
        if validation_paths:
            return _references_from_paths([str(path) for path in validation_paths])
    return parse_reference_args(None)


def config_from_trial_row(row: pd.Series | dict[str, Any]) -> HPOConfig:
    tasks_text = str(row.get("tasks", "")).strip()
    if not tasks_text or tasks_text.lower() == "nan":
        config_key = str(row.get("config_key", ""))
        match = re.search(r"tasks=([^|]+)", config_key)
        tasks_text = match.group(1).replace("-", ",") if match else ",".join(HPOConfig().tasks)
    tasks = tuple(part.strip() for part in tasks_text.split(",") if part.strip())
    return HPOConfig(
        tasks=tasks or HPOConfig().tasks,
        dropout_rate=float(row["dropout_rate"]),
        lam=float(row["lam"]),
        sigma=float(row["sigma"]),
        learning_rate=float(row["learning_rate"]),
    )


def unique_selected_trial_ids(trajectory: pd.DataFrame) -> list[str]:
    if trajectory.empty or "selected_trial_id" not in trajectory:
        return []
    selected: list[str] = []
    seen: set[str] = set()
    for value in trajectory["selected_trial_id"].astype(str):
        trial_id = value.strip()
        if not trial_id or trial_id.lower() == "nan" or trial_id in seen:
            continue
        selected.append(trial_id)
        seen.add(trial_id)
    return selected


def load_existing_screen(screen_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, HPOConfig], list[str]]:
    screen_dir = Path(screen_dir)
    trials_tsv = screen_dir / "hpo_trials.tsv"
    trajectory_tsv = screen_dir / "hpo_objective_trajectory.tsv"
    missing = [str(path) for path in (trials_tsv, trajectory_tsv) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing existing screen artifact(s): {', '.join(missing)}")
    trials_df = pd.read_csv(trials_tsv, sep="\t")
    trajectory = pd.read_csv(trajectory_tsv, sep="\t")
    configs_by_trial = {
        str(row["trial_id"]): config_from_trial_row(row)
        for _, row in trials_df.iterrows()
        if str(row.get("trial_id", "")).strip()
    }
    selected_trial_ids = unique_selected_trial_ids(trajectory)
    missing_configs = [trial_id for trial_id in selected_trial_ids if trial_id not in configs_by_trial]
    if missing_configs:
        raise ValueError(f"Selected trial(s) are absent from hpo_trials.tsv: {', '.join(missing_configs)}")
    return trials_df, trajectory, configs_by_trial, selected_trial_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pareto-guided SMITH HPO on liver sc/snRNA-seq with Visium validation and locked MERFISH testing."
    )
    parser.add_argument("--source-adata", default=str(DEFAULT_SOURCE))
    parser.add_argument("--source-label-column", default="Cell_Type_final")
    parser.add_argument("--merfish-adata", default=str(DEFAULT_MERFISH))
    parser.add_argument("--merfish-label-column", default="Cell_Type")
    parser.add_argument("--visium-reference", action="append", help="Reference as name=/path/to/file.h5ad.")
    parser.add_argument("--visium-label-column", default="cell_type")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/liver_pareto_hpo"))
    parser.add_argument("--gene-universe", choices=["merfish", "source_visium_shared"], default="merfish")
    parser.add_argument("--panel-size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--max-new-trials-per-iteration", type=int, default=8)
    parser.add_argument("--min-screen-iterations", type=int, default=2)
    parser.add_argument("--frontier-patience", type=int, default=2)
    parser.add_argument("--frontier-min-delta", type=float, default=0.001)
    parser.add_argument(
        "--full-from-existing-screen",
        default="",
        help="Run only the full rerun stage from an existing screen output directory.",
    )
    parser.add_argument("--run-full-rerun", action="store_true")
    parser.add_argument("--full-output-dir", default="")
    parser.add_argument("--full-epoch", type=int, default=50)
    parser.add_argument("--full-record", type=int, default=10)
    parser.add_argument("--full-source-max-cells", type=int, default=0)
    parser.add_argument("--full-smith-max-cells", type=int, default=0)
    parser.add_argument("--full-eval-max-cells", type=int, default=0)
    parser.add_argument("--full-merfish-max-cells", type=int, default=0)
    parser.add_argument("--full-batch-size", type=int, default=1024)
    parser.add_argument("--source-max-cells", type=int, default=30000)
    parser.add_argument("--smith-max-cells", type=int, default=0)
    parser.add_argument("--eval-max-cells", type=int, default=0)
    parser.add_argument("--merfish-max-cells", type=int, default=0)
    parser.add_argument("--eval-min-label-count", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--class-weight", default="none", help="LogisticRegression class_weight for split accuracy.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    def cli_has(flag: str) -> bool:
        return flag in sys.argv[1:] or any(item.startswith(f"{flag}=") for item in sys.argv[1:])

    if args.full_from_existing_screen:
        screen_dir = Path(args.full_from_existing_screen)
        manifest_path = screen_dir / "hpo_search_manifest.json"
        screen_manifest: dict[str, Any] = {}
        if manifest_path.exists():
            screen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        full_output_dir = (
            Path(args.full_output_dir)
            if args.full_output_dir
            else Path(args.output_dir)
            if cli_has("--output-dir")
            else screen_dir / "full_rerun"
        )
        output_dir = ensure_dir(full_output_dir)
        source_adata = (
            args.source_adata if cli_has("--source-adata") else _screen_param(screen_manifest, "source_adata", args.source_adata)
        )
        source_label_column = (
            args.source_label_column
            if cli_has("--source-label-column")
            else _screen_param(screen_manifest, "source_label_column", args.source_label_column)
        )
        merfish_adata = (
            args.merfish_adata
            if cli_has("--merfish-adata")
            else _screen_param(screen_manifest, "merfish_adata", args.merfish_adata)
        )
        merfish_label_column = (
            args.merfish_label_column
            if cli_has("--merfish-label-column")
            else _screen_param(screen_manifest, "merfish_label_column", args.merfish_label_column)
        )
        visium_label_column = (
            args.visium_label_column
            if cli_has("--visium-label-column")
            else _screen_param(screen_manifest, "visium_label_column", args.visium_label_column)
        )
        panel_size = int(args.panel_size if cli_has("--panel-size") else _screen_param(screen_manifest, "panel_size", args.panel_size))
        gene_universe_mode = str(
            args.gene_universe
            if cli_has("--gene-universe")
            else _screen_param(screen_manifest, "gene_universe", args.gene_universe)
        )
        references = references_for_full_from_screen(
            visium_reference_args=args.visium_reference,
            screen_manifest=screen_manifest,
        )
        if gene_universe_mode == "merfish":
            gene_universe = _load_merfish_gene_universe(merfish_adata)
            gene_universe_note = "MERFISH var_names used only as a fixed assay-measurability universe."
        elif gene_universe_mode == "source_visium_shared":
            gene_universe = _load_source_visium_shared_gene_universe(source_adata, references)
            gene_universe_note = "Intersection of source and Visium genes; MERFISH genes not used during HPO search."
        else:
            raise ValueError(f"Unsupported gene universe mode: {gene_universe_mode}")

        trials_df, trajectory, configs_by_trial, selected_trial_ids = load_existing_screen(screen_dir)
        full_artifacts = full_rerun_selected_configs(
            output_dir=output_dir,
            selected_trial_ids=selected_trial_ids,
            trajectory=trajectory,
            trials_df=trials_df,
            configs_by_trial=configs_by_trial,
            source_adata=source_adata,
            source_label_column=source_label_column,
            gene_universe=gene_universe,
            references=references,
            merfish_adata=merfish_adata,
            merfish_label_column=merfish_label_column,
            visium_label_column=visium_label_column,
            panel_size=panel_size,
            epoch=args.full_epoch,
            record=args.full_record,
            source_max_cells=args.full_source_max_cells,
            smith_max_cells=args.full_smith_max_cells,
            eval_max_cells=args.full_eval_max_cells,
            merfish_max_cells=args.full_merfish_max_cells,
            eval_min_label_count=args.eval_min_label_count,
            test_size=args.test_size,
            seed=args.seed,
            batch_size=args.full_batch_size,
            device=args.device,
            python_executable=args.python_executable,
            class_weight=_class_weight(args.class_weight),
            force=args.force,
        )
        manifest = {
            "mode": "full_from_existing_screen",
            "screen_dir": str(screen_dir),
            "screen_manifest": str(manifest_path) if manifest_path.exists() else "",
            "algorithm": "Full rerun of unique configs selected by the frozen Visium-driven Pareto trajectory.",
            "leakage_policy": {
                "train_source": str(source_adata),
                "validation_for_search": [str(path) for _, path in references],
                "locked_test": str(merfish_adata),
                "merfish_test_timing": "MERFISH metrics are computed only after the Visium-driven trajectory is fixed.",
                "gene_universe": gene_universe_mode,
                "gene_universe_note": gene_universe_note,
            },
            "full_rerun_policy": {
                "enabled": True,
                "trigger": (
                    "Run after the short-screen trajectory is frozen by frontier stability, no-new-neighbor stop, "
                    "or max-iteration stop."
                ),
                "selected_configs": "Unique selected_trial_id values in hpo_objective_trajectory.tsv.",
                "selected_trial_ids": selected_trial_ids,
            },
            "parameters": vars(args),
            "n_gene_universe": len(gene_universe),
            "artifacts": full_artifacts,
        }
        full_manifest_path = write_json(output_dir / "full_from_existing_screen_manifest.json", manifest)
        print(json.dumps({**full_artifacts, "manifest_json": str(full_manifest_path)}, indent=2))
        return 0

    output_dir = ensure_dir(args.output_dir)
    references = parse_reference_args(args.visium_reference)
    if args.gene_universe == "merfish":
        gene_universe = _load_merfish_gene_universe(args.merfish_adata)
        gene_universe_note = "MERFISH var_names used only as a fixed assay-measurability universe."
    else:
        gene_universe = _load_source_visium_shared_gene_universe(args.source_adata, references)
        gene_universe_note = "Intersection of source and Visium genes; MERFISH genes not used during HPO search."

    source_prepared = prepare_source_input(
        args.source_adata,
        output_dir / "prepared/source_hpo_input.h5ad",
        gene_universe,
        label_column=args.source_label_column,
        max_cells=args.source_max_cells,
        seed=args.seed,
    )
    source_h5ad = source_prepared["output_h5ad"]

    all_trial_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    frontier_progress_rows: list[dict[str, Any]] = []
    expanded_keys: set[str] = set()
    seen_configs: dict[str, str] = {}
    configs_by_trial: dict[str, HPOConfig] = {}
    trial_counter = 0

    def evaluate_trial(config: HPOConfig, iteration: int, parent_trial_id: str, mutation: str) -> dict[str, Any]:
        nonlocal trial_counter
        if config.key in seen_configs:
            existing_id = seen_configs[config.key]
            existing = next(row for row in all_trial_rows if row["trial_id"] == existing_id)
            return existing
        trial_counter += 1
        trial_id = f"trial_{trial_counter:03d}"
        seen_configs[config.key] = trial_id
        configs_by_trial[trial_id] = config
        run = run_smith_config(
            config=config,
            trial_id=trial_id,
            source_h5ad=source_h5ad,
            output_dir=output_dir / "smith_runs",
            panel_size=args.panel_size,
            epoch=args.epoch,
            record=args.record,
            device=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
            max_cells=args.smith_max_cells,
            python_executable=args.python_executable,
            force=args.force,
        )
        row: dict[str, Any] = {
            "trial_index": trial_counter,
            "trial_id": trial_id,
            "iteration": iteration,
            "parent_trial_id": parent_trial_id,
            "mutation": mutation,
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
            panel_path = write_panel_from_rank(
                run["rank_csv"],
                output_dir / "panels" / f"{trial_id}_top_{args.panel_size}_panel.tsv",
                args.panel_size,
            )
            row["panel_path"] = panel_path
            trial_eval_rows: list[dict[str, Any]] = []
            for ref_id, ref_path in references:
                dataset_rows = evaluate_panel_on_dataset(
                    adata_file=ref_path,
                    dataset_id=ref_id,
                    dataset_role="visium_validation",
                    panel_path=panel_path,
                    output_dir=output_dir / "evaluation" / trial_id / "visium",
                    panel_size=args.panel_size,
                    label_column=args.visium_label_column,
                    seed=args.seed,
                    test_size=args.test_size,
                    max_cells=args.eval_max_cells,
                    min_label_count=args.eval_min_label_count,
                    class_weight=_class_weight(args.class_weight),
                    subset_cache_dir=output_dir / "evaluation" / "_shared_subsets",
                )
                for item in dataset_rows:
                    item.update({"trial_id": trial_id, "iteration": iteration, "config_key": config.key})
                trial_eval_rows.extend(dataset_rows)
            eval_rows.extend(trial_eval_rows)
            row.update(aggregate_visium_metrics(trial_eval_rows))
        all_trial_rows.append(row)
        return row

    root = HPOConfig()
    evaluate_trial(root, iteration=0, parent_trial_id="", mutation="root_default")
    trials_df = pd.DataFrame(all_trial_rows)
    archive = pareto_archive(trials_df)
    for record in archive.to_dict(orient="records"):
        archive_record = dict(record)
        archive_record["trial_iteration"] = archive_record.get("iteration")
        archive_record["iteration"] = 0
        archive_rows.append(archive_record)
    frontier_progress_rows.append(frontier_best_row(archive, iteration=0, n_new_trials=1))
    stop_reason = "max_iterations"

    for iteration in range(1, args.max_iterations + 1):
        parent_rows = archive.to_dict(orient="records")
        parent_ids = {str(row["trial_id"]) for row in parent_rows}
        recent_completed = [
            row
            for row in all_trial_rows
            if row["status"] in {"completed", "skipped_existing"}
            and int(row["iteration"]) == iteration - 1
            and str(row["trial_id"]) not in parent_ids
        ]
        parent_rows.extend(recent_completed)
        candidates, newly_expanded_keys = select_candidate_neighbors(
            parent_rows=parent_rows,
            configs_by_trial=configs_by_trial,
            seen_config_keys=set(seen_configs),
            expanded_keys=expanded_keys,
            max_new_trials=args.max_new_trials_per_iteration,
        )
        expanded_keys.update(newly_expanded_keys)
        if not candidates:
            stop_reason = "no_new_neighbors"
            break
        for child_config, parent_id, mutation in candidates:
            evaluate_trial(child_config, iteration=iteration, parent_trial_id=parent_id, mutation=mutation)
        trials_df = pd.DataFrame(all_trial_rows)
        archive = pareto_archive(trials_df)
        for record in archive.to_dict(orient="records"):
            archive_record = dict(record)
            archive_record["trial_iteration"] = archive_record.get("iteration")
            archive_record["iteration"] = iteration
            archive_rows.append(archive_record)
        progress_row = frontier_best_row(archive, iteration=iteration, n_new_trials=len(candidates))
        frontier_progress_rows.append(progress_row)
        if iteration >= args.min_screen_iterations and len(frontier_progress_rows) > args.frontier_patience:
            recent = frontier_progress_rows[-(args.frontier_patience + 1) :]
            stable = True
            for objective in OBJECTIVES:
                prior_best = max(float(row[f"best_{objective}"]) for row in recent[:-1])
                current_best = float(recent[-1][f"best_{objective}"])
                if current_best - prior_best >= args.frontier_min_delta:
                    stable = False
                    break
            if stable:
                stop_reason = "frontier_stable"
                frontier_progress_rows[-1]["stop_reason"] = stop_reason
                break

    trials_df = pd.DataFrame(all_trial_rows)
    archive_history = pd.DataFrame(archive_rows)
    frontier_progress = pd.DataFrame(frontier_progress_rows)
    trajectory = pd.concat(
        [
            select_objective_trajectory(trials_df, archive_history, "cell_type_accuracy"),
            select_objective_trajectory(trials_df, archive_history, "spatial_pearson"),
        ],
        ignore_index=True,
    )

    # Locked MERFISH test: computed only after the Visium-driven search trajectory is fixed.
    merfish_rows: list[dict[str, Any]] = []
    selected_trial_ids = sorted(set(trajectory["selected_trial_id"].astype(str))) if not trajectory.empty else []
    panel_by_trial = dict(zip(trials_df["trial_id"].astype(str), trials_df["panel_path"].astype(str), strict=False))
    for trial_id in selected_trial_ids:
        panel_path = panel_by_trial[trial_id]
        dataset_rows = evaluate_panel_on_dataset(
            adata_file=args.merfish_adata,
            dataset_id="locked_merfish",
            dataset_role="merfish_locked_test",
            panel_path=panel_path,
            output_dir=output_dir / "evaluation" / trial_id / "merfish_locked",
            panel_size=args.panel_size,
            label_column=args.merfish_label_column,
            seed=args.seed,
            test_size=args.test_size,
            max_cells=args.merfish_max_cells,
            min_label_count=args.eval_min_label_count,
            class_weight=_class_weight(args.class_weight),
            subset_cache_dir=output_dir / "evaluation" / "_shared_subsets",
        )
        config_key = str(trials_df.loc[trials_df["trial_id"] == trial_id, "config_key"].iloc[0])
        iteration = int(trials_df.loc[trials_df["trial_id"] == trial_id, "iteration"].iloc[0])
        for item in dataset_rows:
            item.update({"trial_id": trial_id, "iteration": iteration, "config_key": config_key})
        merfish_rows.extend(dataset_rows)
    eval_rows.extend(merfish_rows)
    eval_df = pd.DataFrame(eval_rows)

    if not trajectory.empty and not eval_df.empty:
        merfish_wide = (
            eval_df[eval_df["dataset_role"] == "merfish_locked_test"]
            .pivot_table(index="trial_id", columns="metric", values="value", aggfunc="first")
            .reset_index()
        )
        trajectory = trajectory.merge(merfish_wide, left_on="selected_trial_id", right_on="trial_id", how="left")
        trajectory = trajectory.drop(columns=["trial_id"], errors="ignore")
        trajectory = trajectory.rename(
            columns={
                "cell_type_accuracy": "merfish_cell_type_accuracy",
                "spatial_pearson": "merfish_spatial_pearson",
            }
        )
        trajectory["merfish_value"] = np.where(
            trajectory["objective_metric"] == "cell_type_accuracy",
            trajectory["merfish_cell_type_accuracy"],
            trajectory["merfish_spatial_pearson"],
        )

    trial_tsv = output_dir / "hpo_trials.tsv"
    evaluation_tsv = output_dir / "hpo_evaluation_long.tsv"
    archive_tsv = output_dir / "pareto_archive_by_iteration.tsv"
    frontier_progress_tsv = output_dir / "frontier_progress.tsv"
    trajectory_tsv = output_dir / "hpo_objective_trajectory.tsv"
    trials_df.to_csv(trial_tsv, sep="\t", index=False)
    eval_df.to_csv(evaluation_tsv, sep="\t", index=False)
    archive_history.to_csv(archive_tsv, sep="\t", index=False)
    frontier_progress.to_csv(frontier_progress_tsv, sep="\t", index=False)
    trajectory.to_csv(trajectory_tsv, sep="\t", index=False)

    full_artifacts: dict[str, str] = {}
    if args.run_full_rerun and not trajectory.empty:
        selected_trial_ids = sorted(set(trajectory["selected_trial_id"].astype(str)))
        full_output_dir = Path(args.full_output_dir) if args.full_output_dir else output_dir / "full_rerun"
        full_artifacts = full_rerun_selected_configs(
            output_dir=full_output_dir,
            selected_trial_ids=selected_trial_ids,
            trajectory=trajectory,
            trials_df=trials_df,
            configs_by_trial=configs_by_trial,
            source_adata=args.source_adata,
            source_label_column=args.source_label_column,
            gene_universe=gene_universe,
            references=references,
            merfish_adata=args.merfish_adata,
            merfish_label_column=args.merfish_label_column,
            visium_label_column=args.visium_label_column,
            panel_size=args.panel_size,
            epoch=args.full_epoch,
            record=args.full_record,
            source_max_cells=args.full_source_max_cells,
            smith_max_cells=args.full_smith_max_cells,
            eval_max_cells=args.full_eval_max_cells,
            merfish_max_cells=args.full_merfish_max_cells,
            eval_min_label_count=args.eval_min_label_count,
            test_size=args.test_size,
            seed=args.seed,
            batch_size=args.full_batch_size,
            device=args.device,
            python_executable=args.python_executable,
            class_weight=_class_weight(args.class_weight),
            force=args.force,
        )

    manifest = {
        "algorithm": "Pareto-guided local hyperparameter search over one-step SMITH config perturbations",
        "leakage_policy": {
            "train_source": str(args.source_adata),
            "validation_for_search": [str(path) for _, path in references],
            "locked_test": str(args.merfish_adata),
            "merfish_test_timing": "MERFISH metrics are computed only after the Visium-driven trajectory is fixed.",
            "gene_universe": args.gene_universe,
            "gene_universe_note": gene_universe_note,
        },
        "objectives": list(OBJECTIVES),
        "search_space": {
            "tasks": [",".join(item) for item in TASK_GRID],
            **{key: list(value) for key, value in FLOAT_GRIDS.items()},
        },
        "screen_stop_reason": stop_reason,
        "full_rerun_policy": {
            "enabled": bool(args.run_full_rerun),
            "trigger": (
                "After short-screen trajectory is frozen by frontier stability, no-new-neighbor stop, "
                "or max-iteration stop."
            ),
            "selected_configs": "Unique selected_trial_id values in hpo_objective_trajectory.tsv.",
        },
        "parameters": vars(args),
        "source_prepared": source_prepared,
        "n_gene_universe": len(gene_universe),
        "artifacts": {
            "trials_tsv": str(trial_tsv),
            "evaluation_long_tsv": str(evaluation_tsv),
            "archive_tsv": str(archive_tsv),
            "frontier_progress_tsv": str(frontier_progress_tsv),
            "trajectory_tsv": str(trajectory_tsv),
            **full_artifacts,
        },
    }
    manifest_path = write_json(output_dir / "hpo_search_manifest.json", manifest)
    print(json.dumps({**manifest["artifacts"], "manifest_json": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
