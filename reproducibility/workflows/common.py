from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse

import anndata as ad


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


GENE_COLUMNS = ("gene_symbol", "gene_symbols", "feature_name", "gene_name", "gene_short_name", "symbol")


def clean_gene(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none"} else text.upper()


def gene_symbols(adata: ad.AnnData) -> list[str]:
    for column in GENE_COLUMNS:
        if column in adata.var:
            return [clean_gene(value) for value in adata.var[column].astype(str)]
    return [clean_gene(value) for value in adata.var_names.astype(str)]


def dense_matrix(adata: ad.AnnData, positions: list[int]) -> Any:
    matrix = adata.X[:, positions]
    return matrix.toarray() if sparse.issparse(matrix) else matrix


def read_panel(path: str | Path, panel_size: int | None = None) -> list[str]:
    path = Path(path)
    frame = pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".tab"} else ",")
    column = next(
        (name for name in frame.columns if str(name).lower() in {"marker", "gene", "gene_symbol", "target"}),
        frame.columns[0],
    )
    genes = []
    for value in frame[column]:
        gene = clean_gene(value)
        if gene and gene not in genes:
            genes.append(gene)
    return genes[:panel_size] if panel_size else genes


def parse_int_list(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        values = [int(item) for item in value]
    else:
        values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"Expected positive comma-separated integers, got {value!r}.")
    return list(dict.fromkeys(values))


def parse_key_value_list(values: list[str] | None) -> dict[str, str]:
    parsed = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        if not key.strip() or not item.strip():
            raise ValueError(f"Expected NAME=VALUE, got {value!r}.")
        parsed[key.strip()] = item.strip()
    return parsed


def write_top_panel(ranking_path: str | Path, output_path: str | Path, panel_size: int) -> Path:
    ranking_path = Path(ranking_path)
    frame = pd.read_csv(ranking_path, sep="\t" if ranking_path.suffix.lower() == ".tsv" else ",")
    gene_column = next(
        (name for name in frame.columns if str(name).lower() in {"marker", "gene", "gene_symbol", "target"}),
        frame.columns[0],
    )
    genes = []
    for value in frame[gene_column]:
        gene = clean_gene(value)
        if gene and gene not in genes:
            genes.append(gene)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rank": range(1, min(panel_size, len(genes)) + 1), "gene_symbol": genes[:panel_size]}).to_csv(
        output_path, sep="\t" if output_path.suffix.lower() == ".tsv" else ",", index=False
    )
    return output_path


def panel_positions(adata: ad.AnnData, panel: list[str]) -> tuple[list[int], list[str]]:
    first = {}
    for index, gene in enumerate(gene_symbols(adata)):
        if gene and gene not in first:
            first[gene] = index
    shared = [gene for gene in panel if gene in first]
    if not shared:
        raise ValueError("No panel genes overlap the evaluation dataset.")
    return [first[gene] for gene in shared], shared


def latest_epoch_csv(saving_dir: str | Path) -> Path:
    candidates = list(Path(saving_dir).glob("epoch_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"SMITH did not create an epoch CSV under {saving_dir}")
    return max(candidates, key=lambda path: int(path.stem.split("_", 1)[1]))


def run_command(command: list[str], log_path: str | Path, cwd: str | Path | None = None) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    source_root = REPO_ROOT / "src"
    environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd or REPO_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_smith(
    *,
    adata_file: str | Path,
    output_dir: str | Path,
    tasks: str,
    task_name: str,
    panel_size: int,
    epochs: int,
    device: str,
    seed: int,
    batch_size: int,
    layer: str = "raw",
    time_label: str | None = None,
    max_cells: int | None = None,
    sampling_strategy: str = "random",
    balance_mode: str = "capped",
    balance_cap: int = 500,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    saving_dir = output_dir / "ranking"
    log_dir = output_dir / "training_logs"
    existing = list(saving_dir.glob("epoch_*.csv"))
    if existing and not force:
        ranking = latest_epoch_csv(saving_dir)
        return {"status": "skipped_existing", "ranking_csv": str(ranking), "output_dir": str(output_dir)}

    record = max(1, int(epochs))
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "main.py"),
        "--adata_file", str(Path(adata_file).resolve()),
        "--saving_dir", str(saving_dir),
        "--log_dir", str(log_dir),
        "--tasks", tasks,
        "--task_name", task_name,
        "--layer", layer,
        "--panel_size", str(panel_size),
        "--epoch", str(epochs),
        "--record", str(record),
        "--batch_size", str(batch_size),
        "--device", device,
        "--seed", str(seed),
        "--balance_mode", balance_mode,
        "--balance_cap", str(balance_cap),
        "--sampling_strategy", sampling_strategy,
        "--val",
    ]
    if time_label:
        command.extend(["--time_label", time_label])
    if max_cells:
        command.extend(["--max_cells", str(max_cells)])
    write_json(output_dir / "command.json", {"command": command})
    run_command(command, output_dir / "training.log")
    ranking = latest_epoch_csv(saving_dir)
    panel_path = output_dir / f"panel_top{panel_size}.csv"
    pd.read_csv(ranking).head(panel_size).to_csv(panel_path, index=False)
    return {
        "status": "completed",
        "ranking_csv": str(ranking),
        "panel_csv": str(panel_path),
        "output_dir": str(output_dir),
        "command": command,
    }
