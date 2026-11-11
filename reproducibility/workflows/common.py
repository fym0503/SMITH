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
import yaml
from scipy import sparse

import anndata as ad


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_MODEL_CONFIG = REPO_ROOT / "reproducibility/configs/training_models.yaml"
TUTORIAL_PROFILE_CONFIG = REPO_ROOT / "reproducibility/configs/tutorial_profiles.yaml"


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


def resolve_training_model(model_profile: str) -> dict[str, Any]:
    """Resolve a named training implementation from the checked-in model profiles."""
    payload = yaml.safe_load(TRAINING_MODEL_CONFIG.read_text(encoding="utf-8"))
    try:
        model = dict(payload["models"][model_profile])
    except KeyError as exc:
        raise ValueError(
            f"Unknown training model {model_profile!r}; choose from {', '.join(payload['models'])}"
        ) from exc
    model["profile"] = model_profile
    model["entrypoint"] = REPO_ROOT / model["entrypoint"]
    if model["training_backend"] == "paper_20260428":
        import json

        manifest = json.loads((REPO_ROOT / model["source_manifest"]).read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = model["entrypoint"].parent / item["path"]
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Manuscript backend checksum mismatch: {item['path']}")
    return model


def resolve_tutorial_profile(profile_id: str) -> dict[str, Any]:
    """Resolve a user-facing, task-specific training and evaluation profile."""
    payload = yaml.safe_load(TUTORIAL_PROFILE_CONFIG.read_text(encoding="utf-8"))
    profiles = payload["profiles"]
    aliases = {alias.lower(): key for key, value in profiles.items() for alias in [key, *value.get("aliases", [])]}
    try:
        canonical = aliases[str(profile_id).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown tutorial profile {profile_id!r}; choose from {', '.join(profiles)}") from exc
    profile = dict(profiles[canonical])
    profile["id"] = canonical
    profile["training_model"] = resolve_training_model(profile["training_model"])
    return profile


def ranked_genes(ranking: pd.DataFrame, panel_size: int | None = None) -> list[str]:
    """Return unique normalized genes from an in-memory SMITH ranking."""
    gene_column = next(
        (name for name in ranking.columns if str(name).lower() in {"marker", "gene", "gene_symbol", "target"}),
        ranking.columns[0],
    )
    genes: list[str] = []
    for value in ranking[gene_column]:
        gene = clean_gene(value)
        if gene and gene not in genes:
            genes.append(gene)
    return genes[:panel_size] if panel_size else genes


def write_panel_genes(genes: list[str], output_path: str | Path) -> Path:
    """Persist an in-memory panel without reading a ranking file again."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rank": range(1, len(genes) + 1), "gene_symbol": genes}).to_csv(
        output_path,
        sep="\t" if output_path.suffix.lower() == ".tsv" else ",",
        index=False,
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
    adata_file: str | Path | None,
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
    save_model: bool = False,
    force: bool = False,
    adata: ad.AnnData | None = None,
    include_in_memory: bool = False,
    model_options: dict[str, Any] | None = None,
    validation: bool = True,
    record_interval: int | None = None,
    model_profile: str = "smith_current",
    tutorial_profile: str | None = None,
    training_backend: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if adata is not None:
        input_snapshot = output_dir / "input_snapshot.h5ad"
        adata.write_h5ad(input_snapshot)
        adata_file = input_snapshot
    if adata_file is None:
        raise ValueError("run_smith requires adata_file or an in-memory AnnData object")
    saving_dir = output_dir / "ranking"
    log_dir = output_dir / "training_logs"
    existing = list(saving_dir.glob("epoch_*.csv"))
    if existing and not force:
        if save_model and not list(saving_dir.glob("*-rep-epoch*.pt")):
            force = True
        else:
            ranking = latest_epoch_csv(saving_dir)
            checkpoints = list(saving_dir.glob("*-rep-epoch*.pt"))
            result: dict[str, Any] = {
                "status": "skipped_existing",
                "ranking_csv": str(ranking),
                "output_dir": str(output_dir),
                "checkpoint_file": str(checkpoints[-1]) if checkpoints else None,
            }
            if include_in_memory:
                ranking_frame = pd.read_csv(ranking)
                result["ranking_frame"] = ranking_frame
                result["panel_genes"] = ranked_genes(ranking_frame, panel_size)
            return result

    record = record_interval or max(1, int(epochs))
    selected_tutorial_profile = None
    if tutorial_profile is not None:
        selected_tutorial_profile = resolve_tutorial_profile(tutorial_profile)
        model_profile = selected_tutorial_profile["training_model"]["profile"]
    if training_backend is not None:
        legacy_profiles = {"current": "smith_current", "paper_20260428": "smith_paper_20260428"}
        try:
            model_profile = legacy_profiles[training_backend]
        except KeyError as exc:
            raise ValueError(f"Unknown legacy training backend: {training_backend}") from exc
    model = resolve_training_model(model_profile)
    entrypoint = model["entrypoint"]
    command = [
        sys.executable,
        str(entrypoint),
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
    ]
    if validation:
        command.append("--val")
    allowed = {"learning_rate", "dim", "rep_dim", "rep_hidden_dims", "head_hidden_dims",
               "dropout_rate", "lam", "sigma", "activation", "optimizer"}
    for name, value in (model_options or {}).items():
        if name not in allowed:
            raise ValueError(f"Unsupported model option: {name}")
        command.extend([f"--{name}", str(value)])
    if time_label:
        command.extend(["--time_label", time_label])
    if max_cells:
        command.extend(["--max_cells", str(max_cells)])
    if save_model:
        command.append("--save")
    write_json(output_dir / "command.json", {
        "model_profile": model_profile,
        "tutorial_profile": selected_tutorial_profile["id"] if selected_tutorial_profile else None,
        "tutorial_profile_settings": selected_tutorial_profile.get("parameters", {}) if selected_tutorial_profile else {},
        "training_backend": model["training_backend"],
        "model_description": model["description"],
        "model_options": model_options or {},
        "command": command,
    })
    run_command(command, output_dir / "training.log")
    ranking = latest_epoch_csv(saving_dir)
    panel_path = output_dir / f"panel_top{panel_size}.csv"
    pd.read_csv(ranking).head(panel_size).to_csv(panel_path, index=False)
    epoch = int(ranking.stem.split("_", 1)[1])
    checkpoint = saving_dir / f"{tasks.replace(',', '-')}-seed{seed}-{task_name}-rep-epoch{epoch}.pt"
    result = {
        "status": "completed",
        "ranking_csv": str(ranking),
        "panel_csv": str(panel_path),
        "output_dir": str(output_dir),
        "model_profile": model_profile,
        "tutorial_profile": selected_tutorial_profile["id"] if selected_tutorial_profile else None,
        "command": command,
        "checkpoint_file": str(checkpoint) if checkpoint.is_file() else None,
    }
    if include_in_memory:
        ranking_frame = pd.read_csv(ranking)
        result["ranking_frame"] = ranking_frame
        result["panel_genes"] = ranked_genes(ranking_frame, panel_size)
    return result
