from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from smith_agent.utils import ensure_dir, write_json


@dataclass
class SmithSelectionConfig:
    python_executable: str
    smith_root: Path
    adata_file: Path
    saving_dir: Path
    log_dir: Path
    tasks: str
    task_name: str
    panel_size: int = 64
    epoch: int = 200
    record: int = 50
    device: str = "cpu"
    seed: int = 42
    learning_rate: float = 0.001
    batch_size: int = 32
    dim: int = 32
    rep_dim: int = 32
    rep_hidden_dims: str = "32"
    head_hidden_dims: str = ""
    dropout_rate: float = 0.2
    lam: float = 0.5
    sigma: float = 0.5
    activation: str = "tanh"
    optimizer: str = "Adam"
    extra_args: dict[str, Any] | None = None


def build_smith_command(config: SmithSelectionConfig) -> list[str]:
    main_py = config.smith_root / "main.py"
    command = [
        config.python_executable,
        str(main_py),
        "--adata_file",
        str(config.adata_file),
        "--saving_dir",
        str(config.saving_dir),
        "--log_dir",
        str(config.log_dir),
        "--tasks",
        config.tasks,
        "--task_name",
        config.task_name,
        "--panel_size",
        str(config.panel_size),
        "--epoch",
        str(config.epoch),
        "--record",
        str(config.record),
        "--device",
        config.device,
        "--seed",
        str(config.seed),
        "--learning_rate",
        str(config.learning_rate),
        "--batch_size",
        str(config.batch_size),
        "--dim",
        str(config.dim),
        "--rep_dim",
        str(config.rep_dim),
        "--rep_hidden_dims",
        config.rep_hidden_dims,
        "--head_hidden_dims",
        config.head_hidden_dims,
        "--dropout_rate",
        str(config.dropout_rate),
        "--lam",
        str(config.lam),
        "--sigma",
        str(config.sigma),
        "--activation",
        config.activation,
        "--optimizer",
        config.optimizer,
    ]
    for key, value in sorted((config.extra_args or {}).items()):
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        command.extend([flag, str(value)])
    return command


def run_smith_selection(config: SmithSelectionConfig, execute: bool = False) -> dict[str, Any]:
    ensure_dir(config.saving_dir)
    ensure_dir(config.log_dir)
    command = build_smith_command(config)
    manifest = {
        "command": command,
        "execute": execute,
        "saving_dir": str(config.saving_dir),
        "log_dir": str(config.log_dir),
    }
    manifest_path = write_json(config.saving_dir / "smith_run_manifest.json", manifest)
    if not execute:
        return {
            "status": "planned",
            "command": command,
            "manifest_path": str(manifest_path),
        }

    completed = subprocess.run(
        command,
        cwd=str(config.smith_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "status": "completed",
        "command": command,
        "manifest_path": str(manifest_path),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def load_ranked_genes(epoch_csv: str | Path, limit: int | None = None) -> list[str]:
    df = pd.read_csv(epoch_csv)
    genes = df.iloc[:, 0].astype(str).tolist()
    return genes[:limit] if limit is not None else genes
