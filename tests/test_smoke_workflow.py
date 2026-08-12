from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import anndata as ad


def _env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo / "src") + ((os.pathsep + current) if current else "")
    return env


def test_make_smoke_h5ad(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "smoke.h5ad"
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "make_smoke_h5ad.py"),
            "--output",
            str(output),
            "--n-cells",
            "24",
            "--n-genes",
            "12",
        ],
        check=True,
        cwd=repo,
        env=_env(repo),
    )
    adata = ad.read_h5ad(output)
    assert adata.shape == (24, 12)
    assert {"celltype", "cell_type", "region", "pathology"}.issubset(adata.obs.columns)
    assert "spatial" in adata.obsm
    assert "raw" in adata.layers


def test_smith_one_epoch_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    adata_path = tmp_path / "smoke.h5ad"
    saving_dir = tmp_path / "saving"
    log_dir = tmp_path / "logs"
    env = _env(repo)

    subprocess.run(
        [sys.executable, str(repo / "scripts" / "make_smoke_h5ad.py"), "--output", str(adata_path)],
        check=True,
        cwd=repo,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "main.py"),
            "--adata_file",
            str(adata_path),
            "--saving_dir",
            str(saving_dir),
            "--log_dir",
            str(log_dir),
            "--tasks",
            "recon,cls,region,pathology,coordination",
            "--task_name",
            "celltype",
            "--panel_size",
            "8",
            "--epoch",
            "1",
            "--record",
            "1",
            "--batch_size",
            "16",
            "--rep_dim",
            "8",
            "--rep_hidden_dims",
            "8",
            "--head_hidden_dims",
            "8",
            "--dim",
            "8",
            "--device",
            "cpu",
            "--seed",
            "7",
            "--balance_mode",
            "off",
        ],
        check=True,
        cwd=repo,
        env=env,
    )
    marker_file = saving_dir / "epoch_0.csv"
    assert marker_file.exists()
    text = marker_file.read_text(encoding="utf-8")
    assert text.startswith("marker")
