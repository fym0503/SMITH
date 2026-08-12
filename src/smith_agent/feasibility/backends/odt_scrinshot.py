from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from smith_agent.schemas import BackendResult


ROOT_DIR = Path(os.environ.get("SMITH_PACKAGE_ROOT", Path(__file__).resolve().parents[4]))
ODT_PYTHON = Path(os.environ.get("SMITH_ODT_PYTHON", "python"))
ODT_SCRIPT = Path(os.environ.get("SMITH_ODT_SCRIPT", ROOT_DIR / "scripts" / "odt_scrinshot_feasibility_demo.py"))
ODT_LIGHT_SCRIPT = Path(os.environ.get("SMITH_ODT_LIGHT_SCRIPT", ROOT_DIR / "scripts" / "odt_property_only_runner.py"))


class ODTScrinshotBackend:
    backend_name = "odt_scrinshot"

    def run_gene_symbols(
        self,
        genes: list[str],
        species: str,
        output_dir: str | Path,
        set_size_min: int = 3,
        set_size_opt: int = 5,
        n_sets: int = 20,
    ) -> BackendResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(ODT_PYTHON),
            str(ODT_SCRIPT),
            "--species",
            species,
            "--output-dir",
            str(output_dir),
            "--set-size-min",
            str(set_size_min),
            "--set-size-opt",
            str(set_size_opt),
            "--n-sets",
            str(n_sets),
            "--genes",
            *genes,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)
        summary_path = output_dir / "feasibility_summary.tsv"

        if proc.returncode != 0 or not summary_path.exists():
            return BackendResult(
                backend=self.backend_name,
                status="error",
                input_summary={"genes": genes, "species": species},
                notes=[proc.stderr.strip() or proc.stdout.strip() or "ODT run failed."],
            )

        df = pd.read_csv(summary_path, sep="\t")
        feasible_count = int(df["feasible_property_only"].fillna(False).astype(bool).sum())
        return BackendResult(
            backend=self.backend_name,
            status="ok",
            input_summary={"genes": genes, "species": species},
            metrics={
                "n_genes": int(len(df)),
                "feasible_property_only_count": feasible_count,
                "best_set_size_max": int(df["best_set_size"].fillna(0).max()),
            },
            output_files={
                "summary_tsv": str(summary_path),
                "metadata_json": str(output_dir / "run_metadata.json"),
            },
            notes=[proc.stdout.strip()] if proc.stdout.strip() else [],
        )

    def run_gene_symbols_property_only(
        self,
        genes: list[str],
        species: str,
        output_dir: str | Path,
        set_size_min: int = 2,
    ) -> BackendResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(ODT_PYTHON),
            str(ODT_LIGHT_SCRIPT),
            "--species",
            species,
            "--output-dir",
            str(output_dir),
            "--set-size-min",
            str(set_size_min),
            "--genes",
            *genes,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)
        summary_path = output_dir / "property_only_summary.tsv"
        if proc.returncode != 0 or not summary_path.exists():
            return BackendResult(
                backend=self.backend_name,
                status="error",
                input_summary={"genes": genes, "species": species},
                notes=[proc.stderr.strip() or proc.stdout.strip() or "ODT property-only run failed."],
            )
        df = pd.read_csv(summary_path, sep="\t")
        feasible_count = int(df["feasible_property_only"].fillna(False).astype(bool).sum())
        return BackendResult(
            backend=self.backend_name,
            status="ok",
            input_summary={"genes": genes, "species": species, "mode": "property_filter_only"},
            metrics={
                "n_genes": int(len(df)),
                "feasible_property_only_count": feasible_count,
                "candidate_oligos_after_property_filters_total": int(
                    df["candidate_oligos_after_property_filters"].fillna(0).sum()
                ),
            },
            output_files={
                "summary_tsv": str(summary_path),
                "metadata_json": str(output_dir / "run_metadata.json"),
            },
            notes=[proc.stdout.strip()] if proc.stdout.strip() else [],
        )
