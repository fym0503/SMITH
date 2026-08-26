"""Run the manuscript-scale Agent probe feasibility analysis from live rankings."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from smith_agent.feasibility.manuscript import (
    MANUSCRIPT_FIGURE6G_GENES,
    manuscript_offtarget_examples,
    manuscript_pass_rates,
    ranked_gene_universe,
    write_probe_audit_tables,
)


def _run(command: list[str], *, env: dict[str, str], cwd: Path) -> None:
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Probe feasibility executable is missing: {exc}. Configure the backend environment before running Figure 6f-g."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Probe feasibility command failed with exit code {exc.returncode}: {' '.join(command)}"
        ) from exc


def run_probe_feasibility_loaded(
    ranking_frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    species: str = "homo_sapiens",
    human_reference_dir: str | Path | None = None,
    gene_metadata_h5ad: str | Path | None = None,
    max_genes: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run ODT/OligoMiner/ProbeDealer and return current-run tables in memory.

    The two legacy scripts are used only as backend runners. Their newly created
    files are loaded immediately into memory, summarized, and saved as audit
    artifacts; no prior output is accepted as an input.
    """
    root = Path(project_root).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ranking = ranked_gene_universe(ranking_frame, max_genes=max_genes)
    rank_path = out / "current_smith_ranked_genes.tsv"
    ranking.to_csv(rank_path, sep="\t", index=False)
    probedealer_dir = out / "probedealer_full_scan"
    three_tool_dir = out / "three_tool_scan"
    env = os.environ.copy()
    # The backend runner must use the active environment. Some cluster images
    # expose only python3, so never assume a bare `python` executable exists.
    configured_odt = env.get("SMITH_ODT_PYTHON", "")
    configured_ok = bool(configured_odt) and (
        Path(configured_odt).is_file() or shutil.which(configured_odt) is not None
    )
    if not configured_ok:
        env["SMITH_ODT_PYTHON"] = sys.executable
    if human_reference_dir:
        env["SMITH_HUMAN_REFERENCE_DIR"] = str(Path(human_reference_dir).resolve())
    if gene_metadata_h5ad:
        env["SMITH_SOURCE_GENE_METADATA_H5AD"] = str(Path(gene_metadata_h5ad).resolve())
    probe_script = root / "scripts/agent_examples/run_scrna_full_probedealer_feasibility_scan.py"
    three_tool_script = root / "scripts/agent_examples/run_scrna_full_odt_oligominer_scan.py"
    if not probe_script.is_file() or not three_tool_script.is_file():
        raise FileNotFoundError("The manuscript probe backend runner scripts are missing from the SMITH package.")
    probe_command = [
        sys.executable,
        str(probe_script),
        "--rank-tsv",
        str(rank_path),
        "--output-dir",
        str(probedealer_dir),
        "--top-n",
        str(len(ranking)),
        "--species",
        species,
    ]
    if gene_metadata_h5ad:
        probe_command.extend(["--gene-metadata-h5ad", str(Path(gene_metadata_h5ad).resolve())])
    if force:
        probe_command.append("--force-manifest")
    _run(probe_command, env=env, cwd=root)
    three_tool_command = [
        sys.executable,
        str(three_tool_script),
        "--rank-tsv",
        str(rank_path),
        "--probedealer-scan-dir",
        str(probedealer_dir),
        "--output-dir",
        str(three_tool_dir),
        "--species",
        species,
    ]
    if force:
        three_tool_command.extend(["--force-odt", "--force-oligominer"])
    _run(three_tool_command, env=env, cwd=root)

    feasibility_path = three_tool_dir / "three_tool_feasibility_table.tsv"
    risk_path = probedealer_dir / "probe_risk_summary.tsv"
    required_outputs = {
        "three_tool": feasibility_path,
        "probedealer": risk_path,
    }
    missing_outputs = [name for name, path in required_outputs.items() if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(
            "Probe backends completed without producing the required Figure 6f-g tables: "
            + ", ".join(missing_outputs)
        )
    feasibility = pd.read_csv(feasibility_path, sep="\t")
    risk = pd.read_csv(risk_path, sep="\t")
    pass_rates = manuscript_pass_rates(feasibility)
    examples = manuscript_offtarget_examples(risk, genes=MANUSCRIPT_FIGURE6G_GENES)
    audit = write_probe_audit_tables(out, feasibility, risk, pass_rates, examples)
    return {
        "ranking": ranking,
        "feasibility": feasibility,
        "risk": risk,
        "pass_rates": pass_rates,
        "examples": examples,
        "audit": audit,
        "status": "completed" if not missing_outputs else "incomplete",
        "backend_outputs": {"probedealer": str(probedealer_dir), "three_tool": str(three_tool_dir)},
    }
