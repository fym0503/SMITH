from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from smith_agent.adapters.external import add_sys_paths


def _backend_root(package_root: str | Path) -> Path:
    root = Path(package_root).resolve()
    add_sys_paths([root, root / "src"])
    return root


def run_odt_property_screen(
    package_root: str | Path,
    genes: list[str],
    species: str,
    output_dir: str | Path,
    set_size_min: int = 2,
) -> dict[str, Any]:
    _backend_root(package_root)
    from smith_agent.feasibility.backends.odt_scrinshot import ODTScrinshotBackend

    backend = ODTScrinshotBackend()
    return backend.run_gene_symbols_property_only(
        genes=genes,
        species=species,
        output_dir=Path(output_dir).resolve(),
        set_size_min=set_size_min,
    ).to_dict()


def run_odt_property_batches(
    package_root: str | Path,
    manifest_tsv: str | Path,
    species: str,
    output_dir: str | Path,
    batch_size: int = 10,
    max_workers: int = 8,
    set_size_min: int = 2,
) -> dict[str, Any]:
    root = _backend_root(package_root)
    script = root / "scripts" / "run_odt_property_batches.py"
    python_path = Path(os.environ.get("SMITH_ODT_PYTHON", "python"))
    out_dir = Path(output_dir).resolve()
    import subprocess
    import pandas as pd

    cmd = [
        str(python_path),
        str(script),
        "--manifest",
        str(Path(manifest_tsv).resolve()),
        "--species",
        species,
        "--output-dir",
        str(out_dir),
        "--batch-size",
        str(batch_size),
        "--max-workers",
        str(max_workers),
        "--set-size-min",
        str(set_size_min),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    summary_path = out_dir / "property_only_summary.tsv"
    if proc.returncode != 0 or not summary_path.exists():
        return {
            "backend": "odt_scrinshot_batches",
            "status": "error",
            "input_summary": {"manifest_tsv": str(manifest_tsv), "species": species},
            "metrics": {},
            "output_files": {},
            "notes": [proc.stderr.strip() or proc.stdout.strip() or "ODT batch property run failed."],
        }
    df = pd.read_csv(summary_path, sep="\t")
    return {
        "backend": "odt_scrinshot_batches",
        "status": "ok",
        "input_summary": {"manifest_tsv": str(manifest_tsv), "species": species, "batch_size": batch_size},
        "metrics": {
            "n_genes": int(len(df)),
            "feasible_property_only_count": int(df["feasible_property_only"].fillna(False).astype(bool).sum()),
        },
        "output_files": {"summary_tsv": str(summary_path)},
        "notes": [proc.stdout.strip()] if proc.stdout.strip() else [],
    }


def run_oligominer_specificity_screen(
    package_root: str | Path,
    transcript_fasta: str | Path,
    output_dir: str | Path,
    temperature_c: int = 42,
    species: str = "mus_musculus",
) -> dict[str, Any]:
    _backend_root(package_root)
    from smith_agent.feasibility.backends.oligominer import OligoMinerBackend

    backend = OligoMinerBackend()
    return backend.run_multi_transcript_specificity(
        fasta_path=transcript_fasta,
        output_dir=Path(output_dir).resolve(),
        temperature_c=temperature_c,
        species=species,
    ).to_dict()


def run_probedealer_backend_screen(
    package_root: str | Path,
    transcript_fasta: str | Path,
    output_dir: str | Path,
    use_full_mouse_reference: bool = True,
    use_transcriptome_reference: bool | None = None,
    species: str = "mus_musculus",
) -> dict[str, Any]:
    _backend_root(package_root)
    from smith_agent.feasibility.backends.probedealer import ProbeDealerBackend

    backend = ProbeDealerBackend()
    return backend.run_transcript_fasta(
        fasta_path=transcript_fasta,
        output_dir=Path(output_dir).resolve(),
        use_full_mouse_reference=use_full_mouse_reference,
        use_transcriptome_reference=use_transcriptome_reference,
        species=species,
    ).to_dict()
