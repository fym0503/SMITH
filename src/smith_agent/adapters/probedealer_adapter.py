from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from Bio import SeqIO

from smith_agent.adapters.external import add_sys_paths
from smith_agent.schemas import BackendResult
from smith_agent.utils import ensure_dir, write_json


def run_probedealer_screen_light(
    package_root: str | Path,
    transcript_fasta: str | Path,
    output_dir: str | Path,
    config_overrides: dict[str, Any] | None = None,
) -> BackendResult:
    output_dir = ensure_dir(output_dir)
    add_sys_paths([package_root, Path(package_root) / "src"])

    from smith_agent.probedealer import OligoDesignConfig, build_oligo_array, load_fasta_records

    cfg = OligoDesignConfig(**(config_overrides or {}))
    records = load_fasta_records(transcript_fasta)
    designed = build_oligo_array(records, cfg)

    summary_rows: list[dict[str, Any]] = []
    fasta_records = []
    for transcript_id, probes in designed.items():
        summary_rows.append(
            {
                "transcript_id": transcript_id,
                "initial_probe_count": len(probes),
                "final_probe_count": len(probes),
            }
        )
        for probe in probes:
            fasta_records.append((probe.header, probe.sequence))

    summary_path = output_dir / "probedealer_summary.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)

    fasta_path = output_dir / "final_probes.fa"
    with fasta_path.open("w", encoding="utf-8") as handle:
        for header, sequence in fasta_records:
            handle.write(f">{header}\n{sequence}\n")

    metadata_path = output_dir / "run_metadata.json"
    metadata = {
        "transcript_fasta": str(Path(transcript_fasta).resolve()),
        "transcript_count": len(records),
        "config": config_overrides or {},
    }
    write_json(metadata_path, metadata)

    return BackendResult(
        backend="probedealer",
        status="completed",
        input_summary={
            "transcript_fasta": str(transcript_fasta),
            "transcript_count": len(records),
        },
        metrics={
            "transcript_count": len(records),
            "total_final_probe_count": int(sum(row["final_probe_count"] for row in summary_rows)),
        },
        output_files={
            "probedealer_summary_tsv": str(summary_path),
            "final_probes_fasta": str(fasta_path),
            "run_metadata_json": str(metadata_path),
        },
        notes=["Lightweight ProbeDealer screen completed without transcriptome BLAST filtering."],
    )


def summarize_fasta_records(fasta_path: str | Path) -> dict[str, Any]:
    count = 0
    lengths: list[int] = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        count += 1
        lengths.append(len(record.seq))
    return {
        "record_count": count,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }
