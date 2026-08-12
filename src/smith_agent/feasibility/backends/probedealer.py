from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from smith_agent.probedealer import OligoDesignConfig, build_oligo_array, load_fasta_records
from smith_agent.probedealer.transcriptome import (
    biomart_dataset_for_species,
    fetch_biomart_transcriptome,
    filter_probes_by_transcriptome,
    load_transcript_to_gene,
    make_blast_db,
    reference_slug_for_species,
)
from smith_agent.schemas import BackendResult


ROOT_DIR = Path(os.environ.get("SMITH_PACKAGE_ROOT", Path(__file__).resolve().parents[4]))
BLASTN_PATH = Path(os.environ.get("SMITH_BLASTN", ROOT_DIR / "third_party/envs/probedealer_blast/bin/blastn"))
MAKEBLASTDB_PATH = Path(os.environ.get("SMITH_MAKEBLASTDB", ROOT_DIR / "third_party/envs/probedealer_blast/bin/makeblastdb"))
FULL_MOUSE_REF_DIR = ROOT_DIR / "third_party/reference_data/probedealer_mouse_full"
REFERENCE_DATA_DIR = ROOT_DIR / "third_party/reference_data"


class ProbeDealerBackend:
    backend_name = "probedealer_py"

    def _reference_dir(self, species: str) -> Path:
        slug = reference_slug_for_species(species)
        if slug == "mouse":
            return FULL_MOUSE_REF_DIR
        return REFERENCE_DATA_DIR / f"probedealer_{slug}_full"

    def ensure_mouse_reference(self, transcript_ids: list[str] | None = None) -> tuple[Path, Path, Path]:
        return self.ensure_reference("mus_musculus", transcript_ids=transcript_ids)

    def ensure_reference(
        self,
        species: str = "mus_musculus",
        transcript_ids: list[str] | None = None,
    ) -> tuple[Path, Path, Path]:
        biomart_dataset_for_species(species)
        ref_dir = self._reference_dir(species)
        ref_dir.mkdir(parents=True, exist_ok=True)
        short_fasta = ref_dir / "TxShortHeader.fa"
        mapping_tsv = ref_dir / "transcript_to_gene.tsv"
        db_prefix = ref_dir / "TxShortHeader"

        if not short_fasta.exists() or not mapping_tsv.exists():
            fetch_biomart_transcriptome(ref_dir, transcript_ids=transcript_ids, species=species)
        if not (ref_dir / "TxShortHeader.nsq").exists():
            make_blast_db(short_fasta, db_prefix, MAKEBLASTDB_PATH)
        return short_fasta, mapping_tsv, db_prefix

    def run_transcript_fasta(
        self,
        fasta_path: str | Path,
        output_dir: str | Path,
        use_full_mouse_reference: bool = True,
        use_transcriptome_reference: bool | None = None,
        species: str = "mus_musculus",
    ) -> BackendResult:
        fasta_path = Path(fasta_path).resolve()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records = load_fasta_records(fasta_path)
        designed = build_oligo_array(records, OligoDesignConfig())

        rows = []
        output_fasta = output_dir / "initial_probes.fa"
        with output_fasta.open("w") as handle:
            for sequence_id, probes in designed.items():
                for probe in probes:
                    handle.write(f">{probe.header}\n{probe.sequence}\n")

        use_reference = use_full_mouse_reference if use_transcriptome_reference is None else use_transcriptome_reference
        if use_reference:
            transcript_ids = [sequence_id for sequence_id, _ in records]
            _short_fasta, mapping_tsv, db_prefix = self.ensure_reference(species, transcript_ids=None)
            transcript_to_gene = load_transcript_to_gene(mapping_tsv)
        else:
            transcript_to_gene = {}
            db_prefix = None

        filtered_fasta = output_dir / "final_probes.fa"
        with filtered_fasta.open("w") as handle:
            for sequence_id, _ in records:
                probes = designed[sequence_id]
                filtered = probes
                if db_prefix is not None:
                    filtered = filter_probes_by_transcriptome(
                        probes,
                        transcript_to_gene=transcript_to_gene,
                        blast_db_prefix=db_prefix,
                        blastn_path=BLASTN_PATH,
                        work_dir=output_dir / f"blast_{sequence_id}",
                    )
                for probe in filtered:
                    handle.write(f">{probe.header}\n{probe.sequence}\n")
                rows.append(
                    {
                        "transcript_id": sequence_id,
                        "initial_probe_count": len(probes),
                        "final_probe_count": len(filtered),
                    }
                )

        summary_path = output_dir / "probedealer_summary.tsv"
        pd.DataFrame(rows).to_csv(summary_path, sep="\t", index=False)
        return BackendResult(
            backend=self.backend_name,
            status="ok",
            input_summary={
                "fasta_path": str(fasta_path),
                "n_transcripts": len(records),
                "species": species,
                "reference_db": str(db_prefix) if db_prefix is not None else "",
            },
            metrics={
                "initial_probe_count_total": int(sum(row["initial_probe_count"] for row in rows)),
                "final_probe_count_total": int(sum(row["final_probe_count"] for row in rows)),
            },
            output_files={
                "summary_tsv": str(summary_path),
                "initial_probes_fasta": str(output_fasta),
                "final_probes_fasta": str(filtered_fasta),
            },
        )
