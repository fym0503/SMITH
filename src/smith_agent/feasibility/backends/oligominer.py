from __future__ import annotations

import os
import subprocess
from pathlib import Path

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from smith_agent.probedealer.transcriptome import (
    biomart_dataset_for_species,
    fetch_biomart_transcriptome,
    reference_slug_for_species,
)

from smith_agent.schemas import BackendResult


ROOT_DIR = Path(os.environ.get("SMITH_PACKAGE_ROOT", Path(__file__).resolve().parents[4]))
OLIGOMINER_PYTHON = Path(os.environ.get("SMITH_OLIGOMINER_PYTHON", ROOT_DIR / "third_party/envs/oligominer/bin/python"))
OLIGOMINER_BOWTIE2 = Path(os.environ.get("SMITH_OLIGOMINER_BOWTIE2", ROOT_DIR / "third_party/envs/oligominer/bin/bowtie2"))
OLIGOMINER_BOWTIE2_BUILD = Path(os.environ.get("SMITH_OLIGOMINER_BOWTIE2_BUILD", ROOT_DIR / "third_party/envs/oligominer/bin/bowtie2-build"))
OLIGOMINER_DIR = ROOT_DIR / "third_party/OligoMiner"
DEFAULT_MOUSE_TRANSCRIPTOME_FASTA = ROOT_DIR / "third_party/reference_data/probedealer_mouse_full/TxShortHeader.fa"
DEFAULT_MOUSE_TRANSCRIPTOME_INDEX = ROOT_DIR / "third_party/reference_data/oligominer_mouse_full_tx/mouse_tx"
REFERENCE_DATA_DIR = ROOT_DIR / "third_party/reference_data"


class OligoMinerBackend:
    backend_name = "oligominer"

    def _reference_paths(self, species: str) -> tuple[Path, Path, Path]:
        slug = reference_slug_for_species(species)
        if slug == "mouse":
            return (
                ROOT_DIR / "third_party/reference_data/probedealer_mouse_full",
                DEFAULT_MOUSE_TRANSCRIPTOME_FASTA,
                DEFAULT_MOUSE_TRANSCRIPTOME_INDEX,
            )
        ref_dir = REFERENCE_DATA_DIR / f"probedealer_{slug}_full"
        index_prefix = REFERENCE_DATA_DIR / f"oligominer_{slug}_full_tx" / f"{slug}_tx"
        return ref_dir, ref_dir / "TxShortHeader.fa", index_prefix

    def ensure_mouse_transcriptome_index(self) -> Path:
        return self.ensure_transcriptome_index("mus_musculus")

    def ensure_transcriptome_index(self, species: str = "mus_musculus") -> Path:
        biomart_dataset_for_species(species)
        ref_dir, transcriptome_fasta, index_prefix = self._reference_paths(species)
        mapping_tsv = ref_dir / "transcript_to_gene.tsv"
        if not transcriptome_fasta.exists() or not mapping_tsv.exists():
            fetch_biomart_transcriptome(ref_dir, species=species)

        index_dir = index_prefix.parent
        index_dir.mkdir(parents=True, exist_ok=True)
        if (index_dir / f"{index_prefix.name}.1.bt2").exists():
            return index_prefix

        cmd = [
            str(OLIGOMINER_BOWTIE2_BUILD),
            str(transcriptome_fasta),
            str(index_prefix),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT_DIR)
        return index_prefix

    def run_blockparse(
        self,
        fasta_path: str | Path,
        output_dir: str | Path,
        output_stem: str = "oligominer_run",
    ) -> BackendResult:
        fasta_path = Path(fasta_path).resolve()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        record_count = sum(1 for _ in SeqIO.parse(str(fasta_path), "fasta"))
        if record_count != 1:
            return BackendResult(
                backend=self.backend_name,
                status="error",
                input_summary={"fasta_path": str(fasta_path)},
                notes=["OligoMiner blockParse currently requires a single-entry FASTA."],
            )

        cmd = [
            str(OLIGOMINER_PYTHON),
            str(OLIGOMINER_DIR / "blockParse.py"),
            "-f",
            str(fasta_path),
            "-o",
            output_stem,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
        fastq_path = output_dir / f"{output_stem}.fastq"
        if proc.returncode != 0 or not fastq_path.exists():
            return BackendResult(
                backend=self.backend_name,
                status="error",
                input_summary={"fasta_path": str(fasta_path)},
                notes=[proc.stderr.strip() or proc.stdout.strip() or "OligoMiner blockParse failed."],
            )

        line_count = sum(1 for _ in fastq_path.open())
        candidate_count = line_count // 4
        return BackendResult(
            backend=self.backend_name,
            status="ok",
            input_summary={"fasta_path": str(fasta_path)},
            metrics={"candidate_probe_count": int(candidate_count)},
            output_files={"fastq": str(fastq_path.resolve())},
            notes=[proc.stdout.strip()] if proc.stdout.strip() else [],
        )

    def run_multi_transcript_fasta(
        self,
        fasta_path: str | Path,
        output_dir: str | Path,
    ) -> BackendResult:
        fasta_path = Path(fasta_path).resolve()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        per_transcript = []
        total_candidates = 0
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            transcript_dir = output_dir / record.id
            transcript_dir.mkdir(parents=True, exist_ok=True)
            transcript_fasta = transcript_dir / f"{record.id}.fa"
            with transcript_fasta.open("w") as handle:
                clean_record = SeqRecord(record.seq, id=record.id, description=record.id)
                SeqIO.write([clean_record], handle, "fasta")

            result = self.run_blockparse(
                fasta_path=transcript_fasta,
                output_dir=transcript_dir,
                output_stem=record.id,
            )
            count = int(result.metrics.get("candidate_probe_count", 0))
            total_candidates += count
            per_transcript.append((record.id, count))

        summary_path = output_dir / "oligominer_summary.tsv"
        with summary_path.open("w") as handle:
            handle.write("transcript_id\tcandidate_probe_count\n")
            for transcript_id, count in per_transcript:
                handle.write(f"{transcript_id}\t{count}\n")

        return BackendResult(
            backend=self.backend_name,
            status="ok",
            input_summary={"fasta_path": str(fasta_path), "n_transcripts": len(per_transcript)},
            metrics={"candidate_probe_count_total": int(total_candidates)},
            output_files={"summary_tsv": str(summary_path)},
        )

    def run_multi_transcript_specificity(
        self,
        fasta_path: str | Path,
        output_dir: str | Path,
        temperature_c: int = 42,
        species: str = "mus_musculus",
    ) -> BackendResult:
        fasta_path = Path(fasta_path).resolve()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        index_prefix = self.ensure_transcriptome_index(species)

        per_transcript = []
        total_candidates = 0
        total_specific = 0

        for record in SeqIO.parse(str(fasta_path), "fasta"):
            transcript_dir = output_dir / record.id
            transcript_dir.mkdir(parents=True, exist_ok=True)
            transcript_fasta = transcript_dir / f"{record.id}.fa"
            with transcript_fasta.open("w") as handle:
                clean_record = SeqRecord(record.seq, id=record.id, description=record.id)
                SeqIO.write([clean_record], handle, "fasta")

            blockparse_result = self.run_blockparse(
                fasta_path=transcript_fasta,
                output_dir=transcript_dir,
                output_stem=record.id,
            )
            fastq_path = Path(blockparse_result.output_files["fastq"]).resolve()
            sam_path = (transcript_dir / f"{record.id}.sam").resolve()
            bowtie_cmd = [
                str(OLIGOMINER_BOWTIE2),
                "-x",
                str(index_prefix),
                "-U",
                str(fastq_path),
                "--no-hd",
                "-t",
                "-k",
                "2",
                "--local",
                "-D",
                "20",
                "-R",
                "3",
                "-N",
                "1",
                "-L",
                "20",
                "-i",
                "C,4",
                "--score-min",
                "G,1,4",
                "-S",
                str(sam_path),
            ]
            bowtie_proc = subprocess.run(
                bowtie_cmd,
                capture_output=True,
                text=True,
                cwd=transcript_dir,
            )
            if bowtie_proc.returncode != 0 or not sam_path.exists():
                return BackendResult(
                    backend=self.backend_name,
                    status="error",
                    input_summary={"fasta_path": str(fasta_path)},
                    notes=[bowtie_proc.stderr.strip() or bowtie_proc.stdout.strip() or "bowtie2 failed"],
                )

            clean_cmd = [
                str(OLIGOMINER_PYTHON),
                str(OLIGOMINER_DIR / "outputClean.py"),
                "-f",
                str(sam_path),
                "-T",
                str(temperature_c),
                "-o",
                record.id,
            ]
            # outputClean.py is sensitive to blank lines in SAM files.
            cleaned_sam_path = (transcript_dir / f"{record.id}.clean.sam").resolve()
            with sam_path.open() as src, cleaned_sam_path.open("w") as dst:
                for line in src:
                    if line.strip():
                        dst.write(line)
            clean_cmd[3] = str(cleaned_sam_path)
            clean_proc = subprocess.run(
                clean_cmd,
                capture_output=True,
                text=True,
                cwd=transcript_dir,
            )
            if clean_proc.returncode != 0:
                return BackendResult(
                    backend=self.backend_name,
                    status="error",
                    input_summary={"fasta_path": str(fasta_path)},
                    notes=[clean_proc.stderr.strip() or clean_proc.stdout.strip() or "outputClean failed"],
                )

            bed_path = (transcript_dir / f"{record.id}_probes.bed").resolve()
            if not bed_path.exists():
                alt_bed_path = (transcript_dir / f"{record.id}.bed").resolve()
                if alt_bed_path.exists():
                    bed_path = alt_bed_path
            candidate_count = int(blockparse_result.metrics.get("candidate_probe_count", 0))
            specific_count = 0
            if bed_path.exists():
                specific_count = sum(1 for _ in bed_path.open())

            total_candidates += candidate_count
            total_specific += specific_count
            per_transcript.append((record.id, candidate_count, specific_count))

        summary_path = output_dir / "oligominer_specificity_summary.tsv"
        with summary_path.open("w") as handle:
            handle.write("transcript_id\tcandidate_probe_count\tspecific_probe_count\n")
            for transcript_id, candidate_count, specific_count in per_transcript:
                handle.write(f"{transcript_id}\t{candidate_count}\t{specific_count}\n")

        return BackendResult(
            backend=self.backend_name,
            status="ok",
            input_summary={
                "fasta_path": str(fasta_path),
                "n_transcripts": len(per_transcript),
                "species": species,
                "reference_index": str(index_prefix),
                "temperature_c": int(temperature_c),
            },
            metrics={
                "candidate_probe_count_total": int(total_candidates),
                "specific_probe_count_total": int(total_specific),
            },
            output_files={"summary_tsv": str(summary_path)},
        )
