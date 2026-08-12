from __future__ import annotations

import os
import subprocess
from pathlib import Path

from smith_agent.schemas import BackendResult


ROOT_DIR = Path(os.environ.get("SMITH_PACKAGE_ROOT", Path(__file__).resolve().parents[4]))
PAINTSHOP_SNAKEMAKE = Path(os.environ.get("SMITH_PAINTSHOP_SNAKEMAKE", ROOT_DIR / "third_party/envs/paintshop_pipeline/bin/snakemake"))
PAINTSHOP_EXAMPLE_DIR = ROOT_DIR / "third_party/PaintSHOP_pipeline/example_run"
PAINTSHOP_SNAKEFILE = ROOT_DIR / "third_party/PaintSHOP_pipeline/workflow/Snakefile"
PAINTSHOP_CONDA_PREFIX = ROOT_DIR / "third_party/PaintSHOP_pipeline/shared_conda_envs"


class PaintSHOPBackend:
    backend_name = "paintshop"

    def run_example(self, cores: int = 1) -> BackendResult:
        return self._run_with_config(
            work_dir=PAINTSHOP_EXAMPLE_DIR,
            config_path=PAINTSHOP_EXAMPLE_DIR / "config.yml",
            input_summary={
                "genome_fasta": str(PAINTSHOP_EXAMPLE_DIR / "data/example.fa"),
                "annotation_file": str(PAINTSHOP_EXAMPLE_DIR / "data/example.gtf"),
            },
            cores=cores,
        )

    def run_custom(
        self,
        assembly: str,
        genome_fasta: str | Path,
        annotation_file: str | Path,
        work_dir: str | Path,
        cores: int = 1,
    ) -> BackendResult:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config_path = work_dir / "config.yml"
        config_path.write_text(
            "\n".join(
                [
                    f"assembly: '{assembly}'",
                    f"genome_fasta: '{Path(genome_fasta).resolve()}'",
                    f"annotation_file: '{Path(annotation_file).resolve()}'",
                    "",
                ]
            )
        )
        return self._run_with_config(
            work_dir=work_dir,
            config_path=config_path,
            input_summary={
                "genome_fasta": str(Path(genome_fasta).resolve()),
                "annotation_file": str(Path(annotation_file).resolve()),
            },
            cores=cores,
        )

    def _run_with_config(
        self,
        work_dir: Path,
        config_path: Path,
        input_summary: dict[str, str],
        cores: int,
    ) -> BackendResult:
        cmd = [
            str(PAINTSHOP_SNAKEMAKE),
            "--configfile",
            str(config_path.resolve()),
            "--snakefile",
            str(PAINTSHOP_SNAKEFILE),
            "--use-conda",
            "--conda-prefix",
            str(PAINTSHOP_CONDA_PREFIX),
            "--cores",
            str(cores),
            "--restart-times",
            "1",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
        output_dir = work_dir / "pipeline_output"
        report_path = output_dir / "report.html"

        if proc.returncode != 0:
            return BackendResult(
                backend=self.backend_name,
                status="error",
                input_summary=input_summary,
                notes=[proc.stderr.strip() or proc.stdout.strip() or "PaintSHOP example failed."],
            )

        n_outputs = len(list(output_dir.glob("**/*")))
        return BackendResult(
            backend=self.backend_name,
            status="ok",
            input_summary=input_summary,
            metrics={"output_path_count": int(n_outputs)},
            output_files={"pipeline_output_dir": str(output_dir), "report_html": str(report_path)},
            notes=[proc.stdout.strip()] if proc.stdout.strip() else [],
        )
