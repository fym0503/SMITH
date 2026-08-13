#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from smith_agent.adapters.smith_interface_backends import (
    run_oligominer_specificity_screen,
    run_probedealer_backend_screen,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMITH_INTERFACE_ROOT = Path("/workspace/fanyimin/smith_interface")
HUMAN_REFERENCE_DIR = SMITH_INTERFACE_ROOT / "third_party/reference_data/probedealer_human_full"
ODT_PYTHON = Path("/workspace/fanyimin/spapros_info/.venv312/bin/python")
SOURCE_GENE_METADATA_H5AD = PROJECT_ROOT / "data/cellxgene_liver_scrna/fe4bc7fc-0035-4ebb-919b-2d9097ec5dd4.h5ad"
DEFAULT_PANEL = (
    PROJECT_ROOT
    / "outputs/liver_merfish_benchmark/formal_multi_visium_smith_5seed_panels"
    / "seed_1/panels/multi_visium_top_128_panel.tsv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/agent_feasibility_filtering_top128"
MANUAL_HUMAN_GENE_IDS = {
    # PECAM1 is present in the MERFISH/Visium panel but absent from the source
    # CellxGene feature-name map used below.
    "PECAM1": "ENSG00000261371",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ODT/SCRINSHOT, OligoMiner and ProbeDealer on a human gene panel or h5ad var gene universe."
    )
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--gene-list", default="")
    parser.add_argument("--h5ad-var-names", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--species", default="homo_sapiens")
    parser.add_argument("--panel-size", type=int, default=0)
    parser.add_argument("--odt-batch-size", type=int, default=8)
    parser.add_argument("--odt-max-workers", type=int, default=4)
    parser.add_argument("--min-property-probes", type=int, default=20)
    parser.add_argument("--min-specific-probes", type=int, default=10)
    parser.add_argument("--min-deployment-probes", type=int, default=20)
    parser.add_argument("--label", default="multi_visium_top128")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_panel(panel_path: str | Path, panel_size: int) -> pd.DataFrame:
    df = pd.read_csv(panel_path, sep="\t")
    if "gene_symbol" not in df.columns:
        raise ValueError(f"Panel file must contain `gene_symbol`: {panel_path}")
    df = df.head(panel_size).copy()
    if "rank" not in df.columns:
        df.insert(0, "rank", range(1, len(df) + 1))
    return df[["rank", "gene_symbol"]].copy()


def read_gene_list(gene_list_path: str | Path, panel_size: int | None = None) -> pd.DataFrame:
    path = Path(gene_list_path)
    genes = [
        line.strip().split("\t")[0].split(",")[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if panel_size is not None and panel_size > 0:
        genes = genes[:panel_size]
    return pd.DataFrame({"rank": range(1, len(genes) + 1), "gene_symbol": genes})


def read_h5ad_var_names(h5ad_path: str | Path, panel_size: int | None = None) -> pd.DataFrame:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    genes = [str(item) for item in adata.var_names]
    adata.file.close()
    if panel_size is not None and panel_size > 0:
        genes = genes[:panel_size]
    return pd.DataFrame({"rank": range(1, len(genes) + 1), "gene_symbol": genes})


def load_transcript_to_gene(mapping_tsv: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with Path(mapping_tsv).open() as handle:
        next(handle)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def load_gene_symbol_to_gene_id(h5ad_path: str | Path) -> dict[str, str]:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    var = adata.var.copy()
    adata.file.close()
    if "feature_name" not in var.columns:
        return {}
    mapping: dict[str, str] = {}
    for gene_id, symbol in zip(var.index.astype(str), var["feature_name"].astype(str), strict=False):
        if gene_id.startswith("ENSG") and symbol:
            mapping.setdefault(symbol, gene_id)
    return mapping


def build_local_probe_candidate_manifest(
    output_dir: str | Path,
    panel: pd.DataFrame,
    species: str,
    reference_dir: str | Path = HUMAN_REFERENCE_DIR,
    gene_metadata_h5ad: str | Path = SOURCE_GENE_METADATA_H5AD,
) -> dict[str, str]:
    if species not in {"homo_sapiens", "human", "hsapiens"}:
        raise ValueError("Local reference manifest builder currently supports human panels only.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(reference_dir)
    transcriptome_fasta = reference_dir / "TxShortHeader.fa"
    transcript_to_gene_tsv = reference_dir / "transcript_to_gene.tsv"
    transcript_to_gene = load_transcript_to_gene(transcript_to_gene_tsv)
    symbol_to_gene_id = load_gene_symbol_to_gene_id(gene_metadata_h5ad)
    symbol_to_gene_id.update(MANUAL_HUMAN_GENE_IDS)

    panel_genes = panel["gene_symbol"].astype(str).tolist()
    failures: list[dict[str, str]] = []
    target_gene_ids: dict[str, str] = {}
    for gene in panel_genes:
        gene_id = symbol_to_gene_id.get(gene)
        if gene_id:
            target_gene_ids[gene] = gene_id
        else:
            failures.append({"gene_symbol": gene, "reason": "no ENSG mapping in local source metadata"})

    wanted_gene_ids = set(target_gene_ids.values())
    wanted_transcript_to_gene = {
        transcript_id: gene_id
        for transcript_id, gene_id in transcript_to_gene.items()
        if gene_id in wanted_gene_ids
    }
    wanted_transcript_ids = set(wanted_transcript_to_gene)
    best_by_gene_id: dict[str, tuple[str, str]] = {}
    for record in SeqIO.parse(str(transcriptome_fasta), "fasta"):
        transcript_id = record.id
        if transcript_id not in wanted_transcript_ids:
            continue
        gene_id = wanted_transcript_to_gene[transcript_id]
        sequence = str(record.seq).upper()
        prior = best_by_gene_id.get(gene_id)
        if prior is None or len(sequence) > len(prior[1]):
            best_by_gene_id[gene_id] = (transcript_id, sequence)

    records: list[dict[str, Any]] = []
    fasta_records: list[SeqRecord] = []
    for gene in panel_genes:
        gene_id = target_gene_ids.get(gene)
        if not gene_id:
            continue
        best = best_by_gene_id.get(gene_id)
        if best is None:
            failures.append({"gene_symbol": gene, "gene_id": gene_id, "reason": "no cDNA transcript in local reference"})
            continue
        transcript_id, sequence = best
        records.append(
            {
                "gene_symbol": gene,
                "gene_id": gene_id,
                "transcript_id": transcript_id,
                "sequence_length": len(sequence),
            }
        )
        fasta_records.append(SeqRecord(Seq(sequence), id=transcript_id, description=transcript_id))

    if not records:
        raise ValueError("No panel transcripts could be resolved from local human reference.")

    manifest_path = out_dir / "probe_candidate_manifest.tsv"
    pd.DataFrame(records).to_csv(manifest_path, sep="\t", index=False)

    transcript_fasta = out_dir / "probe_candidate_transcripts.fa"
    with transcript_fasta.open("w", encoding="utf-8") as handle:
        SeqIO.write(fasta_records, handle, "fasta")

    paintshop_fasta = out_dir / "paintshop_pseudogenome.fa"
    with paintshop_fasta.open("w", encoding="utf-8") as handle:
        SeqIO.write(fasta_records, handle, "fasta")

    paintshop_gtf = out_dir / "paintshop_pseudogenome.gtf"
    with paintshop_gtf.open("w", encoding="utf-8") as gtf_handle:
        for record, fasta_record in zip(records, fasta_records, strict=False):
            attrs = (
                f'gene_id "{record["gene_id"]}"; transcript_id "{record["transcript_id"]}"; '
                f'exon_number "1"; exon_id "{record["transcript_id"]}.1"; gene_name "{record["gene_symbol"]}";'
            )
            gtf_handle.write(
                f"{record['transcript_id']}\tcustom\texon\t1\t{len(fasta_record.seq)}\t.\t+\t.\t{attrs}\n"
            )

    failures_json = out_dir / "probe_candidate_failures.json"
    failures_json.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")

    return {
        "manifest_tsv": str(manifest_path),
        "transcript_fasta": str(transcript_fasta),
        "paintshop_fasta": str(paintshop_fasta),
        "paintshop_gtf": str(paintshop_gtf),
        "failures_json": str(failures_json),
    }


def write_local_odt_runner(runner_path: Path) -> None:
    runner_path.write_text(
        r'''
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import MeltingTemp as mt
from oligo_designer_toolsuite.pipelines._scrinshot_probe_designer import TargetProbeDesigner


SCRINSHOT_DEFAULTS = {
    "target_probe_Tm_parameters": {
        "check": True,
        "strict": True,
        "c_seq": None,
        "shift": 0,
        "nn_table": mt.DNA_NN3,
        "tmm_table": mt.DNA_TMM1,
        "imm_table": mt.DNA_IMM1,
        "de_table": mt.DNA_DE1,
        "dnac1": 50,
        "dnac2": 0,
        "selfcomp": False,
        "saltcorr": 7,
        "Na": 39,
        "K": 75,
        "Tris": 20,
        "Mg": 10,
        "dNTPs": 0,
    },
    "target_probe_Tm_chem_correction_parameters": {
        "DMSO": 0,
        "fmd": 20,
        "DMSOfactor": 0.75,
        "fmdfactor": 0.65,
        "fmdmethod": 1,
        "GC": None,
    },
    "target_probe_Tm_salt_correction_parameters": None,
}


def count_oligos_by_region(oligo_database):
    return {
        region_id: len(oligo_database.database[region_id])
        for region_id in oligo_database.get_regionid_list()
    }


def write_odt_fasta(batch_manifest, transcript_fasta, fasta_path):
    sequences = {record.id: str(record.seq).upper() for record in SeqIO.parse(str(transcript_fasta), "fasta")}
    with fasta_path.open("w") as handle:
        for row in batch_manifest.itertuples(index=False):
            sequence = sequences[str(row.transcript_id)]
            header = (
                f"{row.gene_symbol}::transcript_id={row.transcript_id}"
                f"::transcript:1-{len(sequence)}(+)"
            )
            handle.write(f">{header}\n")
            for i in range(0, len(sequence), 80):
                handle.write(sequence[i : i + 80] + "\n")


def run_batch(batch_idx, batch_manifest, transcript_fasta, output_dir, set_size_min):
    batch_dir = output_dir / f"batch_{batch_idx:02d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = batch_dir / "input_transcripts.fa"
    write_odt_fasta(batch_manifest, transcript_fasta, fasta_path)

    defaults = copy.deepcopy(SCRINSHOT_DEFAULTS)
    designer = TargetProbeDesigner(str(batch_dir), n_jobs=1)
    gene_ids = batch_manifest["gene_symbol"].astype(str).tolist()
    database = designer.create_oligo_database(
        gene_ids=gene_ids,
        oligo_length_min=40,
        oligo_length_max=45,
        files_fasta_oligo_database=[str(fasta_path)],
        min_oligos_per_gene=set_size_min,
        isoform_consensus=0,
    )
    counts_initial = count_oligos_by_region(database)
    database = designer.filter_by_property(
        oligo_database=database,
        GC_content_min=40,
        GC_content_max=60,
        Tm_min=65,
        Tm_max=75,
        detect_oligo_length_min=15,
        detect_oligo_length_max=40,
        min_thymines=2,
        arm_length_min=10,
        arm_Tm_dif_max=2,
        arm_Tm_min=50,
        arm_Tm_max=60,
        homopolymeric_base_n={"A": 5, "T": 5, "C": 5, "G": 5},
        Tm_parameters=defaults["target_probe_Tm_parameters"],
        Tm_chem_correction_parameters=defaults["target_probe_Tm_chem_correction_parameters"],
        Tm_salt_correction_parameters=defaults["target_probe_Tm_salt_correction_parameters"],
    )
    counts_property = count_oligos_by_region(database)

    rows = []
    for row in batch_manifest.itertuples(index=False):
        rows.append(
            {
                "gene": row.gene_symbol,
                "status": "ok",
                "reason": None,
                "transcript_id": row.transcript_id,
                "candidate_oligos_initial": int(counts_initial.get(row.gene_symbol, 0)),
                "candidate_oligos_after_property_filters": int(counts_property.get(row.gene_symbol, 0)),
                "feasible_property_only": int(counts_property.get(row.gene_symbol, 0)) >= set_size_min,
            }
        )
    batch_summary = batch_dir / "property_only_summary.tsv"
    pd.DataFrame(rows).to_csv(batch_summary, sep="\t", index=False)
    return batch_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--transcript-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--set-size-min", type=int, default=2)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    transcript_fasta = Path(args.transcript_fasta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batches = [
        manifest.iloc[i : i + args.batch_size].copy()
        for i in range(0, len(manifest), args.batch_size)
    ]
    paths = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_batch, idx, batch, transcript_fasta, output_dir, args.set_size_min): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            paths.append(future.result())
    merged = pd.concat([pd.read_csv(path, sep="\t") for path in paths], ignore_index=True)
    merged = merged.merge(manifest[["gene_symbol"]].assign(_order=range(len(manifest))), left_on="gene", right_on="gene_symbol", how="left")
    merged = merged.sort_values("_order").drop(columns=["gene_symbol", "_order"])
    merged.to_csv(output_dir / "property_only_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
'''.lstrip(),
        encoding="utf-8",
    )


def run_local_odt_property_batches(
    manifest_tsv: str | Path,
    transcript_fasta: str | Path,
    output_dir: str | Path,
    batch_size: int,
    max_workers: int,
    set_size_min: int = 2,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_path = output_dir / "run_local_odt_property.py"
    write_local_odt_runner(runner_path)
    cmd = [
        str(ODT_PYTHON),
        str(runner_path),
        "--manifest",
        str(Path(manifest_tsv).resolve()),
        "--transcript-fasta",
        str(Path(transcript_fasta).resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--batch-size",
        str(batch_size),
        "--max-workers",
        str(max_workers),
        "--set-size-min",
        str(set_size_min),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    summary_path = output_dir / "property_only_summary.tsv"
    if proc.returncode != 0 or not summary_path.exists():
        return {
            "backend": "odt_scrinshot_local_batches",
            "status": "error",
            "metrics": {},
            "output_files": {},
            "notes": [proc.stderr.strip() or proc.stdout.strip() or "local ODT property run failed"],
        }
    df = pd.read_csv(summary_path, sep="\t")
    return {
        "backend": "odt_scrinshot_local_batches",
        "status": "ok",
        "metrics": {
            "n_genes": int(len(df)),
            "feasible_property_only_count": int(df["feasible_property_only"].fillna(False).astype(bool).sum()),
        },
        "output_files": {"summary_tsv": str(summary_path)},
        "notes": [proc.stdout.strip()] if proc.stdout.strip() else [],
    }


def fasta_ids(fasta_path: str | Path) -> list[str]:
    path = Path(fasta_path)
    if not path.exists():
        return []
    return [record.id for record in SeqIO.parse(str(path), "fasta")]


def fastq_ids(fastq_path: str | Path) -> list[str]:
    path = Path(fastq_path)
    if not path.exists():
        return []
    ids: list[str] = []
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if idx % 4 == 0:
                ids.append(line.strip().lstrip("@").split()[0])
    return ids


def parse_two_column_hits(path: str | Path, transcript_to_gene: dict[str, str]) -> dict[str, set[str]]:
    hits: dict[str, set[str]] = {}
    path = Path(path)
    if not path.exists():
        return hits
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("@"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            query_id = parts[0]
            subject_id = parts[2]
            if subject_id == "*":
                continue
            gene_id = transcript_to_gene.get(subject_id)
            if gene_id:
                hits.setdefault(query_id, set()).add(gene_id)
    return hits


def parse_blast_hits(path: str | Path, transcript_to_gene: dict[str, str]) -> dict[str, set[str]]:
    hits: dict[str, set[str]] = {}
    path = Path(path)
    if not path.exists():
        return hits
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            query_id, subject_id = parts[:2]
            gene_id = transcript_to_gene.get(subject_id)
            if gene_id:
                hits.setdefault(query_id, set()).add(gene_id)
    return hits


def summarize_gene_aware_hits(
    query_ids: list[str],
    hits_by_query: dict[str, set[str]],
    target_gene_id: str,
) -> dict[str, int]:
    target_hit = 0
    cross_gene = 0
    off_target_only = 0
    no_hit = 0
    gene_aware_specific = 0

    for query_id in query_ids:
        hit_genes = hits_by_query.get(query_id, set())
        has_target = target_gene_id in hit_genes
        has_cross_gene = any(gene_id != target_gene_id for gene_id in hit_genes)
        if has_target:
            target_hit += 1
        if has_cross_gene:
            cross_gene += 1
        if hit_genes and not has_target:
            off_target_only += 1
        if not hit_genes:
            no_hit += 1
        if has_target and not has_cross_gene:
            gene_aware_specific += 1

    return {
        "target_gene_hit_probe_count": target_hit,
        "cross_gene_probe_count": cross_gene,
        "off_target_only_probe_count": off_target_only,
        "no_hit_probe_count": no_hit,
        "gene_aware_specific_probe_count": gene_aware_specific,
    }


def build_oligominer_gene_aware_summary(
    manifest: pd.DataFrame,
    oligominer_dir: str | Path,
    transcript_to_gene: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    oligominer_dir = Path(oligominer_dir)
    for row in manifest.itertuples(index=False):
        transcript_id = str(row.transcript_id)
        transcript_dir = oligominer_dir / transcript_id
        query_ids = fastq_ids(transcript_dir / f"{transcript_id}.fastq")
        hits = parse_two_column_hits(transcript_dir / f"{transcript_id}.sam", transcript_to_gene)
        summary = summarize_gene_aware_hits(query_ids, hits, str(row.gene_id))
        rows.append({"transcript_id": transcript_id, **summary})
    return pd.DataFrame(rows)


def build_probedealer_target_summary(
    manifest: pd.DataFrame,
    probedealer_dir: str | Path,
    transcript_to_gene: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    probedealer_dir = Path(probedealer_dir)
    for row in manifest.itertuples(index=False):
        transcript_id = str(row.transcript_id)
        blast_dir = probedealer_dir / f"blast_{transcript_id}"
        query_ids = fasta_ids(blast_dir / "probe_queries.fa")
        hits = parse_blast_hits(blast_dir / "blast_hits.tsv", transcript_to_gene)
        summary = summarize_gene_aware_hits(query_ids, hits, str(row.gene_id))
        rows.append({"transcript_id": transcript_id, **summary})
    return pd.DataFrame(rows)


def timed_step(name: str, run: bool, fn) -> dict[str, Any]:
    if not run:
        return {"status": "skipped_existing", "seconds": 0.0}
    start = time.time()
    result = fn()
    elapsed = round(time.time() - start, 3)
    if isinstance(result, dict):
        return {"status": result.get("status", "ok"), "seconds": elapsed, "result": result}
    return {"status": "ok", "seconds": elapsed, "result": result}


def failure_reason(row: Any) -> str:
    reasons: list[str] = []
    if not bool(row.transcript_resolved):
        reasons.append("transcript_unresolved")
    if not bool(row.pass_odt_property_20):
        reasons.append("low_odt_property_probe_count")
    if not bool(row.pass_oligominer_geneaware_10):
        reasons.append("low_oligominer_geneaware_specificity")
    if not bool(row.pass_probedealer_target_20):
        reasons.append("low_probedealer_target_probe_count")
    return ";".join(reasons) if reasons else "pass"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_size = args.panel_size if args.panel_size and args.panel_size > 0 else None
    if args.h5ad_var_names:
        panel = read_h5ad_var_names(args.h5ad_var_names, panel_size=panel_size)
        input_path = str(Path(args.h5ad_var_names).resolve())
        input_type = "h5ad_var_names"
    elif args.gene_list:
        panel = read_gene_list(args.gene_list, panel_size=panel_size)
        input_path = str(Path(args.gene_list).resolve())
        input_type = "gene_list"
    else:
        panel = read_panel(args.panel, panel_size or 128)
        input_path = str(Path(args.panel).resolve())
        input_type = "panel"
    genes = panel["gene_symbol"].astype(str).tolist()
    (out_dir / "selected_genes.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
    panel.to_csv(out_dir / "selected_panel.tsv", sep="\t", index=False)

    probe_dir = out_dir / "probe_candidates"
    manifest_tsv = probe_dir / "probe_candidate_manifest.tsv"
    transcript_fasta = probe_dir / "probe_candidate_transcripts.fa"
    steps: dict[str, Any] = {}
    steps["build_probe_candidate_manifest"] = timed_step(
        "build_probe_candidate_manifest",
        args.force or not (manifest_tsv.exists() and transcript_fasta.exists()),
        lambda: build_local_probe_candidate_manifest(
            output_dir=probe_dir,
            species=args.species,
            panel=panel,
        ),
    )

    odt_dir = out_dir / "odt_property_batches"
    odt_summary = odt_dir / "property_only_summary.tsv"
    steps["run_odt_property_batches"] = timed_step(
        "run_odt_property_batches",
        args.force or not odt_summary.exists(),
        lambda: run_local_odt_property_batches(
            manifest_tsv=manifest_tsv,
            transcript_fasta=transcript_fasta,
            output_dir=odt_dir,
            batch_size=args.odt_batch_size,
            max_workers=args.odt_max_workers,
            set_size_min=2,
        ),
    )

    oligominer_dir = out_dir / "oligominer"
    oligominer_summary = oligominer_dir / "oligominer_specificity_summary.tsv"
    steps["run_oligominer_specificity_screen"] = timed_step(
        "run_oligominer_specificity_screen",
        args.force or not oligominer_summary.exists(),
        lambda: run_oligominer_specificity_screen(
            smith_interface_root=SMITH_INTERFACE_ROOT,
            transcript_fasta=transcript_fasta,
            output_dir=oligominer_dir,
            temperature_c=42,
            species=args.species,
        ),
    )

    probedealer_dir = out_dir / "probedealer_backend"
    probedealer_summary = probedealer_dir / "probedealer_summary.tsv"
    steps["run_probedealer_backend_screen"] = timed_step(
        "run_probedealer_backend_screen",
        args.force or not probedealer_summary.exists(),
        lambda: run_probedealer_backend_screen(
            smith_interface_root=SMITH_INTERFACE_ROOT,
            transcript_fasta=transcript_fasta,
            output_dir=probedealer_dir,
            use_transcriptome_reference=True,
            species=args.species,
        ),
    )

    manifest = pd.read_csv(manifest_tsv, sep="\t")
    odt = pd.read_csv(odt_summary, sep="\t").rename(
        columns={
            "gene": "gene_symbol",
            "candidate_oligos_initial": "odt_initial_probe_count",
            "candidate_oligos_after_property_filters": "odt_property_probe_count",
            "feasible_property_only": "odt_feasible_property_only",
        }
    )
    oligominer = pd.read_csv(oligominer_summary, sep="\t").rename(
        columns={
            "candidate_probe_count": "oligominer_candidate_probe_count",
            "specific_probe_count": "oligominer_strict_specific_probe_count",
        }
    )
    probedealer = pd.read_csv(probedealer_summary, sep="\t").rename(
        columns={
            "initial_probe_count": "probedealer_initial_probe_count",
            "final_probe_count": "probedealer_backend_final_probe_count",
        }
    )
    transcript_to_gene = load_transcript_to_gene(HUMAN_REFERENCE_DIR / "transcript_to_gene.tsv")
    oligominer_gene_aware = build_oligominer_gene_aware_summary(manifest, oligominer_dir, transcript_to_gene)
    probedealer_target = build_probedealer_target_summary(manifest, probedealer_dir, transcript_to_gene)
    oligominer_gene_aware = oligominer_gene_aware.rename(
        columns={
            "target_gene_hit_probe_count": "oligominer_target_gene_hit_probe_count",
            "cross_gene_probe_count": "oligominer_cross_gene_probe_count",
            "off_target_only_probe_count": "oligominer_off_target_only_probe_count",
            "no_hit_probe_count": "oligominer_no_hit_probe_count",
            "gene_aware_specific_probe_count": "oligominer_geneaware_specific_probe_count",
        }
    )
    probedealer_target = probedealer_target.rename(
        columns={
            "target_gene_hit_probe_count": "probedealer_target_gene_hit_probe_count",
            "cross_gene_probe_count": "probedealer_cross_gene_probe_count",
            "off_target_only_probe_count": "probedealer_off_target_only_probe_count",
            "no_hit_probe_count": "probedealer_no_hit_probe_count",
            "gene_aware_specific_probe_count": "probedealer_target_final_probe_count",
        }
    )

    table = panel.merge(manifest, on="gene_symbol", how="left")
    table["transcript_resolved"] = table["transcript_id"].notna()
    table = table.merge(
        odt[["gene_symbol", "status", "reason", "transcript_id", "odt_initial_probe_count", "odt_property_probe_count", "odt_feasible_property_only"]].rename(
            columns={
                "status": "odt_status",
                "reason": "odt_reason",
                "transcript_id": "odt_transcript_id",
            }
        ),
        on="gene_symbol",
        how="left",
    )
    table = table.merge(oligominer, on="transcript_id", how="left")
    table = table.merge(oligominer_gene_aware, on="transcript_id", how="left")
    table = table.merge(probedealer, on="transcript_id", how="left")
    table = table.merge(probedealer_target, on="transcript_id", how="left")

    numeric_cols = [
        "odt_initial_probe_count",
        "odt_property_probe_count",
        "oligominer_candidate_probe_count",
        "oligominer_strict_specific_probe_count",
        "oligominer_target_gene_hit_probe_count",
        "oligominer_cross_gene_probe_count",
        "oligominer_off_target_only_probe_count",
        "oligominer_no_hit_probe_count",
        "oligominer_geneaware_specific_probe_count",
        "probedealer_initial_probe_count",
        "probedealer_backend_final_probe_count",
        "probedealer_target_gene_hit_probe_count",
        "probedealer_cross_gene_probe_count",
        "probedealer_off_target_only_probe_count",
        "probedealer_no_hit_probe_count",
        "probedealer_target_final_probe_count",
    ]
    for col in numeric_cols:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0).astype(int)

    table["pass_odt_property_20"] = table["odt_property_probe_count"] >= args.min_property_probes
    table["pass_oligominer_strict_10"] = table["oligominer_strict_specific_probe_count"] >= args.min_specific_probes
    table["pass_oligominer_geneaware_10"] = table["oligominer_geneaware_specific_probe_count"] >= args.min_specific_probes
    table["pass_probedealer_backend_final_20"] = table["probedealer_backend_final_probe_count"] >= args.min_deployment_probes
    table["pass_probedealer_target_20"] = table["probedealer_target_final_probe_count"] >= args.min_deployment_probes
    table["pass_three_tool_feasibility"] = (
        table["transcript_resolved"]
        & table["pass_odt_property_20"]
        & table["pass_oligominer_geneaware_10"]
        & table["pass_probedealer_target_20"]
    )
    table["primary_failure_reason"] = [failure_reason(row) for row in table.itertuples(index=False)]

    table_path = out_dir / "three_tool_feasibility_table.tsv"
    table.to_csv(table_path, sep="\t", index=False)

    overlap_cols = ["pass_odt_property_20", "pass_oligominer_geneaware_10", "pass_probedealer_target_20"]
    overlap = (
        table.groupby(overlap_cols, dropna=False)
        .size()
        .reset_index(name="gene_count")
        .sort_values("gene_count", ascending=False)
    )
    overlap["tool_pattern"] = overlap.apply(
        lambda row: "+".join(
            name
            for name, passed in [
                ("ODT", row["pass_odt_property_20"]),
                ("OligoMiner", row["pass_oligominer_geneaware_10"]),
                ("ProbeDealer", row["pass_probedealer_target_20"]),
            ]
            if bool(passed)
        )
        or "none",
        axis=1,
    )
    overlap_path = out_dir / "three_tool_overlap_counts.tsv"
    overlap.to_csv(overlap_path, sep="\t", index=False)

    pass_summary = pd.DataFrame(
        [
            {"gate": "transcript_resolved", "pass_count": int(table["transcript_resolved"].sum()), "total_count": int(len(table))},
            {"gate": "ODT_property_ge20", "pass_count": int(table["pass_odt_property_20"].sum()), "total_count": int(len(table))},
            {
                "gate": "OligoMiner_strict_specific_ge10",
                "pass_count": int(table["pass_oligominer_strict_10"].sum()),
                "total_count": int(len(table)),
            },
            {
                "gate": "OligoMiner_geneaware_specific_ge10",
                "pass_count": int(table["pass_oligominer_geneaware_10"].sum()),
                "total_count": int(len(table)),
            },
            {
                "gate": "ProbeDealer_backend_final_ge20",
                "pass_count": int(table["pass_probedealer_backend_final_20"].sum()),
                "total_count": int(len(table)),
            },
            {
                "gate": "ProbeDealer_target_final_ge20",
                "pass_count": int(table["pass_probedealer_target_20"].sum()),
                "total_count": int(len(table)),
            },
            {
                "gate": "three_tool_feasibility",
                "pass_count": int(table["pass_three_tool_feasibility"].sum()),
                "total_count": int(len(table)),
            },
        ]
    )
    pass_summary_path = out_dir / "tool_pass_summary.tsv"
    pass_summary.to_csv(pass_summary_path, sep="\t", index=False)

    run_summary = {
        "input": input_path,
        "input_type": input_type,
        "label": args.label,
        "species": args.species,
        "n_panel_genes": int(len(table)),
        "thresholds": {
            "min_property_probes": args.min_property_probes,
            "min_specific_probes": args.min_specific_probes,
            "min_deployment_probes": args.min_deployment_probes,
        },
        "steps": steps,
        "outputs": {
            "selected_panel_tsv": str(out_dir / "selected_panel.tsv"),
            "manifest_tsv": str(manifest_tsv),
            "odt_summary_tsv": str(odt_summary),
            "oligominer_summary_tsv": str(oligominer_summary),
            "probedealer_summary_tsv": str(probedealer_summary),
            "three_tool_feasibility_table_tsv": str(table_path),
            "three_tool_overlap_counts_tsv": str(overlap_path),
            "tool_pass_summary_tsv": str(pass_summary_path),
        },
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "n_panel_genes": int(len(table)),
        "three_tool_pass": int(table["pass_three_tool_feasibility"].sum()),
        "outputs": run_summary["outputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
