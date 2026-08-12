from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils import MeltingTemp as mt


@dataclass(frozen=True)
class OligoDesignConfig:
    probe_length: int = 30
    min_tm: float = 66.0
    max_tm: float = 100.0
    secondary_structure_tm: float = 76.0
    cross_hyb_tm: float = 72.0
    min_gc: float = 30.0
    max_gc: float = 90.0
    exclude_seq: str = r"GGGGGG|CCCCCC|TTTTTT|AAAAAA"
    # Upstream spreadsheet says 31, but empirically the MATLAB package output
    # aligns much more closely with the published supplementary counts when the
    # effective gap is treated as 32.
    probe_gap: int = 32
    salt_molar: float = 1.0
    primer_conc_molar: float = 1e-6
    enable_secondary_structure_filter: bool = False
    enable_cross_hyb_filter: bool = False


@dataclass(frozen=True)
class ProbeCandidate:
    sequence_id: str
    probe_index: int
    start_1based: int
    sequence: str

    @property
    def frag_id(self) -> str:
        return f"{self.sequence_id}_Seq_{self.probe_index}"

    @property
    def header(self) -> str:
        return f"{self.probe_index}_{self.frag_id}"


def load_fasta_records(fasta_path: str | Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        records.append((record.id, str(record.seq).upper()))
    return records


def _tm_nn(seq: str, cfg: OligoDesignConfig) -> float:
    if len(seq) < 2:
        return float("-inf")
    return float(
        mt.Tm_NN(
            seq,
            nn_table=mt.DNA_NN3,
            dnac1=cfg.primer_conc_molar * 1e9,
            dnac2=cfg.primer_conc_molar * 1e9,
            Na=cfg.salt_molar * 1000,
        )
    )


def _gc_percent(seq: str) -> float:
    return float(gc_fraction(seq) * 100.0)


def _reverse_complement(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def _matched_bases_from_local_alignment(seq_a: str, seq_b: str) -> str:
    from Bio import pairwise2

    alignments = pairwise2.align.localms(
        seq_a,
        seq_b,
        2.0,
        -1.0,
        -2.0,
        -0.5,
        one_alignment_only=True,
    )
    if not alignments:
        return ""
    aligned_a, aligned_b, _, _, _ = alignments[0]
    return "".join(
        base_a
        for base_a, base_b in zip(aligned_a, aligned_b)
        if base_a != "-" and base_b != "-" and base_a == base_b
    )


def _secondary_structure_tm(seq: str, cfg: OligoDesignConfig) -> float:
    try:
        import RNA  # type: ignore
    except ImportError:
        RNA = None

    if RNA is not None:
        structure, _mfe = RNA.fold(seq)
        stem_bases = "".join(base for base, bracket in zip(seq, structure) if bracket == "(")
    else:
        # MATLAB uses rnafold plus oligoprop on stem bases. Fall back to a rough
        # local-alignment proxy when ViennaRNA is unavailable.
        stem_bases = _matched_bases_from_local_alignment(seq, _reverse_complement(seq))

    return _tm_nn(stem_bases, cfg)


def _cross_hyb_tm(seq: str, accepted: Iterable[ProbeCandidate], cfg: OligoDesignConfig) -> float:
    rev = _reverse_complement(seq)
    max_tm = float("-inf")
    for probe in accepted:
        matched = _matched_bases_from_local_alignment(probe.sequence, rev)
        max_tm = max(max_tm, _tm_nn(matched, cfg))
    return max_tm


def build_oligo_array(
    records: Iterable[tuple[str, str]],
    cfg: OligoDesignConfig | None = None,
) -> dict[str, list[ProbeCandidate]]:
    cfg = cfg or OligoDesignConfig()
    exclude_re = re.compile(cfg.exclude_seq, flags=re.IGNORECASE)
    accepted_by_id: dict[str, list[ProbeCandidate]] = {}

    for sequence_id, sequence in records:
        accepted: list[ProbeCandidate] = []
        start0 = 0
        probe_index = 0
        seq = sequence.upper()

        while start0 + cfg.probe_length <= len(seq):
            oligo = seq[start0 : start0 + cfg.probe_length]

            tm = _tm_nn(oligo, cfg)
            if tm < cfg.min_tm or tm > cfg.max_tm:
                start0 += 1
                continue

            gc_pct = _gc_percent(oligo)
            if gc_pct < cfg.min_gc or gc_pct > cfg.max_gc:
                start0 += 1
                continue

            if exclude_re.search(oligo) or "N" in oligo:
                start0 += 1
                continue

            if cfg.enable_secondary_structure_filter and cfg.secondary_structure_tm < 999:
                if _secondary_structure_tm(oligo, cfg) > cfg.secondary_structure_tm:
                    start0 += 1
                    continue

            if accepted and cfg.enable_cross_hyb_filter and cfg.cross_hyb_tm < 999:
                if _cross_hyb_tm(oligo, accepted, cfg) > cfg.cross_hyb_tm:
                    start0 += 1
                    continue

            probe_index += 1
            accepted.append(
                ProbeCandidate(
                    sequence_id=sequence_id,
                    probe_index=probe_index,
                    start_1based=start0 + 1,
                    sequence=oligo,
                )
            )
            start0 += cfg.probe_gap - 1

        accepted_by_id[sequence_id] = accepted

    return accepted_by_id
