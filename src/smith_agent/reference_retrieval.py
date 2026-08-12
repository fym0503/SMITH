from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

from smith_agent.registry import DatasetRegistryEntry, RegistryBundle
from smith_agent.session import AgentSession
from smith_agent.utils import ensure_dir, write_json


GENE_COLUMN_HINTS = (
    "gene",
    "symbol",
    "feature_name",
    "gene_name",
    "gene_short_name",
    "gene_symbol",
    "ensembl",
    "id",
)

ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "cell_type": ("celltype", "cell_type", "cell.type", "cell type", "annotation", "labels", "cluster", "level_"),
    "region": ("region", "domain", "spatial", "leiden", "parcellation", "area"),
    "pathology": ("pathology", "disease", "condition", "diagnosis", "phenotype"),
    "time": ("time", "stage", "development", "embryo"),
    "sample": ("sample", "donor", "patient", "replicate", "dataset", "batch"),
}

TASK_TO_ROLES: dict[str, tuple[str, ...]] = {
    "cls": ("cell_type",),
    "classification": ("cell_type",),
    "cell_type": ("cell_type",),
    "celltype": ("cell_type",),
    "region": ("region",),
    "spatial": ("region",),
    "pathology": ("pathology",),
    "disease": ("pathology",),
    "regression": ("time",),
    "time": ("time",),
    "temporal": ("time",),
    "recon": (),
    "reconstruction": (),
    "transfer": (),
    "panel_selection": (),
}

COMPATIBLE_MODALITY_GROUPS = (
    {"spatial", "st", "merfish", "starmap", "ribomap", "slideseq", "slide-seq", "exseq", "codex"},
    {"scrna-seq", "scrna", "single-cell", "single cell", "single_cell", "rna-seq"},
)

DEFAULT_SCORE_WEIGHTS = {
    "species_score": 20.0,
    "tissue_score": 15.0,
    "modality_score": 15.0,
    "gene_overlap_score": 25.0,
    "label_compatibility_score": 15.0,
    "disease_context_score": 5.0,
    "spatial_support_score": 5.0,
}


@dataclass
class DatasetProfile:
    dataset_id: str = ""
    dataset_path: str = ""
    description: str = ""
    species: str = ""
    tissue: str = ""
    modality: str = ""
    tasks: list[str] = field(default_factory=list)
    n_obs: int = 0
    n_vars: int = 0
    obs_columns: list[str] = field(default_factory=list)
    var_columns: list[str] = field(default_factory=list)
    obsm_keys: list[str] = field(default_factory=list)
    uns_keys: list[str] = field(default_factory=list)
    gene_symbols: set[str] = field(default_factory=set)
    label_roles: dict[str, list[str]] = field(default_factory=dict)
    label_cardinality: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_jsonable(self, include_genes: bool = False) -> dict[str, Any]:
        payload = {
            "dataset_id": self.dataset_id,
            "dataset_path": self.dataset_path,
            "description": self.description,
            "species": self.species,
            "tissue": self.tissue,
            "modality": self.modality,
            "tasks": self.tasks,
            "n_obs": self.n_obs,
            "n_vars": self.n_vars,
            "obs_columns": self.obs_columns,
            "var_columns": self.var_columns,
            "obsm_keys": self.obsm_keys,
            "uns_keys": self.uns_keys,
            "n_gene_symbols": len(self.gene_symbols),
            "label_roles": self.label_roles,
            "label_cardinality": self.label_cardinality,
            "warnings": self.warnings,
        }
        if include_genes:
            payload["gene_symbols"] = sorted(self.gene_symbols)
        return payload


def _normalize_token(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-").replace(" ", "-")


def _metadata_from_path(path: str | Path) -> dict[str, str]:
    text = str(path).lower()
    inferred = {"species": "", "tissue": "", "modality": ""}
    if any(token in text for token in ["human", "htapp", "sc_dataset_processed", "st_dataset_processed"]):
        inferred["species"] = "homo_sapiens"
    elif "elegans" in text:
        inferred["species"] = "caenorhabditis_elegans"
    elif any(token in text for token in ["mouse", "ribomap", "starmap", "mus_"]):
        inferred["species"] = "mus_musculus"

    if any(token in text for token in ["brain", "ad_", "neuro", "ribomap", "starmap"]):
        inferred["tissue"] = "brain"
    elif "elegans" in text:
        inferred["tissue"] = "embryo"
    elif "htapp" in text:
        inferred["tissue"] = "tumor"

    if any(token in text for token in ["scrna", "single_cell", "single-cell", "sc_dataset"]):
        inferred["modality"] = "scRNA-seq"
    elif "ribomap" in text:
        inferred["modality"] = "RIBOMap"
    elif "starmap" in text:
        inferred["modality"] = "STARmap"
    elif "merfish" in text:
        inferred["modality"] = "MERFISH"
    elif "codex" in text:
        inferred["modality"] = "CODEX"
    elif "exseq" in text:
        inferred["modality"] = "ExSeq"
    elif "slide_seq" in text or "slideseq" in text:
        inferred["modality"] = "Slide-seq"
    elif any(token in text for token in ["st_dataset", "spatial"]):
        inferred["modality"] = "spatial"
    elif "tf" in text:
        inferred["modality"] = "TF activity"
    elif "mirna" in text:
        inferred["modality"] = "miRNA activity"
    return inferred


def _metadata_from_entry(entry: DatasetRegistryEntry | None, path: str | Path) -> dict[str, Any]:
    inferred = _metadata_from_path(path)
    if entry is None:
        return {**inferred, "description": "", "tasks": []}
    return {
        "species": entry.species or inferred["species"],
        "tissue": entry.tissue or inferred["tissue"],
        "modality": entry.modality or inferred["modality"],
        "description": entry.description,
        "tasks": list(entry.tasks),
    }


def _clean_gene_values(values: Any) -> set[str]:
    series = pd.Series(values).dropna().astype(str).str.strip()
    if series.empty:
        return set()
    series = series[series.ne("")]
    if series.empty:
        return set()
    numeric_fraction = pd.to_numeric(series, errors="coerce").notna().mean()
    if numeric_fraction > 0.9:
        return set()
    return {item.upper() for item in series if item and item.lower() not in {"nan", "none"}}


def _extract_gene_symbols(adata: ad.AnnData) -> set[str]:
    genes = set()
    genes.update(_clean_gene_values(adata.var_names))
    for column in adata.var.columns:
        lowered = str(column).lower()
        if any(hint in lowered for hint in GENE_COLUMN_HINTS):
            genes.update(_clean_gene_values(adata.var[column]))
    return genes


def _detect_label_roles(obs_columns: list[str]) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    for column in obs_columns:
        normalized = str(column).lower().replace("_", " ")
        compact = str(column).lower()
        for role, patterns in ROLE_PATTERNS.items():
            if any(pattern in normalized or pattern in compact for pattern in patterns):
                roles.setdefault(role, []).append(str(column))
    return roles


def _label_cardinality(adata: ad.AnnData, label_roles: dict[str, list[str]]) -> dict[str, int]:
    cardinality: dict[str, int] = {}
    for columns in label_roles.values():
        for column in columns[:3]:
            if column in cardinality or column not in adata.obs:
                continue
            try:
                cardinality[column] = int(adata.obs[column].nunique(dropna=True))
            except Exception:  # noqa: BLE001
                continue
    return cardinality


def profile_dataset(
    dataset_path: str | Path,
    dataset_id: str = "",
    entry: DatasetRegistryEntry | None = None,
) -> DatasetProfile:
    path = Path(dataset_path)
    metadata = _metadata_from_entry(entry, path)
    profile = DatasetProfile(
        dataset_id=dataset_id,
        dataset_path=str(path),
        description=str(metadata.get("description", "")),
        species=str(metadata.get("species", "")),
        tissue=str(metadata.get("tissue", "")),
        modality=str(metadata.get("modality", "")),
        tasks=[str(item) for item in metadata.get("tasks", [])],
    )
    try:
        adata = ad.read_h5ad(path, backed="r")
    except Exception as exc:  # noqa: BLE001
        profile.warnings.append(f"failed_to_read_h5ad: {exc}")
        return profile

    try:
        profile.n_obs = int(adata.n_obs)
        profile.n_vars = int(adata.n_vars)
        profile.obs_columns = [str(item) for item in adata.obs.columns]
        profile.var_columns = [str(item) for item in adata.var.columns]
        profile.obsm_keys = [str(item) for item in adata.obsm.keys()]
        profile.uns_keys = [str(item) for item in adata.uns.keys()]
        profile.gene_symbols = _extract_gene_symbols(adata)
        profile.label_roles = _detect_label_roles(profile.obs_columns)
        profile.label_cardinality = _label_cardinality(adata, profile.label_roles)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    return profile


def _resolve_query_profile(
    registries: RegistryBundle,
    session: AgentSession,
    request: dict[str, Any],
) -> DatasetProfile:
    dataset_id = str(request.get("dataset_id", "") or "").strip() or session.active_dataset_id
    dataset_path = str(request.get("dataset_path", "") or "").strip()
    entry = None
    resolved_id = registries.resolve_dataset_id(dataset_id)
    if resolved_id:
        entry = registries.datasets[resolved_id]
        dataset_path = dataset_path or entry.path
        dataset_id = resolved_id
    elif dataset_id in session.state.get("mounted_datasets", {}):
        dataset_path = dataset_path or str(session.state["mounted_datasets"][dataset_id]["path"])
    if not dataset_path:
        raise ValueError("score_reference_transferability requires a dataset_path, dataset_id, or active dataset.")

    profile = profile_dataset(dataset_path, dataset_id=dataset_id, entry=entry)
    profile.species = str(request.get("species") or profile.species)
    profile.tissue = str(request.get("tissue") or profile.tissue)
    profile.modality = str(request.get("modality") or profile.modality)
    request_tasks = [str(item) for item in request.get("tasks", []) if str(item).strip()]
    if request_tasks:
        profile.tasks = request_tasks
    return profile


def _requested_roles(request: dict[str, Any], query_profile: DatasetProfile) -> set[str]:
    roles: set[str] = set()
    for task in request.get("tasks", []) or query_profile.tasks:
        roles.update(TASK_TO_ROLES.get(str(task).lower(), ()))
    objective = str(request.get("objective", "")).lower()
    for token, mapped_roles in TASK_TO_ROLES.items():
        if token in objective:
            roles.update(mapped_roles)
    if not roles:
        roles.update(role for role in ("cell_type", "pathology", "region", "time") if query_profile.label_roles.get(role))
    return roles


def _same_or_empty(a: str, b: str) -> bool:
    return bool(a and b and _normalize_token(a) == _normalize_token(b))


def _contains_match(a: str, b: str) -> bool:
    left = _normalize_token(a)
    right = _normalize_token(b)
    return bool(left and right and (left in right or right in left))


def _species_score(query: DatasetProfile, candidate: DatasetProfile) -> float:
    if not query.species:
        return 0.5
    return 1.0 if _same_or_empty(query.species, candidate.species) else 0.0


def _tissue_score(query: DatasetProfile, candidate: DatasetProfile) -> float:
    if not query.tissue:
        return 0.5
    if _same_or_empty(query.tissue, candidate.tissue):
        return 1.0
    return 0.5 if _contains_match(query.tissue, candidate.tissue) else 0.0


def _modality_group(modality: str) -> set[str]:
    normalized = _normalize_token(modality)
    for group in COMPATIBLE_MODALITY_GROUPS:
        if normalized in group or any(item in normalized for item in group):
            return group
    return {normalized} if normalized else set()


def _modality_score(query: DatasetProfile, candidate: DatasetProfile) -> float:
    if not query.modality:
        return 0.5
    if _same_or_empty(query.modality, candidate.modality):
        return 1.0
    query_group = _modality_group(query.modality)
    candidate_group = _modality_group(candidate.modality)
    if query_group and candidate_group and query_group == candidate_group:
        return 0.8
    if query_group and candidate_group:
        return 0.6
    return 0.3 if candidate.modality else 0.0


def _gene_overlap(query: DatasetProfile, candidate: DatasetProfile) -> tuple[int, float, float]:
    if not query.gene_symbols or not candidate.gene_symbols:
        return 0, 0.0, 0.0
    raw_overlap = len(query.gene_symbols.intersection(candidate.gene_symbols))
    # AnnData objects often expose both gene symbols and Ensembl IDs. Cap the
    # overlap by feature counts so coverage remains interpretable per target.
    overlap = min(raw_overlap, max(0, query.n_vars), max(0, candidate.n_vars))
    query_coverage = overlap / max(1, query.n_vars or len(query.gene_symbols))
    union_features = max(1, (query.n_vars or len(query.gene_symbols)) + (candidate.n_vars or len(candidate.gene_symbols)) - overlap)
    jaccard = overlap / union_features
    return overlap, query_coverage, jaccard


def _label_score(query_roles: set[str], query: DatasetProfile, candidate: DatasetProfile) -> tuple[float, list[str], list[str]]:
    if not query_roles:
        return 0.5, [], []
    matched_roles = sorted(role for role in query_roles if candidate.label_roles.get(role))
    missing_roles = sorted(role for role in query_roles if not candidate.label_roles.get(role))
    role_score = len(matched_roles) / max(1, len(query_roles))
    exact_shared = set(query.obs_columns).intersection(candidate.obs_columns)
    shared_label_columns = sorted(
        column
        for column in exact_shared
        if any(column in columns for columns in query.label_roles.values())
        or any(column in columns for columns in candidate.label_roles.values())
    )
    if shared_label_columns:
        role_score = min(1.0, role_score + 0.1)
    return role_score, matched_roles, missing_roles


def _disease_context_score(query: DatasetProfile, candidate: DatasetProfile) -> float:
    query_has = bool(query.label_roles.get("pathology"))
    candidate_has = bool(candidate.label_roles.get("pathology"))
    if not query_has:
        return 0.5
    return 1.0 if candidate_has else 0.0


def _has_spatial_support(profile: DatasetProfile) -> bool:
    modality_group = _modality_group(profile.modality)
    if modality_group == _modality_group("spatial"):
        return True
    if any("spatial" in key.lower() for key in profile.obsm_keys):
        return True
    lowered_columns = {column.lower() for column in profile.obs_columns}
    coordinate_pairs = (
        {"x", "y"},
        {"center_x", "center_y"},
        {"xcoord", "ycoord"},
        {"row", "column"},
    )
    return any(pair.issubset(lowered_columns) for pair in coordinate_pairs)


def _spatial_support_score(query: DatasetProfile, candidate: DatasetProfile) -> float:
    query_needs_spatial = bool(query.label_roles.get("region") or _has_spatial_support(query))
    if not query_needs_spatial:
        return 0.5
    return 1.0 if _has_spatial_support(candidate) else 0.0


def _recommend_transfer_mode(query: DatasetProfile, candidate: DatasetProfile) -> str:
    query_modality = _normalize_token(query.modality)
    candidate_modality = _normalize_token(candidate.modality)
    query_is_sc = query_modality in _modality_group("scRNA-seq")
    candidate_is_sc = candidate_modality in _modality_group("scRNA-seq")
    candidate_is_spatial = candidate_modality in _modality_group("spatial")
    query_is_spatial = query_modality in _modality_group("spatial")
    if query_is_sc and candidate_is_spatial:
        return "SMITH-SC+ST spatial-context transfer prior"
    if candidate_is_sc and query_is_spatial:
        return "SMITH-SC source prior for spatial target"
    if candidate_is_spatial and query_is_spatial:
        return "SMITH-ST same-modality spatial transfer"
    if candidate_is_sc and not query_is_spatial:
        return "SMITH-SC single-cell reference"
    if candidate_is_spatial:
        return "spatial reference prior"
    return "cross-modality reference prior"


def _rationale(row: dict[str, Any]) -> str:
    reasons = []
    if row["species_score"] >= 1:
        reasons.append("same species")
    if row["tissue_score"] >= 1:
        reasons.append("same tissue")
    if row["modality_score"] >= 1:
        reasons.append("same modality")
    elif row["modality_score"] >= 0.6:
        reasons.append("transfer-compatible modality")
    if row["gene_overlap_count"]:
        reasons.append(f"{row['gene_overlap_count']} shared genes")
    if row["matched_label_roles"]:
        reasons.append(f"labels: {row['matched_label_roles']}")
    if row["missing_label_roles"]:
        reasons.append(f"missing: {row['missing_label_roles']}")
    return "; ".join(reasons) if reasons else "low available compatibility evidence"


def _score_candidate(
    query: DatasetProfile,
    candidate: DatasetProfile,
    request: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, Any]:
    query_roles = _requested_roles(request, query)
    overlap_count, overlap_score, jaccard = _gene_overlap(query, candidate)
    label_score, matched_roles, missing_roles = _label_score(query_roles, query, candidate)
    components = {
        "species_score": _species_score(query, candidate),
        "tissue_score": _tissue_score(query, candidate),
        "modality_score": _modality_score(query, candidate),
        "gene_overlap_score": overlap_score,
        "label_compatibility_score": label_score,
        "disease_context_score": _disease_context_score(query, candidate),
        "spatial_support_score": _spatial_support_score(query, candidate),
    }
    total_weight = sum(weights.values())
    weighted_score = sum(components[key] * weights[key] for key in weights) / total_weight * 100.0
    row: dict[str, Any] = {
        "dataset_id": candidate.dataset_id,
        "score": round(float(weighted_score), 3),
        "species": candidate.species,
        "tissue": candidate.tissue,
        "modality": candidate.modality,
        "n_obs": candidate.n_obs,
        "n_vars": candidate.n_vars,
        "tasks": ",".join(candidate.tasks),
        "gene_overlap_count": int(overlap_count),
        "query_gene_coverage": round(float(overlap_score), 4),
        "gene_jaccard": round(float(jaccard), 4),
        "matched_label_roles": ",".join(matched_roles),
        "missing_label_roles": ",".join(missing_roles),
        "candidate_label_columns": ";".join(
            f"{role}:{','.join(columns[:3])}" for role, columns in sorted(candidate.label_roles.items())
        ),
        "recommended_transfer_mode": _recommend_transfer_mode(query, candidate),
        **{key: round(float(value), 4) for key, value in components.items()},
    }
    row["rationale"] = _rationale(row)
    return row


def _plot_reference_scores(score_matrix: pd.DataFrame, output_png: str | Path) -> str:
    output = Path(output_png)
    ensure_dir(output.parent)
    if score_matrix.empty:
        return ""
    components = [column for column in score_matrix.columns if column.endswith("_score")]
    plot_df = score_matrix.set_index("dataset_id")[components]
    height = max(2.5, 0.45 * len(plot_df) + 1.3)
    width = max(6.5, 0.8 * len(components) + 2)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(plot_df.to_numpy(dtype=float), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([item.replace("_score", "").replace("_", "\n") for item in components], fontsize=8)
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(list(plot_df.index), fontsize=8)
    ax.set_title("Reference Transferability Score Decomposition")
    for row_idx in range(plot_df.shape[0]):
        for col_idx in range(plot_df.shape[1]):
            value = plot_df.iat[row_idx, col_idx]
            if math.isfinite(float(value)):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7, color="#1b1b1b")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="component score")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def score_reference_transferability(
    registries: RegistryBundle,
    session: AgentSession,
    working_dir: str | Path,
    request: dict[str, Any] | None = None,
    candidate_dataset_ids: list[str] | None = None,
) -> dict[str, Any]:
    request = dict(request or {})
    output_dir = ensure_dir(request.get("output_dir") or Path(working_dir))
    include_query_dataset = bool(request.get("include_query_dataset", False))
    query = _resolve_query_profile(registries, session, request)
    query_path = str(Path(query.dataset_path).resolve())
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    weights.update({str(k): float(v) for k, v in dict(request.get("score_weights", {})).items() if str(k) in weights})

    candidates = candidate_dataset_ids or request.get("candidate_dataset_ids") or list(registries.datasets.keys())
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for raw_candidate in candidates:
        dataset_id = registries.resolve_dataset_id(str(raw_candidate))
        if not dataset_id:
            skipped.append({"candidate": str(raw_candidate), "reason": "unknown_dataset_id"})
            continue
        entry = registries.datasets[dataset_id]
        candidate_path = str(Path(entry.path).resolve())
        if not include_query_dataset and candidate_path == query_path:
            skipped.append({"candidate": dataset_id, "reason": "query_dataset"})
            continue
        candidate_profile = profile_dataset(entry.path, dataset_id=dataset_id, entry=entry)
        row = _score_candidate(query, candidate_profile, request, weights)
        rows.append(row)

    ranked = pd.DataFrame(rows)
    if not ranked.empty:
        ranked = ranked.sort_values(["score", "gene_overlap_count", "dataset_id"], ascending=[False, False, True])

    candidates_tsv = Path(output_dir) / "reference_candidates.tsv"
    score_matrix_tsv = Path(output_dir) / "reference_score_matrix.tsv"
    ranked.to_csv(candidates_tsv, sep="\t", index=False)
    component_columns = ["dataset_id", *[column for column in ranked.columns if column.endswith("_score")]]
    score_matrix = ranked[component_columns] if not ranked.empty else pd.DataFrame(columns=component_columns)
    score_matrix.to_csv(score_matrix_tsv, sep="\t", index=False)
    score_plot_png = _plot_reference_scores(score_matrix, Path(output_dir) / "reference_score_matrix.png")

    top = ranked.head(1).to_dict(orient="records")[0] if not ranked.empty else None
    selection = {
        "query_dataset": query.to_jsonable(include_genes=False),
        "score_weights": weights,
        "selected_reference": top,
        "skipped_candidates": skipped,
        "artifacts": {
            "reference_candidates_tsv": str(candidates_tsv),
            "reference_score_matrix_tsv": str(score_matrix_tsv),
            "reference_score_matrix_png": score_plot_png,
        },
    }
    selection_json = Path(output_dir) / "reference_selection.json"
    write_json(selection_json, selection)
    return {
        "reference_candidates_tsv": str(candidates_tsv),
        "reference_score_matrix_tsv": str(score_matrix_tsv),
        "reference_score_matrix_png": score_plot_png,
        "reference_selection_json": str(selection_json),
        "query_dataset": query.to_jsonable(include_genes=False),
        "top_candidates": ranked.head(5).to_dict(orient="records") if not ranked.empty else [],
        "skipped_candidates": skipped,
    }
