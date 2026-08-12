from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import requests

from smith_agent.utils import ensure_dir, write_json


DISCOVER_INDEX_URL = "https://api.cellxgene.cziscience.com/dp/v1/datasets/index"

IMAGING_SPATIAL_ASSAY_TOKENS = (
    "merfish",
    "merscope",
    "xenium",
    "cosmx",
    "seqfish",
    "osmfish",
    "starmap",
    "starfish",
)

SEQUENCING_SPATIAL_ASSAY_TOKENS = (
    "visium",
    "slide-seq",
    "slideseq",
    "stereo-seq",
    "stereoseq",
    "dbit",
    "spatial gene expression",
)

SPATIAL_ASSAY_FAMILIES = ("imaging_spatial", "sequencing_spatial")

PANEL_SOURCE_QUERY_ROLES = {
    "panel design source",
    "panel source",
    "panel source reference",
    "panel design source reference",
    "panel-design source",
    "panel-design source reference",
    "panel_design_source",
    "panel_design_source_reference",
    "source reference",
    "source_reference",
    "scrna panel source",
    "scrna_panel_source",
}

SINGLE_CELL_RNA_ASSAY_TOKENS = (
    "10x",
    "smart-seq",
    "drop-seq",
    "seq-well",
    "indrop",
    "sci-rna",
    "cel-seq",
    "mars-seq",
    "scrb-seq",
    "bd rhapsody",
    "parse",
    "hive",
    "rna-seq",
    "transcription profiling",
    "gene expression",
)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _labels(record: dict[str, Any], key: str) -> list[str]:
    values = record.get(key, [])
    if not isinstance(values, list):
        return []
    labels = []
    for item in values:
        if isinstance(item, dict) and item.get("label"):
            labels.append(str(item["label"]))
    return labels


def _normalize(value: str) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def _term_matches(value: str, terms: list[str], *, exact: bool = False) -> bool:
    if not terms:
        return True
    normalized = _normalize(value)
    for term in terms:
        query = _normalize(term)
        if not query:
            continue
        if exact and normalized == query:
            return True
        if not exact and (normalized == query or query in normalized or normalized in query):
            return True
    return False


def _any_term_matches(values: list[str], terms: list[str], *, exact: bool = False) -> bool:
    if not terms:
        return True
    return any(_term_matches(value, terms, exact=exact) for value in values)


def classify_assay_family(assay_labels: list[str]) -> str:
    text = " ".join(assay_labels).lower()
    if any(token in text for token in IMAGING_SPATIAL_ASSAY_TOKENS):
        return "imaging_spatial"
    if any(token in text for token in SEQUENCING_SPATIAL_ASSAY_TOKENS):
        return "sequencing_spatial"
    if any(token in text for token in SINGLE_CELL_RNA_ASSAY_TOKENS):
        return "single_cell_rna"
    if "atac" in text:
        return "chromatin_accessibility"
    if "methyl" in text or "mch" in text:
        return "methylation"
    return "other"


def _query_role(query: dict[str, Any]) -> str:
    for key in ("query_role", "dataset_role", "reference_role", "intended_use"):
        value = str(query.get(key, "") or "").strip()
        if value:
            return _normalize(value)
    return ""


def _normalize_panel_source_query(query: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    normalized = dict(query)
    notes: list[str] = []
    role = _query_role(normalized)
    is_panel_source = bool(normalized.get("panel_design_source")) or role in PANEL_SOURCE_QUERY_ROLES
    if not is_panel_source:
        return normalized, notes

    desired_families = _as_list(normalized.get("assay_families") or normalized.get("assay_family"))
    if desired_families and "single_cell_rna" not in desired_families:
        notes.append(f"panel_design_source_forced_single_cell_rna_from:{','.join(desired_families)}")
    if desired_families != ["single_cell_rna"]:
        normalized.pop("assay_families", None)
        normalized["assay_family"] = "single_cell_rna"

    excluded_families = set(_as_list(normalized.get("exclude_assay_families") or normalized.get("exclude_assay_family")))
    excluded_families.update(SPATIAL_ASSAY_FAMILIES)
    normalized.pop("exclude_assay_family", None)
    normalized["exclude_assay_families"] = sorted(excluded_families)
    normalized["query_role"] = "panel_design_source"
    normalized.setdefault("target_data_policy", "held_out_imaging_spatial_target_not_used_for_source_retrieval")
    return normalized, notes


def _fetch_discover_index(url: str = DISCOVER_INDEX_URL) -> list[dict[str, Any]]:
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("CELLxGENE Discover index did not return a dataset list.")
    return [item for item in payload if isinstance(item, dict)]


def _discover_h5ad_asset(record: dict[str, Any], preferred_filetype: str = "H5AD") -> dict[str, Any] | None:
    assets = record.get("dataset_assets", [])
    if not isinstance(assets, list):
        return None
    normalized_preferred = preferred_filetype.upper()
    for filetype in [normalized_preferred, "H5AD", "RAW_H5AD"]:
        for asset in assets:
            if isinstance(asset, dict) and str(asset.get("filetype", "")).upper() == filetype:
                return dict(asset)
    return None


def _discover_h5ad_asset_url(dataset_id: str, asset_id: str) -> str:
    response = requests.get(
        f"https://api.cellxgene.cziscience.com/dp/v1/datasets/{dataset_id}/asset/{asset_id}",
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    url = str(payload.get("url", ""))
    if not url:
        raise ValueError(f"CELLxGENE Discover asset response did not include a URL for {dataset_id}/{asset_id}.")
    return url


def _stream_download(url: str, output_path: str | Path) -> str:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return str(output_path)


def _census_dataset_ids(census_version: str = "stable") -> set[str]:
    try:
        import cellxgene_census
    except ImportError:
        return set()
    with cellxgene_census.open_soma(census_version=census_version) as census:
        datasets = census["census_info"]["datasets"].read(column_names=["dataset_id"]).concat().to_pandas()
    return set(datasets["dataset_id"].astype(str))


def _tissue_tier(tissues: list[str], query_tissues: list[str]) -> str:
    if not query_tissues:
        return "unconstrained"
    matching = [tissue for tissue in tissues if _term_matches(tissue, query_tissues)]
    if not matching:
        return "no_match"
    if tissues and len(matching) == len(tissues):
        return "tissue_specific"
    return "tissue_containing"


def _score_record(record: dict[str, Any], query: dict[str, Any], assay_family: str, tissue_tier: str) -> tuple[float, dict[str, float]]:
    organisms = _labels(record, "organism")
    tissues = _labels(record, "tissue")
    diseases = _labels(record, "disease")
    organism_terms = _as_list(query.get("organism") or query.get("species"))
    tissue_terms = _as_list(query.get("tissue") or query.get("tissues"))
    disease_terms = _as_list(query.get("disease") or query.get("diseases"))

    organism_score = 1.0 if _any_term_matches(organisms, organism_terms, exact=True) else 0.0
    if not organism_terms:
        organism_score = 0.5
    tissue_score = 1.0 if tissue_tier == "tissue_specific" else 0.75 if tissue_tier == "tissue_containing" else 0.0
    if not tissue_terms:
        tissue_score = 0.5
    if disease_terms:
        disease_score = 1.0 if _any_term_matches(diseases, disease_terms) else 0.0
    else:
        disease_score = 0.5
    desired_families = _as_list(query.get("assay_families") or query.get("assay_family"))
    assay_score = 1.0 if not desired_families or assay_family in desired_families else 0.0
    primary_raw = record.get("is_primary_data")
    primary_score = 1.0 if primary_raw is True or str(primary_raw).upper() == "PRIMARY" else 0.5
    cell_count = int(record.get("primary_cell_count") or record.get("cell_count") or 0)
    size_score = min(1.0, math.log10(max(10, cell_count)) / 6.0)
    tissue_specificity_score = 1.0 if tissue_tier == "tissue_specific" else 0.4 if tissue_tier == "tissue_containing" else 0.0
    components = {
        "organism_score": organism_score,
        "tissue_score": tissue_score,
        "disease_score": disease_score,
        "assay_score": assay_score,
        "primary_score": primary_score,
        "size_score": size_score,
        "tissue_specificity_score": tissue_specificity_score,
    }
    weights = {
        "organism_score": 0.20,
        "tissue_score": 0.25,
        "disease_score": 0.10,
        "assay_score": 0.15,
        "primary_score": 0.10,
        "size_score": 0.10,
        "tissue_specificity_score": 0.10,
    }
    score = 100.0 * sum(components[name] * weights[name] for name in weights)
    return score, components


def _record_to_row(record: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    assays = _labels(record, "assay")
    tissues = _labels(record, "tissue")
    diseases = _labels(record, "disease")
    organism = _labels(record, "organism")
    assay_family = classify_assay_family(assays)
    tier = _tissue_tier(tissues, _as_list(query.get("tissue") or query.get("tissues")))
    score, components = _score_record(record, query, assay_family, tier)
    h5ad_asset = _discover_h5ad_asset(record, preferred_filetype=str(query.get("preferred_filetype", "H5AD")))
    return {
        "dataset_id": record.get("id", ""),
        "collection_id": record.get("collection_id", ""),
        "score": round(score, 3),
        "name": record.get("name", ""),
        "organism": ";".join(organism),
        "tissue": ";".join(tissues),
        "disease": ";".join(diseases),
        "assay": ";".join(assays),
        "assay_family": assay_family,
        "tissue_match_tier": tier,
        "cell_count": int(record.get("cell_count") or 0),
        "primary_cell_count": int(record.get("primary_cell_count") or 0),
        "is_primary_data": record.get("is_primary_data"),
        "suspension_type": ";".join(_labels(record, "suspension_type")),
        "explorer_url": record.get("explorer_url", ""),
        "discover_h5ad_asset_id": h5ad_asset.get("id", "") if h5ad_asset else "",
        "discover_h5ad_filetype": h5ad_asset.get("filetype", "") if h5ad_asset else "",
        "discover_h5ad_s3_uri": h5ad_asset.get("s3_uri", "") if h5ad_asset else "",
        **{key: round(float(value), 4) for key, value in components.items()},
    }


def query_cellxgene_metadata(
    output_dir: str | Path,
    query: dict[str, Any] | None = None,
    max_results: int = 50,
    index_url: str = DISCOVER_INDEX_URL,
) -> dict[str, Any]:
    query, normalization_notes = _normalize_panel_source_query(dict(query or {}))
    output_dir = ensure_dir(output_dir)
    raw_records = _fetch_discover_index(index_url)

    organism_terms = _as_list(query.get("organism") or query.get("species"))
    tissue_terms = _as_list(query.get("tissue") or query.get("tissues"))
    disease_terms = _as_list(query.get("disease") or query.get("diseases"))
    collection_ids = set(_as_list(query.get("collection_ids") or query.get("collection_id")))
    exclude_dataset_ids = set(_as_list(query.get("exclude_dataset_ids") or query.get("exclude_dataset_id")))
    include_assays = _as_list(query.get("include_assays"))
    exclude_assays = _as_list(query.get("exclude_assays"))
    desired_families = _as_list(query.get("assay_families") or query.get("assay_family"))
    exclude_families = _as_list(query.get("exclude_assay_families") or query.get("exclude_assay_family"))
    min_cells = int(query.get("min_cells", 0) or 0)
    primary_only = bool(query.get("primary_only", False))
    materializable_only = bool(query.get("materializable_only", True))
    census_version = str(query.get("census_version", "stable"))
    tissue_mode = str(query.get("tissue_mode", "containing") or "containing").lower()
    materializable_ids = _census_dataset_ids(census_version) if materializable_only else set()

    rows = []
    skipped_counts: dict[str, int] = {}
    for record in raw_records:
        assays = _labels(record, "assay")
        tissues = _labels(record, "tissue")
        diseases = _labels(record, "disease")
        organisms = _labels(record, "organism")
        assay_family = classify_assay_family(assays)
        cell_count = int(record.get("primary_cell_count") or record.get("cell_count") or 0)
        dataset_id = str(record.get("id", ""))
        collection_id = str(record.get("collection_id", ""))

        def skip(reason: str) -> None:
            skipped_counts[reason] = skipped_counts.get(reason, 0) + 1

        has_discover_h5ad = _discover_h5ad_asset(record, preferred_filetype=str(query.get("preferred_filetype", "H5AD"))) is not None
        if exclude_dataset_ids and dataset_id in exclude_dataset_ids:
            skip("exclude_dataset_id")
            continue
        if collection_ids and collection_id not in collection_ids:
            skip("collection_id")
            continue
        if materializable_only and materializable_ids and dataset_id not in materializable_ids and not has_discover_h5ad:
            skip("not_in_census")
            continue
        if organism_terms and not _any_term_matches(organisms, organism_terms, exact=True):
            skip("organism")
            continue
        if tissue_terms and not _any_term_matches(tissues, tissue_terms):
            skip("tissue")
            continue
        tier = _tissue_tier(tissues, tissue_terms)
        if tissue_terms and tissue_mode == "specific" and tier != "tissue_specific":
            skip("tissue_specificity")
            continue
        if disease_terms and not _any_term_matches(diseases, disease_terms):
            skip("disease")
            continue
        if include_assays and not _any_term_matches(assays, include_assays):
            skip("include_assays")
            continue
        if exclude_assays and _any_term_matches(assays, exclude_assays):
            skip("exclude_assays")
            continue
        if exclude_families and assay_family in exclude_families:
            skip("exclude_assay_family")
            continue
        if desired_families and assay_family not in desired_families:
            skip("assay_family")
            continue
        if min_cells and cell_count < min_cells:
            skip("min_cells")
            continue
        primary_raw = record.get("is_primary_data")
        is_primary = primary_raw is True or str(primary_raw).upper() == "PRIMARY"
        if primary_only and not is_primary:
            skip("primary_only")
            continue
        row = _record_to_row(record, query)
        row["materializable_in_census"] = bool(materializable_ids and dataset_id in materializable_ids)
        row["downloadable_h5ad"] = bool(has_discover_h5ad)
        rows.append(row)

    ranked = pd.DataFrame(rows)
    if not ranked.empty:
        ranked = ranked.sort_values(["score", "primary_cell_count", "cell_count"], ascending=[False, False, False])
        ranked = ranked.head(max_results)

    candidates_tsv = Path(output_dir) / "cellxgene_reference_candidates.tsv"
    candidates_json = Path(output_dir) / "cellxgene_reference_candidates.json"
    ranked.to_csv(candidates_tsv, sep="\t", index=False)
    write_json(
        candidates_json,
        {
            "query": query,
            "index_url": index_url,
            "census_version": census_version,
            "materializable_only": materializable_only,
            "n_index_records": len(raw_records),
            "n_candidates": int(ranked.shape[0]),
            "skipped_counts": skipped_counts,
            "normalization_notes": normalization_notes,
            "top_candidates": ranked.head(10).to_dict(orient="records") if not ranked.empty else [],
        },
    )
    return {
        "cellxgene_reference_candidates_tsv": str(candidates_tsv),
        "cellxgene_reference_candidates_json": str(candidates_json),
        "n_index_records": len(raw_records),
        "n_candidates": int(ranked.shape[0]),
        "top_candidates": ranked.head(10).to_dict(orient="records") if not ranked.empty else [],
        "skipped_counts": skipped_counts,
        "normalization_notes": normalization_notes,
    }


def _quote_filter_value(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _or_filter(column: str, values: list[str]) -> str:
    values = [value for value in values if str(value).strip()]
    if not values:
        return ""
    return "(" + " or ".join(f"{column} == {_quote_filter_value(value)}" for value in values) + ")"


def build_obs_value_filter(
    dataset_id: str,
    tissue: Any = None,
    tissue_general: Any = None,
    disease: Any = None,
    cell_type: Any = None,
    assay: Any = None,
    suspension_type: Any = None,
    is_primary_data: bool | None = None,
    extra_filter: str = "",
) -> str:
    clauses = [f"dataset_id == {_quote_filter_value(dataset_id)}"]
    for column, values in [
        ("tissue", _as_list(tissue)),
        ("tissue_general", _as_list(tissue_general)),
        ("disease", _as_list(disease)),
        ("cell_type", _as_list(cell_type)),
        ("assay", _as_list(assay)),
        ("suspension_type", _as_list(suspension_type)),
    ]:
        clause = _or_filter(column, values)
        if clause:
            clauses.append(clause)
    if is_primary_data is not None:
        clauses.append(f"is_primary_data == {str(bool(is_primary_data)).lower()}")
    if extra_filter.strip():
        clauses.append(f"({extra_filter.strip()})")
    return " and ".join(clauses)


def _build_var_filter(gene_symbols: list[str], extra_filter: str = "") -> str:
    clauses = []
    gene_clause = _or_filter("feature_name", gene_symbols)
    if gene_clause:
        clauses.append(gene_clause)
    if extra_filter.strip():
        clauses.append(f"({extra_filter.strip()})")
    return " and ".join(clauses)


def _local_subset_h5ad(
    source_h5ad: str | Path,
    output_h5ad: str | Path,
    tissue: Any = None,
    disease: Any = None,
    cell_type: Any = None,
    assay: Any = None,
    gene_symbols: list[str] | None = None,
    max_cells: int = 50000,
    seed: int = 42,
) -> dict[str, Any]:
    source_h5ad = Path(source_h5ad)
    output_h5ad = Path(output_h5ad)
    ensure_dir(output_h5ad.parent)
    adata = ad.read_h5ad(source_h5ad, backed="r")
    mask = pd.Series(True, index=adata.obs_names)
    for column, values in [
        ("tissue", _as_list(tissue)),
        ("disease", _as_list(disease)),
        ("cell_type", _as_list(cell_type)),
        ("assay", _as_list(assay)),
    ]:
        if values and column in adata.obs:
            allowed = {_normalize(value) for value in values}
            mask &= adata.obs[column].astype(str).map(lambda item: _normalize(item) in allowed).to_numpy()
    indices = np.where(mask.to_numpy())[0]
    if max_cells and len(indices) > max_cells:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_cells, replace=False))
    var_indices: slice | list[int] = slice(None)
    if gene_symbols:
        genes = {_normalize(gene) for gene in gene_symbols}
        var_names = pd.Index([_normalize(item) for item in adata.var_names])
        selected = np.where(var_names.isin(genes))[0].tolist()
        if not selected:
            for column in ["feature_name", "gene_name", "gene_symbol"]:
                if column in adata.var:
                    values = pd.Index([_normalize(item) for item in adata.var[column].astype(str)])
                    selected = np.where(values.isin(genes))[0].tolist()
                    if selected:
                        break
        var_indices = selected
    try:
        if isinstance(var_indices, slice):
            subset = adata[indices.tolist(), :].to_memory()
        else:
            subset = adata[indices.tolist(), var_indices].to_memory()
    except Exception:  # noqa: BLE001
        # Some Discover H5AD files expose backed sparse datasets whose fancy
        # indexing is incompatible with the scipy version in this environment.
        # Loading the file into memory first is acceptable for the small
        # post-filtered Visium/Slide-seq pilot datasets and keeps the generic
        # local subsetting path usable.
        adata_mem = adata.to_memory()
        if isinstance(var_indices, slice):
            subset = adata_mem[indices.tolist(), :].copy()
        else:
            subset = adata_mem[indices.tolist(), var_indices].copy()
    subset.write_h5ad(output_h5ad)
    if getattr(adata, "file", None) is not None:
        adata.file.close()
    return {
        "n_obs": int(subset.n_obs),
        "n_vars": int(subset.n_vars),
        "output_h5ad": str(output_h5ad),
    }


def materialize_cellxgene_dataset(
    output_dir: str | Path,
    dataset_id: str,
    organism: str,
    census_version: str = "stable",
    tissue: Any = None,
    tissue_general: Any = None,
    disease: Any = None,
    cell_type: Any = None,
    assay: Any = None,
    suspension_type: Any = None,
    is_primary_data: bool | None = None,
    gene_symbols: list[str] | None = None,
    max_cells: int = 50000,
    seed: int = 42,
    output_h5ad: str | Path | None = None,
    extra_obs_filter: str = "",
    extra_var_filter: str = "",
    download_source: bool = False,
    source: str = "census",
    discover_asset_id: str = "",
    discover_filetype: str = "H5AD",
    source_h5ad: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    output_h5ad = Path(output_h5ad or output_dir / f"cellxgene_{dataset_id}.h5ad")
    ensure_dir(output_h5ad.parent)

    if download_source or source == "discover":
        cellxgene_census = None
        if source == "census":
            try:
                import cellxgene_census as _cellxgene_census

                cellxgene_census = _cellxgene_census
            except ImportError as exc:
                raise RuntimeError("CELLxGENE Census source downloads require `cellxgene-census`. Use source='discover' to download Discover H5AD assets without Census.") from exc
        source_path = Path(source_h5ad) if source_h5ad else output_dir / f"cellxgene_{dataset_id}.source.h5ad"
        if not source_h5ad:
            if source == "census":
                try:
                    if cellxgene_census is None:
                        raise RuntimeError("cellxgene_census import failed.")
                    cellxgene_census.download_source_h5ad(dataset_id, str(source_path), census_version=census_version)
                except Exception:  # noqa: BLE001
                    source = "discover"
            if source == "discover":
                if not discover_asset_id:
                    records = _fetch_discover_index()
                    record = next((item for item in records if str(item.get("id", "")) == dataset_id), None)
                    if record is None:
                        raise ValueError(f"Dataset {dataset_id} was not found in CELLxGENE Discover index.")
                    asset = _discover_h5ad_asset(record, preferred_filetype=discover_filetype)
                    if asset is None:
                        raise ValueError(f"Dataset {dataset_id} does not expose an H5AD asset in CELLxGENE Discover.")
                    discover_asset_id = str(asset["id"])
                url = _discover_h5ad_asset_url(dataset_id, discover_asset_id)
                _stream_download(url, source_path)
        subset_result = _local_subset_h5ad(
            source_h5ad=source_path,
            output_h5ad=output_h5ad,
            tissue=tissue,
            disease=disease,
            cell_type=cell_type,
            assay=assay,
            gene_symbols=gene_symbols,
            max_cells=max_cells,
            seed=seed,
        )
        manifest = {
            "dataset_id": dataset_id,
            "organism": organism,
            "census_version": census_version,
            "download_source": True,
            "source": source,
            "source_h5ad": str(source_path),
            "discover_asset_id": discover_asset_id,
            **subset_result,
        }
        manifest_path = output_dir / "cellxgene_materialization.json"
        write_json(manifest_path, manifest)
        return {**manifest, "cellxgene_materialization_json": str(manifest_path)}

    try:
        import cellxgene_census
    except ImportError as exc:
        raise RuntimeError("CELLxGENE Census subset materialization requires `cellxgene-census`. Use source='discover' and download_source=true to use Discover H5AD assets without Census.") from exc

    obs_filter = build_obs_value_filter(
        dataset_id=dataset_id,
        tissue=tissue,
        tissue_general=tissue_general,
        disease=disease,
        cell_type=cell_type,
        assay=assay,
        suspension_type=suspension_type,
        is_primary_data=is_primary_data,
        extra_filter=extra_obs_filter,
    )
    var_filter = _build_var_filter(gene_symbols or [], extra_filter=extra_var_filter)

    with cellxgene_census.open_soma(census_version=census_version) as census:
        obs = cellxgene_census.get_obs(
            census,
            organism,
            value_filter=obs_filter,
            column_names=["soma_joinid"],
        )
        n_obs_before = int(obs.shape[0])
        if n_obs_before == 0:
            raise ValueError(f"CELLxGENE query matched zero cells: {obs_filter}")
        if max_cells and n_obs_before > max_cells:
            rng = np.random.default_rng(seed)
            obs = obs.iloc[np.sort(rng.choice(n_obs_before, size=max_cells, replace=False))]
        obs_coords = obs["soma_joinid"].to_numpy()
        adata = cellxgene_census.get_anndata(
            census=census,
            organism=organism,
            obs_coords=obs_coords,
            var_value_filter=var_filter or None,
            obs_column_names=[
                "dataset_id",
                "assay",
                "cell_type",
                "tissue",
                "tissue_general",
                "disease",
                "suspension_type",
                "is_primary_data",
            ],
            var_column_names=["feature_id", "feature_name", "feature_length"],
        )
    adata.write_h5ad(output_h5ad)
    manifest = {
        "dataset_id": dataset_id,
        "organism": organism,
        "census_version": census_version,
        "download_source": False,
        "obs_value_filter": obs_filter,
        "var_value_filter": var_filter,
        "n_obs_before_sampling": n_obs_before,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "output_h5ad": str(output_h5ad),
    }
    manifest_path = output_dir / "cellxgene_materialization.json"
    write_json(manifest_path, manifest)
    return {**manifest, "cellxgene_materialization_json": str(manifest_path)}
