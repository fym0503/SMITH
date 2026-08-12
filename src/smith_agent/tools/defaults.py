from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from smith_agent.adapters.probedealer_adapter import run_probedealer_screen_light
from smith_agent.adapters.feasibility_backends import (
    run_odt_property_batches,
    run_odt_property_screen,
    run_oligominer_specificity_screen,
    run_probedealer_backend_screen,
)
from smith_agent.bridge.transcripts import build_probe_candidate_manifest
from smith_agent.cellxgene_backend import materialize_cellxgene_dataset, query_cellxgene_metadata
from smith_agent.adapters.smith_eval import evaluate_cross_dataset_panel
from smith_agent.adapters.smith_runner import SmithSelectionConfig, run_smith_selection
from smith_agent.feasibility.integration import (
    IntegrationThresholds,
    apply_hard_constraints,
    build_integration_summary,
    write_passing_targets_json,
)
from smith_agent.feasibility.workflows import (
    build_three_backend_feasibility_summary,
    write_three_backend_outputs,
)
from smith_agent.panel_rank_aggregation import aggregate_reference_panel_ranks
from smith_agent.benchmarking import evaluate_panel_cell_type_classification, evaluate_panel_coordinate_regression
from smith_agent.reference_retrieval import profile_dataset, score_reference_transferability
from smith_agent.reporting.builder import build_run_report
from smith_agent.reporting.plots import plot_dataset_umap, plot_evaluation_summary
from smith_agent.search import search_file_tree, search_registry_entries, search_session_artifacts
from smith_agent.tool_registry import ToolRegistry, ToolSpec
from smith_agent.utils import ensure_dir, write_json


def _resource_root_from_config(config) -> Path:
    for directory in (config.skills_dir, config.tools_dir, config.models_dir):
        try:
            candidate = Path(directory).resolve().parents[2]
        except IndexError:
            continue
        if (candidate / "configs" / "agent").exists():
            return candidate
    return config.repo_root


def _resolve_project_path(config, relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path

    candidates = [config.repo_root / path]
    resource_root = _resource_root_from_config(config)
    if resource_root != config.repo_root:
        candidates.append(resource_root / path)
    for root in config.external_roots.values():
        candidate = Path(root) / path
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_model_entrypoint(runtime, model_entry) -> Path:
    return _resolve_project_path(runtime.config, model_entry.entrypoint)


def _handle_list_skills(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "skills": [entry.to_dict() for entry in runtime.registries.skills.values()],
    }


def _handle_list_tools(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "tools": [entry.to_dict() for entry in runtime.registries.tools.values()],
    }


def _handle_list_datasets(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "datasets": [entry.to_dict() for entry in runtime.registries.datasets.values()],
    }


def _handle_list_models(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "models": [entry.to_dict() for entry in runtime.registries.models.values()],
    }


def _handle_search_smith_agent(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("search_smith_agent requires a non-empty query.")
    scopes = [str(item).strip().lower() for item in arguments.get("scopes", []) if str(item).strip()]
    if not scopes:
        scopes = ["registries", "docs", "configs", "sessions", "outputs", "feasibility"]
    limit = int(arguments.get("limit", 20) or 20)

    payload: dict[str, Any] = {"query": query, "scopes": scopes}
    if "registries" in scopes:
        payload["registry_results"] = search_registry_entries(runtime.registries, query, limit=limit)
    if any(scope in scopes for scope in ["docs", "configs", "workspace", "feasibility"]):
        roots = []
        if "docs" in scopes or "workspace" in scopes:
            roots.append(_resolve_project_path(runtime.config, "docs"))
        if "configs" in scopes or "workspace" in scopes:
            roots.append(_resolve_project_path(runtime.config, "configs"))
        if "feasibility" in scopes:
            roots.append(_resolve_project_path(runtime.config, "src/smith_agent/feasibility"))
        payload["file_results"] = search_file_tree(query, roots=roots, limit=limit)
    if "sessions" in scopes or "outputs" in scopes:
        payload["artifact_results"] = search_session_artifacts(
            query=query,
            sessions_root=runtime.config.sessions_root if "sessions" in scopes else runtime.config.sessions_root.parent / "_none_",
            outputs_root=runtime.config.outputs_root if "outputs" in scopes else runtime.config.outputs_root.parent / "_none_",
            limit=limit,
        )
    runtime.session.state["last_search"] = payload
    runtime.session_store.save(runtime.session)
    return payload


def _handle_set_active_dataset(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    dataset_id = runtime.registries.resolve_dataset_id(arguments.get("dataset_id"))
    if not dataset_id:
        raise KeyError(f"Unknown dataset: {arguments.get('dataset_id')}")
    runtime.session.active_dataset_id = dataset_id
    runtime.session_store.save(runtime.session)
    return {"active_dataset_id": dataset_id}


def _handle_set_active_model(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    model_id = runtime.registries.resolve_model_id(arguments.get("model_id"))
    if not model_id:
        raise KeyError(f"Unknown model: {arguments.get('model_id')}")
    runtime.session.active_model_id = model_id
    runtime.session_store.save(runtime.session)
    return {"active_model_id": model_id}


def _handle_inspect_session_state(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    del arguments
    return {
        "session_id": runtime.session.session_id,
        "active_dataset_id": runtime.session.active_dataset_id,
        "active_model_id": runtime.session.active_model_id,
        "state": runtime.session.state,
        "memory": runtime.session.memory,
    }


def _handle_describe_skill(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    skill_id = runtime.registries.resolve_skill_id(arguments.get("skill_id"))
    if not skill_id:
        raise KeyError(f"Unknown skill: {arguments.get('skill_id')}")
    return runtime.registries.skills[skill_id].to_dict()


def _handle_resolve_dataset_context(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    dataset_path = arguments.get("dataset_path")
    dataset_entry = None
    dataset_id = ""
    if not dataset_path and arguments.get("dataset_id"):
        raw_dataset_id = str(arguments["dataset_id"])
        dataset_id = runtime.registries.resolve_dataset_id(raw_dataset_id) or raw_dataset_id
        if dataset_id in runtime.registries.datasets:
            dataset_entry = runtime.registries.datasets[dataset_id]
            dataset_path = dataset_entry.path
        elif dataset_id in runtime.session.state.get("mounted_datasets", {}):
            dataset_path = runtime.session.state["mounted_datasets"][dataset_id]["path"]
        else:
            raise KeyError(f"Unknown dataset: {arguments['dataset_id']}")
    if not dataset_path:
        raise ValueError("dataset_path or dataset_id is required")

    profile = profile_dataset(dataset_path, dataset_id=dataset_id, entry=dataset_entry)
    payload = profile.to_jsonable(include_genes=False)
    output_path = runtime.working_dir / "context_summary.json"
    write_json(output_path, payload)
    runtime.session.state["context_summary"] = str(output_path)
    runtime.session_store.save(runtime.session)
    return payload


def _handle_score_reference_transferability(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    result = score_reference_transferability(
        registries=runtime.registries,
        session=runtime.session,
        working_dir=runtime.working_dir,
        request=dict(arguments.get("request", {})),
        candidate_dataset_ids=arguments.get("candidate_dataset_ids"),
    )
    runtime.session.state["reference_candidates"] = result["reference_candidates_tsv"]
    runtime.session.state["reference_selection"] = result["reference_selection_json"]
    if result.get("reference_score_matrix_png"):
        runtime.session.state["reference_score_matrix_plot"] = result["reference_score_matrix_png"]
    runtime.session_store.save(runtime.session)
    return result


def _handle_query_cellxgene_metadata(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    result = query_cellxgene_metadata(
        output_dir=arguments.get("output_dir", runtime.working_dir / "cellxgene_retrieval"),
        query=dict(arguments.get("query", {})),
        max_results=int(arguments.get("max_results", 50)),
        index_url=str(arguments.get("index_url", "https://api.cellxgene.cziscience.com/dp/v1/datasets/index")),
    )
    runtime.session.state["cellxgene_reference_candidates"] = result["cellxgene_reference_candidates_tsv"]
    runtime.session.state["cellxgene_reference_candidates_json"] = result["cellxgene_reference_candidates_json"]
    runtime.session_store.save(runtime.session)
    return result


def _handle_materialize_cellxgene_dataset(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    dataset_id = str(arguments.get("dataset_id", "")).strip()
    if not dataset_id:
        candidates_path = runtime.session.state.get("cellxgene_reference_candidates")
        if candidates_path:
            candidates = pd.read_csv(candidates_path, sep="\t")
            if not candidates.empty:
                dataset_id = str(candidates.iloc[0]["dataset_id"])
    if not dataset_id:
        raise ValueError("materialize_cellxgene_dataset requires dataset_id or prior CELLxGENE candidates.")
    result = materialize_cellxgene_dataset(
        output_dir=arguments.get("output_dir", runtime.working_dir / "cellxgene_materialized"),
        dataset_id=dataset_id,
        organism=str(arguments.get("organism", arguments.get("species", "Homo sapiens"))),
        census_version=str(arguments.get("census_version", "stable")),
        tissue=arguments.get("tissue"),
        tissue_general=arguments.get("tissue_general"),
        disease=arguments.get("disease"),
        cell_type=arguments.get("cell_type"),
        assay=arguments.get("assay"),
        suspension_type=arguments.get("suspension_type"),
        is_primary_data=arguments.get("is_primary_data"),
        gene_symbols=[str(item) for item in arguments.get("gene_symbols", [])],
        max_cells=int(arguments.get("max_cells", 50000)),
        seed=int(arguments.get("seed", 42)),
        output_h5ad=arguments.get("output_h5ad"),
        extra_obs_filter=str(arguments.get("extra_obs_filter", "")),
        extra_var_filter=str(arguments.get("extra_var_filter", "")),
        download_source=bool(arguments.get("download_source", False)),
        source=str(arguments.get("source", "census")),
        discover_asset_id=str(arguments.get("discover_asset_id", "")),
        discover_filetype=str(arguments.get("discover_filetype", "H5AD")),
        source_h5ad=arguments.get("source_h5ad"),
    )
    runtime.session.state["cellxgene_materialized_h5ad"] = result["output_h5ad"]
    runtime.session.state["cellxgene_materialization"] = result["cellxgene_materialization_json"]
    runtime.session_store.save(runtime.session)
    return result


def _handle_aggregate_feasibility_results(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    thresholds = IntegrationThresholds(
        min_property_probes=int(arguments.get("min_property_probes", 20)),
        min_specific_probes=int(arguments.get("min_specific_probes", 10)),
        min_deployment_probes=int(arguments.get("min_deployment_probes", 20)),
        min_paintshop_on_target_mean=float(arguments.get("min_paintshop_on_target_mean", 95.0)),
        max_paintshop_off_target_mean=float(arguments.get("max_paintshop_off_target_mean", 10.0)),
        max_paintshop_off_target_max=float(arguments.get("max_paintshop_off_target_max", 100.0)),
        require_transcript_gate=bool(arguments.get("require_transcript_gate", False)),
    )
    df = build_integration_summary(
        manifest_tsv=arguments["manifest_tsv"],
        odt_summary_tsv=arguments["odt_summary_tsv"],
        oligominer_specificity_tsv=arguments["oligominer_summary_tsv"],
        probedealer_summary_tsv=arguments["probedealer_summary_tsv"],
        paintshop_rna_probe_tsv=arguments["paintshop_rna_probe_tsv"],
        thresholds=thresholds,
    )
    output_path = Path(arguments.get("output_tsv", runtime.working_dir / "feasibility_table.tsv"))
    ensure_dir(output_path.parent)
    df.to_csv(output_path, sep="\t", index=False)
    passing_json = Path(arguments.get("passing_targets_json", runtime.working_dir / "passing_targets.json"))
    write_passing_targets_json(df, passing_json)
    runtime.session.state["feasibility_table"] = str(output_path)
    runtime.session.state["passing_targets"] = str(passing_json)
    runtime.session_store.save(runtime.session)
    return {
        "feasibility_table_tsv": str(output_path),
        "passing_targets_json": str(passing_json),
        "passing_count": int(df["overall_pass"].sum()),
        "total_count": int(df.shape[0]),
    }


def _handle_apply_feasibility_policy(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    table_path = Path(arguments["feasibility_table_tsv"])
    df = pd.read_csv(table_path, sep="\t")
    thresholds = IntegrationThresholds(
        min_property_probes=int(arguments.get("min_property_probes", 20)),
        min_specific_probes=int(arguments.get("min_specific_probes", 10)),
        min_deployment_probes=int(arguments.get("min_deployment_probes", 20)),
        min_paintshop_on_target_mean=float(arguments.get("min_paintshop_on_target_mean", 95.0)),
        max_paintshop_off_target_mean=float(arguments.get("max_paintshop_off_target_mean", 10.0)),
        max_paintshop_off_target_max=float(arguments.get("max_paintshop_off_target_max", 100.0)),
        require_transcript_gate=bool(arguments.get("require_transcript_gate", False)),
    )
    updated = apply_hard_constraints(df, thresholds)
    output_path = Path(arguments.get("output_tsv", runtime.working_dir / "filter_decisions.tsv"))
    ensure_dir(output_path.parent)
    updated.to_csv(output_path, sep="\t", index=False)
    return {
        "filter_decisions_tsv": str(output_path),
        "passing_count": int(updated["overall_pass"].sum()),
        "drop_count": int((~updated["overall_pass"]).sum()),
    }


def _handle_aggregate_reference_panel_ranks(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    result = aggregate_reference_panel_ranks(
        output_dir=arguments.get("output_dir", runtime.working_dir / "reference_panel_rank_aggregation"),
        source_adata_file=arguments.get("source_adata_file"),
        source_rank_file=arguments.get("source_rank_file"),
        reference_adata_files=[str(item) for item in arguments.get("reference_adata_files", [])],
        reference_rank_files=[str(item) for item in arguments.get("reference_rank_files", [])],
        reference_ids=[str(item) for item in arguments.get("reference_ids", [])] or None,
        panel_size=int(arguments.get("panel_size", 64)),
        source_weight=float(arguments.get("source_weight", 0.55)),
        reference_weight=float(arguments.get("reference_weight", 0.45)),
        max_cells=int(arguments.get("max_cells", 50000)),
        seed=int(arguments.get("seed", 42)),
        min_detection_rate=float(arguments.get("min_detection_rate", 0.0)),
        min_reference_support=int(arguments.get("min_reference_support", 1)),
        gene_universe=str(arguments.get("gene_universe", "source")),
        exclude_gene_patterns=[str(item) for item in arguments.get("exclude_gene_patterns", [])],
        restrict_gene_symbols=[str(item) for item in arguments.get("restrict_gene_symbols", [])],
    )
    runtime.session.state["reference_panel_rank_aggregation"] = result["integrated_rank_tsv"]
    runtime.session.state["reference_panel_top_panel"] = result["integrated_top_panel_tsv"]
    runtime.session.state["reference_panel_rank_comparison"] = result["comparison_json"]
    runtime.session_store.save(runtime.session)
    return result


def _handle_run_probedealer_screen_light(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    package_root = _resource_root_from_config(runtime.config)
    result = run_probedealer_screen_light(
        package_root=package_root,
        transcript_fasta=arguments["transcript_fasta"],
        output_dir=arguments.get("output_dir", runtime.working_dir / "probedealer_light"),
        config_overrides=dict(arguments.get("config_overrides", {})),
    )
    return result.to_dict()


def _handle_run_odt_property_screen(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    genes = [str(item).strip() for item in arguments.get("genes", []) if str(item).strip()]
    if not genes:
        raise ValueError("run_odt_property_screen requires a non-empty `genes` list.")
    package_root = _resource_root_from_config(runtime.config)
    result = run_odt_property_screen(
        package_root=package_root,
        genes=genes,
        species=str(arguments.get("species", "mus_musculus")),
        output_dir=arguments.get("output_dir", runtime.working_dir / "odt_property"),
        set_size_min=int(arguments.get("set_size_min", 2)),
    )
    if result.get("output_files", {}).get("summary_tsv"):
        runtime.session.state["odt_summary_tsv"] = result["output_files"]["summary_tsv"]
        runtime.session_store.save(runtime.session)
    return result


def _handle_run_odt_property_batches(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    manifest_tsv = str(arguments.get("manifest_tsv", "")).strip() or str(runtime.session.state.get("probe_candidate_manifest", ""))
    if not manifest_tsv:
        raise ValueError("run_odt_property_batches requires manifest_tsv or a prior probe_candidate_manifest artifact.")
    package_root = _resource_root_from_config(runtime.config)
    result = run_odt_property_batches(
        package_root=package_root,
        manifest_tsv=manifest_tsv,
        species=str(arguments.get("species", "mus_musculus")),
        output_dir=arguments.get("output_dir", runtime.working_dir / "odt_property_batches"),
        batch_size=int(arguments.get("batch_size", 10)),
        max_workers=int(arguments.get("max_workers", 8)),
        set_size_min=int(arguments.get("set_size_min", 2)),
    )
    if result.get("output_files", {}).get("summary_tsv"):
        runtime.session.state["odt_summary_tsv"] = result["output_files"]["summary_tsv"]
        runtime.session_store.save(runtime.session)
    return result


def _handle_build_probe_candidate_manifest(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    output_dir = arguments.get("output_dir", runtime.working_dir / "probe_candidates")
    genes = arguments.get("genes")
    panel_path = arguments.get("panel_path")
    species = str(arguments.get("species", "mus_musculus"))
    if genes is None and not panel_path:
        manifest_path = runtime.session.state.get("smith_selection_manifest")
        if manifest_path:
            panel_path = str(Path(manifest_path).parent / "epoch_0.csv")
    result = build_probe_candidate_manifest(
        output_dir=output_dir,
        species=species,
        genes=genes,
        panel_path=panel_path,
        panel_size=int(arguments.get("panel_size", 64)),
    )
    runtime.session.state["probe_candidate_manifest"] = result["manifest_tsv"]
    runtime.session.state["probe_candidate_transcript_fasta"] = result["transcript_fasta"]
    runtime.session_store.save(runtime.session)
    return result


def _handle_run_oligominer_specificity_screen(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    transcript_fasta = str(arguments.get("transcript_fasta", "")).strip()
    if not transcript_fasta:
        raise ValueError("run_oligominer_specificity_screen requires `transcript_fasta`.")
    package_root = _resource_root_from_config(runtime.config)
    result = run_oligominer_specificity_screen(
        package_root=package_root,
        transcript_fasta=transcript_fasta,
        output_dir=arguments.get("output_dir", runtime.working_dir / "oligominer"),
        temperature_c=int(arguments.get("temperature_c", 42)),
        species=str(arguments.get("species", "mus_musculus")),
    )
    if result.get("output_files", {}).get("summary_tsv"):
        runtime.session.state["oligominer_summary_tsv"] = result["output_files"]["summary_tsv"]
        runtime.session_store.save(runtime.session)
    return result


def _handle_run_probedealer_backend_screen(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    transcript_fasta = str(arguments.get("transcript_fasta", "")).strip()
    if not transcript_fasta:
        raise ValueError("run_probedealer_backend_screen requires `transcript_fasta`.")
    package_root = _resource_root_from_config(runtime.config)
    result = run_probedealer_backend_screen(
        package_root=package_root,
        transcript_fasta=transcript_fasta,
        output_dir=arguments.get("output_dir", runtime.working_dir / "probedealer_backend"),
        use_full_mouse_reference=bool(arguments.get("use_full_mouse_reference", True)),
        use_transcriptome_reference=arguments.get("use_transcriptome_reference"),
        species=str(arguments.get("species", "mus_musculus")),
    )
    if result.get("output_files", {}).get("summary_tsv"):
        runtime.session.state["probedealer_summary_tsv"] = result["output_files"]["summary_tsv"]
        runtime.session_store.save(runtime.session)
    return result


def _handle_run_three_backend_feasibility(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    skip_property_gate = bool(arguments.get("skip_property_gate", False))
    thresholds = IntegrationThresholds(
        min_property_probes=int(arguments.get("min_property_probes", 20 if not skip_property_gate else 0)),
        min_specific_probes=int(arguments.get("min_specific_probes", 10)),
        min_deployment_probes=int(arguments.get("min_deployment_probes", 20)),
        require_transcript_gate=False,
    )
    df = build_three_backend_feasibility_summary(
        manifest_tsv=arguments["manifest_tsv"],
        odt_summary_tsv=arguments.get("odt_summary_tsv"),
        oligominer_summary_tsv=arguments["oligominer_summary_tsv"],
        probedealer_summary_tsv=arguments["probedealer_summary_tsv"],
        thresholds=thresholds,
        skip_property_gate=skip_property_gate,
    )
    outputs = write_three_backend_outputs(df, arguments.get("output_dir", runtime.working_dir / "feasibility_three_backend"))
    runtime.session.state["feasibility_table"] = outputs["integration_summary_tsv"]
    runtime.session.state["passing_targets"] = outputs["passing_targets_json"]
    runtime.session.state["feasibility_summary"] = {
        "integration_summary_tsv": outputs["integration_summary_tsv"],
        "passing_targets_json": outputs["passing_targets_json"],
        "passing_count": int(df["overall_pass"].sum()),
        "total_count": int(df.shape[0]),
    }
    runtime.session_store.save(runtime.session)
    return {
        **outputs,
        "passing_count": int(df["overall_pass"].sum()),
        "total_count": int(df.shape[0]),
    }


def _handle_plot_dataset_umap(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    dataset_path = str(arguments.get("dataset_path", "")).strip()
    if not dataset_path and arguments.get("dataset_id"):
        dataset_id = runtime.registries.resolve_dataset_id(arguments.get("dataset_id"))
        if not dataset_id:
            raise KeyError(f"Unknown dataset: {arguments.get('dataset_id')}")
        dataset_path = runtime.registries.datasets[dataset_id].path
    if not dataset_path:
        active_dataset_id = runtime.session.active_dataset_id
        if active_dataset_id in runtime.registries.datasets:
            dataset_path = runtime.registries.datasets[active_dataset_id].path
        elif active_dataset_id in runtime.session.state.get("mounted_datasets", {}):
            dataset_path = runtime.session.state["mounted_datasets"][active_dataset_id]["path"]
    if not dataset_path:
        raise ValueError("plot_dataset_umap requires a dataset_path or dataset_id.")
    output_path = Path(arguments.get("output_png", runtime.working_dir / "umap.png"))
    rendered = plot_dataset_umap(
        adata_file=dataset_path,
        out_path=output_path,
        color=arguments.get("color"),
        basis=str(arguments.get("basis", "X_umap")),
        max_cells=int(arguments.get("max_cells", 50000)),
        title=arguments.get("title"),
    )
    runtime.session.state["last_umap_plot"] = rendered
    runtime.session_store.save(runtime.session)
    return {"umap_png": rendered}


def _handle_plot_evaluation_summary(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    evaluation_csv = str(arguments.get("evaluation_csv", "")).strip() or str(runtime.session.state.get("cross_dataset_evaluation", ""))
    if not evaluation_csv:
        raise ValueError("plot_evaluation_summary requires an evaluation_csv or a prior cross_dataset_evaluation artifact.")
    output_path = Path(arguments.get("output_png", runtime.working_dir / "evaluation_summary.png"))
    rendered = plot_evaluation_summary(
        evaluation_csv=evaluation_csv,
        out_path=output_path,
        title=str(arguments.get("title", "Cross-Dataset Evaluation Summary")),
    )
    runtime.session.state["last_evaluation_plot"] = rendered
    runtime.session_store.save(runtime.session)
    return {"evaluation_summary_png": rendered}


def _handle_run_smith_selection(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    model_entry = runtime.registries.models["smith_default"]
    entrypoint = _resolve_model_entrypoint(runtime, model_entry)
    smith_root = entrypoint.parent
    output_dir = Path(arguments.get("output_dir", runtime.working_dir / "smith_selection"))
    extra_args = dict(arguments.get("extra_args", {}))
    known_overrides = {
        "learning_rate": float,
        "batch_size": int,
        "dim": int,
        "rep_dim": int,
        "rep_hidden_dims": str,
        "head_hidden_dims": str,
        "dropout_rate": float,
        "lam": float,
        "sigma": float,
        "activation": str,
        "optimizer": str,
    }
    resolved_overrides: dict[str, Any] = {}
    for key, caster in known_overrides.items():
        if key in arguments:
            resolved_overrides[key] = caster(arguments[key])
        elif key in extra_args:
            resolved_overrides[key] = caster(extra_args.pop(key))
    config = SmithSelectionConfig(
        python_executable=str(
            model_entry.metadata.get("python_executable", "python3")
        ),
        smith_root=smith_root,
        adata_file=Path(arguments["adata_file"]),
        saving_dir=output_dir / "saving",
        log_dir=output_dir / "logs",
        tasks=str(arguments["tasks"]),
        task_name=str(arguments["task_name"]),
        panel_size=int(arguments.get("panel_size", 64)),
        epoch=int(arguments.get("epoch", 200)),
        record=int(arguments.get("record", 50)),
        device=str(arguments.get("device", "cpu")),
        seed=int(arguments.get("seed", 42)),
        learning_rate=float(resolved_overrides.get("learning_rate", 0.001)),
        batch_size=int(resolved_overrides.get("batch_size", 32)),
        dim=int(resolved_overrides.get("dim", 32)),
        rep_dim=int(resolved_overrides.get("rep_dim", 32)),
        rep_hidden_dims=str(resolved_overrides.get("rep_hidden_dims", "32")),
        head_hidden_dims=str(resolved_overrides.get("head_hidden_dims", "")),
        dropout_rate=float(resolved_overrides.get("dropout_rate", 0.2)),
        lam=float(resolved_overrides.get("lam", 0.5)),
        sigma=float(resolved_overrides.get("sigma", 0.5)),
        activation=str(resolved_overrides.get("activation", "tanh")),
        optimizer=str(resolved_overrides.get("optimizer", "Adam")),
        extra_args=extra_args,
    )
    result = run_smith_selection(config, execute=bool(arguments.get("execute", False)))
    runtime.session.state["smith_selection_manifest"] = result["manifest_path"]
    runtime.session_store.save(runtime.session)
    return result


def _handle_build_decision_trace(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    selected_panel = pd.read_csv(arguments["selected_panel_tsv"], sep="\t")
    feasibility = pd.read_csv(arguments["feasibility_table_tsv"], sep="\t")
    panel_gene_column = arguments.get("panel_gene_column", selected_panel.columns[0])
    merged = feasibility.merge(
        selected_panel.rename(columns={panel_gene_column: "gene_symbol"}),
        on="gene_symbol",
        how="left",
        suffixes=("", "_panel"),
    )
    merged["status"] = merged["gene_symbol"].isin(selected_panel[panel_gene_column].astype(str)).map(
        {True: "selected", False: "not_selected"}
    )
    merged["final_reason"] = merged["reason"].fillna("no feasibility decision recorded")
    output_path = Path(arguments.get("output_tsv", runtime.working_dir / "decision_trace.tsv"))
    ensure_dir(output_path.parent)
    merged.to_csv(output_path, sep="\t", index=False)
    runtime.session.state["decision_trace"] = str(output_path)
    runtime.session_store.save(runtime.session)
    return {
        "decision_trace_tsv": str(output_path),
        "selected_count": int((merged["status"] == "selected").sum()),
    }


def _handle_evaluate_cross_dataset_panel(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    model_entry = runtime.registries.models["smith_default"]
    entrypoint = _resolve_model_entrypoint(runtime, model_entry)
    smith_root = entrypoint.parent
    results = evaluate_cross_dataset_panel(
        smith_root=smith_root,
        panel_path=arguments["panel_path"],
        train_adata_file=arguments["train_adata_file"],
        test_adata_file=arguments["test_adata_file"],
        panel_size=int(arguments.get("panel_size", 32)),
        label=arguments.get("label"),
        obsm_key=arguments.get("obsm_key", "X_pca"),
        time_label=arguments.get("time_label"),
        n_neighbors=int(arguments.get("n_neighbors", 5)),
    )
    output_path = Path(arguments.get("output_csv", runtime.working_dir / "cross_dataset_evaluation.csv"))
    ensure_dir(output_path.parent)
    results.to_csv(output_path, index=False)
    runtime.session.state["cross_dataset_evaluation"] = str(output_path)
    runtime.session_store.save(runtime.session)
    return {
        "cross_dataset_evaluation_csv": str(output_path),
        "rows": results.to_dict(orient="records"),
    }


def _handle_evaluate_panel_coordinate_regression(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_panel_coordinate_regression(
        adata_file=arguments["adata_file"],
        panel_path=arguments["panel_path"],
        output_dir=arguments.get("output_dir", runtime.working_dir / "coordinate_regression"),
        panel_size=int(arguments.get("panel_size", 64)),
        test_size=float(arguments.get("test_size", 0.2)),
        seed=int(arguments.get("seed", 42)),
        alpha=float(arguments.get("alpha", 1.0)),
    )
    runtime.session.state["coordinate_regression"] = str(Path(arguments.get("output_dir", runtime.working_dir / "coordinate_regression")) / "coordinate_regression_result.json")
    runtime.session_store.save(runtime.session)
    return result.to_dict()


def _handle_evaluate_panel_cell_type_classification(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_panel_cell_type_classification(
        adata_file=arguments["adata_file"],
        panel_path=arguments["panel_path"],
        output_dir=arguments.get("output_dir", runtime.working_dir / "cell_type_classification"),
        panel_size=int(arguments.get("panel_size", 64)),
        label_column=str(arguments.get("label_column", "Cell_Type")),
        test_size=float(arguments.get("test_size", 0.2)),
        seed=int(arguments.get("seed", 42)),
        max_iter=int(arguments.get("max_iter", 1000)),
        class_weight=arguments.get("class_weight", "balanced"),
    )
    runtime.session.state["cell_type_classification"] = str(
        Path(arguments.get("output_dir", runtime.working_dir / "cell_type_classification"))
        / "cell_type_classification_result.json"
    )
    runtime.session_store.save(runtime.session)
    return result.to_dict()


def _handle_build_run_report(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    manifest_path = str(arguments.get("manifest_path", "")).strip() or str(runtime.session.state.get("smith_selection_manifest", ""))
    if not manifest_path:
        raise ValueError("build_run_report requires a manifest_path or a prior smith_selection_manifest artifact.")
    evaluation_csv = str(arguments.get("evaluation_csv", "")).strip() or str(runtime.session.state.get("cross_dataset_evaluation", ""))
    train_adata_file = str(arguments.get("train_adata_file", "")).strip()
    test_adata_file = str(arguments.get("test_adata_file", "")).strip()

    active_dataset_id = runtime.session.active_dataset_id
    if not train_adata_file:
        if active_dataset_id in runtime.registries.datasets:
            train_adata_file = runtime.registries.datasets[active_dataset_id].path
        elif active_dataset_id in runtime.session.state.get("mounted_datasets", {}):
            train_adata_file = runtime.session.state["mounted_datasets"][active_dataset_id]["path"]
    if not test_adata_file:
        mention_ids = runtime.session.state.get("last_dataset_mentions", [])
        mounted = runtime.session.state.get("mounted_datasets", {})
        if len(mention_ids) >= 2 and mention_ids[1] in mounted:
            test_adata_file = mounted[mention_ids[1]]["path"]
        elif "smith_st_processed" in runtime.registries.datasets:
            test_adata_file = runtime.registries.datasets["smith_st_processed"].path

    report_paths = build_run_report(
        output_dir=Path(arguments.get("output_dir", runtime.working_dir / "report")),
        title=str(arguments.get("title", "Smith-Agent Run Report")),
        manifest_path=manifest_path,
        panel_size=int(arguments.get("panel_size", 64)),
        evaluation_csv=evaluation_csv or None,
        feasibility_summary=runtime.session.state.get("feasibility_summary") if isinstance(runtime.session.state.get("feasibility_summary"), dict) else None,
        train_adata_file=train_adata_file or None,
        test_adata_file=test_adata_file or None,
        train_color=str(arguments.get("train_color", "pathology")),
        test_color=str(arguments.get("test_color", "pathology")),
    )
    runtime.session.state["last_report"] = report_paths
    runtime.session_store.save(runtime.session)
    return report_paths


def _handle_export_manifest(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "session_id": runtime.session.session_id,
        "state": runtime.session.state,
        "extra_artifacts": dict(arguments.get("extra_artifacts", {})),
    }
    output_path = Path(arguments.get("output_json", runtime.working_dir / "deliverable_manifest.json"))
    write_json(output_path, payload)
    return {
        "deliverable_manifest_json": str(output_path),
    }


def build_default_tool_registry(runtime) -> ToolRegistry:
    tools = []
    for entry in runtime.registries.tools.values():
        handler_name = entry.metadata.get("handler")
        if not handler_name:
            continue
        handler = {
            "list_skills": _handle_list_skills,
            "list_tools": _handle_list_tools,
            "list_datasets": _handle_list_datasets,
            "list_models": _handle_list_models,
            "search_smith_agent": _handle_search_smith_agent,
            "set_active_dataset": _handle_set_active_dataset,
            "set_active_model": _handle_set_active_model,
            "inspect_session_state": _handle_inspect_session_state,
            "describe_skill": _handle_describe_skill,
            "resolve_dataset_context": _handle_resolve_dataset_context,
            "score_reference_transferability": _handle_score_reference_transferability,
            "query_cellxgene_metadata": _handle_query_cellxgene_metadata,
            "materialize_cellxgene_dataset": _handle_materialize_cellxgene_dataset,
            "aggregate_feasibility_results": _handle_aggregate_feasibility_results,
            "apply_feasibility_policy": _handle_apply_feasibility_policy,
            "aggregate_reference_panel_ranks": _handle_aggregate_reference_panel_ranks,
            "build_probe_candidate_manifest": _handle_build_probe_candidate_manifest,
            "run_probedealer_screen_light": _handle_run_probedealer_screen_light,
            "run_odt_property_screen": _handle_run_odt_property_screen,
            "run_odt_property_batches": _handle_run_odt_property_batches,
            "run_oligominer_specificity_screen": _handle_run_oligominer_specificity_screen,
            "run_probedealer_backend_screen": _handle_run_probedealer_backend_screen,
            "run_three_backend_feasibility": _handle_run_three_backend_feasibility,
            "plot_dataset_umap": _handle_plot_dataset_umap,
            "plot_evaluation_summary": _handle_plot_evaluation_summary,
            "run_smith_selection": _handle_run_smith_selection,
            "build_decision_trace": _handle_build_decision_trace,
            "evaluate_cross_dataset_panel": _handle_evaluate_cross_dataset_panel,
            "evaluate_panel_coordinate_regression": _handle_evaluate_panel_coordinate_regression,
            "evaluate_panel_cell_type_classification": _handle_evaluate_panel_cell_type_classification,
            "build_run_report": _handle_build_run_report,
            "export_manifest": _handle_export_manifest,
        }[handler_name]
        tools.append(
            ToolSpec(
                name=entry.name,
                description=entry.description,
                input_schema=entry.input_schema,
                outputs=entry.outputs,
                handler=handler,
            )
        )
    return ToolRegistry(tools)
